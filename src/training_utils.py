'''
训练工具模块

提供训练相关的工具类和函数，供其他训练脚本导入使用：
- StepLRScheduler: 两段式学习率调度器（预热 + 退火，按步更新）
- GradientMonitor: 梯度监控器
- DynamicWeightedBCE: 动态加权BCE损失函数
- EarlyStopping: 早停机制
- evaluate_model: 模型评估函数
- save_model_with_metadata: 带元数据的模型保存
- print_dispersion_sparkline: 预测值分布可视化
- create_scheduler_from_config: 根据配置创建学习率调度器
'''

import os,torch,torch.nn as nn,numpy as np,math
import contextlib
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import DataConfig,LossConfig,TrainingConfig,RegularizationConfig
import torch.optim as optim

class StepLRScheduler:
    """
    两段式学习率调度器：预热 → 退火，每个优化步（batch）更新一次。

    - 预热：start_lr 线性升到 peak_lr，共 warmup_steps 步
    - 退火：peak_lr 按 cosine / linear 降到 final_lr，共 decay_steps 步
    两段步数之和等于总步数时，学习率正好在最后一步落到 final_lr。

    多 param_group 按构造瞬间各组学习率相对基准组的比例同步缩放：
    例如 Embedding 组是基准组的 0.2 倍，则全程保持 0.2 倍。
    """

    def __init__(self, optimizer, start_lr, peak_lr, final_lr,
                 warmup_steps, decay_steps, shape='cosine'):
        if shape not in ('cosine', 'linear'):
            raise ValueError(f"退火曲线只支持 'cosine' / 'linear'，当前 {shape!r}")
        if peak_lr <= 0:
            raise ValueError("peak_lr 必须为正数")
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.decay_steps = max(0, int(decay_steps))
        self.shape = shape

        base = optimizer.param_groups[0]['lr']
        self.group_scales = [g['lr'] / base if base else 1.0 for g in optimizer.param_groups]
        self.last_step = -1

    def lr_at(self, step):
        """返回第 step 步的基准学习率（不含 param_group 倍率）"""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            t = step / self.warmup_steps
            return self.start_lr + (self.peak_lr - self.start_lr) * t
        if self.decay_steps <= 0:
            return self.peak_lr
        u = min(max((step - self.warmup_steps) / self.decay_steps, 0.0), 1.0)
        factor = (1.0 - u) if self.shape == 'linear' else 0.5 * (1.0 + math.cos(math.pi * u))
        return self.final_lr + (self.peak_lr - self.final_lr) * factor

    def phase_name(self, step):
        return '预热' if (self.warmup_steps > 0 and step < self.warmup_steps) else '退火'

    def step(self, step):
        """把第 step 步的学习率写入各 param_group，返回基准组学习率"""
        lr = self.lr_at(step)
        for group, scale in zip(self.optimizer.param_groups, self.group_scales):
            group['lr'] = lr * scale
        self.last_step = step
        return lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def _draw_sparkline(data, num_bins=20):
    """绘制单行终端直方图并返回统计信息"""
    data = np.array(data)
    counts, _ = np.histogram(data, bins=num_bins, range=(0, 1))
    max_count = max(counts) if max(counts) > 0 else 1
    
    chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    hist_line = ""
    for count in counts:
        idx = int(count / max_count * (len(chars) - 1))
        idx = min(max(idx, 0), len(chars) - 1)
        hist_line += chars[idx]

    mean = float(np.mean(data)) if len(data) > 0 else 0
    std = float(np.std(data)) if len(data) > 0 else 0
    min_val = float(np.min(data)) if len(data) > 0 else 0
    max_val = float(np.max(data)) if len(data) > 0 else 0

    return hist_line, mean, std, min_val, max_val, len(data)


def print_dispersion_sparkline(all_preds, epoch_returns_history=None, all_targets=None):
    """
    打印预测值在0-1区间上的分布直方图（终端字符可视化）

    Args:
        all_preds: 所有样本的预测值数组
        epoch_returns_history: 历史epoch记录列表（用于显示趋势）
        all_targets: 样本标签数组（0/1），若提供则额外显示正负样本各自的分布
    """
    all_preds = np.array(all_preds)

    pos_ratio = float(np.mean(all_preds >= 0.5)) * 100
    high_conf_ratio = float(np.mean(all_preds >= 0.7)) * 100

    def _print_block(title, data):
        hist_line, mean, std, min_val, max_val, total = _draw_sparkline(data)
        print(f' 【{title}】  0.0  {hist_line}  1.0   均值{mean:.3f},标准差{std:.4f},范围[{min_val:.3f},{max_val:.3f}],N={total}')

    _print_block('总样本预测值分布', all_preds)

    if all_targets is not None:
        all_targets = np.array(all_targets)
        pos_preds = all_preds[all_targets == 1]
        neg_preds = all_preds[all_targets == 0]

        if len(pos_preds) > 0:
            _print_block('正样本预测值分布', pos_preds)
        else:
            print(f' 【正样本预测值分布】（无正样本）')

        if len(neg_preds) > 0:
            _print_block('负样本预测值分布', neg_preds)
        else:
            print(f' 【负样本预测值分布】（无负样本）')

    # ─── 趋势 ───
    if epoch_returns_history and len(epoch_returns_history) >= 2:
        stds = [e.get('dispersion_std', 0) for e in epoch_returns_history]
        returns = [e.get('return', 0) for e in epoch_returns_history]
        
        window_size = min(10, len(stds))
        baseline_std = np.mean(stds[-window_size:-1]) if window_size > 1 else stds[-2]
        baseline_return = np.mean(returns[-window_size:-1]) if window_size > 1 else returns[-2]
        
        std_change = (stds[-1] - baseline_std) / baseline_std * 100 if baseline_std > 1e-6 else 0
        return_change = (returns[-1] - baseline_return) / abs(baseline_return) * 100 if abs(baseline_return) > 1e-6 else 0
        
        if std_change < -20:
            status = "⚠️ 分散度下降"
        elif std_change > 10:
            status = "📈 分散度上升"
        else:
            status = "➡️ 分散度稳定"
        
        print(f'    趋势: {status} ({std_change:+.1f}%) | 收益率变化: ({return_change:+.1f}%)')

def create_eval_criterion(eval_targets):
    """
    统一构造评估损失函数（与 train.py 训练末段测试集损失一致的口径）

    训练与诊断脚本过去各自复制粘贴同一段"按 LossConfig.LOSS_TYPE 选损失类、
    按正负比例校准权重"的样板代码，新增 LossConfig.LOSS_TYPE 分支时极易遗漏。
    统一入口让新增分支只动一处。

    Args:
        eval_targets: 评估集标签序列（标量/数组均可，内部统一转 np.array）

    Returns:
        配好权重的损失函数实例：
        - dynamic_bce / pairwise_bce → DynamicWeightedBCE
        - balanced_bce → BalancedBCE

    Raises:
        ValueError: 当 LossConfig.LOSS_TYPE 不在已知分支中
    """
    targets_arr = np.asarray(eval_targets)
    if LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        pos_count = int(np.sum(targets_arr >= 0.5))
        neg_count = int(np.sum(targets_arr < 0.5))
        if pos_count > 0 and neg_count > 0:
            neg_w = LossConfig.POS_WEIGHT * (pos_count / neg_count)
        elif pos_count == 0:
            neg_w = float(LossConfig.POS_WEIGHT)
        else:
            neg_w = 0.1
        criterion.weight_0_0.fill_(neg_w)
        return criterion
    if LossConfig.LOSS_TYPE.lower() == 'balanced_bce':
        criterion = BalancedBCE(reduction='mean')
        criterion.update_weights(targets_arr)
        return criterion
    if LossConfig.LOSS_TYPE.lower() == 'pairwise_bce':
        # Pairwise 仅用于训练；评估口径与 train.py 旧版一致：纯 DynamicWeightedBCE
        # 按正负比例填 neg_weight。
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        pos_count = int(np.sum(targets_arr >= 0.5))
        neg_count = int(np.sum(targets_arr < 0.5))
        if pos_count > 0 and neg_count > 0:
            neg_w = LossConfig.POS_WEIGHT * (pos_count / neg_count)
        elif pos_count == 0:
            neg_w = float(LossConfig.POS_WEIGHT)
        else:
            neg_w = 0.1
        criterion.weight_0_0.fill_(neg_w)
        return criterion
    raise ValueError(
        f"未知 LOSS_TYPE: {LossConfig.LOSS_TYPE} "
        "(支持: dynamic_bce / pairwise_bce / balanced_bce)")


class DynamicWeightedBCE(nn.Module):
    """
    动态加权BCE损失函数：按标签桶分配权重
    - 标签1.0固定权重4.0
    - 标签0.6/0.3/0.0按样本数量动态分配权重（样本少=权重高）
    """
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(DynamicWeightedBCE, self).__init__()
        self.reduction = reduction
        
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        
        self.register_buffer('weight_0_6', torch.tensor(1.0))
        self.register_buffer('weight_0_3', torch.tensor(1.0))
        self.register_buffer('weight_0_0', torch.tensor(1.0))
        
    def update_weights(self, targets):
        """
        二分类动态权重：根据正负样本比例动态调整
        targets: [batch_size] 标签 (1.0/0.0)
        """
        if isinstance(targets, torch.Tensor):
            targets = targets.float().cpu().numpy()
        
        count_positive = np.sum(targets >= 0.5)
        count_negative = np.sum(targets < 0.5)
        
        if count_positive > 0 and count_negative > 0:
            neg_weight = float(self.pos_weight) * (count_positive / count_negative)
            self.weight_0_0.fill_(neg_weight)
        elif count_positive == 0:
            self.weight_0_0.fill_(float(self.pos_weight))
        else:
            self.weight_0_0.fill_(0.1)
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, 1] 模型输出的logits
        targets: [batch_size] 真实标签 (1.0/0.0)
        """
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)

        inputs_fp32 = inputs.float()
        targets_fp32 = targets.float()

        loss = F.binary_cross_entropy_with_logits(inputs_fp32, targets_fp32, reduction='none')
        
        pos_weight = self.pos_weight.to(dtype=loss.dtype, device=loss.device)
        neg_weight = self.weight_0_0.to(dtype=loss.dtype, device=loss.device)

        weights = torch.where(targets_fp32 >= 0.5, pos_weight, neg_weight)
        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedBCE(nn.Module):
    """
    平衡二元交叉熵：正负样本各占一半损失贡献
    w_pos = (a+b) / (2a),  w_neg = (a+b) / (2b)
    使得 weighted_loss 的总量纲与标准 BCE 一致（均值≈1）
    """
    def __init__(self, reduction='mean'):
        super(BalancedBCE, self).__init__()
        self.reduction = reduction
        self.register_buffer('pos_weight', torch.tensor(1.0))
        self.register_buffer('neg_weight', torch.tensor(1.0))

    def update_weights(self, targets):
        if isinstance(targets, torch.Tensor):
            targets = targets.float().cpu().numpy()
        pos_count = np.sum(targets >= 0.5)
        neg_count = np.sum(targets < 0.5)
        total = pos_count + neg_count
        if pos_count > 0 and neg_count > 0:
            self.pos_weight.fill_(total / (2.0 * pos_count))
            self.neg_weight.fill_(total / (2.0 * neg_count))
        else:
            self.pos_weight.fill_(1.0)
            self.neg_weight.fill_(1.0)

    def forward(self, inputs, targets):
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)
        inputs_fp32 = inputs.float()
        targets_fp32 = targets.float()
        loss = F.binary_cross_entropy_with_logits(inputs_fp32, targets_fp32, reduction='none')
        pw = self.pos_weight.to(dtype=loss.dtype, device=loss.device)
        nw = self.neg_weight.to(dtype=loss.dtype, device=loss.device)
        weights = torch.where(targets_fp32 >= 0.5, pw, nw)
        loss = loss * weights
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class PairwiseWeightedBCE(DynamicWeightedBCE):
    """
    动态加权BCE + Top-K%区域Pairwise排序损失

    继承DynamicWeightedBCE的动态加权机制，额外在模型预测的Top K%区域内
    构建正负对，使用RankNet风格的排序损失锐化排序边界。

    训练前warmup_epochs轮仅使用BCE，之后渐进引入Pairwise分量。
    接口与DynamicWeightedBCE完全一致（update_weights/forward），
    仅新增set_epoch用于控制warmup。
    """

    def __init__(self, pos_weight=4.0, reduction='mean',
                 pairwise_weight=0.5, pairwise_top_k=0.10,
                 pairwise_pos_weight=2.0, warmup_epochs=8,
                 sigma=1.0, num_neg=2):
        super().__init__(pos_weight=pos_weight, reduction=reduction)

        # Pairwise参数
        self.pairwise_weight = pairwise_weight
        self.pairwise_top_k = pairwise_top_k
        self.pairwise_pos_weight = pairwise_pos_weight
        self.sigma = sigma
        self.num_neg = num_neg
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前epoch，用于控制Pairwise的渐进引入"""
        self._current_epoch = epoch

    @property
    def pairwise_active(self):
        """Pairwise分量是否已激活"""
        return self._current_epoch >= self.warmup_epochs

    def _compute_pairwise(self, logits, targets):
        """
        Top-K%区域Pairwise排序损失（RankNet风格）

        在模型预测的Top K%样本中，选取正样本与负样本构建pair，
        优化目标：正样本的分数应高于负样本。

        Args:
            logits: [B] 模型原始输出（未经sigmoid）
            targets: [B] 真实标签（0.0/1.0）

        Returns:
            scalar loss，或None（无有效pair时）
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            k = max(2, int(len(probs) * self.pairwise_top_k))
            _, top_indices = torch.topk(probs, k)

        top_logits = logits[top_indices]
        top_targets = targets[top_indices]

        pos_mask = top_targets >= 0.5
        neg_mask = ~pos_mask

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        if n_pos == 0 or n_neg == 0:
            return None

        pos_logits = top_logits[pos_mask]
        neg_logits = top_logits[neg_mask]

        # 每个正样本随机配对num_neg个负样本（控制计算量 + 引入随机性）
        n_sample = min(self.num_neg, n_neg)
        perm = torch.randperm(n_neg, device=logits.device)[:n_sample]
        neg_sampled = neg_logits[perm]

        # RankNet: L = -log(σ(σ * (s_pos - s_neg)))
        # 广播: [n_pos, 1] - [1, n_sample]
        diff = self.sigma * (pos_logits.unsqueeze(1) - neg_sampled.unsqueeze(0))
        loss = -torch.log(torch.sigmoid(diff) + 1e-8)

        return loss.mean() * self.pairwise_pos_weight

    def forward(self, inputs, targets):
        """
        inputs: [batch_size] 或 [batch_size, 1] 模型输出的logits
        targets: [batch_size] 真实标签 (1.0/0.0)
        """
        # BCE部分（父类计算）
        bce_loss = super().forward(inputs, targets)

        if not self.pairwise_active:
            return bce_loss

        # 准备pairwise输入（与父类相同的squeeze处理）
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)

        pw_loss = self._compute_pairwise(inputs.float(), targets.float())
        if pw_loss is None:
            return bce_loss

        return bce_loss + self.pairwise_weight * pw_loss


def _rankdata(x):
    """平均秩（并列取平均），等价于 scipy.stats.rankdata(method='average')。

    自己实现是为了让评估链路不强依赖 scipy：评估脚本要能在任意环境跑起来。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (ranks[order[i]] + ranks[order[j]])
        i = j + 1
    return ranks


def _spearman(a, b):
    """Spearman 秩相关。样本不足或某一列为常数时返回 nan —— 不静默返回 0。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 5:
        return float('nan')
    ra, rb = _rankdata(a), _rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def _newey_west_t(x, lag=2):
    """序列均值相对 0 的 t 值，方差用 Newey-West 修正（处理重叠收益的自相关）。

    持有期 FUTURE_DAYS=3 意味着相邻三天的样本共用同一段行情，IC 序列必然自相关，
    普通标准误会低估、t 值虚高。lag 取 FUTURE_DAYS-1 是常规做法。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return float('nan')
    xc = x - x.mean()
    g0 = float(np.dot(xc, xc) / n)
    var = g0
    for l in range(1, min(lag, n - 1) + 1):
        gl = float(np.dot(xc[:-l], xc[l:]) / n)
        var += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    if var <= 1e-18:
        return float('nan')
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 1e-15 else float('nan')


def daily_discriminative_stats(all_preds, all_targets, all_returns, day_indices,
                               top_pct=1.0, nw_lag=2):
    """逐日口径的判别力指标。与 pooled 口径的区别见下方说明。

    为什么要单独算一份：
        roc_auc_score(all_targets, all_preds) 把整个评估期的样本拼在一起算一次，
        它把「哪几天整体好」也算成了模型的判别力。但选股能用的只有
        「同一天里，买的那几只和没买的有没有区别」—— 跨日的部分无法交易。

        实测（eval_protocol.py --trap，2026H1）：
          只有跨日信号的打分器  pooled AUC 0.6197 / 日均超额 +0.08%
          只有截面信号的打分器  pooled AUC 0.6464 / 日均超额 +1.53%
        两者 pooled 只差 0.027，实际收益差 19 倍。pooled 口径会掩盖这个区别。

    返回：
        daily_auc             逐日 AUC 的平均（每天单独算一次再平均）
        ic_mean               逐日 Spearman(预测, 未来收益) 的平均，即 IC
        icir                  ic_mean / IC 的日间标准差，衡量稳定性
        ic_t                  ic_mean 的 t 值（Newey-West 修正后）
        daily_precision       逐日 Top top_pct% 内正样本比例的平均
        n_days                参与统计的天数
    """
    out = {'daily_auc': float('nan'), 'ic_mean': float('nan'), 'icir': float('nan'),
           'ic_t': float('nan'), 'daily_precision': float('nan'), 'n_days': 0}

    days = np.asarray(day_indices)
    if days.size == 0 or days.size != np.asarray(all_preds).size:
        return out

    preds = np.asarray(all_preds, dtype=np.float64)
    targets = np.asarray(all_targets, dtype=np.float64)
    rets = np.asarray(all_returns, dtype=np.float64)

    aucs, ics, precs = [], [], []
    for day in np.unique(days):
        m = days == day
        p, y, r = preds[m], targets[m], rets[m]
        ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(r)
        p, y, r = p[ok], y[ok], r[ok]
        if p.size < 20:            # 当天样本太少，AUC/IC 都是噪声
            continue
        if (y > 0).sum() >= 3 and (y <= 0).sum() >= 3:
            try:
                aucs.append(roc_auc_score(y, p))
            except ValueError:
                pass
        ic = _spearman(p, r)
        if np.isfinite(ic):
            ics.append(ic)
        k = max(1, int(p.size * top_pct / 100.0))
        precs.append(float(np.mean(y[np.argsort(p)[::-1][:k]])))

    out['n_days'] = len(ics)
    if aucs:
        out['daily_auc'] = float(np.mean(aucs))
    if precs:
        out['daily_precision'] = float(np.mean(precs))
    if len(ics) >= 2:
        arr = np.asarray(ics, dtype=np.float64)
        mean, sd = float(arr.mean()), float(arr.std(ddof=1))
        out['ic_mean'] = mean
        out['icir'] = mean / sd if sd > 1e-12 else float('nan')
        out['ic_t'] = _newey_west_t(arr, lag=nw_lag)
    return out


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns,
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name="", eval_day_indices=None, top_n_per_day=None, eval_daily_returns=None,
                   criterion=None, enable_portfolio_simulation=False, tradeable_mask=None,
                   predict_fn=None):
    """
    模型评估函数
    涨停样本已在向量化批处理中过滤，无需再次过滤

    分批处理，减少显存占用

    Args:
        predict_fn: 可选的自定义推理函数 (model, eval_inputs, device) -> preds(np.ndarray)。
                    提供时（如 MC Dropout 推理）预测由它完成，损失仍用 eval 模式确定性前向计算；
                    为 None 时走默认单次前向推理。

    返回统计字典，包含：
        auc：AUC得分
        top_return：Top1%收益率
        top_count：Top1%样本数
        top_threshold：Top1%最低置信度
        high_conf_count：高置信(>0.7)样本数
        low_conf_count：低置信(<0.2)样本数
        pred_mean：预测均值
        pred_std：预测标准差
        filtered_count：被过滤的涨停样本数
        realistic_stats：实战收益率统计（如果提供了eval_day_indices）
        test_loss：测试集损失（如果提供了criterion）
    """
    model.eval()

    num_samples = len(eval_inputs)
    if num_samples == 0:
        return {
            'auc': 0.5, 'top_return': 0.0, 'daily_top_return': None, 'top_count': 0, 'top_threshold': 0.0,
            'high_conf_count': 0, 'low_conf_count': 0, 'pred_mean': 0.0,
            'pred_std': 0.0, 'filtered_count': 0, 'realistic_stats': None,
            'test_loss': None,
            'daily_auc': float('nan'), 'ic_mean': float('nan'), 'icir': float('nan'),
            'ic_t': float('nan'), 'daily_precision': float('nan'), 'eval_days': 0,
        }

    all_preds = []
    total_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    amp_ctx = _get_amp_context(device)

    if predict_fn is not None:
        # 自定义推理路径（如 MC Dropout）：predict_fn 内部自行管理 train/eval 模式
        all_preds = predict_fn(model, eval_inputs, device)
        if criterion is not None:
            model.eval()  # 损失用确定性 eval 前向计算
            with torch.no_grad():
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)

                    batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
                    batch_targets = torch.tensor(eval_targets[start_idx:end_idx], dtype=torch.float32, device=device)
                    with amp_ctx:
                        logits = model(batch_inputs)
                    logits = logits.float()
                    loss = criterion(logits.squeeze(-1), batch_targets)
                    total_loss += loss.item() * (end_idx - start_idx)
                    del batch_inputs, batch_targets, logits
    else:
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
                with amp_ctx:
                    logits = model(batch_inputs)
                logits = logits.float()
                scores = torch.sigmoid(logits)
                all_preds.append(scores.cpu().numpy().flatten())

                if criterion is not None:
                    batch_targets = torch.tensor(eval_targets[start_idx:end_idx], dtype=torch.float32, device=device)
                    loss = criterion(logits.squeeze(-1), batch_targets)
                    total_loss += loss.item() * (end_idx - start_idx)
                    del batch_targets

                del batch_inputs, logits

        all_preds = np.concatenate(all_preds)
    all_targets = np.array(eval_targets)
    all_returns = np.array(eval_cumulative_returns)

    # 过滤：排除T+1开盘价接近涨停的不可交易样本
    filtered_count = 0
    if tradeable_mask is not None and not np.all(tradeable_mask):
        filtered_count = int(np.sum(~tradeable_mask))
        all_preds = all_preds[tradeable_mask]
        all_targets = all_targets[tradeable_mask]
        all_returns = all_returns[tradeable_mask]
        if eval_day_indices is not None:
            eval_day_indices = [d for d, m in zip(eval_day_indices, tradeable_mask) if m]
        if eval_daily_returns is not None:
            eval_daily_returns = [r for r, m in zip(eval_daily_returns, tradeable_mask) if m]

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    # 逐日口径判别力（daily AUC / IC / ICIR）。
    # pooled AUC 把「哪几天整体好」也算成判别力，那部分对选股不可用；
    # 这是 AUC 高但收益低的已知成因之一，所以两个口径必须同时报。
    # 没有日期索引时无法逐日分组 —— 明确留 nan，不退回 pooled 冒充逐日值。
    # percent 必须在这之前定义：下面的逐日指标与后面的 Top-K% 统计共用同一个口径。
    percent = DataConfig.TOP_K
    _nw_lag = max(1, getattr(DataConfig, 'FUTURE_DAYS', 3) - 1)
    if eval_day_indices is not None:
        _dd = daily_discriminative_stats(all_preds, all_targets, all_returns,
                                         eval_day_indices, top_pct=percent,
                                         nw_lag=_nw_lag)
    else:
        _dd = {'daily_auc': float('nan'), 'ic_mean': float('nan'),
               'icir': float('nan'), 'ic_t': float('nan'),
               'daily_precision': float('nan'), 'n_days': 0}

    sorted_indices = np.argsort(all_preds)[::-1]

    # 头部分类性：Top K% 样本的正样本比例（Precision@K%）
    base_positive_rate = float(np.mean(all_targets))
    precision_at = {'base_positive_rate': base_positive_rate}
    for top_pct in [10, 3, 1]:
        k = max(10, int(len(all_preds) * top_pct / 100))
        subset_targets = all_targets[sorted_indices[:k]]
        precision_at[f'precision_top{top_pct}'] = float(np.mean(subset_targets))

    top_k = max(1, int(len(all_preds) * percent / 100))
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    top_return = np.mean(top_returns)
    top_threshold = all_preds[sorted_indices[top_k - 1]]

    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2

    test_loss = total_loss / num_samples if criterion is not None else None

    daily_top_return = None
    if eval_day_indices is not None:
        all_day_idx = np.array(eval_day_indices)
        unique_days = np.unique(all_day_idx)
        daily_top_indices_list = []
        for day in unique_days:
            day_mask = all_day_idx == day
            day_preds = all_preds[day_mask]
            day_local_k = max(1, int(len(day_preds) * percent / 100))
            day_sorted = np.argsort(day_preds)[::-1]
            day_top_local = day_sorted[:day_local_k]
            day_global_indices = np.where(day_mask)[0][day_top_local]
            daily_top_indices_list.append(day_global_indices)
        if daily_top_indices_list:
            all_daily_top_indices = np.concatenate(daily_top_indices_list)
            daily_top_return = float(np.mean(all_returns[all_daily_top_indices]))

    stats = {
        'auc': auc,
        **precision_at,
        # 逐日口径：与上面的 pooled auc 并列，哪个是「能用的判别力」看这一行
        'daily_auc': _dd['daily_auc'],
        'ic_mean': _dd['ic_mean'],
        'icir': _dd['icir'],
        'ic_t': _dd['ic_t'],
        'daily_precision': _dd['daily_precision'],
        'eval_days': _dd['n_days'],
        'top_return': top_return,
        'daily_top_return': daily_top_return,
        'top_count': top_k,
        'top_threshold': top_threshold,
        'high_conf_count': np.sum(high_conf),
        'low_conf_count': np.sum(low_conf),
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
        'filtered_count': filtered_count,
        'dispersion_std': float(np.std(all_preds)),
        'dispersion_range': float(np.max(all_preds) - np.min(all_preds)),
        'dispersion_iqr': float(np.percentile(all_preds, 75) - np.percentile(all_preds, 25)),
        'test_loss': test_loss,
    }

    if eval_day_indices is not None:
        actual_top_n = top_n_per_day if top_n_per_day is not None else DataConfig.TOP_N_PER_DAY
        if actual_top_n == 0:
            actual_top_n = None
        stats['realistic_stats'] = calculate_realistic_return(all_preds, all_returns, eval_day_indices, percent, actual_top_n)
        
        if enable_portfolio_simulation and eval_daily_returns is not None:
            stats['portfolio_stats'] = calculate_portfolio_simulation(
                all_preds, all_returns, eval_daily_returns, eval_day_indices, percent, actual_top_n)
        else:
            stats['portfolio_stats'] = None
    else:
        stats['realistic_stats'] = None
        stats['portfolio_stats'] = None

    stats['all_preds'] = all_preds
    stats['all_targets'] = all_targets
    return stats


def calculate_realistic_return(all_preds, all_returns, all_day_indices, top_percent=1.0, top_n_per_day=None):
    """
    计算实战收益率（高仿实战）
    
    支持两种模式:
    1. 全局阈值模式（top_n_per_day=None）: 按全局Top%确定阈值，每天选超过阈值的股票
    2. 每日Top N模式（top_n_per_day指定）: 每天选预测分数最高的前N只股票
    """
    unique_days = np.unique(all_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    daily_returns = []
    
    if top_n_per_day is not None:
        for day in unique_days:
            day_mask = all_day_indices == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) == 0:
                daily_stats.append((0, 0.0))
                continue
            
            day_preds = all_preds[day_indices]
            day_returns = all_returns[day_indices]
            
            sorted_local_indices = np.argsort(day_preds)[::-1]
            select_count = min(top_n_per_day, len(day_indices))
            top_local_indices = sorted_local_indices[:select_count]
            
            day_return = np.mean(day_returns[top_local_indices])
            daily_returns.append(day_return)
            daily_stats.append((select_count, day_return))
        
        threshold = None
    else:
        top_k = max(1, int(len(all_preds) * top_percent / 100))
        sorted_indices = np.argsort(all_preds)[::-1]
        threshold = all_preds[sorted_indices[top_k - 1]]
        
        above_threshold_mask = all_preds > threshold
        max_select = DataConfig.MAX_SELECT_PER_DAY
        
        for day in unique_days:
            day_mask = all_day_indices == day
            day_above_threshold = above_threshold_mask & day_mask
            day_indices = np.where(day_above_threshold)[0]
            
            count = len(day_indices)
            if count > 0:
                if max_select > 0 and count > max_select:
                    day_preds = all_preds[day_indices]
                    top_local = np.argsort(day_preds)[::-1][:max_select]
                    selected_indices = day_indices[top_local]
                    count = max_select
                else:
                    selected_indices = day_indices
                day_return = np.mean(all_returns[selected_indices])
                daily_returns.append(day_return)
                daily_stats.append((count, day_return))
            else:
                daily_stats.append((0, 0.0))
    
    if len(daily_returns) > 0:
        avg_realistic_return = np.mean(daily_returns)
        cumulative_return = np.sum(daily_returns)
    else:
        avg_realistic_return = 0.0
        cumulative_return = 0.0
    
    return {
        'threshold': threshold,
        'daily_stats': daily_stats,
        # daily_stats 的第 i 项对应 unique_days[i]（按升序排列的预测日 day_index）
        # 供 run.py 将每个交易日映射回日历日期、并按 --begin 过滤
        'day_indices': unique_days.tolist(),
        'cumulative_return': cumulative_return,
        'valid_days': len(daily_returns),
        'avg_realistic_return': avg_realistic_return,
        'mode': 'top_n_per_day' if top_n_per_day else 'global_threshold'
    }


def calculate_portfolio_simulation(all_preds, all_cumulative_returns, all_daily_returns,
                                    all_day_indices, top_percent=1.0, top_n_per_day=None):
    """
    实战资金管理模拟（逐日滚动买卖）

    模拟A股实盘规则：
    - 每天收盘后模型打分，生成推荐列表
    - 次日开盘：用开盘前可用资金买入推荐股票（受 MAX_BUY_PER_DAY 限制）
    - 开盘/收盘：所有卖出资金一律次日可用（买入以开盘价成交，与卖出同时发生）
    - 最大并发持仓：MAX_HOLDINGS（独立于 MAX_SELECT_PER_DAY）
    - 每日最大买入：MAX_BUY_PER_DAY（0=不限制，填满空位）

    卖出时机（从 daily_returns 长度推导）：
    - len=1: Day1大跌止损 → 次日开盘卖出
    - len=2: 累计亏损/弱势止损 → 次日收盘卖出
    - len=3: 正常持有满3天 → 第3天收盘卖出
    """
    max_holdings = DataConfig.MAX_HOLDINGS if DataConfig.MAX_HOLDINGS > 0 else \
                   DataConfig.MAX_SELECT_PER_DAY if DataConfig.MAX_SELECT_PER_DAY > 0 else 4
    max_buy_per_day = DataConfig.MAX_BUY_PER_DAY if DataConfig.MAX_BUY_PER_DAY > 0 else max_holdings

    unique_days = np.sort(np.unique(all_day_indices))
    
    daily_selections = {}
    
    if top_n_per_day is not None:
        for day in unique_days:
            day_mask = all_day_indices == day
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                daily_selections[day] = []
                continue
            day_preds = all_preds[day_indices]
            sorted_local = np.argsort(day_preds)[::-1]
            select_count = min(top_n_per_day, len(day_indices))
            selected_global = day_indices[sorted_local[:select_count]]
            daily_selections[day] = selected_global.tolist()
    else:
        top_k = max(1, int(len(all_preds) * top_percent / 100))
        sorted_indices = np.argsort(all_preds)[::-1]
        threshold = all_preds[sorted_indices[top_k - 1]]
        above_threshold_mask = all_preds > threshold
        max_select = DataConfig.MAX_SELECT_PER_DAY
        
        for day in unique_days:
            day_mask = all_day_indices == day
            day_above = above_threshold_mask & day_mask
            day_indices = np.where(day_above)[0]
            if len(day_indices) == 0:
                daily_selections[day] = []
                continue
            if max_select > 0 and len(day_indices) > max_select:
                day_preds = all_preds[day_indices]
                top_local = np.argsort(day_preds)[::-1][:max_select]
                selected_global = day_indices[top_local]
                daily_selections[day] = selected_global.tolist()
            else:
                daily_selections[day] = day_indices.tolist()
    
    all_trade_info = []
    for day in unique_days:
        for idx in daily_selections.get(day, []):
            dr = all_daily_returns[idx]
            dr_len = len(dr) if dr is not None else 0
            buy_day = day + 1
            if dr_len <= 1:
                sell_day = buy_day + 1
                sell_type = 'open'
            elif dr_len == 2:
                sell_day = buy_day + 1
                sell_type = 'close'
            else:
                sell_day = buy_day + 2
                sell_type = 'close'
            
            all_trade_info.append({
                'sample_idx': idx,
                'predict_day': day,
                'buy_day': buy_day,
                'sell_day': sell_day,
                'sell_type': sell_type,
                'return': all_cumulative_returns[idx],
            })
    
    if not all_trade_info:
        return {
            'total_days': 0,
            'final_value': 1.0,
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'trade_count': 0,
            'daily_portfolio_values': [],
            'daily_trades': [],
            'max_drawdown': 0.0,
        }
    
    recommendation_map = {}
    for t in all_trade_info:
        pd = t['predict_day']
        if pd not in recommendation_map:
            recommendation_map[pd] = []
        recommendation_map[pd].append(t)
    
    all_event_days = set()
    for t in all_trade_info:
        all_event_days.add(t['buy_day'])
        all_event_days.add(t['sell_day'])
    for day in unique_days:
        all_event_days.add(day + 1)
    all_event_days = sorted(all_event_days)
    
    min_event_day = min(all_event_days)
    max_event_day = max(all_event_days)
    
    cash = 1.0
    holdings = []
    pending_cash = 0.0
    
    daily_portfolio_values = []
    daily_trades = []
    
    total_buy_count = 0
    total_sell_count = 0
    
    for current_day in range(min_event_day, max_event_day + 1):
        day_open_sells = 0
        day_close_sells = 0

        # === Phase 1: 资金到账 ===
        # 昨日所有卖出（开盘+收盘）的资金今日可用
        cash += pending_cash
        pending_cash = 0.0

        # === Phase 2: 开盘买入 ===
        # 买卖都是在开盘前挂单的，是同时进行的，因此卖出时已经是开盘后了
        # 所以买入时只能用昨日已到账的资金，且卖出释放的仓位在买入时还不可用
        predict_day = current_day - 1
        recommendations = recommendation_map.get(predict_day, [])
        
        available_slots = max_holdings - len(holdings)
        max_buy = min(available_slots, max_buy_per_day)
        actual_buys = 0
        invested_today = 0.0

        if max_buy > 0 and len(recommendations) > 0 and cash > 1e-10:
            buy_count = min(max_buy, len(recommendations))
            amount_per_stock = cash / buy_count
            
            for i in range(buy_count):
                t = recommendations[i]
                holdings.append({
                    'amount': amount_per_stock,
                    'return': t['return'],
                    'sell_day': t['sell_day'],
                    'sell_type': t['sell_type'],
                })
                invested_today += amount_per_stock
                actual_buys += 1
                total_buy_count += 1
            cash -= invested_today

        # === Phase 3: 开盘卖出 ===
        # 止损卖出，资金次日可用
        open_sell_released = 0.0
        remaining = []
        for h in holdings:
            if h['sell_day'] == current_day and h['sell_type'] == 'open':
                released = h['amount'] * (1.0 + h['return'])
                open_sell_released += released
                day_open_sells += 1
                total_sell_count += 1
            else:
                remaining.append(h)
        holdings = remaining
        pending_cash += open_sell_released

        # === Phase 4: 收盘卖出 ===
        close_sell_released = 0.0
        remaining = []
        for h in holdings:
            if h['sell_day'] == current_day and h['sell_type'] == 'close':
                released = h['amount'] * (1.0 + h['return'])
                close_sell_released += released
                day_close_sells += 1
                total_sell_count += 1
            else:
                remaining.append(h)
        holdings = remaining
        pending_cash += close_sell_released

        # === Phase 5: 日终快照 ===
        holdings_value = sum(h['amount'] * (1.0 + h['return']) for h in holdings)
        portfolio_value = cash + pending_cash + holdings_value

        daily_portfolio_values.append({
            'day': current_day,
            'portfolio_value': portfolio_value,
            'cash': cash + pending_cash,
            'holdings_count': len(holdings),
            'buy_count': actual_buys,
            'sell_count': day_open_sells + day_close_sells,
            'invested': invested_today,
        })
        
        daily_trades.append({
            'day': current_day,
            'buys': actual_buys,
            'open_sells': day_open_sells,
            'close_sells': day_close_sells,
            'portfolio_value': portfolio_value,
            'cash_ratio': (cash + pending_cash) / portfolio_value if portfolio_value > 0 else 0,
            'holdings_count': len(holdings),
        })
    
    if daily_portfolio_values:
        final_value = daily_portfolio_values[-1]['portfolio_value']
    else:
        final_value = 1.0
    
    total_return = final_value - 1.0
    
    return {
        'total_days': len(daily_portfolio_values),
        'final_value': final_value,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'trade_count': total_buy_count,
        'buy_count': total_buy_count,
        'sell_count': total_sell_count,
        'daily_portfolio_values': daily_portfolio_values,
        'daily_trades': daily_trades,
        'max_drawdown': _calculate_max_drawdown(daily_portfolio_values),
    }


def _calculate_max_drawdown(daily_portfolio_values):
    if not daily_portfolio_values:
        return 0.0
    peak = daily_portfolio_values[0]['portfolio_value']
    max_dd = 0.0
    for d in daily_portfolio_values:
        pv = d['portfolio_value']
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd



def create_run_dir(output_dir):
    """
    在 output_dir 下为本次训练 run 创建独立目录（run 目录）。

    目录即模型：一个 run 目录容纳该次训练的全部产物
    （model_loss_*.pth / model_realistic_*.pth / epoch_returns_*.csv），
    后续评估产物（daily_stats JSON、分类可视化等）也落回所选模型所在的
    run 目录——产物跟着模型走，永不互串。

    目录名取当前时间戳（YYYY-MM-DD_HHMMSS）；同一秒内重复训练时
    自动追加 _2、_3 后缀，绝不覆盖旧 run。

    Returns:
        新创建的 run 目录绝对路径。
    """
    tag = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = os.path.join(output_dir, tag)
    suffix = 2
    while os.path.exists(run_dir):
        run_dir = os.path.join(output_dir, f"{tag}_{suffix}")
        suffix += 1
    os.makedirs(run_dir)
    return run_dir


def save_model_with_metadata(model_state_dict, top_return, top_threshold, auc,
                             epoch, model_prefix="model", extra_info="",
                             output_dir=DataConfig.OUTPUT_DIR):
    """
    通用的模型保存函数，带详细元数据

    保存格式为包含以下键的字典：
    - 'model_arch': 模型架构参数（用于 run.py 自动重建正确大小的模型）
    - 'train_params': 训练超参数快照
    - 'eval_stats': 评估指标
    - 'state_dict': 模型权重

    Args:
        output_dir: 保存目录。典型用法传 train.py 的 run 目录
                    （create_run_dir 创建的 out/<日期戳>/），让每次训练
                    run 的产物集中在一个目录里。
    """
    from config import ModelConfig, TrainingConfig, LossConfig
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")

    return_str = f"{top_return*100:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
    thr_str = f"{top_threshold:.3f}".replace('.', '_')
    auc_str = f"{auc:.4f}".replace('.', '_')

    if extra_info:
        filename = f"{model_prefix}_top{DataConfig.TOP_K}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{extra_info}_{timestamp}.pth"
    else:
        filename = f"{model_prefix}_top{DataConfig.TOP_K}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{timestamp}.pth"

    checkpoint = {
        'model_arch': {
            'input_dim':        ModelConfig.INPUT_DIM,
            'd_model':          ModelConfig.D_MODEL,
            'ffn_expand_ratio': ModelConfig.FFN_EXPAND_RATIO,
            'nhead':            ModelConfig.NHEAD,
            'num_layers':       ModelConfig.NUM_LAYERS,
            'output_dim':       ModelConfig.OUTPUT_DIM,
            'dropout_rate':     ModelConfig.DROPOUT_RATE,
            'attention_dropout':ModelConfig.ATTENTION_DROPOUT,
            'context_length':   DataConfig.CONTEXT_LENGTH,
        },
        'train_params': {
            'epochs':           TrainingConfig.EPOCHS,
            'learning_rate':    TrainingConfig.get_base_lr(),
            'batch_size':       TrainingConfig.BATCH_SIZE,
                'weight_decay':     TrainingConfig.get_base_wd(),
            'loss_type':        LossConfig.LOSS_TYPE,
            'pos_weight':       LossConfig.POS_WEIGHT,
            'embed_lr_ratio':   RegularizationConfig.EMBED_LR_RATIO,
            'lr_start':         float(TrainingConfig.LR_START),
            'lr_peak':          float(TrainingConfig.LR_PEAK),
            'lr_final':         float(TrainingConfig.LR_FINAL),
            'lr_warmup_steps':  int(TrainingConfig.LR_WARMUP_STEPS),
            'lr_decay_steps':   int(TrainingConfig.LR_DECAY_STEPS),
            'lr_decay_shape':   TrainingConfig.LR_DECAY_SHAPE,
            'visreg_weight':    float(RegularizationConfig.VISREG_MAIN_WEIGHT),
            'visreg_aux_batch_size': int(RegularizationConfig.VISREG_AUX_BATCH_SIZE),
            'visreg_target_std': float(RegularizationConfig.TARGET_STD),
        },
        'eval_stats': {
            'top_return':   float(top_return),
            'top_threshold':float(top_threshold),
            'auc':          float(auc),
            'epoch':        int(epoch),
            'top_k':        DataConfig.TOP_K,
        },
        'state_dict': model_state_dict,
    }

    save_path = os.path.join(output_dir, filename)
    torch.save(checkpoint, save_path)

    return save_path

class EarlyStopping:
    """
    早停机制类

    监控指标：
    - avg_loss: 平均损失（越低越好）
    - top_return: Top1%收益率（越高越好）

    任意一个指标改善即重置计数器
    """
    def __init__(self, patience=10):
        """
        Args:
            patience: 容忍无改善的轮数
        """
        self.patience = patience
        self.no_improve_count = 0

        self.best_loss = float('inf')
        self.best_return = -float('inf')
        self.best_ic = -float('inf')
        self.best_return_auc = 0.0
        self.best_return_threshold = 0.0

    def check_improve(self, avg_loss=None, top_return=None, auc=None, threshold=None,
                      ic=None):
        """
        检查是否有改善

        Args:
            avg_loss: 平均损失
            ic: 逐日 IC（首选改善判据；日间方差比收益小一到两个量级）
            top_return: Top1%收益率（**仅在 ic 不可用时**才作为判据）
            auc: AUC得分（仅当改善时更新）
            threshold: Top阈值（仅当改善时更新）

        Returns:
            improved: 是否有改善
            reason: 改善原因字符串
        """
        improved = False
        reasons = []

        if avg_loss is not None and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            improved = True
            reasons.append(f'损失改善: {avg_loss:.4f}')

        # 收益不再是改善判据。原来只要某轮收益更高就重置计数器，而收益在
        # 200 来天的窗口上标准误约 0.5% —— 等于让噪声不断刷新 patience，
        # 早停几乎永不触发（总有一轮更高）。改用 IC 后判据才真正反映模型在进步。
        if ic is not None and np.isfinite(ic):
            if ic > self.best_ic:
                self.best_ic = ic
                improved = True
                if auc is not None:
                    self.best_return_auc = auc
                if threshold is not None:
                    self.best_return_threshold = threshold
                reasons.append(f'判别力改善: IC {ic:+.4f}')
        elif top_return is not None and top_return > self.best_return:
            # 没有 IC 可用时的退化路径：明说是退化，不要静默当成一样可靠
            self.best_return = top_return
            improved = True
            if auc is not None:
                self.best_return_auc = auc
            if threshold is not None:
                self.best_return_threshold = threshold
            reasons.append(f'收益率改善(退化判据): {top_return*100:+.2f}%')

        if improved:
            self.no_improve_count = 0
            return True, ' & '.join(reasons)
        else:
            self.no_improve_count += 1
            return False, None

    def should_stop(self):
        """是否应该停止训练"""
        return self.no_improve_count >= self.patience

    def get_progress(self):
        """获取当前进度"""
        return self.no_improve_count, self.patience

    def get_best_metrics(self):
        """获取最佳指标"""
        return {
            'best_loss': self.best_loss,
            'best_return': self.best_return,
            'best_return_auc': self.best_return_auc,
            'best_return_threshold': self.best_return_threshold
        }

class GradientMonitor:
    """
    梯度监控器：检测梯度爆炸和梯度消失
    在每个batch的backward后收集各层梯度统计信息
    """
    def __init__(self):
        self.grad_stats = {}
        self.hooks = []

    def _create_hook(self, name):
        def hook(grad):
            if grad is None:
                return grad

            grad_flat = grad.data.abs().flatten()

            grad_norm = grad_flat.norm(2).float().item()
            grad_max = grad_flat.max().float().item()
            grad_mean = grad_flat.mean().float().item()
            has_nan = torch.isnan(grad.data).any().item()
            has_inf = torch.isinf(grad.data).any().item()

            if name not in self.grad_stats:
                self.grad_stats[name] = {
                    'norm': [],
                    'max': [],
                    'mean': [],
                    'nan_count': 0,
                    'inf_count': 0,
                    'zero_count': 0
                }

            stats = self.grad_stats[name]
            stats['norm'].append(grad_norm)
            stats['max'].append(grad_max)
            stats['mean'].append(grad_mean)

            if len(stats['norm']) > 100:
                stats['norm'].pop(0)
                stats['max'].pop(0)
                stats['mean'].pop(0)

            if has_nan:
                stats['nan_count'] += 1
            if has_inf:
                stats['inf_count'] += 1
            if grad_norm < 1e-8:
                stats['zero_count'] += 1

            return grad
        return hook

    def register_hooks(self, model):
        """为模型所有参数注册梯度hook"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._create_hook(name))
                self.hooks.append(hook)
        print(f"  已为 {len(self.hooks)} 个参数注册梯度监控hook")

    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_epoch_summary(self):
        """获取当前epoch的梯度统计摘要"""
        summary = {}
        for name, stats in self.grad_stats.items():
            if stats['norm']:
                summary[name] = {
                    'avg_norm': np.mean(stats['norm']),
                    'max_norm': np.max(stats['norm']),
                    'avg_max': np.mean(stats['max']),
                    'avg_mean': np.mean(stats['mean']),
                    'nan_count': stats['nan_count'],
                    'inf_count': stats['inf_count'],
                    'zero_count': stats['zero_count'],
                    'total_batches': len(stats['norm'])
                }
        return summary

    def reset(self):
        """重置统计信息（新epoch开始时调用）"""
        self.grad_stats.clear()

    def diagnose(self):
        """
        诊断梯度问题，返回报告
        返回: (爆炸层列表, 消失层列表, 异常层列表)
        """
        exploding = []
        vanishing = []
        abnormal = []

        summary = self.get_epoch_summary()

        for name, stats in summary.items():
            if stats['avg_norm'] > 10 or stats['max_norm'] > 100:
                exploding.append((name, stats))

            elif stats['avg_norm'] < 1e-5:
                vanishing.append((name, stats))

            if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                abnormal.append((name, stats))

        return exploding, vanishing, abnormal


def _get_amp_context(device):
    """获取AMP自动混合精度上下文管理器（BF16）"""
    if TrainingConfig.USE_AMP and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def training_step(model, optimizer, loss_fn):
    """
    通用训练步，自动处理梯度裁剪和AMP混合精度。

    统一封装前向-反向-更新流程：
    zero_grad → loss_fn(AMP) → backward → clip → step

    Args:
        model: PyTorch模型
        optimizer: 优化器实例（AdamW）
        loss_fn: 无参回调函数，返回 (loss_tensor, output_tensor)。
                 调用方在闭包中封装具体的loss计算逻辑。

    Returns:
        tuple: (loss_value, output) — loss_value为Python float，output为首次前向输出
    """
    device = next(model.parameters()).device
    amp_ctx = _get_amp_context(device)

    optimizer.zero_grad()
    with amp_ctx:
        loss, output = loss_fn()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
    optimizer.step()

    return loss.item(), output


def create_optimizer_from_config_for_params(params, lr=None):
    """
    创建 AdamW 优化器（接受参数列表或 PyTorch param_groups）

    Args:
        params: 可迭代的参数列表，或 [{'params': [...], 'lr': ...}, ...] 形式的
                param_groups（用于给 Embedding 设置独立学习率）
        lr: 学习率（param_groups 中未指定 lr 的组使用此值）

    Returns:
        torch.optim.AdamW
    """
    actual_lr = lr if lr is not None else TrainingConfig.get_base_lr()
    wd = TrainingConfig.get_base_wd()

    params = list(params)
    has_groups = bool(params) and isinstance(params[0], dict)
    n_groups = len(params) if has_groups else 1

    optimizer = optim.AdamW(params, lr=actual_lr, weight_decay=wd)
    print(f"优化器: AdamW (lr={actual_lr}, wd={wd}, param_groups={n_groups})")
    return optimizer


def create_scheduler_from_config(optimizer):
    """
    按 TrainingConfig 的两段式配置创建学习率调度器。

    预热步数 + 退火步数必须等于总步数（EPOCHS × BATCHES_PER_EPOCH），
    由 TrainingConfig.validate_lr_schedule() 校验。

    Returns:
        StepLRScheduler: 每个优化步调用一次 step(step)
    """
    total = TrainingConfig.validate_lr_schedule()
    scheduler = StepLRScheduler(
        optimizer,
        start_lr=TrainingConfig.LR_START,
        peak_lr=TrainingConfig.LR_PEAK,
        final_lr=TrainingConfig.LR_FINAL,
        warmup_steps=TrainingConfig.LR_WARMUP_STEPS,
        decay_steps=TrainingConfig.LR_DECAY_STEPS,
        shape=TrainingConfig.LR_DECAY_SHAPE,
    )
    scheduler.step(0)
    print(f"  学习率调度: 预热 {TrainingConfig.LR_WARMUP_STEPS} 步 "
          f"{TrainingConfig.LR_START:g}→{TrainingConfig.LR_PEAK:g}，退火 "
          f"{TrainingConfig.LR_DECAY_STEPS} 步 {TrainingConfig.LR_PEAK:g}→{TrainingConfig.LR_FINAL:g}"
          f"（{TrainingConfig.LR_DECAY_SHAPE}，共 {total} 步，按步更新）")
    return scheduler
    return warmup_scheduler, main_scheduler, warmup_epochs
