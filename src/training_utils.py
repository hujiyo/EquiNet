'''
训练工具模块

提供训练相关的工具类和函数，供其他训练脚本导入使用：
- WarmupScheduler: 学习率预热调度器
- GradientMonitor: 梯度监控器
- DynamicWeightedBCE: 动态加权BCE损失函数
- EarlyStopping: 早停机制
- evaluate_model: 模型评估函数
- generate_pseudo_labels: 伪标签生成
- save_model_with_metadata: 带元数据的模型保存
- print_dispersion_sparkline: 预测值分布可视化
- create_optimizer_from_config: 根据配置创建优化器
- create_scheduler_from_config: 根据配置创建学习率调度器
'''

import os,torch,torch.nn as nn,numpy as np,math
import contextlib
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import DataConfig,LossConfig,TrainingConfig
import torch.optim as optim
from lion_pytorch import Lion
from muon import SingleDeviceMuonWithAuxAdam as MuonWithAuxAdam

class WarmupScheduler:
    """
    学习率预热调度器
    在前几轮训练中，学习率从很小的值逐步增加到目标学习率
    这有助于模型在训练初期更稳定地收敛
    """
    def __init__(self, optimizer, warmup_epochs, target_lr, start_lr=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.start_lr = start_lr if start_lr is not None else target_lr / 100
        self.current_epoch = 0

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        target_ratio = target_lr / self.base_lrs[0] if self.base_lrs[0] != 0 else 1.0
        start_ratio = self.start_lr / target_lr if target_lr != 0 else 1.0

        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            group['lr'] = base_lr * target_ratio * start_ratio
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            progress = (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = 1.0

        target_ratio = self.target_lr / self.base_lrs[0] if self.base_lrs[0] != 0 else 1.0
        start_ratio = self.start_lr / self.target_lr if self.target_lr != 0 else 1.0

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            scale = start_ratio + (1.0 - start_ratio) * progress
            group['lr'] = base_lr * target_ratio * scale
        
        return self.optimizer.param_groups[0]['lr']
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def is_warmup_phase(self):
        return self.current_epoch < self.warmup_epochs


class CosineAnnealFreezeLR:
    """
    余弦退火 + 固定学习率调度器

    学习率变化：warmup → 余弦退火 → 固定在 eta_min
    余弦退火仅持续 anneal_epochs 轮，之后学习率固定不再下降。

    支持多 param_group 不同学习率：按各自 base_lr 的比例独立退火。
    eta_min 为第一个 group 的最低学习率，其余 group 按比例计算。
    base_lrs 在第一次 step() 时才记录，确保拿到预热结束后的真实目标 lr。
    """
    def __init__(self, optimizer, anneal_epochs, eta_min):
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.anneal_epochs = max(1, anneal_epochs)
        self.current_step = 0
        self.base_lrs = None
        self.eta_mins = None

    def _ensure_base_lrs(self):
        if self.base_lrs is None:
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
            self.eta_mins = []
            if self.base_lrs and self.base_lrs[0] != 0:
                for base_lr in self.base_lrs:
                    self.eta_mins.append(self.eta_min * base_lr / self.base_lrs[0])
            else:
                self.eta_mins = [self.eta_min] * len(self.base_lrs)

    def step(self):
        self._ensure_base_lrs()
        self.current_step += 1
        if self.current_step <= self.anneal_epochs:
            for param_group, base_lr, em in zip(self.optimizer.param_groups, self.base_lrs, self.eta_mins):
                cosine_factor = 0.5 * (1 + math.cos(math.pi * self.current_step / self.anneal_epochs))
                param_group['lr'] = em + (base_lr - em) * cosine_factor

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


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns,
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name="", eval_day_indices=None, top_n_per_day=None, eval_daily_returns=None,
                   criterion=None, enable_portfolio_simulation=False):
    """
    模型评估函数
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

    分批处理，减少显存占用

    返回统计字典，包含：
        auc：AUC得分
        top_return：Top1%收益率
        top_count：Top1%样本数
        top_threshold：Top1%最低置信度
        high_conf_count：高置信(>0.7)样本数
        low_conf_count：低置信(<0.2)样本数
        pred_mean：预测均值
        pred_std：预测标准差
        filtered_count：被过滤的涨停样本数（始终为0，因已在生成阶段过滤）
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
            'test_loss': None
        }

    all_preds = []
    total_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    amp_ctx = _get_amp_context(device)

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
            with amp_ctx:
                logits = model(batch_inputs)
            scores = torch.sigmoid(logits.float())
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

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    sorted_indices = np.argsort(all_preds)[::-1]

    # 头部分类性：Top K% 样本的正样本比例（Precision@K%）
    base_positive_rate = float(np.mean(all_targets))
    precision_at = {'base_positive_rate': base_positive_rate}
    for top_pct in [10, 5, 3]:
        k = max(10, int(len(all_preds) * top_pct / 100))
        subset_targets = all_targets[sorted_indices[:k]]
        precision_at[f'precision_top{top_pct}'] = float(np.mean(subset_targets))

    percent = DataConfig.TOP_K
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
        'top_return': top_return,
        'daily_top_return': daily_top_return,
        'top_count': top_k,
        'top_threshold': top_threshold,
        'high_conf_count': np.sum(high_conf),
        'low_conf_count': np.sum(low_conf),
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
        'filtered_count': 0,
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
        'cumulative_return': cumulative_return,
        'valid_days': len(daily_returns),
        'avg_realistic_return': avg_realistic_return,
        'mode': 'top_n_per_day' if top_n_per_day else 'global_threshold'
    }


def calculate_portfolio_simulation(all_preds, all_cumulative_returns, all_daily_returns,
                                    all_day_indices, top_percent=1.0, top_n_per_day=None):
    """
    实战资金管理模拟（串行逐日模拟买卖过程）
    
    核心规则：
    1. 任何时刻最多持有 MAX_SELECT_PER_DAY（默认4）只股票
    2. 第D天收盘后运行模型得到推荐列表（按分数排序）
    3. 第D+1天开盘：
       a. 先处理今天的卖出（开盘止损卖出）
       b. 所有卖出资金（含昨日收盘卖出、今日开盘卖出）合并为可用资金
       c. 根据当前持仓空位数，从推荐列表中取前N只（N=空位数）
       d. 用可用资金等权买入这N只股票
    4. 第D+1天收盘：处理收盘卖出，资金累计到次日
    5. 如果某天推荐列表为空（模型无推荐），则该天空仓，资金闲置
    
    
    卖出时机（从 daily_returns 长度推导）：
    - len=1: Day1大跌止损 → 买入次日开盘卖出
    - len=2: 累计亏损/弱势止损 → 买入次日收盘卖出
    - len=3: 正常持有满3天 → 买入第3天收盘卖出
    """
    max_holdings = DataConfig.MAX_SELECT_PER_DAY if DataConfig.MAX_SELECT_PER_DAY > 0 else 4
    
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
        
        # === Phase 1: 开盘卖出 ===
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
        
        # 开盘卖出的资金当天不能买入，放入 pending_cash
        pending_cash += open_sell_released
        
        # === Phase 2: 开盘买入 ===
        # 前一天的 predict_day = current_day - 1
        predict_day = current_day - 1
        recommendations = recommendation_map.get(predict_day, [])
        
        available_slots = max_holdings - len(holdings)
        actual_buys = 0
        invested_today = 0.0
        
        if available_slots > 0 and len(recommendations) > 0 and cash > 1e-10:
            buy_count = min(available_slots, len(recommendations))
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
            cash = 0.0
        
        # === Phase 3: 收盘卖出 ===
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
        
        # 收盘卖出的资金次日可用，放入 pending_cash
        pending_cash += close_sell_released
        
        # === Phase 4: 日终结算 ===
        # pending_cash 转为明日可用的 cash
        cash += pending_cash
        pending_cash = 0.0
        
        holdings_value = sum(h['amount'] * (1.0 + h['return']) for h in holdings)
        portfolio_value = cash + holdings_value
        
        daily_portfolio_values.append({
            'day': current_day,
            'portfolio_value': portfolio_value,
            'cash': cash,
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
            'cash_ratio': cash / portfolio_value if portfolio_value > 0 else 0,
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



def generate_pseudo_labels(pred_scores, original_targets,
                           pseudo_pos_ratio=0.01,
                           pseudo_neg_ratio=0.05):
    """
    统一的伪标签生成函数（按数量取Top-K%方式）

    核心思想：
    - 按预测分数排序，取前 pseudo_pos_ratio 比例的样本 → 强制标签=1.0（伪正）
    - 按预测分数排序，取倒数 pseudo_neg_ratio 比例的样本 → 强制标签=0.0（伪负）
    - 其余样本保持原始标签不变
    """
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.float().detach().cpu().numpy()
    if isinstance(original_targets, torch.Tensor):
        original_targets = original_targets.float().detach().cpu().numpy()

    pred_scores = np.asarray(pred_scores).flatten()
    original_targets = np.asarray(original_targets).copy()

    if len(pred_scores) == 0:
        stats = {
            'pseudo_pos_count': 0,
            'pseudo_neg_count': 0,
            'unchanged_count': 0,
            'threshold_pos': 0.0,
            'threshold_neg': 0.0,
        }
        return original_targets, stats

    k_pos = max(1, int(len(pred_scores) * pseudo_pos_ratio))
    k_pos = min(k_pos, len(pred_scores))
    threshold_pos = np.sort(pred_scores)[-k_pos]

    k_neg = max(1, int(len(pred_scores) * pseudo_neg_ratio))
    k_neg = min(k_neg, len(pred_scores))
    threshold_neg = np.sort(pred_scores)[k_neg - 1]

    pseudo_targets = original_targets.copy()

    high_mask = pred_scores >= threshold_pos
    pseudo_targets[high_mask] = 1.0

    low_mask = pred_scores <= threshold_neg
    pseudo_targets[low_mask] = 0.0

    stats = {
        'pseudo_pos_count': int(np.sum(high_mask)),
        'pseudo_neg_count': int(np.sum(low_mask)),
        'unchanged_count': int(len(pred_scores) - np.sum(high_mask) - np.sum(low_mask)),
        'threshold_pos': float(threshold_pos),
        'threshold_neg': float(threshold_neg),
    }

    return pseudo_targets, stats


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
            'use_adamw':        TrainingConfig.OPTIMIZER_TYPE == 'adamw',
            'use_mano':         TrainingConfig.OPTIMIZER_TYPE == 'mano',
            'optimizer_type':   TrainingConfig.OPTIMIZER_TYPE,
            'weight_decay':     TrainingConfig.get_base_wd(),
            'loss_type':        LossConfig.LOSS_TYPE,
            'pos_weight':       LossConfig.POS_WEIGHT,
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
        self.best_return_auc = 0.0
        self.best_return_threshold = 0.0

    def check_improve(self, avg_loss=None, top_return=None, auc=None, threshold=None):
        """
        检查是否有改善

        Args:
            avg_loss: 平均损失
            top_return: Top1%收益率
            auc: AUC得分（仅当收益率改善时更新）
            threshold: Top阈值（仅当收益率改善时更新）

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

        if top_return is not None and top_return > self.best_return:
            self.best_return = top_return
            improved = True
            if auc is not None:
                self.best_return_auc = auc
            if threshold is not None:
                self.best_return_threshold = threshold
            reasons.append(f'收益率改善: {top_return*100:+.2f}%')

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
        optimizer: 优化器实例（AdamW、Lion、Mano等）
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


def create_optimizer_from_config(model, lr=None):
    """
    根据配置创建优化器（统一入口）

    Args:
        model: PyTorch模型
        lr: 学习率，如果为None则自动使用当前优化器的默认学习率

    Returns:
        optimizer: 创建的优化器实例
    """
    from optimizers import create_optimizer

    actual_lr = lr if lr is not None else TrainingConfig.get_base_lr()
    wd = TrainingConfig.get_base_wd()
    optimizer_type = TrainingConfig.OPTIMIZER_TYPE.lower()

    if optimizer_type == 'mano':
        optimizer = create_optimizer(
            model,
            optimizer_type='mano',
            lr=actual_lr,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=wd,
            betas=TrainingConfig.MANO_ADAMW_BETAS,
            nesterov=TrainingConfig.MANO_NESTEROV,
            dual_dim_projection=TrainingConfig.MANO_DUAL_DIM_PROJECTION
        )
    elif optimizer_type == 'lion':
        optimizer = Lion(model.parameters(), lr=actual_lr, betas=TrainingConfig.LION_BETAS, weight_decay=wd)
        print(f"优化器: Lion (lr={actual_lr}, wd={wd}, betas={TrainingConfig.LION_BETAS})")
    elif optimizer_type == 'muon':
        muon_lr = actual_lr
        muon_wd = TrainingConfig.MUON_WEIGHT_DECAY
        muon_momentum = TrainingConfig.MUON_MOMENTUM
        adamw_lr = TrainingConfig.MUON_ADAMW_LR_RATIO * muon_lr

        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        other_params = [p for p in model.parameters() if p.ndim < 2]

        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=muon_lr, momentum=muon_momentum, weight_decay=muon_wd),
            dict(params=other_params, use_muon=False,
                 lr=adamw_lr, betas=TrainingConfig.MUON_ADAMW_BETAS, weight_decay=muon_wd),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        print(f"优化器: MuonWithAuxAdam (Muon lr={muon_lr}, AdamW lr={adamw_lr:.6f}, wd={muon_wd})")
        print(f"  2D权重(Muon): {len(hidden_weights)} 个参数")
        print(f"  1D参数(AdamW): {len(other_params)} 个参数")
    else:  # 'adamw' 或其他
        optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=wd)
        print(f"优化器: AdamW (lr={actual_lr}, wd={wd})")

    return optimizer


def create_optimizer_from_config_for_params(params, lr=None):
    """
    根据配置创建优化器（接受参数列表而非模型，用于冻结embedding等场景）

    Args:
        params: 可迭代的参数列表（通常来自 [p for p in model.parameters() if p.requires_grad]）
        lr: 学习率

    Returns:
        optimizer: 创建的优化器实例
    """
    actual_lr = lr if lr is not None else TrainingConfig.get_base_lr()
    wd = TrainingConfig.get_base_wd()
    optimizer_type = TrainingConfig.OPTIMIZER_TYPE.lower()

    if optimizer_type == 'mano':
        from optimizers import create_optimizer

        class _ParamContainer:
            """包装参数列表，让 create_optimizer 能像 model 一样使用"""
            def __init__(self, params):
                self._params = list(params)
            def parameters(self):
                return iter(self._params)
            def named_parameters(self):
                return ((f"param_{i}", p) for i, p in enumerate(self._params))

        optimizer = create_optimizer(
            _ParamContainer(params),
            optimizer_type='mano',
            lr=actual_lr,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=wd,
            betas=TrainingConfig.MANO_ADAMW_BETAS,
            nesterov=TrainingConfig.MANO_NESTEROV,
            dual_dim_projection=TrainingConfig.MANO_DUAL_DIM_PROJECTION
        )
    elif optimizer_type == 'lion':
        optimizer = Lion(params, lr=actual_lr, betas=TrainingConfig.LION_BETAS, weight_decay=wd)
        print(f"优化器(部分参数): Lion (lr={actual_lr}, wd={wd}, betas={TrainingConfig.LION_BETAS})")
    elif optimizer_type == 'muon':
        params_list = list(params)
        hidden_weights = [p for p in params_list if p.ndim >= 2]
        other_params = [p for p in params_list if p.ndim < 2]
        muon_lr = actual_lr
        muon_wd = TrainingConfig.MUON_WEIGHT_DECAY
        muon_momentum = TrainingConfig.MUON_MOMENTUM
        adamw_lr = TrainingConfig.MUON_ADAMW_LR_RATIO * muon_lr

        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=muon_lr, momentum=muon_momentum, weight_decay=muon_wd),
            dict(params=other_params, use_muon=False,
                 lr=adamw_lr, betas=TrainingConfig.MUON_ADAMW_BETAS, weight_decay=muon_wd),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        print(f"优化器(部分参数): MuonWithAuxAdam (Muon lr={muon_lr}, AdamW lr={adamw_lr:.6f})")
    else:
        optimizer = optim.AdamW(params, lr=actual_lr, weight_decay=wd)
        print(f"优化器(部分参数): AdamW (lr={actual_lr}, wd={wd})")

    return optimizer


def create_scheduler_from_config(optimizer, epochs, lr=None, eta_min=None, warmup_start_lr=None):
    """
    根据配置创建学习率调度器（预热 + 余弦退火）

    Args:
        optimizer: PyTorch优化器
        epochs: 总训练轮数
        lr: 目标学习率，如果为None则自动使用当前优化器的默认学习率
        eta_min: 余弦退火最小学习率，如果为None则使用 TrainingConfig.COSINE_ETA_MIN
        warmup_start_lr: 预热起始学习率，如果为None则使用 TrainingConfig.WARMUP_START_LR

    Returns:
        tuple: (warmup_scheduler, main_scheduler, warmup_epochs)
            - warmup_scheduler: 预热调度器
            - main_scheduler: 余弦退火调度器
            - warmup_epochs: 预热轮数（用于打印状态）
    """
    actual_lr = lr if lr is not None else TrainingConfig.get_base_lr()
    actual_eta_min = eta_min if eta_min is not None else TrainingConfig.COSINE_ETA_MIN
    actual_warmup_start_lr = warmup_start_lr if warmup_start_lr is not None else TrainingConfig.WARMUP_START_LR

    warmup_epochs = TrainingConfig.WARMUP_EPOCHS

    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        target_lr=actual_lr,
        start_lr=actual_warmup_start_lr
    )

    total_main_epochs = max(1, epochs - warmup_epochs)
    anneal_epochs = TrainingConfig.COSINE_ANNEAL_EPOCHS

    if anneal_epochs < total_main_epochs:
        main_scheduler = CosineAnnealFreezeLR(
            optimizer,
            anneal_epochs=anneal_epochs,
            eta_min=actual_eta_min
        )
    else:
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_main_epochs,
            eta_min=actual_eta_min
        )

    return warmup_scheduler, main_scheduler, warmup_epochs
