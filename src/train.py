'''
训练脚本

评分制度（收益率制度，以代码实现为准）：
采用排序能力评估，更贴近真实选股场景。
按预测概率从高到低排序，统计Top-K%样本的收益：
每个区间统计：样本数、平均收益、累计收益、上涨准确率、非负率
'''

import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
import random
import csv
from datetime import datetime
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import (ModelConfig, TrainingConfig, DataConfig,
                   DeviceConfig, ModelSaveConfig, LossConfig,
                   print_config_summary)

from model import create_model

from data import (
    load_and_preprocess_data,
    create_sampler, sample_with_pools,
    generate_sample_from_index,
    create_fixed_evaluation_dataset,
    create_train_evaluation_dataset,
    normalize_data_for_prediction,
    predict_single_stock,
    predict_multiple_stocks
)

# 学习率预热调度器
class WarmupScheduler:
    """
    学习率预热调度器
    在前几轮训练中，学习率从很小的值逐步增加到目标学习率
    这有助于模型在训练初期更稳定地收敛
    """
    def __init__(self, optimizer, warmup_epochs, target_lr, start_lr=None):
        """
        Args:
            optimizer: PyTorch优化器
            warmup_epochs: 预热轮数
            target_lr: 目标学习率（预热结束后的学习率）
            start_lr: 预热起始学习率，如果为None则使用target_lr的1/100
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.start_lr = start_lr if start_lr is not None else target_lr / 100
        self.current_epoch = 0
        
        # 设置初始学习率为预热起始学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.start_lr
    
    def step(self, epoch=None):
        """
        更新学习率
        Args:
            epoch: 当前轮数，如果为None则使用内部计数器
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.start_lr + (self.target_lr - self.start_lr) * ((self.current_epoch + 1) / self.warmup_epochs)
        else:
            # 预热结束后保持目标学习率
            lr = self.target_lr
        
        # 更新优化器的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_last_lr(self):
        """获取当前学习率（兼容PyTorch调度器接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def is_warmup_phase(self):
        """判断是否还在预热阶段"""
        return self.current_epoch < self.warmup_epochs

def print_dispersion_sparkline(all_preds, epoch_returns_history=None):
    """
    打印预测值在0-1区间上的分布直方图（终端字符可视化）
    
    Args:
        all_preds: 所有样本的预测值数组
        epoch_returns_history: 历史epoch记录列表（用于显示趋势）
    """
    print(f'  【预测值分布直方图】')
    
    all_preds = np.array(all_preds)
    
    num_bins = 20
    counts, _ = np.histogram(all_preds, bins=num_bins, range=(0, 1))
    max_count = max(counts) if max(counts) > 0 else 1
    
    chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    
    hist_line = ""
    for count in counts:
        idx = int(count / max_count * (len(chars) - 1))
        idx = min(max(idx, 0), len(chars) - 1)
        hist_line += chars[idx]
    
    print(f'    0.0  {hist_line}  1.0')
    print(f'         ├────────────────────┤')
    
    std = float(np.std(all_preds))
    mean = float(np.mean(all_preds))
    min_val = float(np.min(all_preds))
    max_val = float(np.max(all_preds))
    pos_ratio = float(np.mean(all_preds >= 0.5)) * 100
    high_conf_ratio = float(np.mean(all_preds >= 0.7)) * 100
    
    print(f'    均值={mean:.3f}, 标准差={std:.4f}, 范围=[{min_val:.3f}, {max_val:.3f}]')
    print(f'    >0.5: {pos_ratio:.1f}%, >0.7: {high_conf_ratio:.1f}%')
    
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

# 动态加权BCE损失函数实现
class DynamicWeightedBCE(nn.Module):
    """
    动态加权BCE损失函数：按标签桶分配权重
    - 标签1.0固定权重4.0
    - 标签0.6/0.3/0.0按样本数量动态分配权重（样本少=权重高）
    """
    def __init__(self, pos_weight=4.0, reduction='mean'):
        super(DynamicWeightedBCE, self).__init__()
        self.reduction = reduction
        
        # 固定正样本权重
        self.register_buffer('pos_weight', torch.tensor(pos_weight))
        
        # 动态负样本权重（按标签桶分配）
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
        
        # 统计正负样本数量
        count_positive = np.sum(targets >= 0.5)  # 上涨样本（≥5%）
        count_negative = np.sum(targets < 0.5)   # 不上涨样本（<5%）
        
        if count_positive > 0 and count_negative > 0:
            # 动态调整负样本权重，保持正负样本对总损失的贡献平衡
            # neg_weight = pos_weight * (正样本数 / 负样本数)
            neg_weight = float(self.pos_weight) * (count_positive / count_negative)

            # 更新负样本权重（复用weight_0_0变量）
            self.weight_0_0.fill_(neg_weight)
        elif count_positive == 0:
            # 没有正样本，负样本权重设为正样本权重
            self.weight_0_0.fill_(float(self.pos_weight))
        else:
            # 没有负样本，权重设为较小值
            self.weight_0_0.fill_(0.1)
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, 1] 模型输出的logits
        targets: [batch_size] 真实标签 (1.0/0.0)
        """
        # 确保输入形状正确：如果是 [batch_size, 1] 则 squeeze(-1) 变成 [batch_size]
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)

        # 使用FP32计算loss确保数值稳定性
        inputs_fp32 = inputs.float()
        targets_fp32 = targets.float()

        # 计算BCE loss（带logits）
        loss = F.binary_cross_entropy_with_logits(inputs_fp32, targets_fp32, reduction='none')
        
        # 二分类动态权重：正样本和负样本分别使用动态权重
        pos_weight = self.pos_weight.to(dtype=loss.dtype, device=loss.device)
        neg_weight = self.weight_0_0.to(dtype=loss.dtype, device=loss.device)

        # 根据标签分配权重：正样本用pos_weight，负样本用动态neg_weight
        weights = torch.where(targets_fp32 >= 0.5, pos_weight, neg_weight)
        loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ==================== 数据处理已移至 data.py ====================
# 以下函数/类已迁移到 src/data.py:
# - process_single_file
# - load_and_preprocess_data
# - TemporalSampler
# - generate_sample_from_index
# - sample_with_pools
# - create_fixed_evaluation_dataset
# - create_train_evaluation_dataset
# - normalize_data_for_prediction
# - predict_single_stock
# - predict_multiple_stocks


def evaluate_model_batch(model, eval_inputs, eval_targets, eval_cumulative_returns, device, 
                         batch_size=256, eval_day_indices=None, top_n_per_day=None, eval_daily_returns=None):
    """
    批量评估模型性能（详细版，用于train.py主训练流程）
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

    优化版本：分批处理，减少显存占用

    返回:
        total: 总样本数
        class_correct: [不上涨正确数, 上涨正确数]
        class_total: [不上涨总数, 上涨总数]
        pred_positive_correct: 预测上涨且真实上涨的数量
        pred_positive_total: 预测上涨的总数量
        pred_non_negative: 预测上涨且真实收益>=0的数量
        auc_score: AUC得分
        confidence_stats: 置信度区间统计
        top_stats: Top N% 收益统计
        realistic_stats: 实战收益率统计（如果提供了eval_day_indices）
    """
    model.eval()

    num_samples = len(eval_inputs)
    if num_samples == 0:
        return 0, [0, 0], [0, 0], 0, 0, 0, 0.5, {}, {'count': 0, 'avg_return': 0, 'total_return': 0, 'filtered_count': 0}, None, None, {}
    
    all_preds = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_preds = torch.sigmoid(model(batch_inputs))
            all_preds.append(batch_preds.cpu().numpy().flatten())
            del batch_inputs
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(eval_targets)
    all_returns = np.array(eval_cumulative_returns)

    total = len(all_preds)

    pred_labels = (all_preds >= 0.5).astype(int)
    true_labels = (all_targets >= 0.5).astype(int)

    class_correct = [0, 0]
    class_total = [0, 0]

    for i in range(2):
        mask = true_labels == i
        class_total[i] = np.sum(mask)
        class_correct[i] = np.sum((pred_labels == i) & mask)

    pred_positive_mask = pred_labels == 1
    pred_positive_total = np.sum(pred_positive_mask)
    pred_positive_correct = np.sum(pred_positive_mask & (true_labels == 1))
    pred_non_negative = np.sum(pred_positive_mask & (all_returns >= 0))

    try:
        auc_score = roc_auc_score(true_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    confidence_intervals = ['0.50-0.55', '0.55-0.58', '0.58-0.60', '0.60-0.70', '0.70-1.00']
    confidence_bounds = [(0.50, 0.55), (0.55, 0.58), (0.58, 0.60), (0.60, 0.70), (0.70, 1.00)]
    confidence_stats = {}

    for interval, (low, high) in zip(confidence_intervals, confidence_bounds):
        mask = (all_preds >= low) & (all_preds < high)
        total_in_interval = np.sum(mask)
        correct_in_interval = np.sum(mask & (true_labels == 1))
        non_negative_in_interval = np.sum(mask & (all_returns >= 0))
        confidence_stats[interval] = (correct_in_interval, total_in_interval, non_negative_in_interval)

    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(all_preds) * percent / 100))
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    avg_return = np.mean(top_returns)
    total_return = np.sum(top_returns)

    top_stats = {
        'count': top_k,
        'avg_return': avg_return,
        'total_return': total_return,
        'filtered_count': 0
    }

    dispersion_stats = {
        'std': float(np.std(all_preds)),
        'mean': float(np.mean(all_preds)),
        'min': float(np.min(all_preds)),
        'max': float(np.max(all_preds)),
        'range': float(np.max(all_preds) - np.min(all_preds)),
        'iqr': float(np.percentile(all_preds, 75) - np.percentile(all_preds, 25)),
        'q25': float(np.percentile(all_preds, 25)),
        'q50': float(np.percentile(all_preds, 50)),
        'q75': float(np.percentile(all_preds, 75)),
        'pos_ratio': float(np.mean(all_preds >= 0.5)),
        'high_conf_ratio': float(np.mean(all_preds >= 0.7)),
        'low_conf_ratio': float(np.mean(all_preds < 0.3)),
    }

    realistic_stats = None
    smart_exit_stats = None
    if eval_day_indices is not None:
        actual_top_n = top_n_per_day if top_n_per_day is not None else DataConfig.TOP_N_PER_DAY
        if actual_top_n == 0:
            actual_top_n = None
        realistic_stats = calculate_realistic_return(all_preds, all_returns, eval_day_indices, percent, actual_top_n)
        
        if eval_daily_returns is not None and actual_top_n is not None:
            smart_exit_stats = calculate_smart_exit_return(all_preds, eval_daily_returns, eval_day_indices, actual_top_n)

    return total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats, realistic_stats, smart_exit_stats, dispersion_stats, all_preds


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns,
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name="", eval_day_indices=None, top_n_per_day=None, eval_daily_returns=None):
    """
    简化版模型评估函数（用于train_clone.py和train_evolve.py）
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

    优化版本：分批处理，减少显存占用

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
        smart_exit_stats：智能止损策略统计（如果提供了eval_daily_returns）
    """
    model.eval()

    num_samples = len(eval_inputs)
    if num_samples == 0:
        return {
            'auc': 0.5, 'top_return': 0.0, 'top_count': 0, 'top_threshold': 0.0,
            'high_conf_count': 0, 'low_conf_count': 0, 'pred_mean': 0.0, 
            'pred_std': 0.0, 'filtered_count': 0, 'realistic_stats': None, 'smart_exit_stats': None
        }

    all_preds = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], dtype=torch.float32, device=device)
            batch_preds = torch.sigmoid(model(batch_inputs))
            all_preds.append(batch_preds.cpu().numpy().flatten())
            del batch_inputs
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.array(eval_targets)
    all_returns = np.array(eval_cumulative_returns)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(all_preds) * percent / 100))
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    top_return = np.mean(top_returns)
    top_threshold = all_preds[sorted_indices[top_k - 1]]

    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2

    stats = {
        'auc': auc,
        'top_return': top_return,
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
    }

    if eval_day_indices is not None:
        actual_top_n = top_n_per_day if top_n_per_day is not None else DataConfig.TOP_N_PER_DAY
        if actual_top_n == 0:
            actual_top_n = None
        stats['realistic_stats'] = calculate_realistic_return(all_preds, all_returns, eval_day_indices, percent, actual_top_n)
        
        if eval_daily_returns is not None and actual_top_n is not None:
            stats['smart_exit_stats'] = calculate_smart_exit_return(all_preds, eval_daily_returns, eval_day_indices, actual_top_n)
        else:
            stats['smart_exit_stats'] = None
    else:
        stats['realistic_stats'] = None
        stats['smart_exit_stats'] = None

    stats['all_preds'] = all_preds
    return stats


def calculate_realistic_return(all_preds, all_returns, all_day_indices, top_percent=1.0, top_n_per_day=None):
    """
    计算实战收益率（高仿实战）
    
    支持两种模式:
    1. 全局阈值模式（top_n_per_day=None）: 按全局Top%确定阈值，每天选超过阈值的股票
    2. 每日Top N模式（top_n_per_day指定）: 每天选预测分数最高的前N只股票
    
    Args:
        all_preds: 所有样本的预测分数
        all_returns: 所有样本的收益率
        all_day_indices: 每个样本对应的预测日在测试集中的相对偏移量
        top_percent: Top百分比，默认1%（仅全局阈值模式使用）
        top_n_per_day: 每天选股数量，如4表示每天选前4只（优先于top_percent）
    
    Returns:
        realistic_stats: 包含每日统计和平均实战收益率的字典
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


def calculate_smart_exit_return(all_preds, all_daily_returns, all_day_indices, top_n_per_day=4,
                                 stop_loss_day1=-0.05, stop_loss_cum=-0.05, take_profit=0.08,
                                 sell_at_day2_close=True):
    """
    智能止损策略收益率计算（A股T+1规则）
    
    注意：此函数使用"收益率"而非"涨跌幅"
    - 收益率：基准是买入价（T+1开盘价），用于计算投资回报
    - r1 + r2 + r3 = 累计收益率
    
    交易规则：
    - T日晚预测 → T+1日开盘买入
    - Day1（T+1）：买入后持有，无法卖出（A股T+1）
    - Day2（T+2）：可以卖出
    - Day3（T+3）：可以卖出
    
    策略规则：
    1. Day1大跌止损：如果Day1收益率 < stop_loss_day1（如-5%），Day2卖出
       - sell_at_day2_close=True: Day2收盘卖出，收益=r1+r2
       - sell_at_day2_close=False: Day2开盘卖出，收益≈r1
    2. 累计止损：如果Day1+Day2累计 < stop_loss_cum（如-5%），Day3卖出
    3. 止盈：如果累计收益 >= take_profit（如8%），Day3卖出
    4. 正常持有：否则持有到Day3收盘
    
    Args:
        all_preds: 所有样本的预测分数
        all_daily_returns: 每日收益率列表 [[r1, r2, r3], ...]
            - 基准是买入价（T+1开盘价）
            - r1 = (T+1收盘 - T+1开盘) / T+1开盘
            - r2 = (T+2收盘 - T+1收盘) / T+1开盘
            - r3 = (T+3收盘 - T+2收盘) / T+1开盘
        all_day_indices: 每个样本对应的预测日索引
        top_n_per_day: 每天选股数量
        stop_loss_day1: Day1大跌止损阈值（默认-5%，Day1跌超这个值Day2卖出）
        stop_loss_cum: 累计止损阈值（默认-5%）
        take_profit: 止盈阈值（默认8%）
        sell_at_day2_close: Day1大跌后是否在Day2收盘卖出（True=收盘卖，False=开盘卖）
    
    Returns:
        stats: 包含策略统计的字典
    """
    unique_days = np.unique(all_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    daily_returns = []
    
    total_trades = 0
    stop_loss_day1_count = 0
    stop_loss_cum_count = 0
    take_profit_count = 0
    normal_exit_count = 0
    
    for day in unique_days:
        day_mask = all_day_indices == day
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0:
            daily_stats.append((0, 0.0, 'none'))
            continue
        
        day_preds = all_preds[day_indices]
        day_daily_returns = [all_daily_returns[i] for i in day_indices]
        
        sorted_local_indices = np.argsort(day_preds)[::-1]
        select_count = min(top_n_per_day, len(day_indices))
        top_local_indices = sorted_local_indices[:select_count]
        
        day_trade_returns = []
        day_exit_types = {'stop_day1': 0, 'stop_cum': 0, 'profit': 0, 'normal': 0}
        
        for idx in top_local_indices:
            daily_ret = day_daily_returns[idx]
            r1, r2, r3 = daily_ret[0], daily_ret[1], daily_ret[2]
            
            total_trades += 1
            
            if r1 < stop_loss_day1:
                if sell_at_day2_close:
                    final_ret = r1 + r2
                else:
                    final_ret = r1
                stop_loss_day1_count += 1
                day_exit_types['stop_day1'] += 1
            elif r1 + r2 < stop_loss_cum:
                final_ret = r1 + r2
                stop_loss_cum_count += 1
                day_exit_types['stop_cum'] += 1
            elif r1 + r2 + r3 >= take_profit:
                final_ret = r1 + r2 + r3
                take_profit_count += 1
                day_exit_types['profit'] += 1
            else:
                final_ret = r1 + r2 + r3
                normal_exit_count += 1
                day_exit_types['normal'] += 1
            
            day_trade_returns.append(final_ret)
        
        avg_day_return = np.mean(day_trade_returns)
        daily_returns.append(avg_day_return)
        
        exit_type = max(day_exit_types, key=day_exit_types.get)
        daily_stats.append((select_count, avg_day_return, exit_type))
    
    if len(daily_returns) > 0:
        avg_realistic_return = np.mean(daily_returns)
        cumulative_return = np.sum(daily_returns)
    else:
        avg_realistic_return = 0.0
        cumulative_return = 0.0
    
    return {
        'daily_stats': daily_stats,
        'cumulative_return': cumulative_return,
        'valid_days': len(daily_returns),
        'avg_realistic_return': avg_realistic_return,
        'total_trades': total_trades,
        'stop_loss_day1_count': stop_loss_day1_count,
        'stop_loss_cum_count': stop_loss_cum_count,
        'take_profit_count': take_profit_count,
        'normal_exit_count': normal_exit_count,
        'stop_loss_day1_ratio': stop_loss_day1_count / total_trades if total_trades > 0 else 0,
        'stop_loss_cum_ratio': stop_loss_cum_count / total_trades if total_trades > 0 else 0,
        'take_profit_ratio': take_profit_count / total_trades if total_trades > 0 else 0,
        'strategy': f'smart_exit(stop_day1={stop_loss_day1*100:.1f}%, stop_cum={stop_loss_cum*100:.1f}%, profit={take_profit*100:.1f}%)'
    }


def generate_pseudo_labels(pred_scores, original_targets,
                           pseudo_pos_ratio=0.01,
                           pseudo_neg_ratio=0.05):
    """
    统一的伪标签生成函数（按数量取Top-K%方式）

    核心思想：
    - 按预测分数排序，取前 pseudo_pos_ratio 比例的样本 → 强制标签=1.0（伪正）
    - 按预测分数排序，取倒数 pseudo_neg_ratio 比例的样本 → 强制标签=0.0（伪负）
    - 其余样本保持原始标签不变

    优点：
    - 每轮伪标签数量固定（按比例），训练更稳定
    - 过滤掉教师模型"不确定"的样本（中间部分）

    Args:
        pred_scores: 教师模型的预测分数 [batch_size] 或 numpy array
        original_targets: 原始标签 [batch_size] 或 numpy array
        pseudo_pos_ratio: 伪正标签比例（如0.01=前1%）
        pseudo_neg_ratio: 伪负标签比例（如0.05=倒数5%）

    Returns:
        pseudo_targets: 伪标签数组，与original_targets形状相同
        stats: 统计信息字典
    """
    # 转为numpy数组
    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.float().detach().cpu().numpy()
    if isinstance(original_targets, torch.Tensor):
        original_targets = original_targets.float().detach().cpu().numpy()

    pred_scores = np.asarray(pred_scores).flatten()
    original_targets = np.asarray(original_targets).copy()

    # 边界检查：如果样本数为0，直接返回空结果
    if len(pred_scores) == 0:
        stats = {
            'pseudo_pos_count': 0,
            'pseudo_neg_count': 0,
            'unchanged_count': 0,
            'threshold_pos': 0.0,
            'threshold_neg': 0.0,
        }
        return original_targets, stats

    # 计算伪正阈值：按数量取前pseudo_pos_ratio
    k_pos = max(1, int(len(pred_scores) * pseudo_pos_ratio))
    k_pos = min(k_pos, len(pred_scores))  # 确保不超过数组长度
    threshold_pos = np.sort(pred_scores)[-k_pos]  # 第k_pos大的值

    # 计算伪负阈值：按数量取倒数pseudo_neg_ratio
    k_neg = max(1, int(len(pred_scores) * pseudo_neg_ratio))
    k_neg = min(k_neg, len(pred_scores))  # 确保不超过数组长度
    threshold_neg = np.sort(pred_scores)[k_neg - 1]  # 第k_neg小的值

    # 生成伪标签
    pseudo_targets = original_targets.copy()

    # 伪正：预测值 >= threshold_pos → 强制标签=1.0
    high_mask = pred_scores >= threshold_pos
    pseudo_targets[high_mask] = 1.0

    # 伪负：预测值 <= threshold_neg → 强制标签=0.0
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

    Args:
        model_state_dict: 模型state_dict
        top_return: Top1%收益率（小数，如0.015）
        top_threshold: Top1%阈值
        auc: AUC得分
        epoch: 轮次
        model_prefix: 模型前缀（如"modelA", "modelB", "evolved"）
        extra_info: 额外信息（如教师数量）
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")

    return_str = f"{top_return*100:+.2f}".replace('+', 'p').replace('-', 'n').replace('.', '_')
    thr_str = f"{top_threshold:.3f}".replace('.', '_')
    auc_str = f"{auc:.4f}".replace('.', '_')

    if extra_info:
        filename = f"{model_prefix}_top{DataConfig.TOP_PERCENT}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{extra_info}_{timestamp}.pth"
    else:
        filename = f"{model_prefix}_top{DataConfig.TOP_PERCENT}_{return_str}pct_thr{thr_str}_auc{auc_str}_ep{epoch}_{timestamp}.pth"

    save_path = os.path.join(output_dir, filename)
    torch.save(model_state_dict, save_path)

    return save_path

def calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=1024):
    """
    计算测试集损失（官方标准：除以样本数）

    优化版本：
    - 权重在训练开始时已设置，此处直接使用
    - 支持大batch_size，提高GPU利用率

    计算方式：
    - 每个batch的loss.item()是该batch内每样本的平均损失（reduction='mean'）
    - 累加时乘以batch_size，得到所有样本的总损失
    - 最终除以总样本数，得到每样本平均损失
    """
    model.eval()
    total_loss = 0.0
    num_samples = len(eval_inputs)
    
    if num_samples == 0:
        return 0.0
    
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx],
                                       dtype=torch.float32).to(device)
            batch_targets = torch.tensor(eval_targets[start_idx:end_idx],
                                        dtype=torch.float32).to(device)

            outputs = model(batch_inputs)
            loss = criterion(outputs.squeeze(-1), batch_targets)
            total_loss += loss.item() * (end_idx - start_idx)

    return total_loss / num_samples


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

        # 用于min模式（如loss）
        self.best_loss = float('inf')
        # 用于max模式（如return）
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

        # 检查loss改善（min模式）
        if avg_loss is not None and avg_loss < self.best_loss:
            self.best_loss = avg_loss
            improved = True
            reasons.append(f'损失改善: {avg_loss:.4f}')

        # 检查收益率改善（max模式）
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

def print_sample_predictions(model, eval_inputs, eval_targets, device, num_samples=5, epoch=1):
    """
    随机打印几个样本的预测值，用于调试
    """
    model.eval()
    
    indices = random.sample(range(len(eval_inputs)), min(num_samples, len(eval_inputs)))
    
    print(f"  样本预测示例 (Epoch {epoch}):")
    with torch.no_grad():
        for idx in indices:
            input_tensor = torch.tensor(eval_inputs[idx:idx+1], dtype=torch.float32).to(device)
            pred = torch.sigmoid(model(input_tensor)).item()
            target = eval_targets[idx]
            print(f"    样本{idx}: 预测={pred:.4f}, 真实={target:.1f}")

# ==================== 梯度检测工具 ====================

class GradientMonitor:
    """
    梯度监控器：检测梯度爆炸和梯度消失
    在每个batch的backward后收集各层梯度统计信息
    """
    def __init__(self):
        self.grad_stats = {}  # {layer_name: {'norm': [], 'max': [], 'mean': [], 'nan_count': 0}}
        self.hooks = []

    def _create_hook(self, name):
        def hook(grad):
            if grad is None:
                return grad

            grad_flat = grad.data.abs().flatten()

            # 统计信息（转为float确保数值精度）
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

            # 只保留最近100个batch的统计（避免内存占用过大）
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
            # 梯度爆炸：平均范数 > 10 或 最大范数 > 100
            if stats['avg_norm'] > 10 or stats['max_norm'] > 100:
                exploding.append((name, stats))

            # 梯度消失：平均范数 < 1e-5
            elif stats['avg_norm'] < 1e-5:
                vanishing.append((name, stats))

            # 异常：出现NaN或Inf
            if stats['nan_count'] > 0 or stats['inf_count'] > 0:
                abnormal.append((name, stats))

        return exploding, vanishing, abnormal

# ==================== 训练函数 ====================

# 改进的训练函数
def train_model(model, train_stock_info, test_stock_info, epochs=TrainingConfig.EPOCHS,
               learning_rate=TrainingConfig.LEARNING_RATE, device=None,
               batch_size=TrainingConfig.BATCH_SIZE, batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH):
    """
    使用预计算训练数据集和固定评估集的训练函数（使用滚动窗口标准化避免数据泄露）
    提高训练效率，确保评估的一致性

    注意：本训练函数使用 FP32 (float32) 精度进行训练
    - 确保预测值有足够的精度进行排序
    - 避免预测值碰撞问题
    """
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)
    print("训练精度: FP32 (Float 32)")
    print("数据标准化: 滚动窗口标准化（避免数据泄露）")
    print("采样策略: 采样头在多股票上同步前进，使用正负样本池平衡")
    print(f"数据划分: 按时间划分，最近{DataConfig.TEST_DAYS}天作为测试集")
    print("="*60 + "\n")
    
    # 创建采样器（根据配置选择策略）
    print("正在初始化采样器...")
    sampler = create_sampler(train_stock_info)
    
    # 设置训练随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)

    # 创建固定的评估数据集（训练开始前创建一次，使用滚动窗口标准化）
    print("\n创建评估数据集...")
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = create_fixed_evaluation_dataset(test_stock_info)
    train_eval_inputs, train_eval_targets, train_eval_returns, _ = create_train_evaluation_dataset(train_stock_info, first_n_days=80)

    # 损失函数：由全局配置控制
    if LossConfig.use_dynamic_bce():
        print("损失函数: DynamicWeightedBCE (正样本权重4.0，负样本动态调整)")
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        
        # 测试集权重：开局算一次，整个训练过程复用（保证测试loss稳定可比）
        test_targets = np.array(eval_targets)
        test_pos_count = np.sum(test_targets >= 0.5)
        test_neg_count = np.sum(test_targets < 0.5)
        if test_pos_count > 0 and test_neg_count > 0:
            test_neg_weight = LossConfig.POS_WEIGHT * (test_pos_count / test_neg_count)
        elif test_pos_count == 0:
            test_neg_weight = float(LossConfig.POS_WEIGHT)
        else:
            test_neg_weight = 0.1
        eval_criterion.weight_0_0.fill_(test_neg_weight)
        print(f"测试集权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={test_neg_weight:.4f} (正负比例={test_pos_count}:{test_neg_count})")
    else:
        print("损失函数: 简单BCE (BCEWithLogitsLoss)")
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    # 根据配置选择优化器：AdamW相比Adam有更好的泛化性能，Mano是流形优化器
    if TrainingConfig.USE_MANO:
        # 使用HybridManoAdamW混合优化器
        from optimizers import create_optimizer
        optimizer = create_optimizer(
            model, 
            optimizer_type='mano', 
            lr=learning_rate,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            betas=TrainingConfig.MANO_ADAMW_BETAS
        )
    elif TrainingConfig.USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
        print(f"优化器: AdamW (weight_decay={TrainingConfig.WEIGHT_DECAY})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TrainingConfig.WEIGHT_DECAY)
        print(f"优化器: Adam (weight_decay={TrainingConfig.WEIGHT_DECAY})")
    
    # 计算动态预热轮次（总轮数的10%）
    warmup_epochs = max(1, int(epochs * TrainingConfig.WARMUP_RATIO))

    # 创建预热调度器
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        target_lr=learning_rate,
        start_lr=TrainingConfig.WARMUP_START_LR
    )

    # 创建主调度器
    # 注意：虽然warmup_scheduler已经将optimizer的学习率设置为start_lr，
    # 但主调度器应该基于target_lr来工作。
    # 我们在创建主调度器前先临时设置为target_lr，这样主调度器就会以正确的学习率为基准
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    # 根据配置选择主调度器
    if TrainingConfig.USE_COSINE_ANNEALING:
        # 修复：使用总轮数-预热轮数作为T_max，确保余弦退火覆盖整个训练过程
        # 避免在训练后期学习率再次上升
        total_main_epochs = epochs - warmup_epochs
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_main_epochs,  # 使用实际的主训练轮数
            eta_min=TrainingConfig.COSINE_ETA_MIN
        )
        scheduler_type = f"余弦退火(周期={total_main_epochs}轮)"
    else:
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=TrainingConfig.SCHEDULER_STEP_SIZE, 
            gamma=TrainingConfig.SCHEDULER_GAMMA
        )
        scheduler_type = "阶梯衰减"
    
    # 创建主调度器后，需要将学习率重新设置回start_lr，因为训练从预热开始
    for param_group in optimizer.param_groups:
        param_group['lr'] = TrainingConfig.WARMUP_START_LR
    
    print(f"学习率调度策略: {scheduler_type}")
    
    best_loss = float('inf')  # 使用测试集loss作为保存标准（越低越好）
    best_model_state = None  # 缓存最佳模型状态（内存中）
    best_epoch = 0  # 记录最佳模型所在轮次

    # 早停机制（patience = EPOCHS * 0.25）
    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    # 创建训练用的随机数生成器
    train_rng = random.Random(DataConfig.RANDOM_SEED)

    # 记录每轮收益率
    epoch_returns = []  # 格式: [{'turn': 1, 'return': 1.62}, ...]

    # 创建并注册梯度监控器
    print("\n正在初始化梯度监控器...")
    grad_monitor = GradientMonitor()
    grad_monitor.register_hooks(model)

    try:
        for epoch in range(epochs):
            # 新epoch开始，重置梯度统计
            grad_monitor.reset()

            model.train()
            total_loss = 0
            num_valid_steps = 0

            # 训练阶段 - 更新学习率
            if warmup_scheduler.is_warmup_phase():
                # 预热阶段：使用预热调度器
                current_lr = warmup_scheduler.step(epoch)
                lr_status = f"预热阶段 ({epoch + 1}/{warmup_epochs})"
            else:
                # 预热结束后：使用主调度器获取当前学习率
                current_lr = main_scheduler.get_last_lr()[0]
                lr_status = "正常训练"

            # 显示当前采样进度
            current_samples, total_samples = sampler.get_progress()
            progress_pct = current_samples / total_samples * 100 if total_samples > 0 else 0
            print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status}), 采样进度: {current_samples}/{total_samples} ({progress_pct:.1f}%)')

            # 使用采样器生成当前epoch的训练数据
            print(f'  使用时间顺序采样器生成数据...')
            epoch_inputs, epoch_targets = sample_with_pools(
                sampler, train_stock_info, batch_size, batches_per_epoch, train_rng
            )

            # 将数据转换为tensor并移到设备上 (使用FP32精度)
            epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.float32).to(device)
            epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.float32).to(device)

            # 计算实际可用的batch数量（防止索引越界）
            actual_batches = len(epoch_inputs_tensor) // batch_size
            if actual_batches < batches_per_epoch:
                print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

            # 训练循环：使用实际的batch数量，而不是固定的batches_per_epoch
            num_samples = len(epoch_inputs)
            for step in range(actual_batches):
                start_idx = step * batch_size
                end_idx = (step + 1) * batch_size  # 不需要min，因为actual_batches已经保证了不越界

                batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
                batch_targets = epoch_targets_tensor[start_idx:end_idx]

                # 动态更新权重：根据当前batch的正负样本比例（仅DynamicWeightedBCE需要）
                if hasattr(criterion, 'update_weights'):
                    criterion.update_weights(batch_targets)

                optimizer.zero_grad()
                output = model(batch_inputs)
                loss = criterion(output.squeeze(), batch_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
                optimizer.step()

                # 累加loss时乘以batch_size，得到该batch的总损失
                total_loss += loss.item() * (end_idx - start_idx)

                # 实时更新进度显示
                progress = (step + 1) / actual_batches * 100
                # 使用已处理的样本数计算当前平均损失
                processed_samples = (step + 1) * batch_size
                avg_loss = total_loss / processed_samples
                print(f'\r  训练进度: {progress:.1f}% ({step + 1}/{actual_batches}), 平均损失: {avg_loss:.4f}', end='', flush=True)

            # 训练循环结束，计算最终的训练集平均损失
            # 除以总样本数（官方标准），与测试损失计算方式保持一致
            # total_loss已经是所有样本的总损失（累加时乘以了batch_size）
            train_loss_epoch = total_loss / num_samples

            print()  # 换行
            print()  # 空行

            # 清理数据以释放内存
            del epoch_inputs_tensor, epoch_targets_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 更新学习率
            # 注意：预热阶段的学习率已经在epoch开始时由warmup_scheduler.step()更新
            # 只有预热结束后才使用主调度器
            if not warmup_scheduler.is_warmup_phase():
                main_scheduler.step()  # 更新主调度器（余弦退火或阶梯衰减）

            # ==================== 梯度诊断报告 ====================
            print("  梯度状态检查:")
            exploding, vanishing, abnormal = grad_monitor.diagnose()

            if exploding:
                print(f'    🔴 梯度爆炸警告 ({len(exploding)}层):')
                for name, stats in exploding[:3]:  # 只显示前3层
                    print(f'      - {name}: avg_norm={stats["avg_norm"]:.4f}, max_norm={stats["max_norm"]:.4f}')
                if len(exploding) > 3:
                    print(f'      ... 还有 {len(exploding) - 3} 层')

            if vanishing:
                print(f'    🟡 梯度消失警告 ({len(vanishing)}层):')
                for name, stats in vanishing[:3]:  # 只显示前3层
                    print(f'      - {name}: avg_norm={stats["avg_norm"]:.2e}, avg_max={stats["avg_max"]:.2e}')
                if len(vanishing) > 3:
                    print(f'      ... 还有 {len(vanishing) - 3} 层')

            if abnormal:
                print(f'    ⚠️ 梯度异常 ({len(abnormal)}层):')
                for name, stats in abnormal[:3]:
                    issues = []
                    if stats['nan_count'] > 0:
                        issues.append(f'NaN×{stats["nan_count"]}')
                    if stats['inf_count'] > 0:
                        issues.append(f'Inf×{stats["inf_count"]}')
                    print(f'      - {name}: {", ".join(issues)}')
                if len(abnormal) > 3:
                    print(f'      ... 还有 {len(abnormal) - 3} 层')

            if not exploding and not vanishing and not abnormal:
                print(f'    ✅ 梯度正常')

            # 打印各层梯度摘要（前5层 + 输出层）
            summary = grad_monitor.get_epoch_summary()
            layer_names = list(summary.keys())
            print('    各层梯度范数摘要:')
            for name in layer_names[:3]:
                s = summary[name]
                print(f'      {name}: avg_norm={s["avg_norm"]:.6f}, max={s["avg_max"]:.6f}')
            if len(layer_names) > 3:
                print(f'      ... (共 {len(layer_names)} 层)')
            # 显示输出层
            for name in layer_names:
                if 'output' in name.lower():
                    s = summary[name]
                    print(f'      {name}: avg_norm={s["avg_norm"]:.6f}, max={s["avg_max"]:.6f}')
            # ===================================================

            # 固定评估集评估
            total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats, realistic_stats, smart_exit_stats, dispersion_stats, all_preds = evaluate_model_batch(
                model, eval_inputs, eval_targets, eval_cumulative_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE, eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns
            )

            # 计算训练集收益率（用于检测过拟合）
            _, _, _, _, _, _, _, _, train_top_stats, _, _, _, _ = evaluate_model_batch(
                model, train_eval_inputs, train_eval_targets, train_eval_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE
            )

            # 计算测试集损失（使用全局权重，batch_size=1024）
            test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device)

            # 记录当前轮次收益率（必须在test_loss计算之后）
            epoch_return = {
                'turn': epoch + 1,
                'return': top_stats['avg_return'] * 100,
                'train_loss': train_loss_epoch,
                'test_loss': test_loss,
                'dispersion_std': dispersion_stats['std'],
                'dispersion_range': dispersion_stats['range'],
                'dispersion_iqr': dispersion_stats['iqr'],
                'pos_ratio': dispersion_stats['pos_ratio'],
                'high_conf_ratio': dispersion_stats['high_conf_ratio'],
            }
            epoch_returns.append(epoch_return)

            # 随机挑选5组样本打印模型输出值
            print_sample_predictions(model, eval_inputs, eval_targets, device, num_samples=5, epoch=epoch+1)

            # 打印详细结果
            class_names = ['不上涨', '上涨']
            for i in range(2):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    print(f'  {class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.3f}')
                else:
                    print(f'  {class_names[i]}: 0/0 = 0.000 (无样本)')

            # 计算上涨准确率（预测上涨后真上涨的概率）
            if pred_positive_total > 0:
                precision = pred_positive_correct / pred_positive_total
                non_negative_rate = pred_non_negative / pred_positive_total
                print(f'  上涨准确率: {pred_positive_correct}/{pred_positive_total} = {precision:.3f} 准确率: {pred_non_negative}/{pred_positive_total} = {non_negative_rate:.3f}')
            else:
                print(f'  上涨准确率: 0/0 = 0.000 (无预测上涨)')

            # 打印置信度区间的精确度统计
            print(f'  置信度区间精确度:')
            for interval in ['0.50-0.55', '0.55-0.58', '0.58-0.60', '0.60-0.70', '0.70-1.00']:
                correct, total_pred, non_negative = confidence_stats[interval]
                if total_pred > 0:
                    precision = correct / total_pred
                    non_negative_rate = non_negative / total_pred
                    print(f'    {interval}: 上涨准确={correct}/{total_pred}={precision:.3f}, 非负准确={non_negative}/{total_pred}={non_negative_rate:.3f}')
                else:
                    print(f'    {interval}: 无预测')

            overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0

            print(f'  总体准确率: {overall_acc:.3f}')

            # 收益率对比（训练集 vs 测试集）- 用于检测过拟合
            train_return_pct = train_top_stats["avg_return"] * 100
            test_return_pct = top_stats["avg_return"] * 100
            return_gap = train_return_pct - test_return_pct

            print(f'  【过拟合检测】Top{DataConfig.TOP_PERCENT}%收益率对比:')
            print(f'    训练集: {train_return_pct:+.2f}% (样本数={train_top_stats["count"]})')
            print(f'    测试集: {test_return_pct:+.2f}% (样本数={top_stats["count"]})')
            print(f'    差距: {return_gap:+.2f}% ', end='')
            if return_gap > 1.0:
                print('⚠️ 过拟合风险：训练集明显高于测试集')
            elif return_gap < -0.5:
                print('⚠️ 欠拟合：测试集高于训练集（罕见）')
            else:
                print('✓ 正常')
            
            # 实战收益率统计
            if realistic_stats is not None:
                daily_stats_str = ', '.join([f'({count},{day_ret*100:.1f}%)' for count, day_ret in realistic_stats['daily_stats']])
                mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if realistic_stats.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
                print(f'  【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
                print(f'  【实战收益率({mode_str})】平均实战收益率: {realistic_stats["avg_realistic_return"]*100:.1f}%')
            
            # 智能止损策略统计
            if smart_exit_stats is not None:
                print(f'  【智能止损策略】平均收益率: {smart_exit_stats["avg_realistic_return"]*100:.1f}%')
                print(f'  【智能止损策略】总交易: {smart_exit_stats["total_trades"]}次')
                print(f'  【智能止损策略】Day1止损: {smart_exit_stats["stop_loss_day1_count"]}次({smart_exit_stats["stop_loss_day1_ratio"]*100:.1f}%)')
                print(f'  【智能止损策略】累计止损: {smart_exit_stats["stop_loss_cum_count"]}次({smart_exit_stats["stop_loss_cum_ratio"]*100:.1f}%)')
                print(f'  【智能止损策略】止盈: {smart_exit_stats["take_profit_count"]}次({smart_exit_stats["take_profit_ratio"]*100:.1f}%)')
            
            print(f'  AUC得分: {auc_score:.4f}')
            print(f'  训练集损失: {train_loss_epoch:.4f}, 测试集损失: {test_loss:.4f}')
            
            print_dispersion_sparkline(all_preds, epoch_returns)

            # 早停检测
            improved, improve_reason = early_stopping.check_improve(
                avg_loss=test_loss,
                top_return=top_stats['avg_return'],
                auc=auc_score,
                threshold=top_stats.get('threshold', 0.0)
            )

            if improved:
                no_improve_count, patience_limit = early_stopping.get_progress()
                print(f'  ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
            else:
                no_improve_count, patience_limit = early_stopping.get_progress()
                print(f'  ⚠ 无改善 ({no_improve_count}/{patience_limit})')

            # 保存最佳模型（使用测试集loss作为主要标准，同时监控AUC）
            MIN_AUC = DataConfig.MIN_AUC

            # 判断是否保存模型
            should_save = False
            save_reason = ""

            if auc_score < MIN_AUC:
                print(f'  ⚠ AUC过低({auc_score:.4f}<{MIN_AUC})，模型分类能力不足，暂不更新')
            elif test_loss < best_loss:
                should_save = True
                save_reason = f'测试集Loss降低: {best_loss:.4f} → {test_loss:.4f}'

            if should_save:
                best_loss = test_loss
                best_epoch = epoch + 1
                # 缓存模型状态到内存（深拷贝），不立即写入磁盘
                import copy
                best_model_state = copy.deepcopy(model.state_dict())
                print(f'  ✓ 发现更好的模型！{save_reason}（已缓存到内存）')
                print(f'    详情: AUC={auc_score:.4f}, Top{DataConfig.TOP_PERCENT}%收益: 平均={top_stats["avg_return"]*100:+.2f}%, 累计={top_stats["total_return"]*100:+.2f}%')

            # 早停检查
            if early_stopping.should_stop():
                print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
                break

            print("-" * 50)

    finally:
        # 训练结束或异常时，移除梯度监控hooks
        grad_monitor.remove_hooks()
        print("\n梯度监控器已移除")

    # 训练结束后，将最佳模型保存到磁盘
    if best_model_state is not None:
        print("\n" + "=" * 50)
        print(f"训练完成！正在保存最佳模型...")
        print(f"最佳模型来自第 {best_epoch} 轮，测试集Loss: {best_loss:.4f}")
        torch.save(best_model_state, ModelSaveConfig.get_best_model_path())
        print(f"✓ 最佳模型已保存到: {ModelSaveConfig.get_best_model_path()}")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠ 警告：未找到符合条件的最佳模型（AUC要求未达标）")
        print("=" * 50)

    # 保存每轮收益率到CSV（使用时间戳避免多模型训练时覆盖）
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"baseline_epoch_returns_{timestamp}.csv")
    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['turn', 'return', 'train_loss', 'test_loss', 'dispersion_std', 'dispersion_range', 'dispersion_iqr', 'pos_ratio', 'high_conf_ratio'])
        writer.writeheader()

        for epoch_return in epoch_returns:
            row = {
                'turn': epoch_return['turn'],
                'return': f"{epoch_return['return']:.2f}",
                'train_loss': f"{epoch_return['train_loss']:.4f}",
                'test_loss': f"{epoch_return['test_loss']:.4f}",
                'dispersion_std': f"{epoch_return['dispersion_std']:.4f}",
                'dispersion_range': f"{epoch_return['dispersion_range']:.4f}",
                'dispersion_iqr': f"{epoch_return['dispersion_iqr']:.4f}",
                'pos_ratio': f"{epoch_return['pos_ratio']:.4f}",
                'high_conf_ratio': f"{epoch_return['high_conf_ratio']:.4f}",
            }
            writer.writerow(row)

    print(f"✓ 每轮收益率已保存: {os.path.basename(returns_csv_path)}")
    print(f"  共记录 {len(epoch_returns)} 轮训练数据")
    print("=" * 50)

if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 打印配置摘要
    print_config_summary()
    
    # 获取设备信息
    device = DeviceConfig.print_device_info()

    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    
    # 使用改进的数据加载函数（按时间划分，避免数据泄露）
    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()

    # 打印数据集统计信息
    print("\n" + "="*60)
    print("数据集划分统计")
    print("="*60)
    
    train_lengths = [info['data_length'] for info in train_stock_info]
    test_lengths = [info['data_length'] for info in test_stock_info]
    
    print(f"训练集:")
    print(f"  股票数量: {len(train_stock_info)}")
    print(f"  数据长度: 最小={min(train_lengths)}, 最大={max(train_lengths)}, 平均={np.mean(train_lengths):.1f}")

    print(f"\n测试集:")
    print(f"  股票数量: {len(test_stock_info)}")
    print(f"  数据长度: 最小={min(test_lengths)}, 最大={max(test_lengths)}, 平均={np.mean(test_lengths):.1f}")
    print(f"  时间范围: 每只股票的最近 {DataConfig.TEST_DAYS} 天")

    print(f"\n前3只股票示例:")
    for i in range(min(3, len(train_stock_info))):
        train_info = train_stock_info[i]
        print(f"  {train_info['file_name']}: 训练集长度={train_info['data_length']}")

    print("="*60)

    print("正在创建 Transformer 模型 (FP32精度)...")
    model = create_model().to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    print("开始训练...")
    # 使用带固定评估集的训练函数（使用滚动窗口标准化）
    train_model(model, train_stock_info, test_stock_info, device=device)
    
    # 保存最终模型（训练结束时的状态）
    final_model_path = ModelSaveConfig.get_final_model_path(ModelConfig.D_MODEL)
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")