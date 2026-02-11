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
                   DeviceConfig, ModelSaveConfig,
                   print_config_summary)

# 从 model.py 导入模型定义
from model import EnhancedStockTransformer

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
            # BF16需要先转为FP32再转numpy
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
        inputs: [batch_size, 1] 模型输出的logits (BF16)
        targets: [batch_size] 真实标签 (1.0/0.0) (BF16)
        """
        # 确保输入形状正确：如果是 [batch_size, 1] 则 squeeze(-1) 变成 [batch_size]
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(-1)

        # BF16训练时，这里用FP32计算loss更稳定
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

# ==================== 模型定义已移至 model.py ====================
# 以下类已迁移到 src/model.py:
# - PositionalEncoding
# - MultiHeadAttention
# - TransformerLayer
# - EnhancedStockTransformer
# - init_weights

# 单个文件处理函数（用于多进程）
def process_single_file(args):
    """
    处理单个文件，返回原始数据（不做全局标准化，避免数据泄露）
    
    采样边界设计（确保训练集和测试集完全不交叠）：
    - 测试集：最后 test_days (80) 天，完全冻结
    - 训练集最后一个样本：需要 REQUIRED_LENGTH (63) 天（60上下文+3预测）
    - 指针到达末尾后该股票不再参与训练
    - 为了不交叠：训练集末位置 = 总长度 - test_days - REQUIRED_LENGTH = 总长度 - 143
    - 最低数据长度：至少 REQUIRED_LENGTH + test_days = 143 天
    
    指针位置：
    - 训练集指针初始位置：2021年起始位置（如果还未上市则为数据第一天）
    - 训练集指针末位置：总长度 - test_days - REQUIRED_LENGTH
    """
    file_path, file_name, test_days, train_start_year = args
    try:
        df = pd.read_csv(file_path)
        
        # 🔑 关键：数据文件是按时间倒序排列的（最新在前），需要反转为正序（最早在前）
        df = df.iloc[::-1].reset_index(drop=True)
        
        # 使用6维特征：OHLC + volume + exchange(换手率)
        data = df[['start', 'max', 'min', 'end', 'volume', 'exchange']].values
        times = df['time'].values  # 日期列，格式为YYYYMMDD整数
        
        data_length = len(data)
        required_length = DataConfig.REQUIRED_LENGTH  # 63天
        
        # 🔑 边界检查1：最低数据长度
        # 为了生成至少一个训练样本，需要：
        # - 测试集：test_days (80天)
        # - 训练样本：REQUIRED_LENGTH (63天)
        # - 采样起始需要前一天数据：1天
        # 总计：80 + 63 + 1 = 144天（保守估计，实际可能需要更多）
        # 但为了简化，我们使用 test_days + required_length 作为初步筛选
        min_required_length = required_length + test_days  # 63 + 80 = 143
        if data_length < min_required_length:
            return None  # 数据不足143天，直接丢弃
        
        # 🔑 训练集末位置 = 总长度 - test_days - REQUIRED_LENGTH
        # 这样训练集最后一个样本的最后一天不会进入测试集的80天范围
        train_end_idx = data_length - test_days - required_length
        
        # 测试集起始位置（用于评估时的采样）
        test_split_point = data_length - test_days
        
        # 🔑 计算训练集起始位置：2021年起始位置（如果还未上市则为数据第一天）
        train_start_date = train_start_year * 10000 + 101  # 如2021 -> 20210101
        valid_indices = np.where(times >= train_start_date)[0]
        
        if len(valid_indices) > 0:
            # 股票在2021年前已上市，从2021年开始
            train_start_idx = valid_indices[0]
        else:
            # 股票在2021年后才上市，从第一天开始
            train_start_idx = 0
        
        # 🔑 边界检查2：训练集起始位置必须在末位置之前
        if train_start_idx >= train_end_idx:
            return None  # 没有有效的训练数据
        
        # 🔑 边界检查3：训练数据长度必须足够生成至少一个样本
        train_length = train_end_idx - train_start_idx
        if train_length < required_length:
            return None  # 训练数据不足，丢弃
        
        # 提取训练数据（保留完整数据，记录起止索引供采样器使用）
        train_data = data.copy()  # 保留完整数据
        test_data = data.copy()   # 测试集也需要完整数据（用于上下文）
        
        stock_info = {
            'file_name': file_name,
            'data_length': data_length,
            'train_data': train_data,
            'test_data': test_data,
            'train_start_idx': train_start_idx,  # 训练集采样起始索引
            'train_end_idx': train_end_idx,      # 训练集采样末索引（不含）
            'train_length': train_length,
            'test_split_point': test_split_point  # 测试集起始位置（总长度-80）
        }
        
        return stock_info
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return None

# 数据预处理函数（按时间划分训练集和测试集）
def load_and_preprocess_data(data_dir=DataConfig.DATA_DIR, test_days=DataConfig.TEST_DAYS, train_start_year=DataConfig.TRAIN_START_YEAR):
    """
    数据加载和预处理，使用多进程并行加载
    
    采样边界设计：
    - 训练集：从2021年（或上市日）到 总长度-test_days-REQUIRED_LENGTH
    - 测试集：最近test_days天
    - 最低数据要求：test_days + REQUIRED_LENGTH = 143天
    """
    from multiprocessing import Pool, cpu_count
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    all_files.sort()
    
    print(f"总共 {len(all_files)} 只股票文件")
    print(f"划分策略:")
    print(f"  - 训练集: {train_start_year}年起（或上市日）到 总长度-{test_days}-{DataConfig.REQUIRED_LENGTH}")
    print(f"  - 测试集: 最近 {test_days} 天")
    print(f"  - 最低数据要求: {test_days + DataConfig.REQUIRED_LENGTH} 天（测试集{test_days}天 + 训练样本{DataConfig.REQUIRED_LENGTH}天）")
    
    # 处理所有文件
    file_args = [(os.path.join(data_dir, f), f, test_days, train_start_year) for f in all_files]
    num_workers = min(cpu_count(), 8)
    
    with Pool(num_workers) as pool:
        all_stock_info = [r for r in pool.map(process_single_file, file_args) if r is not None]
    
    # 统计丢弃数量
    discarded_count = len(all_files) - len(all_stock_info)
    print(f"有效股票: {len(all_stock_info)} 只，丢弃: {discarded_count} 只")
    
    # 分离训练和测试数据
    train_stock_info = []
    test_stock_info = []
    
    for stock_info in all_stock_info:
        # 训练集：保留完整数据和采样边界索引
        train_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['train_data'],
            'data_length': stock_info['data_length'],
            'train_start_idx': stock_info['train_start_idx'],  # 采样起始索引
            'train_end_idx': stock_info['train_end_idx'],      # 采样末索引
        })
        
        # 测试集：保留完整数据和测试起始位置
        test_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['test_data'],
            'data_length': stock_info['data_length'],
            'test_split_point': stock_info['test_split_point']
        })
    
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    
    return train_stock_info, test_stock_info

# 计算股票选择权重
def calculate_stock_weights(stock_info_list):
    """
    计算每只股票的采样权重
    数据量越大的股票权重越大，但最大不超过平均值的1.5倍
    """
    data_lengths = [info['data_length'] for info in stock_info_list]
    avg_length = np.mean(data_lengths)
    
    # 计算权重：数据长度 / 平均长度，但限制在1.0到1.5之间
    weights = []
    for length in data_lengths:
        weight = length / avg_length
        weight = max(1.0, min(1.5, weight))  # 限制在1.0到1.5之间
        weights.append(weight)
    
    # 归一化权重，使其总和为1.0（np.random.choice要求）
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    return normalized_weights

# 解决样本交叠导致的数据泄露问题：时间只向前推进，不回头

class TemporalSampler:
    """
    时间顺序采样器：采样头在多个股票上同步向前移动，不回头
    
    采样边界设计：
    - 每只股票的指针初始位置 = train_start_idx（2021年起始位置，或上市第一天）
    - 每只股票的指针末位置 = train_end_idx（总长度-80-63=总长度-143）
    - 指针到达末尾后该股票不再参与训练
    
    核心算法：
    1. 计算总样本数和每个epoch需要的样本数
    2. 将总样本数均匀分配到各个epoch
    3. 每个epoch采样固定数量的"轮次"，确保最后一个epoch恰好到达最新时间
    4. 每轮从所有股票当前位置各取一个样本，然后指针前进
    """
    def __init__(self, stock_info_list):
        self.stock_info_list = stock_info_list
        self.required_length = DataConfig.REQUIRED_LENGTH

        # 每只股票的采样位置指针（初始化为各自的train_start_idx + 1，因为需要前一天数据）
        self.stock_positions = []

        # 每只股票的初始采样位置（用于计算进度）
        self.stock_start_positions = []

        # 每只股票的最大可用位置
        self.stock_max_positions = []
        
        # 是否可以循环采样（数据长度 > 600）
        self.can_loop = []
        
        # 记录每个股票的循环次数
        self.loop_counts = [0] * len(stock_info_list)
        
        for stock_info in stock_info_list:
            train_start_idx = stock_info.get('train_start_idx', 0)
            train_end_idx = stock_info.get('train_end_idx', len(stock_info['data']))
            data_length = stock_info.get('data_length', 0)
            
            # 采样起始位置：train_start_idx + 1（需要前一天数据作为基准）
            start_pos = max(1, train_start_idx + 1)
            
            # 采样末位置：直接使用train_end_idx
            # train_end_idx = data_length - test_days - REQUIRED_LENGTH 已经考虑了边界
            # start_idx 最大就是 train_end_idx，不需要再减 REQUIRED_LENGTH
            max_pos = train_end_idx
            
            # 确保起始位置不超过末位置
            if start_pos > max_pos:
                start_pos = max_pos + 1  # 设置为无效，该股票不参与采样
            
            self.stock_positions.append(start_pos)
            self.stock_start_positions.append(start_pos)  # 保存初始位置用于进度计算
            self.stock_max_positions.append(max_pos)
            
            # 只有数据长度 > 600 的股票才允许循环采样
            self.can_loop.append(data_length > 600)

        # 统计信息
        valid_stocks = sum(1 for i in range(len(stock_info_list)) 
                         if self.stock_positions[i] <= self.stock_max_positions[i])
        total_samples = sum(max(0, self.stock_max_positions[i] - self.stock_positions[i] + 1) 
                          for i in range(len(stock_info_list)))
        
        # 检查是否有有效股票
        if valid_stocks == 0:
            raise ValueError(
                f"没有有效的训练股票！\n"
                f"  总股票数: {len(stock_info_list)}\n"
                f"  请检查数据质量或调整参数"
            )
        
        print(f"  初始化采样器: {valid_stocks}只有效股票, 总样本数={total_samples}")
        print(f"  采样策略: 时间顺序前进，支持循环采样（数据长度>600的股票）")

    def sample_batch_rounds(self, num_rounds):
        """
        批量采样多轮：一次性生成多轮的样本索引，提高效率

        参数:
            num_rounds: 要采样的轮数

        返回: [(stock_idx, start_idx), ...] 所有轮次的样本索引列表
        """
        all_samples = []

        for _ in range(num_rounds):
            for stock_idx in range(len(self.stock_info_list)):
                current_pos = self.stock_positions[stock_idx]
                max_pos = self.stock_max_positions[stock_idx]

                # 如果到达末尾且允许循环，重置到起点
                if current_pos > max_pos and self.can_loop[stock_idx]:
                    current_pos = self.stock_start_positions[stock_idx]
                    self.stock_positions[stock_idx] = current_pos
                    self.loop_counts[stock_idx] += 1

                # 检查是否还有可用样本
                if current_pos <= max_pos:
                    all_samples.append((stock_idx, current_pos))
                    # 指针前进一步
                    self.stock_positions[stock_idx] += 1

            # 如果所有股票都已到达终点（且无法循环），提前退出
            if not any(self.stock_positions[i] <= self.stock_max_positions[i] 
                      for i in range(len(self.stock_info_list))):
                break

        return all_samples
    
    def get_progress(self):
        """获取当前采样进度"""
        total_samples = 0
        current_samples = 0
        for start_pos, pos, max_pos in zip(self.stock_start_positions, self.stock_positions, self.stock_max_positions):
            if start_pos <= max_pos:
                total_samples += max_pos - start_pos + 1
                current_samples += max(0, min(pos, max_pos + 1) - start_pos)
        return current_samples, total_samples

    def get_loop_stats(self):
        """获取循环统计信息"""
        looped_stocks_count = sum(1 for c in self.loop_counts if c > 0)
        total_loops = sum(self.loop_counts)
        return looped_stocks_count, total_loops

def generate_sample_from_index(stock_info_list, stock_idx, start_idx):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引

    返回: (input_seq, target) 或 None（如果样本无效）
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    required_length = DataConfig.REQUIRED_LENGTH

    # 提取原始数据窗口（包含前一天用于计算第一天的涨跌幅）
    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    prev_day_data = stock_data[start_idx - 1]

    # 快速检查：避免除零
    prev_close = prev_day_data[3]
    prev_volume = prev_day_data[4]
    if prev_close == 0 or prev_volume == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    # 向量化检查：所有收盘价和成交量都不为0
    closes = input_seq_raw[:, 3]
    volumes = input_seq_raw[:, 4]
    if np.any(closes == 0) or np.any(volumes == 0):
        return None

    # 🔥 新增（第一阶段）：过滤非10%涨跌幅限制的股票样本
    # 检查整个样本窗口（CONTEXT_LENGTH + FUTURE_DAYS）内的任意一天涨跌幅是否超过11%
    # 如果超过，说明该股票可能是科创板/创业板/北交所等（20%或30%涨跌幅限制），踢除整个样本
    # 使用11%作为阈值（留1%余量，避免误判正常10%股票的微小波动）
    sample_window_start = start_idx - 1  # 包含前一天（用于计算第1天涨跌幅）
    sample_window_end = start_idx + required_length  # 包含未来FUTURE_DAYS
    sample_data = stock_data[sample_window_start:sample_window_end]

    limit_threshold = 0.11  # 11%阈值

    for day_idx in range(1, len(sample_data)):
        today_close = sample_data[day_idx, 3]
        yesterday_close = sample_data[day_idx - 1, 3]

        if yesterday_close > 0:
            daily_return = (today_close - yesterday_close) / yesterday_close
            # 检查是否超过涨跌幅限制（涨幅或跌幅）
            if abs(daily_return) > limit_threshold:
                return None  # 过滤掉非10%涨跌幅限制的股票样本

    # 🔥 新增（第二阶段）：过滤涨停样本（在计算标签之前）
    # 检查第60天（最后一天）的涨幅是否>=9.5%
    last_day_idx = start_idx + context_length - 1
    prev_day_idx = start_idx + context_length - 2
    prev_day_close = stock_data[prev_day_idx, 3]
    last_day_close = stock_data[last_day_idx, 3]

    if prev_day_close > 0:
        last_day_return = (last_day_close - prev_day_close) / prev_day_close
        # 涨停阈值：9.5%
        if last_day_return >= 0.095:
            return None  # 过滤掉涨停样本

    # 特征标准化：向量化计算
    input_seq = np.empty((context_length, 6), dtype=np.float32)
    
    # 价格特征：相对涨跌幅（向量化）
    # 第1天相对于前一天
    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    # 第2-N天相对于前一天（向量化）
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]
    
    # 成交量特征：相对变化（向量化）
    input_seq[0, 4] = (volumes[0] - prev_volume) / prev_volume
    if context_length > 1:
        input_seq[1:, 4] = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    
    # 换手率：直接归一化（向量化）
    input_seq[:, 5] = input_seq_raw[:, 5] / 100.0
    
    # 数值范围限制（向量化）
    # 价格特征：限制在[-0.1, 0.1]，对应±10%涨跌幅
    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])
    np.clip(input_seq[:, 4], -5.0, 5.0, out=input_seq[:, 4])
    input_seq[:, 4] = input_seq[:, 4] / 10.0 + 0.5
    np.clip(input_seq[:, 4:6], 0.0, 1.0, out=input_seq[:, 4:6])

    # NaN/Inf检测（向量化）
    if np.any(~np.isfinite(input_seq)):
        return None

    # 计算标签
    original_start_price = closes[-1]  # 当前收盘价
    original_end_price = stock_data[start_idx + required_length - 1, 3]  # N天后收盘价

    if original_start_price == 0:
        return None

    cumulative_return = (original_end_price - original_start_price) / original_start_price

    # 二分类标签
    target = 1.0 if cumulative_return >= DataConfig.UPRISE_THRESHOLD else 0.0

    return input_seq, target

def sample_with_pools(sampler, stock_info_list, batch_size, batches_per_epoch, rng):
    """
    使用样本池机制采样（流式处理版）：
    1. 按时间顺序遍历样本索引
    2. 实时填充正负样本池
    3. 一旦正样本达到配额且负样本足够，立即生成Batch并清空负样本池
    4. 确保Batch之间的时间有序性，严格防止未来数据泄露到过去的Batch中
    5. 支持循环采样：数据到达末尾后自动循环回起点
    6. 动态生成索引：按需生成，直到batch数量满足要求
    """
    positive_ratio = 0.25
    pos_quota = max(1, int(batch_size * positive_ratio))
    neg_quota = batch_size - pos_quota

    pos_pool_inputs = []
    pos_pool_targets = []
    neg_pool_inputs = []
    neg_pool_targets = []

    all_batch_inputs = []
    all_batch_targets = []

    batches_generated = 0
    
    # 初始估算：每次生成一批索引
    initial_rounds = 50  # 先生成50轮试试
    total_rounds_generated = 0
    total_indices_generated = 0
    
    print(f"    动态采样策略：按需生成索引，直到满足{batches_per_epoch}个batch...")
    
    # 动态生成循环：不断生成新索引，直到batch够了
    while batches_generated < batches_per_epoch:
        # 生成一批新索引
        sample_indices = sampler.sample_batch_rounds(initial_rounds)
        
        if len(sample_indices) == 0:
            print(f"\n    ⚠ 警告：采样头已到达所有股票终点且无法循环，停止采样")
            break
        
        total_rounds_generated += initial_rounds
        total_indices_generated += len(sample_indices)
        
        # 批量处理索引（优化：减少函数调用开销）
        for stock_idx, start_idx in sample_indices:
            # 如果已经生成了足够的Batch，提前结束
            if batches_generated >= batches_per_epoch:
                break

            sample = generate_sample_from_index(stock_info_list, stock_idx, start_idx)
            if sample is None:
                continue

            input_seq, target = sample

            if target >= 0.5:  # 正样本
                pos_pool_inputs.append(input_seq)
                pos_pool_targets.append(target)
            else:  # 负样本
                neg_pool_inputs.append(input_seq)
                neg_pool_targets.append(target)
            
            # 检查是否可以生成一个Batch
            # 触发条件：正样本达到配额 AND 负样本也足够
            if len(pos_pool_inputs) >= pos_quota and len(neg_pool_inputs) >= neg_quota:
                # 1. 取正样本：按时间顺序取最早进入池子的样本（FIFO）
                batch_pos_inputs = pos_pool_inputs[:pos_quota]
                batch_pos_targets = pos_pool_targets[:pos_quota]
                
                # 2. 取负样本：从当前积累的负样本池中随机抽取（实现"随机丢弃"逻辑）
                neg_indices = rng.sample(range(len(neg_pool_inputs)), neg_quota)
                batch_neg_inputs = [neg_pool_inputs[i] for i in neg_indices]
                batch_neg_targets = [neg_pool_targets[i] for i in neg_indices]
                
                # 3. 合并并打乱
                batch_inputs = batch_pos_inputs + batch_neg_inputs
                batch_targets = batch_pos_targets + batch_neg_targets
                
                combined = list(zip(batch_inputs, batch_targets))
                rng.shuffle(combined)
                b_inputs, b_targets = zip(*combined)
                
                all_batch_inputs.extend(b_inputs)
                all_batch_targets.extend(b_targets)
                
                batches_generated += 1
                
                # 4. 更新池子状态
                pos_pool_inputs = pos_pool_inputs[pos_quota:]
                pos_pool_targets = pos_pool_targets[pos_quota:]
                neg_pool_inputs = []
                neg_pool_targets = []
        
        # 打印进度（移到外层，减少打印次数）
        print(f"    已生成 {batches_generated}/{batches_per_epoch} 个Batch (已采样{total_rounds_generated}轮)", end='\r', flush=True)
        
        # 如果这批索引处理完了还不够，继续生成下一批
        if batches_generated < batches_per_epoch:
            # 根据当前进度调整下次生成的轮数
            remaining_batches = batches_per_epoch - batches_generated
            if batches_generated > 0:
                estimated_rounds = max(20, int(remaining_batches / batches_generated * total_rounds_generated * 1.2))
                initial_rounds = min(estimated_rounds, 100)
            else:
                initial_rounds = 100

    print(f"\n    已生成 {batches_generated}/{batches_per_epoch} 个batch (总共采样{total_rounds_generated}轮, {total_indices_generated}个索引)")
    
    if batches_generated < batches_per_epoch:
        print(f"    ⚠ 警告：样本不足，仅生成 {batches_generated} 个Batch (目标: {batches_per_epoch})")
        # 如果生成的不够，也返回已有的，而不是报错（防止训练中断）
        if batches_generated == 0:
             raise ValueError(f"样本严重不足：无法生成任何Batch")

    return np.asarray(all_batch_inputs), np.asarray(all_batch_targets)

def create_fixed_evaluation_dataset(test_stock_info):
    """创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）"""
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []

    required_length = DataConfig.REQUIRED_LENGTH
    context_length = DataConfig.CONTEXT_LENGTH

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))

        start_min = max(1, test_split_point)
        start_max = data_length - required_length
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx)
            if sample is None:
                continue

            input_seq, target = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)

            # 计算未来收益率
            original_start_price = stock_data[start_idx + context_length - 1, 3]
            original_end_price = stock_data[start_idx + required_length - 1, 3]
            if original_start_price == 0:
                continue
            cumulative_return = (original_end_price - original_start_price) / original_start_price
            eval_cumulative_returns.append(float(cumulative_return))

    if len(eval_inputs) == 0:
        raise ValueError("固定评估集为空：test_stock_info中没有可用样本")

    return np.asarray(eval_inputs), np.asarray(eval_targets), np.asarray(eval_cumulative_returns)


def create_train_evaluation_dataset(train_stock_info, first_n_days=80):
    """
    创建训练集评估数据集，用于检测过拟合
    使用每个股票的前N个交易日作为训练集评估样本

    Args:
        train_stock_info: 训练股票信息列表
        first_n_days: 使用前多少个交易日，默认80

    Returns:
        eval_inputs, eval_targets, eval_cumulative_returns
    """
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []

    required_length = DataConfig.REQUIRED_LENGTH
    context_length = DataConfig.CONTEXT_LENGTH

    for stock_info in train_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        train_start_idx = stock_info.get('train_start_idx', 0)

        # 使用每个股票的前80个交易日（从train_start_idx开始）
        start_min = max(1, train_start_idx + 1)
        start_max = min(train_start_idx + first_n_days, data_length - required_length)
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx)
            if sample is None:
                continue

            input_seq, target = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)

            # 计算未来收益率
            original_start_price = stock_data[start_idx + context_length - 1, 3]
            original_end_price = stock_data[start_idx + required_length - 1, 3]
            if original_start_price == 0:
                continue
            cumulative_return = (original_end_price - original_start_price) / original_start_price
            eval_cumulative_returns.append(float(cumulative_return))

    if len(eval_inputs) == 0:
        raise ValueError("训练集评估集为空：train_stock_info中没有可用样本")

    print(f"    训练集评估数据集已生成: {len(eval_inputs)}个样本 (每股票前{first_n_days}交易日)")
    return np.asarray(eval_inputs), np.asarray(eval_targets), np.asarray(eval_cumulative_returns)


def evaluate_model_batch(model, eval_inputs, eval_targets, eval_cumulative_returns, device, batch_size=100):
    """
    批量评估模型性能（详细版，用于train.py主训练流程）
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

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
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_returns = []

    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx],
                                       dtype=torch.bfloat16).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            batch_returns = eval_cumulative_returns[start_idx:end_idx]

            preds = torch.sigmoid(model(batch_inputs))

            all_preds.extend(preds.float().cpu().numpy().flatten())
            all_targets.extend(batch_targets)
            all_returns.extend(batch_returns)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)

    total = len(all_preds)

    # 计算分类准确率
    pred_labels = (all_preds >= 0.5).astype(int)
    true_labels = (all_targets >= 0.5).astype(int)

    class_correct = [0, 0]
    class_total = [0, 0]

    for i in range(2):
        mask = true_labels == i
        class_total[i] = np.sum(mask)
        class_correct[i] = np.sum((pred_labels == i) & mask)

    # 预测上涨的统计
    pred_positive_mask = pred_labels == 1
    pred_positive_total = np.sum(pred_positive_mask)
    pred_positive_correct = np.sum(pred_positive_mask & (true_labels == 1))
    pred_non_negative = np.sum(pred_positive_mask & (all_returns >= 0))

    # 计算 AUC
    try:
        auc_score = roc_auc_score(true_labels, all_preds)
    except ValueError:
        auc_score = 0.5

    # 置信度区间统计
    confidence_intervals = ['0.50-0.55', '0.55-0.58', '0.58-0.60', '0.60-0.70', '0.70-1.00']
    confidence_bounds = [(0.50, 0.55), (0.55, 0.58), (0.58, 0.60), (0.60, 0.70), (0.70, 1.00)]
    confidence_stats = {}

    for interval, (low, high) in zip(confidence_intervals, confidence_bounds):
        mask = (all_preds >= low) & (all_preds < high)
        total_in_interval = np.sum(mask)
        correct_in_interval = np.sum(mask & (true_labels == 1))
        non_negative_in_interval = np.sum(mask & (all_returns >= 0))
        confidence_stats[interval] = (correct_in_interval, total_in_interval, non_negative_in_interval)

    # Top N% 收益统计（涨停样本已在生成阶段过滤）
    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(all_preds) * percent / 100))
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    avg_return = np.mean(top_returns)
    total_return = np.sum(top_returns)
    compound_return = np.prod(1 + top_returns) ** (1 / len(top_returns)) - 1  # 几何平均复利收益率

    top_stats = {
        'count': top_k,
        'avg_return': avg_return,
        'total_return': total_return,
        'compound_return': compound_return,
        'filtered_count': 0  # 已在生成阶段过滤，这里为0
    }

    return total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats


def evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns,
                   device, batch_size=DataConfig.EVAL_BATCH_SIZE, model_name=""):
    """
    简化版模型评估函数（用于train_clone.py和train_evolve.py）
    涨停样本已在generate_sample_from_index中过滤，无需再次过滤

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
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_returns = []

    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx],
                                       dtype=torch.bfloat16).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            batch_returns = eval_cumulative_returns[start_idx:end_idx]

            preds = torch.sigmoid(model(batch_inputs))

            all_preds.extend(preds.float().cpu().numpy().flatten())
            all_targets.extend(batch_targets)
            all_returns.extend(batch_returns)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_returns = np.array(all_returns)

    # 计算 AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    # 计算 Top N% 收益（涨停样本已在生成阶段过滤）
    percent = DataConfig.TOP_PERCENT
    top_k = max(1, int(len(all_preds) * percent / 100))
    sorted_indices = np.argsort(all_preds)[::-1]
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    top_return = np.mean(top_returns)
    top_return_compound = np.prod(1 + top_returns) ** (1 / len(top_returns)) - 1  # 几何平均复利收益率
    top_threshold = all_preds[sorted_indices[top_k - 1]]

    # 统计高置信样本
    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2

    stats = {
        'auc': auc,
        'top_return': top_return,
        'top_return_compound': top_return_compound,
        'top_count': top_k,
        'top_threshold': top_threshold,
        'high_conf_count': np.sum(high_conf),
        'low_conf_count': np.sum(low_conf),
        'pred_mean': np.mean(all_preds),
        'pred_std': np.std(all_preds),
        'filtered_count': 0  # 已在生成阶段过滤
    }

    return stats


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
    # 转为numpy数组（BF16需要先转float32）
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

def calculate_test_loss(model, eval_inputs, eval_targets, criterion, device, batch_size=100):
    """
    计算测试集损失
    """
    model.eval()
    total_loss = 0.0
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx],
                                       dtype=torch.bfloat16).to(device)
            batch_targets = torch.tensor(eval_targets[start_idx:end_idx],
                                        dtype=torch.bfloat16).to(device)

            # 动态更新权重：根据当前batch的正负样本比例
            criterion.update_weights(batch_targets)

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
            input_tensor = torch.tensor(eval_inputs[idx:idx+1], dtype=torch.bfloat16).to(device)
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

            # 统计信息（转为float避免BF16问题）
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
def train_model(model, train_stock_info, test_stock_info, train_weights, epochs=TrainingConfig.EPOCHS,
               learning_rate=TrainingConfig.LEARNING_RATE, device=None,
               batch_size=TrainingConfig.BATCH_SIZE, batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH):
    """
    使用预计算训练数据集和固定评估集的训练函数（使用滚动窗口标准化避免数据泄露）
    提高训练效率，确保评估的一致性

    注意：本训练函数使用 BF16 (bfloat16) 精度进行训练
    - 训练速度比FP32快约2倍
    - 内存占用减半
    - 模型精度与FP32相当
    """
    print("\n" + "="*60)
    print("训练配置")
    print("="*60)
    print("训练精度: BF16 (Brain Floating Point 16)")
    print("数据标准化: 滚动窗口标准化（避免数据泄露）")
    print("采样策略: 采样头在多股票上同步前进，使用正负样本池平衡")
    print(f"数据划分: 按时间划分，最近{DataConfig.TEST_DAYS}天作为测试集")
    print("="*60 + "\n")
    
    # 创建时间顺序采样器
    print("正在初始化时间顺序采样器...")
    sampler = TemporalSampler(train_stock_info)
    
    # 设置训练随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)

    # 创建固定的评估数据集（训练开始前创建一次，使用滚动窗口标准化）
    print("\n创建评估数据集...")
    eval_inputs, eval_targets, eval_cumulative_returns = create_fixed_evaluation_dataset(test_stock_info)
    train_eval_inputs, train_eval_targets, train_eval_returns = create_train_evaluation_dataset(train_stock_info, first_n_days=80)

    # 使用自定义动态加权BCE损失函数
    # 特点：1. 根据batch正负样本比例动态调整负样本权重 2. 对预测偏差大的样本指数级惩罚
    criterion = DynamicWeightedBCE(pos_weight=4.0, reduction='mean')

    # 测试集损失同样使用动态加权BCE，保持评估一致性
    eval_criterion = DynamicWeightedBCE(pos_weight=4.0, reduction='mean')
    
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
    
    # 创建预热调度器
    warmup_scheduler = WarmupScheduler(
        optimizer, 
        warmup_epochs=TrainingConfig.WARMUP_EPOCHS,
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
        total_main_epochs = epochs - TrainingConfig.WARMUP_EPOCHS
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
                lr_status = f"预热阶段 ({epoch + 1}/{TrainingConfig.WARMUP_EPOCHS})"
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

            # 将数据转换为tensor并移到设备上 (使用BF16精度)
            epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.bfloat16).to(device)
            epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.bfloat16).to(device)

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

                # 动态更新权重：根据当前batch的正负样本比例
                criterion.update_weights(batch_targets)

                optimizer.zero_grad()
                output = model(batch_inputs)
                loss = criterion(output.squeeze(), batch_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
                optimizer.step()

                total_loss += loss.item()

                # 实时更新进度显示
                progress = (step + 1) / actual_batches * 100
                avg_loss = total_loss / (step + 1)
                print(f'\r  训练进度: {progress:.1f}% ({step + 1}/{actual_batches}), 平均损失: {avg_loss:.4f}', end='', flush=True)

            # 训练循环结束，计算最终的训练集平均损失
            # 使用 actual_batches 确保与训练循环中的 avg_loss 一致
            train_loss_epoch = total_loss / actual_batches

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
            total, class_correct, class_total, pred_positive_correct, pred_positive_total, pred_non_negative, auc_score, confidence_stats, top_stats = evaluate_model_batch(
                model, eval_inputs, eval_targets, eval_cumulative_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE
            )

            # 计算训练集收益率（用于检测过拟合）
            _, _, _, _, _, _, _, _, train_top_stats = evaluate_model_batch(
                model, train_eval_inputs, train_eval_targets, train_eval_returns, device, batch_size=DataConfig.EVAL_BATCH_SIZE
            )

            # 计算测试集损失（使用固定权重的eval_criterion，保证可比性）
            test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device, batch_size=DataConfig.EVAL_BATCH_SIZE)

            # 记录当前轮次收益率（必须在test_loss计算之后）
            epoch_return = {
                'turn': epoch + 1,
                'return': top_stats['avg_return'] * 100,  # 转换为百分比
                'return_compound': top_stats['compound_return'] * 100,  # 复利收益率百分比
                'train_loss': train_loss_epoch,  # 训练集损失（基于实际batch数）
                'test_loss': test_loss  # 测试集损失
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
            train_compound_pct = train_top_stats["compound_return"] * 100
            test_compound_pct = top_stats["compound_return"] * 100
            return_gap = train_return_pct - test_return_pct

            print(f'  【过拟合检测】Top{DataConfig.TOP_PERCENT}%收益率对比:')
            print(f'    训练集: {train_return_pct:+.2f}% | 复利: {train_compound_pct:+.2f}% (样本数={train_top_stats["count"]})')
            print(f'    测试集: {test_return_pct:+.2f}% | 复利: {test_compound_pct:+.2f}% (样本数={top_stats["count"]})')
            print(f'    差距: {return_gap:+.2f}% ', end='')
            if return_gap > 1.0:
                print('⚠️ 过拟合风险：训练集明显高于测试集')
            elif return_gap < -0.5:
                print('⚠️ 欠拟合：测试集高于训练集（罕见）')
            else:
                print('✓ 正常')
            print(f'  AUC得分: {auc_score:.4f}')
            print(f'  训练集损失: {train_loss_epoch:.4f}, 测试集损失: {test_loss:.4f}')

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
                print(f'    详情: AUC={auc_score:.4f}, Top{DataConfig.TOP_PERCENT}%收益: 平均={top_stats["avg_return"]*100:+.2f}% | 复利={top_stats["compound_return"]*100:+.2f}%, 累计={top_stats["total_return"]*100:+.2f}%')

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
        writer = csv.DictWriter(f, fieldnames=['turn', 'return', 'return_compound', 'train_loss', 'test_loss'])
        writer.writeheader()

        for epoch_return in epoch_returns:
            row = {
                'turn': epoch_return['turn'],
                'return': f"{epoch_return['return']:.2f}",
                'return_compound': f"{epoch_return['return_compound']:.2f}",
                'train_loss': f"{epoch_return['train_loss']:.4f}",
                'test_loss': f"{epoch_return['test_loss']:.4f}"
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

    # 计算股票选择权重
    train_weights = calculate_stock_weights(train_stock_info)
    test_weights = calculate_stock_weights(test_stock_info)
    
    # 打印数据集统计信息
    print("\n" + "="*60)
    print("数据集划分统计")
    print("="*60)
    
    train_lengths = [info['data_length'] for info in train_stock_info]
    test_lengths = [info['data_length'] for info in test_stock_info]
    
    print(f"训练集:")
    print(f"  股票数量: {len(train_stock_info)}")
    print(f"  数据长度: 最小={min(train_lengths)}, 最大={max(train_lengths)}, 平均={np.mean(train_lengths):.1f}")
    print(f"  采样权重: {min(train_weights):.3f} - {max(train_weights):.3f}")
    
    print(f"\n测试集:")
    print(f"  股票数量: {len(test_stock_info)}")
    print(f"  数据长度: 最小={min(test_lengths)}, 最大={max(test_lengths)}, 平均={np.mean(test_lengths):.1f}")
    print(f"  时间范围: 每只股票的最近 {DataConfig.TEST_DAYS} 天")
    
    print(f"\n前3只股票示例:")
    for i in range(min(3, len(train_stock_info))):
        train_info = train_stock_info[i]
        print(f"  {train_info['file_name']}: 训练集长度={train_info['data_length']}, 权重={train_weights[i]:.3f}")
    
    print("="*60)

    print("正在创建 Transformer 模型 (BF16精度)...")
    model = EnhancedStockTransformer(
        input_dim=ModelConfig.INPUT_DIM, 
        d_model=ModelConfig.D_MODEL, 
        nhead=ModelConfig.NHEAD, 
        num_layers=ModelConfig.NUM_LAYERS, 
        output_dim=ModelConfig.OUTPUT_DIM,
        seq_len=ModelConfig.SEQ_LEN
    ).to(device)
    
    # 将模型参数转换为BF16精度
    model = model.to(dtype=torch.bfloat16)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    print("开始训练...")
    # 使用带固定评估集的训练函数（使用滚动窗口标准化）
    train_model(model, train_stock_info, test_stock_info, train_weights, device=device)
    
    # 保存最终模型（训练结束时的状态）
    final_model_path = ModelSaveConfig.get_final_model_path(ModelConfig.D_MODEL)
    torch.save(model.state_dict(), final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")

# ==================== 统一预测函数 ====================
def normalize_data_for_prediction(data):
    """
    统一的数据归一化函数（滚动窗口标准化）
    用于所有预测场景，确保与训练时完全一致
    
    Args:
        data: numpy array, shape [seq_len, 6] (OHLC + volume + exchange)
        
    Returns:
        normalized_data: numpy array, shape [seq_len-1, 6] 或 None（如果数据无效）
    """
    if len(data) < 2:
        return None
    
    normalized_data = np.zeros_like(data, dtype=np.float64)
    
    # 滚动窗口标准化：每天相对于前一天的涨跌幅
    for i in range(1, len(data)):
        yesterday_close = data[i-1, 3]  # 前一天的收盘价
        yesterday_volume = data[i-1, 4]  # 前一天的成交量
        
        if yesterday_close == 0 or yesterday_volume == 0:
            return None  # 数据异常
        
        # 价格特征：相对于前一天收盘价的涨跌幅
        normalized_data[i, :4] = (data[i, :4] - yesterday_close) / yesterday_close
        # 流动性特征
        normalized_data[i, 4] = (data[i, 4] - yesterday_volume) / yesterday_volume
        normalized_data[i, 5] = np.clip(data[i, 5] / 100.0, 0.0, 1.0)  # 换手率
    
    # 数值范围限制
    # 价格特征：限制在[-0.1, 0.1]，对应±10%涨跌幅
    normalized_data[:, :4] = np.clip(normalized_data[:, :4], -0.1, 0.1)
    normalized_data[:, 4] = np.clip(normalized_data[:, 4], -5.0, 5.0)
    normalized_data[:, 4] = np.clip(normalized_data[:, 4] / 10.0 + 0.5, 0.0, 1.0)
    normalized_data[:, 5] = np.clip(normalized_data[:, 5], 0.0, 1.0)
    
    # NaN/Inf检测
    result = normalized_data[1:]
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        return None
    
    # 只返回标准化后的数据（去掉第0天基准数据）
    return result

def predict_single_stock(model_path, stock_data, device=None):
    """
    统一的单股票预测函数
    
    Args:
        model_path: 模型文件路径
        stock_data: numpy array, shape [seq_len, 6] (OHLC + volume + exchange)，至少需要CONTEXT_LENGTH+1天数据
        device: 计算设备
        
    Returns:
        probability: float, 预测概率 [0, 1]，如果预测失败返回None
    """
    if device is None:
        device = DeviceConfig.get_device()
    
    # 检查数据长度
    if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
        return None
    
    # 取最新数据
    recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
    
    # 归一化
    normalized_data = normalize_data_for_prediction(recent_data)
    if normalized_data is None:
        return None
    
    # 加载模型
    try:
        model = EnhancedStockTransformer(
            input_dim=ModelConfig.INPUT_DIM,
            d_model=ModelConfig.D_MODEL,
            nhead=ModelConfig.NHEAD,
            num_layers=ModelConfig.NUM_LAYERS,
            output_dim=ModelConfig.OUTPUT_DIM,
            seq_len=ModelConfig.SEQ_LEN
        ).to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    # 预测
    try:
        input_tensor = torch.tensor(normalized_data, dtype=torch.bfloat16).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probability = torch.sigmoid(output).float().cpu().item()
        
        return probability
    except Exception as e:
        print(f"预测失败: {e}")
        return None

def predict_multiple_stocks(model_path, stock_files_data, device=None):
    """
    统一的多股票预测函数
    
    Args:
        model_path: 模型文件路径
        stock_files_data: dict, {文件名: numpy_array}
        device: 计算设备
        
    Returns:
        predictions: list of (filename, probability)
    """
    if device is None:
        device = DeviceConfig.get_device()
    
    predictions = []
    
    # 加载模型（只加载一次）
    try:
        model = EnhancedStockTransformer(
            input_dim=ModelConfig.INPUT_DIM,
            d_model=ModelConfig.D_MODEL,
            nhead=ModelConfig.NHEAD,
            num_layers=ModelConfig.NUM_LAYERS,
            output_dim=ModelConfig.OUTPUT_DIM,
            seq_len=ModelConfig.SEQ_LEN
        ).to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return predictions
    
    # 批量预测
    with torch.no_grad():
        for filename, stock_data in stock_files_data.items():
            # 检查数据长度
            if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
                continue
            
            # 取最新数据并归一化
            recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
            normalized_data = normalize_data_for_prediction(recent_data)
            if normalized_data is None:
                continue
            
            try:
                # 预测
                input_tensor = torch.tensor(normalized_data, dtype=torch.bfloat16).unsqueeze(0).to(device)
                output = model(input_tensor)
                probability = torch.sigmoid(output).float().cpu().item()
                
                predictions.append((filename, probability))
            except Exception as e:
                print(f"{filename} 预测失败: {e}")
                continue
    
    return predictions