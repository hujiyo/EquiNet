"""
EquiNet 数据处理模块

包含所有数据相关的功能：
- 数据加载和预处理
- 样本生成
- 时间顺序采样器
- 评估数据集创建
- 预测函数
"""

import os
import random
import numpy as np
import pandas as pd
import torch

from config import DataConfig, DeviceConfig, ModelConfig


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
        
        df = df.iloc[::-1].reset_index(drop=True)
        
        data = df[['start', 'max', 'min', 'end', 'volume', 'exchange']].values
        times = df['time'].values
        
        data_length = len(data)
        required_length = DataConfig.REQUIRED_LENGTH
        
        min_required_length = required_length + test_days
        if data_length < min_required_length:
            return None
        
        train_end_idx = data_length - test_days - required_length
        
        test_split_point = data_length - test_days
        
        train_start_date = train_start_year * 10000 + 101
        valid_indices = np.where(times >= train_start_date)[0]
        
        if len(valid_indices) > 0:
            train_start_idx = valid_indices[0]
        else:
            train_start_idx = 0
        
        if train_start_idx >= train_end_idx:
            return None
        
        train_length = train_end_idx - train_start_idx
        if train_length < required_length:
            return None
        
        train_data = data.copy()
        test_data = data.copy()
        
        stock_info = {
            'file_name': file_name,
            'data_length': data_length,
            'train_data': train_data,
            'test_data': test_data,
            'train_start_idx': train_start_idx,
            'train_end_idx': train_end_idx,
            'train_length': train_length,
            'test_split_point': test_split_point,
            'times': times
        }
        
        return stock_info
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return None


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
    
    file_args = [(os.path.join(data_dir, f), f, test_days, train_start_year) for f in all_files]
    num_workers = min(cpu_count(), 8)
    
    with Pool(num_workers) as pool:
        all_stock_info = [r for r in pool.map(process_single_file, file_args) if r is not None]
    
    discarded_count = len(all_files) - len(all_stock_info)
    print(f"有效股票: {len(all_stock_info)} 只，丢弃: {discarded_count} 只")
    
    train_stock_info = []
    test_stock_info = []
    
    for stock_info in all_stock_info:
        train_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['train_data'],
            'data_length': stock_info['data_length'],
            'train_start_idx': stock_info['train_start_idx'],
            'train_end_idx': stock_info['train_end_idx'],
        })
        
        test_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['test_data'],
            'data_length': stock_info['data_length'],
            'test_split_point': stock_info['test_split_point']
        })
    
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    
    return train_stock_info, test_stock_info


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

        self.stock_positions = []
        self.stock_start_positions = []
        self.stock_max_positions = []
        self.can_loop = []
        self.loop_counts = [0] * len(stock_info_list)
        
        for stock_info in stock_info_list:
            train_start_idx = stock_info.get('train_start_idx', 0)
            train_end_idx = stock_info.get('train_end_idx', len(stock_info['data']))
            data_length = stock_info.get('data_length', 0)
            
            start_pos = max(1, train_start_idx + 1)
            max_pos = train_end_idx
            
            if start_pos > max_pos:
                start_pos = max_pos + 1
            
            self.stock_positions.append(start_pos)
            self.stock_start_positions.append(start_pos)
            self.stock_max_positions.append(max_pos)
            self.can_loop.append(data_length > 600)

        valid_stocks = sum(1 for i in range(len(stock_info_list)) 
                         if self.stock_positions[i] <= self.stock_max_positions[i])
        total_samples = sum(max(0, self.stock_max_positions[i] - self.stock_positions[i] + 1) 
                          for i in range(len(stock_info_list)))
        
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

                if current_pos > max_pos and self.can_loop[stock_idx]:
                    current_pos = self.stock_start_positions[stock_idx]
                    self.stock_positions[stock_idx] = current_pos
                    self.loop_counts[stock_idx] += 1

                if current_pos <= max_pos:
                    all_samples.append((stock_idx, current_pos))
                    self.stock_positions[stock_idx] += 1

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


def check_strong_signal(daily_returns):
    """
    强势信号检测：判断是否存在强势买入信号
    
    标签=1的条件（满足任一即可）：
    1. 单日爆发：Day1涨幅 ≥ 5%
    2. 双日接力：Day1+Day2累计 ≥ 6% 且 Day1>1%, Day2>1%
    3. 稳健上涨：Day1 ≥ 1% 且 Day2 ≥ 1% 且 Day3 ≥ 1% 且 累计 ≥ 5%
    4. 爆发后延续：任意一天 ≥ 8% 且 累计 ≥ 6%
    5. 累计达标：3天累计涨幅 ≥ 8%（基础条件）
    
    Args:
        daily_returns: list或np.array, 3天的日收益率 [Day1, Day2, Day3]
        
    Returns:
        int: 1表示存在强势信号，0表示无信号
    """
    if len(daily_returns) < 3:
        return 0
    
    r1, r2, r3 = daily_returns[0], daily_returns[1], daily_returns[2]
    cum_2day = r1 + r2
    cum_3day = r1 + r2 + r3
    
    if r1 >= DataConfig.SIGNAL_DAY1_BURST:
        return 1
    
    if cum_2day >= DataConfig.SIGNAL_TWO_DAY_CUM and r1 > DataConfig.SIGNAL_DAY_MIN and r2 > DataConfig.SIGNAL_DAY_MIN:
        return 1
    
    if (r1 >= DataConfig.SIGNAL_DAY_MIN and 
        r2 >= DataConfig.SIGNAL_DAY_MIN and 
        r3 >= DataConfig.SIGNAL_DAY_MIN and 
        cum_3day >= DataConfig.SIGNAL_THREE_DAY_CUM):
        return 1
    
    max_day = max(r1, r2, r3)
    if max_day >= DataConfig.SIGNAL_ANY_BURST and cum_3day >= DataConfig.SIGNAL_BURST_CUM:
        return 1
    
    if cum_3day >= DataConfig.UPRISE_THRESHOLD:
        return 1
    
    return 0


def generate_sample_from_index(stock_info_list, stock_idx, start_idx):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引

    返回: (input_seq, target, cumulative_return) 或 None（如果样本无效）
    
    收益率计算（实战视角）:
        T日晚上运行模型 → T+1日开盘买入
        Day1收益率 = (T+1收盘 - T+1开盘) / T+1开盘  (日内涨幅)
        Day2收益率 = (T+2收盘 - T+1收盘) / T+1收盘
        Day3收益率 = (T+3收盘 - T+2收盘) / T+2收盘
        累计收益率 = (T+3收盘 - T+1开盘) / T+1开盘
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    required_length = DataConfig.REQUIRED_LENGTH

    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    prev_day_data = stock_data[start_idx - 1]

    prev_close = prev_day_data[3]
    prev_volume = prev_day_data[4]
    if prev_close == 0 or prev_volume == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    closes = input_seq_raw[:, 3]
    volumes = input_seq_raw[:, 4]
    if np.any(closes == 0) or np.any(volumes == 0):
        return None

    sample_window_start = start_idx - 1
    sample_window_end = start_idx + required_length
    sample_data = stock_data[sample_window_start:sample_window_end]

    limit_threshold = 0.11

    for day_idx in range(1, len(sample_data)):
        today_close = sample_data[day_idx, 3]
        yesterday_close = sample_data[day_idx - 1, 3]

        if yesterday_close > 0:
            daily_return = (today_close - yesterday_close) / yesterday_close
            if abs(daily_return) > limit_threshold:
                return None

    last_day_idx = start_idx + context_length - 1
    prev_day_idx = start_idx + context_length - 2
    prev_day_close = stock_data[prev_day_idx, 3]
    last_day_close = stock_data[last_day_idx, 3]

    if prev_day_close > 0:
        last_day_return = (last_day_close - prev_day_close) / prev_day_close
        if last_day_return >= 0.095:
            return None

    input_seq = np.empty((context_length, 6), dtype=np.float32)
    
    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]
    
    input_seq[0, 4] = (volumes[0] - prev_volume) / prev_volume
    if context_length > 1:
        input_seq[1:, 4] = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    
    input_seq[:, 5] = input_seq_raw[:, 5] / 100.0
    
    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])
    np.clip(input_seq[:, 4], -5.0, 5.0, out=input_seq[:, 4])
    input_seq[:, 4] = input_seq[:, 4] / 10.0 + 0.5
    np.clip(input_seq[:, 4:6], 0.0, 1.0, out=input_seq[:, 4:6])

    if np.any(~np.isfinite(input_seq)):
        return None

    t1_open = stock_data[start_idx + context_length, 0]
    t1_close = stock_data[start_idx + context_length, 3]
    t2_close = stock_data[start_idx + context_length + 1, 3]
    t3_close = stock_data[start_idx + context_length + 2, 3]

    if t1_open == 0:
        return None

    cumulative_return = (t3_close - t1_open) / t1_open

    future_closes = stock_data[start_idx + context_length:start_idx + required_length, 3]
    daily_returns = []
    prev_close_for_future = closes[-1]
    for future_close in future_closes:
        if prev_close_for_future > 0:
            daily_ret = (future_close - prev_close_for_future) / prev_close_for_future
            daily_returns.append(daily_ret)
        prev_close_for_future = future_close
    
    target = float(check_strong_signal(daily_returns))

    return input_seq, target, cumulative_return


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
    
    initial_rounds = 50
    total_rounds_generated = 0
    total_indices_generated = 0
    
    print(f"    动态采样策略：按需生成索引，直到满足{batches_per_epoch}个batch...")
    
    while batches_generated < batches_per_epoch:
        sample_indices = sampler.sample_batch_rounds(initial_rounds)
        
        if len(sample_indices) == 0:
            print(f"\n    ⚠ 警告：采样头已到达所有股票终点且无法循环，停止采样")
            break
        
        total_rounds_generated += initial_rounds
        total_indices_generated += len(sample_indices)
        
        for stock_idx, start_idx in sample_indices:
            if batches_generated >= batches_per_epoch:
                break

            sample = generate_sample_from_index(stock_info_list, stock_idx, start_idx)
            if sample is None:
                continue

            input_seq, target, _ = sample

            if target >= 0.5:
                pos_pool_inputs.append(input_seq)
                pos_pool_targets.append(target)
            else:
                neg_pool_inputs.append(input_seq)
                neg_pool_targets.append(target)
            
            if len(pos_pool_inputs) >= pos_quota and len(neg_pool_inputs) >= neg_quota:
                batch_pos_inputs = pos_pool_inputs[:pos_quota]
                batch_pos_targets = pos_pool_targets[:pos_quota]
                
                neg_indices = rng.sample(range(len(neg_pool_inputs)), neg_quota)
                batch_neg_inputs = [neg_pool_inputs[i] for i in neg_indices]
                batch_neg_targets = [neg_pool_targets[i] for i in neg_indices]
                
                batch_inputs = batch_pos_inputs + batch_neg_inputs
                batch_targets = batch_pos_targets + batch_neg_targets
                
                combined = list(zip(batch_inputs, batch_targets))
                rng.shuffle(combined)
                b_inputs, b_targets = zip(*combined)
                
                all_batch_inputs.extend(b_inputs)
                all_batch_targets.extend(b_targets)
                
                batches_generated += 1
                
                pos_pool_inputs = pos_pool_inputs[pos_quota:]
                pos_pool_targets = pos_pool_targets[pos_quota:]
                neg_pool_inputs = []
                neg_pool_targets = []
        
        print(f"    已生成 {batches_generated}/{batches_per_epoch} 个Batch (已采样{total_rounds_generated}轮)", end='\r', flush=True)
        
        if batches_generated < batches_per_epoch:
            remaining_batches = batches_per_epoch - batches_generated
            if batches_generated > 0:
                estimated_rounds = max(20, int(remaining_batches / batches_generated * total_rounds_generated * 1.2))
                initial_rounds = min(estimated_rounds, 100)
            else:
                initial_rounds = 100

    print(f"\n    已生成 {batches_generated}/{batches_per_epoch} 个batch (总共采样{total_rounds_generated}轮, {total_indices_generated}个索引)")
    
    if batches_generated < batches_per_epoch:
        print(f"    ⚠ 警告：样本不足，仅生成 {batches_generated} 个Batch (目标: {batches_per_epoch})")
        if batches_generated == 0:
             raise ValueError(f"样本严重不足：无法生成任何Batch")

    return np.asarray(all_batch_inputs), np.asarray(all_batch_targets)


def create_fixed_evaluation_dataset(test_stock_info):
    """
    创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）
    
    返回:
        eval_inputs: 输入序列
        eval_targets: 标签
        eval_cumulative_returns: 累计收益率
        eval_day_indices: 每个样本对应的预测日实际日期（如20241201）
    """
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []
    eval_day_indices = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        times = stock_info.get('times', None)
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))

        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.REQUIRED_LENGTH
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx)
            if sample is None:
                continue

            input_seq, target, cumulative_return = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)
            eval_cumulative_returns.append(float(cumulative_return))
            
            predict_day_idx = start_idx + DataConfig.CONTEXT_LENGTH
            if times is not None and predict_day_idx < len(times):
                day_index = times[predict_day_idx]
            else:
                day_index = predict_day_idx - test_split_point
            eval_day_indices.append(day_index)

    if len(eval_inputs) == 0:
        raise ValueError("固定评估集为空：test_stock_info中没有可用样本")

    return (np.asarray(eval_inputs), np.asarray(eval_targets), 
            np.asarray(eval_cumulative_returns), np.asarray(eval_day_indices))


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

    for stock_info in train_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        train_start_idx = stock_info.get('train_start_idx', 0)

        start_min = max(1, train_start_idx + 1)
        start_max = min(train_start_idx + first_n_days, data_length - DataConfig.REQUIRED_LENGTH)
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx)
            if sample is None:
                continue

            input_seq, target, cumulative_return = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)
            eval_cumulative_returns.append(float(cumulative_return))

    if len(eval_inputs) == 0:
        raise ValueError("训练集评估集为空：train_stock_info中没有可用样本")

    print(f"    训练集评估数据集已生成: {len(eval_inputs)}个样本 (每股票前{first_n_days}交易日)")
    return np.asarray(eval_inputs), np.asarray(eval_targets), np.asarray(eval_cumulative_returns)


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
    
    for i in range(1, len(data)):
        yesterday_close = data[i-1, 3]
        yesterday_volume = data[i-1, 4]
        
        if yesterday_close == 0 or yesterday_volume == 0:
            return None
        
        normalized_data[i, :4] = (data[i, :4] - yesterday_close) / yesterday_close
        normalized_data[i, 4] = (data[i, 4] - yesterday_volume) / yesterday_volume
        normalized_data[i, 5] = np.clip(data[i, 5] / 100.0, 0.0, 1.0)
    
    normalized_data[:, :4] = np.clip(normalized_data[:, :4], -0.1, 0.1)
    normalized_data[:, 4] = np.clip(normalized_data[:, 4], -5.0, 5.0)
    normalized_data[:, 4] = np.clip(normalized_data[:, 4] / 10.0 + 0.5, 0.0, 1.0)
    normalized_data[:, 5] = np.clip(normalized_data[:, 5], 0.0, 1.0)
    
    result = normalized_data[1:]
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        return None
    
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
    from model import create_model
    
    if device is None:
        device = DeviceConfig.get_device()
    
    if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
        return None
    
    recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
    
    normalized_data = normalize_data_for_prediction(recent_data)
    if normalized_data is None:
        return None
    
    try:
        model = create_model().to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.float()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None
    
    try:
        input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)
        
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
    from model import create_model
    
    if device is None:
        device = DeviceConfig.get_device()
    
    predictions = []
    
    try:
        model = create_model().to(device)
        
        model = model.to(dtype=torch.bfloat16)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model = model.float()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return predictions
    
    with torch.no_grad():
        for filename, stock_data in stock_files_data.items():
            if len(stock_data) < DataConfig.CONTEXT_LENGTH + 1:
                continue
            
            recent_data = stock_data[-(DataConfig.CONTEXT_LENGTH + 1):]
            normalized_data = normalize_data_for_prediction(recent_data)
            if normalized_data is None:
                continue
            
            try:
                input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(input_tensor)
                probability = torch.sigmoid(output).float().cpu().item()
                
                predictions.append((filename, probability))
            except Exception as e:
                print(f"{filename} 预测失败: {e}")
                continue
    
    return predictions
