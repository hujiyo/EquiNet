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
from config import DataConfig, DeviceConfig
from multiprocessing import Pool, cpu_count


def process_single_file(args):
    file_path, file_name, test_days, train_start_year = args
    """
    处理单个股票CSV文件，返回包含训练和测试数据的字典
    
    数据处理流程：
    1. 读取CSV并反转时间顺序
    2. 提取OHLCV数据：['start', 'max', 'min', 'end', 'volume', 'exchange']
    3. 验证数据长度是否满足最低要求
    
    数据分割策略（确保训练集和测试集严格分离）：
    - 测试集：最后 test_days 天的数据，完全冻结用于评估
    - 训练集：从 train_start_year 开始到 train_end_idx 结束
    - 缓冲区：训练集结束后有 REQUIRED_LENGTH 天的缓冲区，防止数据泄露
    
    关键索引计算：
    - train_end_idx: 训练集最后一个可用位置 = data_length - test_days - required_length
    - test_split_point: 测试集起始位置 = data_length - test_days
    - train_start_idx: 根据 train_start_year 找到的实际起始位置
    
    返回数据包含完整的训练和测试数据副本，使用时需根据索引切片访问。
    
    Args:
        file_path: CSV文件路径
        file_name: 文件名
        test_days: 测试集天数
        train_start_year: 训练开始年份
    
    Returns:
        dict or None: 包含股票信息的字典，数据不足时返回None
    """
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
            'file_name': file_name,              # 股票文件名，用于识别不同股票
            'data_length': data_length,          # 总数据长度（天数），用于验证数据充足性
            'train_data': train_data,            # 完整数据副本，训练时根据[train_start_idx:train_end_idx]切片访问
            'test_data': test_data,              # 完整数据副本，测试时根据[test_split_point:]切片访问
            'train_start_idx': train_start_idx,  # 训练集起始索引，根据train_start_year计算得出
            'train_end_idx': train_end_idx,      # 训练集结束索引，确保与测试集有缓冲区
            'train_length': train_length,        # 可用训练数据长度，用于验证训练集是否充足
            'test_split_point': test_split_point # 测试集起始索引，固定为最后test_days天的开始位置
        }
        
        return stock_info
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        return None


def load_and_preprocess_data(data_dir=DataConfig.DATA_DIR, test_days=DataConfig.TEST_DAYS, train_start_year=DataConfig.TRAIN_START_YEAR):
    """
    数据加载和预处理，使用多进程并行加载
    
    采样边界设计：
    - 训练集：从TRAIN_START_YEAR年（或上市日）到 总长度-test_days-REQUIRED_LENGTH
    - 测试集：最近test_days天
    - 最低数据要求：test_days + REQUIRED_LENGTH
    """
    
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
    - 每只股票的指针初始位置 = train_start_idx（TRAIN_START_YEAR年起始位置+1，或上市第一天+1）
    - 每只股票的指针末位置 = train_end_idx（总长度 - TEST_DAYS - REQUIRED_LENGTH）
    
    关键设计：start_pos = max(1, train_start_idx + 1)
    原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
    因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
    
    核心算法：
    1. 计算总样本数和每个epoch需要的样本数
    2. 将总样本数均匀分配到各个epoch
    3. 每轮从所有股票当前位置各取一个样本，然后指针前进
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
            
            # 关键设计：start_pos = max(1, train_start_idx + 1)
            # 原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
            # 因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
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


class RandomSampler:
    """
    随机采样器：每次随机选择股票和位置进行采样
    
    与TemporalSampler的区别：
    - TemporalSampler: 时间顺序前进，指针不回头（除非循环）
    - RandomSampler: 每次完全随机选择样本，无时间顺序
    
    适用场景：
    - 对比实验：评估时间顺序采样对模型效果的影响
    - 数据增强：打破时间依赖，增加样本多样性
    """
    def __init__(self, stock_info_list):
        self.stock_info_list = stock_info_list
        self.required_length = DataConfig.REQUIRED_LENGTH
        
        self.valid_stock_indices = []
        self.stock_sample_ranges = []
        
        for stock_idx, stock_info in enumerate(stock_info_list):
            train_start_idx = stock_info.get('train_start_idx', 0)
            train_end_idx = stock_info.get('train_end_idx', len(stock_info['data']))
            
            # 关键设计：start_pos = max(1, train_start_idx + 1)
            # 原因：每个样本需要前一天数据作为归一化基准（prev_day_data = stock_data[start_idx-1]）
            # 因此第一个有效样本必须从 index=1 开始，确保 index=0 存在作为基准日
            start_pos = max(1, train_start_idx + 1)
            max_pos = train_end_idx
            
            if start_pos <= max_pos:
                self.valid_stock_indices.append(stock_idx)
                self.stock_sample_ranges.append((start_pos, max_pos))
        
        valid_stocks = len(self.valid_stock_indices)
        total_samples = sum(max_pos - start_pos + 1 
                          for start_pos, max_pos in self.stock_sample_ranges)
        
        if valid_stocks == 0:
            raise ValueError(
                f"没有有效的训练股票！\n"
                f"  总股票数: {len(stock_info_list)}\n"
                f"  请检查数据质量或调整参数"
            )
        
        print(f"  初始化随机采样器: {valid_stocks}只有效股票, 总样本数={total_samples}")
        print(f"  采样策略: 完全随机采样，每次随机选择股票和位置")

    def sample_batch_rounds(self, num_rounds, rng=None):
        """
        随机采样多轮：按样本数量加权随机采样

        参数:
            num_rounds: 要采样的轮数（每轮采样 valid_stocks 个样本）
            rng: 随机数生成器（用于可复现性）

        返回: [(stock_idx, start_idx), ...] 所有轮次的样本索引列表
        
        加权策略：
            - 每只股票被选中的概率与其样本数量成正比
            - 样本多的股票被采样次数多，样本少的股票被采样次数少
            - 确保每个样本被采样的期望概率相等
        """
        if rng is None:
            rng = random.Random()
        
        all_samples = []
        num_samples_per_round = len(self.valid_stock_indices)
        total_samples_to_generate = num_rounds * num_samples_per_round
        
        sample_weights = [max_pos - start_pos + 1 
                         for start_pos, max_pos in self.stock_sample_ranges]
        
        stock_to_range = {
            stock_idx: self.stock_sample_ranges[i]
            for i, stock_idx in enumerate(self.valid_stock_indices)
        }
        
        for _ in range(total_samples_to_generate):
            stock_idx = rng.choices(
                self.valid_stock_indices,
                weights=sample_weights,
                k=1
            )[0]
            
            start_pos, max_pos = stock_to_range[stock_idx]
            start_idx = rng.randint(start_pos, max_pos)
            all_samples.append((stock_idx, start_idx))
        
        return all_samples
    
    def get_progress(self):
        """随机采样器无进度概念，返回 (0, 1) 表示无限采样"""
        return 0, 1

    def get_loop_stats(self):
        """随机采样器无循环概念"""
        return 0, 0


def create_sampler(stock_info_list, strategy=None):
    """
    根据配置创建采样器
    
    参数:
        stock_info_list: 股票信息列表
        strategy: 采样策略，可选 'temporal' 或 'random'，默认使用 DataConfig.SAMPLING_STRATEGY
    
    返回:
        sampler: TemporalSampler 或 RandomSampler 实例
    """
    if strategy is None:
        strategy = DataConfig.SAMPLING_STRATEGY
    
    if strategy == 'random':
        print("使用随机采样策略")
        return RandomSampler(stock_info_list)
    else:
        print("使用时间顺序采样策略")
        return TemporalSampler(stock_info_list)


def check_strong_signal(daily_price_changes):
    """
    强势信号检测：判断是否存在强势买入信号（风险优化版）
    
    注意：此函数使用"涨跌幅"而非"收益率"
    - 涨跌幅：基准是前一日收盘价，用于判断股票走势强弱
    - 收益率：基准是买入价，用于计算投资回报
    
    标签=1的条件（满足任一即可）：
    1. 单日爆发：Day1涨跌幅 ≥ 5% 且 累计 ≥ 2%
    2. 双日接力：Day1+Day2累计涨跌幅 ≥ 6% 且 Day1>1%, Day2>1% 且 累计 ≥ 2%
    3. 稳健上涨：Day1 ≥ 1% 且 Day2 ≥ 1% 且 Day3 ≥ 1% 且 累计 ≥ 5%
    4. 爆发后延续：任意一天涨跌幅 ≥ 8% 且 累计 ≥ 6% 且 Day1 ≥ -2%
    5. 累计达标：3天累计涨跌幅 ≥ 8% 且 Day1 ≥ -2%
    
    风险控制：
    - 条件1、2增加累计≥2%兜底，过滤8.13%和1.47%的累计亏损样本
    - 条件4、5增加Day1≥-2%限制，过滤25.60%和4.65%的"买入当天就亏"样本
    - 条件3天然安全，无需修改
    
    Args:
        daily_price_changes: list或np.array, 3天的涨跌幅 [Day1, Day2, Day3]
            Day1涨跌幅 = (T+1收盘 - T日收盘) / T日收盘
            Day2涨跌幅 = (T+2收盘 - T+1收盘) / T+1收盘
            Day3涨跌幅 = (T+3收盘 - T+2收盘) / T+2收盘
        
    Returns:
        int: 1表示存在强势信号，0表示无信号
    """
    if len(daily_price_changes) < 3:
        return 0
    
    r1, r2, r3 = daily_price_changes[0], daily_price_changes[1], daily_price_changes[2]
    cum_2day = r1 + r2
    cum_3day = r1 + r2 + r3
    
    # 条件1：单日爆发 + 累计兜底
    if r1 >= DataConfig.SIGNAL_DAY1_BURST and cum_3day >= DataConfig.SIGNAL_MIN_CUM_RETURN:
        return 1
    
    # 条件2：双日接力 + 累计兜底
    if (cum_2day >= DataConfig.SIGNAL_TWO_DAY_CUM and 
        r1 > DataConfig.SIGNAL_DAY_MIN and 
        r2 > DataConfig.SIGNAL_DAY_MIN and 
        cum_3day >= DataConfig.SIGNAL_MIN_CUM_RETURN):
        return 1
    
    # 条件3：稳健上涨（天然安全，无需修改）
    if (r1 >= DataConfig.SIGNAL_DAY_MIN and 
        r2 >= DataConfig.SIGNAL_DAY_MIN and 
        r3 >= DataConfig.SIGNAL_DAY_MIN and 
        cum_3day >= DataConfig.SIGNAL_THREE_DAY_CUM):
        return 1
    
    # 条件4：爆发后延续 + Day1保护
    max_day = max(r1, r2, r3)
    if (max_day >= DataConfig.SIGNAL_ANY_BURST and 
        cum_3day >= DataConfig.SIGNAL_BURST_CUM and 
        r1 >= DataConfig.SIGNAL_DAY1_MAX_DROP):
        return 1
    
    # 条件5：累计达标 + Day1保护
    if cum_3day >= DataConfig.UPRISE_THRESHOLD and r1 >= DataConfig.SIGNAL_DAY1_MAX_DROP:
        return 1
    
    return 0


def generate_sample_from_index(stock_info_list, stock_idx, start_idx):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引

    返回: (input_seq, target, cumulative_return, daily_returns) 或 None（如果样本无效）
    
    核心概念区分：
        【涨跌幅】用于标签生成，判断股票走势强弱
            - 基准是前一日收盘价
            - Day1涨跌幅 = (T+1收盘 - T日收盘) / T日收盘
            - Day2涨跌幅 = (T+2收盘 - T+1收盘) / T+1收盘
            - Day3涨跌幅 = (T+3收盘 - T+2收盘) / T+2收盘
        
        【收益率】用于计算投资回报，评估模型表现，EquiNet默认用户是Day1以开盘价买入，Day3再以收盘价卖出，因此收益率计算逻辑如下：
            - 基准是买入价（T+1开盘价）
            - Day1收益率 = (T+1收盘 - T+1开盘) / T+1开盘（日内收益）
            - Day2收益率贡献 = (T+2收盘 - T+1收盘) / T+1开盘
            - Day3收益率贡献 = (T+3收盘 - T+2收盘) / T+1开盘
            - 累计收益率 = Day1 + Day2 + Day3 = (T+3收盘 - T+1开盘) / T+1开盘
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

    # 规则7：上下文最后一天涨停过滤（可通过配置开关控制）
    if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP:
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

    # ========== 涨跌幅计算（用于标签生成）==========
    # 基准是前一日收盘价，用于判断股票走势强弱
    daily_price_changes = []
    day1_price_change = (t1_close - closes[-1]) / closes[-1]  # (T+1收盘 - T日收盘) / T日收盘
    daily_price_changes.append(day1_price_change)
    day2_price_change = (t2_close - t1_close) / t1_close      # (T+2收盘 - T+1收盘) / T+1收盘
    daily_price_changes.append(day2_price_change)
    day3_price_change = (t3_close - t2_close) / t2_close      # (T+3收盘 - T+2收盘) / T+2收盘
    daily_price_changes.append(day3_price_change)
    
    # ========== 收益率计算（用于评估模型表现）==========
    # 基准是买入价（T+1开盘价），用于计算投资回报
    daily_returns = []
    day1_return = (t1_close - t1_open) / t1_open              # Day1日内收益
    daily_returns.append(day1_return)
    day2_return = (t2_close - t1_close) / t1_open             # Day2收益贡献
    daily_returns.append(day2_return)
    day3_return = (t3_close - t2_close) / t1_open             # Day3收益贡献
    daily_returns.append(day3_return)
    
    # 标签生成使用涨跌幅
    target = float(check_strong_signal(daily_price_changes))

    return input_seq, target, cumulative_return, daily_returns


def generate_sample_from_index_partial(stock_info_list, stock_idx, start_idx):
    """
    生成样本，支持不完整的未来数据（用于最近几天的临时评估）
    
    与generate_sample_from_index的区别：
    - 由run.py使用，与模型训练阶段脚本无关
    - 允许未来数据不足3天
    - 返回可用天数信息
    
    返回: (input_seq, target, cumulative_return, daily_returns, available_days) 或 None
        available_days: 可用的未来天数 (1, 2, 或 3)
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    data_length = len(stock_data)

    # 安全检查：确保存在前一天数据作为归一化基准
    if start_idx < 1:
        return None  # 无法获取 stock_data[start_idx-1] 作为基准日
    
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
    sample_window_end = min(data_length, start_idx + DataConfig.REQUIRED_LENGTH)
    sample_data = stock_data[sample_window_start:sample_window_end]

    limit_threshold = 0.11
    for day_idx in range(1, len(sample_data)):
        today_close = sample_data[day_idx, 3]
        yesterday_close = sample_data[day_idx - 1, 3]
        if yesterday_close > 0:
            daily_return = (today_close - yesterday_close) / yesterday_close
            if abs(daily_return) > limit_threshold:
                return None

    # 上下文最后一天涨停过滤（通过配置开关控制）
    if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP:
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

    t1_idx = start_idx + context_length
    t2_idx = start_idx + context_length + 1
    t3_idx = start_idx + context_length + 2
    
    available_days = 0
    if t1_idx < data_length:
        available_days = 1
    if t2_idx < data_length:
        available_days = 2
    if t3_idx < data_length:
        available_days = 3
    
    if available_days == 0:
        return None
    
    t1_open = stock_data[t1_idx, 0]
    t1_close = stock_data[t1_idx, 3]

    if t1_open == 0:
        return None

    daily_returns = []
    cumulative_return = 0.0
    
    day1_return = (t1_close - t1_open) / t1_open
    daily_returns.append(day1_return)
    cumulative_return = day1_return
    
    t2_close = None
    t3_close = None
    
    if available_days >= 2:
        t2_close = stock_data[t2_idx, 3]
        day2_return = (t2_close - t1_close) / t1_open
        daily_returns.append(day2_return)
        cumulative_return = day1_return + day2_return
    
    if available_days >= 3:
        t3_close = stock_data[t3_idx, 3]
        day3_return = (t3_close - t2_close) / t1_open
        daily_returns.append(day3_return)
        cumulative_return = day1_return + day2_return + day3_return
    
    daily_price_changes = []
    day1_price_change = (t1_close - closes[-1]) / closes[-1]
    daily_price_changes.append(day1_price_change)
    
    if available_days >= 2:
        day2_price_change = (t2_close - t1_close) / t1_close
        daily_price_changes.append(day2_price_change)
    
    if available_days >= 3:
        day3_price_change = (t3_close - t2_close) / t2_close
        daily_price_changes.append(day3_price_change)
    
    target = float(check_strong_signal(daily_price_changes))

    return input_seq, target, cumulative_return, daily_returns, available_days


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
    pos_pool_returns = []
    neg_pool_inputs = []
    neg_pool_targets = []
    neg_pool_returns = []

    all_batch_inputs = []
    all_batch_targets = []
    all_batch_returns = []

    batches_generated = 0
    
    initial_rounds = 50
    total_rounds_generated = 0
    total_indices_generated = 0
    
    print(f"    动态采样策略：按需生成索引，直到满足{batches_per_epoch}个batch...")
    
    while batches_generated < batches_per_epoch:
        if isinstance(sampler, RandomSampler):
            sample_indices = sampler.sample_batch_rounds(initial_rounds, rng)
        else:
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

            input_seq, target, cumulative_return, _ = sample

            if target >= 0.5:
                pos_pool_inputs.append(input_seq)
                pos_pool_targets.append(target)
                pos_pool_returns.append(cumulative_return)
            else:
                neg_pool_inputs.append(input_seq)
                neg_pool_targets.append(target)
                neg_pool_returns.append(cumulative_return)
            
            if len(pos_pool_inputs) >= pos_quota and len(neg_pool_inputs) >= neg_quota:
                batch_pos_inputs = pos_pool_inputs[:pos_quota]
                batch_pos_targets = pos_pool_targets[:pos_quota]
                batch_pos_returns = pos_pool_returns[:pos_quota]
                
                neg_indices = rng.sample(range(len(neg_pool_inputs)), neg_quota)
                batch_neg_inputs = [neg_pool_inputs[i] for i in neg_indices]
                batch_neg_targets = [neg_pool_targets[i] for i in neg_indices]
                batch_neg_returns = [neg_pool_returns[i] for i in neg_indices]
                
                batch_inputs = batch_pos_inputs + batch_neg_inputs
                batch_targets = batch_pos_targets + batch_neg_targets
                batch_returns = batch_pos_returns + batch_neg_returns
                
                combined = list(zip(batch_inputs, batch_targets, batch_returns))
                rng.shuffle(combined)
                b_inputs, b_targets, b_returns = zip(*combined)
                
                all_batch_inputs.extend(b_inputs)
                all_batch_targets.extend(b_targets)
                all_batch_returns.extend(b_returns)
                
                batches_generated += 1
                
                pos_pool_inputs = pos_pool_inputs[pos_quota:]
                pos_pool_targets = pos_pool_targets[pos_quota:]
                pos_pool_returns = pos_pool_returns[pos_quota:]
                neg_pool_inputs = []
                neg_pool_targets = []
                neg_pool_returns = []
        
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

    return np.asarray(all_batch_inputs), np.asarray(all_batch_targets), np.asarray(all_batch_returns)


def create_fixed_evaluation_dataset(test_stock_info):
    """
    创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）
    
    只包含完整样本（available_days == 3），用于模型评估
    
    返回:
        eval_inputs: 输入序列
        eval_targets: 标签
        eval_cumulative_returns: 累计收益率 = (T+3收盘 - T+1开盘) / T+1开盘
        eval_day_indices: 每个样本对应的预测日在测试集中的相对偏移量（用于实战收益率按天分组）
        eval_daily_returns: 每日收益率列表 [[r1, r2, r3], ...]
            - 基准是买入价（T+1开盘价），用于计算投资回报
            - r1 = (T+1收盘 - T+1开盘) / T+1开盘（Day1日内收益）
            - r2 = (T+2收盘 - T+1收盘) / T+1开盘（Day2收益贡献）
            - r3 = (T+3收盘 - T+2收盘) / T+1开盘（Day3收益贡献）
            - r1 + r2 + r3 = 累计收益率
    """
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []
    eval_day_indices = []
    eval_daily_returns = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
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

            input_seq, target, cumulative_return, daily_returns = sample
            eval_inputs.append(input_seq)
            eval_targets.append(target)
            eval_cumulative_returns.append(float(cumulative_return))
            eval_daily_returns.append(daily_returns)
            
            predict_day_idx = start_idx + DataConfig.CONTEXT_LENGTH
            day_index = predict_day_idx - test_split_point
            eval_day_indices.append(day_index)

    if len(eval_inputs) == 0:
        raise ValueError("固定评估集为空：test_stock_info中没有可用样本")

    return (np.asarray(eval_inputs), np.asarray(eval_targets), 
            np.asarray(eval_cumulative_returns), np.asarray(eval_day_indices),
            eval_daily_returns)


def create_recent_days_dataset(test_stock_info):
    """
    创建最近几天的临时评估数据集（用于run.py中显示最近几天的实战收益率）
    
    包含完整样本和临时样本，用于展示最近几天的选股情况
    - 完整样本（available_days == 3）：与 create_fixed_evaluation_dataset 一致
    - 临时样本（available_days < 3）：仅用于展示，方便用户决策
    
    返回:
        recent_inputs: 输入序列
        recent_cumulative_returns: 累计收益率（可能不完整）
        recent_day_indices: 预测日索引
        recent_available_days: 可用天数 (1, 2, 或 3)
    """
    recent_inputs = []
    recent_cumulative_returns = []
    recent_day_indices = []
    recent_available_days = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))
        
        # 和 create_fixed_evaluation_dataset 一样的起点，但扩展到包含最近的临时数据
        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.CONTEXT_LENGTH - 1
        
        if start_max < start_min:
            continue

        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index_partial([stock_info], 0, start_idx)
            if sample is None:
                continue

            input_seq, target, cumulative_return, daily_returns, available_days = sample
            
            predict_day_idx = start_idx + DataConfig.CONTEXT_LENGTH
            
            recent_inputs.append(input_seq)
            recent_cumulative_returns.append(float(cumulative_return))
            recent_available_days.append(available_days)
            
            day_index = predict_day_idx - test_split_point
            recent_day_indices.append(day_index)

    if len(recent_inputs) == 0:
        return None, None, None, None

    return (np.asarray(recent_inputs), np.asarray(recent_cumulative_returns), 
            np.asarray(recent_day_indices), np.asarray(recent_available_days))
