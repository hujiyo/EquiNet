"""
EquiNet 数据处理模块

包含所有数据相关的功能：
- 数据加载和预处理
- 样本生成
- 时间顺序采样器
- 评估数据集创建
- 预测函数
- 特征归一化模块
"""

import os
import sys
import random
import argparse
import pickle
import numpy as np
import pandas as pd
from config import DataConfig, generate_label, calculate_returns
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from typing import Dict, List, Tuple


class FeatureNormalizer:
    """
    特征归一化器 - 两阶段归一化

    阶段1: QuantileTransformer → 处理偏态和集中度问题
    阶段2: StandardScaler → 确保均值0方差1

    使用方法：
        # 训练阶段
        normalizer = FeatureNormalizer()
        normalizer.fit(train_stock_info)
        normalizer.save('normalizer.pkl')

        # 推理阶段
        normalizer = FeatureNormalizer.load('normalizer.pkl')
        normalized_data = normalizer.transform(raw_data)
    """

    def __init__(self,
                 output_distribution='normal',
                 n_quantiles=1000,
                 random_state=42):
        """
        Args:
            output_distribution: 'normal' 或 'uniform'
                - 'normal': 输出符合标准正态分布（推荐）
                - 'uniform': 输出符合 [0, 1] 均匀分布
            n_quantiles: 分位数数量，越多越精确但越慢
            random_state: 随机种子
        """
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.random_state = random_state

        # 为每个特征组创建独立的 pipeline
        self.ohl_pipeline = self._create_pipeline()
        self.volume_pipeline = self._create_pipeline()
        self.exchange_pipeline = self._create_pipeline()
        self.index_pipeline = self._create_pipeline()

        self.is_fitted = False

    def _create_pipeline(self):
        """
        创建两阶段归一化 pipeline

        为什么需要 StandardScaler？
        - QuantileTransformer 的输出虽然是正态分布，但均值和方差可能不是 0 和 1
        - StandardScaler 确保最终输出严格满足：均值=0，标准差=1
        """
        from sklearn.pipeline import Pipeline

        return Pipeline([
            ('quantile', QuantileTransformer(
                output_distribution=self.output_distribution,
                n_quantiles=self.n_quantiles,
                random_state=self.random_state,
                subsample=100000
            )),
            ('scaler', StandardScaler())
        ])

    def _collect_training_features(self, train_stock_info: List[Dict], index_data: Dict = None, times: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从训练集收集所有特征值（避免数据泄漏）

        关键：只使用每只股票的训练集部分（train_end_idx 之前）
        
        使用 data.py 中的 coarse_normalize_context_window() 进行粗处理，
        确保与训练时的数据处理逻辑完全一致。

        Args:
            train_stock_info: 训练集股票信息列表
            index_data: 大盘数据字典 {date: index_value}
            times: 股票时间映射字典 {stock_code: times_array}

        Returns:
            ohl_data: OHLC 特征 [N_samples * 30 * 4]
            volume_data: Volume 特征 [N_samples * 30]
            exchange_data: Exchange 特征 [N_samples * 30]
            index_data_collected: Index 特征 [N_samples * 30]
        """
        from data import coarse_normalize_context_window, DataConfig
        
        ohl_data = []
        volume_data = []
        exchange_data = []
        index_data_collected = []
        
        context_length = DataConfig.CONTEXT_LENGTH

        for stock in train_stock_info:
            data = stock['data']
            stock_code = stock.get('file_name', '')
            train_start_idx = stock.get('train_start_idx', 1)
            train_end_idx = stock.get('train_end_idx', len(data))            
            stock_times = times.get(stock_code) if times else None

            for i in range(train_start_idx, train_end_idx - context_length):
                input_seq = coarse_normalize_context_window(
                    data, i, context_length,
                    check_limit_up=False,
                    required_length=context_length,
                    index_data=index_data,
                    times=stock_times
                )
                
                if input_seq is None:
                    continue
                
                ohl_data.append(input_seq[:, :4].flatten())
                volume_data.append(input_seq[:, 4].flatten())
                exchange_data.append(input_seq[:, 5].flatten())
                index_data_collected.append(input_seq[:, 6].flatten())

        ohl_data = np.concatenate(ohl_data) if ohl_data else np.array([])
        volume_data = np.concatenate(volume_data) if volume_data else np.array([])
        exchange_data = np.concatenate(exchange_data) if exchange_data else np.array([])
        index_data_collected = np.concatenate(index_data_collected) if index_data_collected else np.array([])

        print(f"[FeatureNormalizer] 收集到的训练数据:")
        print(f"  OHLC: {len(ohl_data)} 个值")
        print(f"  Volume: {len(volume_data)} 个值")
        print(f"  Exchange: {len(exchange_data)} 个值")
        print(f"  Index: {len(index_data_collected)} 个值")

        return ohl_data, volume_data, exchange_data, index_data_collected

    def fit(self, train_stock_info: List[Dict], index_data: Dict = None, times: Dict = None):
        """
        在训练集上拟合归一化器

        ⚠️ 重要：此函数必须在训练集上调用，且只能调用一次
        测试集不能调用此函数，否则会导致数据泄漏

        Args:
            train_stock_info: 训练集股票信息列表
            index_data: 大盘数据字典 {date: index_value}
            times: 股票时间映射字典 {stock_code: times_array}
        """
        print("\n[FeatureNormalizer] 开始拟合归一化器...")
        print(f"  输出分布: {self.output_distribution}")
        print(f"  分位数数量: {self.n_quantiles}")

        # 收集训练数据
        ohl_data, volume_data, exchange_data, index_data_collected = self._collect_training_features(
            train_stock_info, index_data, times
        )

        # 拟合每个特征组的 pipeline
        print("\n[FeatureNormalizer] 拟合 OHLC 特征...")
        self.ohl_pipeline.fit(ohl_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Volume 特征...")
        self.volume_pipeline.fit(volume_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Exchange 特征...")
        self.exchange_pipeline.fit(exchange_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Index 特征...")
        if len(index_data_collected) == 0:
            raise RuntimeError("Index特征数据为空！请确保大盘数据已正确加载。")
        self.index_pipeline.fit(index_data_collected.reshape(-1, 1))

        self.is_fitted = True

        self._print_transform_stats(ohl_data, volume_data, exchange_data, index_data_collected)

        print("\n[FeatureNormalizer] ✓ 拟合完成！")

    def _print_transform_stats(self, ohl_data, volume_data, exchange_data, index_data_collected=None):
        """
        打印变换后的统计信息，验证归一化效果
        """
        print("\n[FeatureNormalizer] 变换后的统计信息:")

        # OHLC
        ohl_transformed = self.ohl_pipeline.transform(ohl_data.reshape(-1, 1)).flatten()
        print(f"  OHLC:")
        print(f"    均值: {ohl_transformed.mean():.6f}")
        print(f"    标准差: {ohl_transformed.std():.6f}")
        print(f"    范围: [{ohl_transformed.min():.6f}, {ohl_transformed.max():.6f}]")

        # Volume
        volume_transformed = self.volume_pipeline.transform(volume_data.reshape(-1, 1)).flatten()
        print(f"  Volume:")
        print(f"    均值: {volume_transformed.mean():.6f}")
        print(f"    标准差: {volume_transformed.std():.6f}")
        print(f"    范围: [{volume_transformed.min():.6f}, {volume_transformed.max():.6f}]")

        # Exchange
        exchange_transformed = self.exchange_pipeline.transform(exchange_data.reshape(-1, 1)).flatten()
        print(f"  Exchange:")
        print(f"    均值: {exchange_transformed.mean():.6f}")
        print(f"    标准差: {exchange_transformed.std():.6f}")
        print(f"    范围: [{exchange_transformed.min():.6f}, {exchange_transformed.max():.6f}]")

        if index_data_collected is not None and len(index_data_collected) > 0:
            index_transformed = self.index_pipeline.transform(index_data_collected.reshape(-1, 1)).flatten()
            print(f"  Index:")
            print(f"    均值: {index_transformed.mean():.6f}")
            print(f"    标准差: {index_transformed.std():.6f}")
            print(f"    范围: [{index_transformed.min():.6f}, {index_transformed.max():.6f}]")
    def transform(self, input_seq: np.ndarray) -> np.ndarray:
        """
        对单个样本应用归一化

        ⚠️ 重要：此函数可以在训练集、验证集、测试集上调用
        因为它只使用 fit() 时学到的参数，不会产生数据泄漏

        Args:
            input_seq: [context_length, 6] 原始输入序列

        Returns:
            normalized_seq: [context_length, 6] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        normalized = np.empty_like(input_seq, dtype=np.float32)

        # 展平以便转换
        ohl_flat = input_seq[:, :4].flatten()  # [context_length * 4]
        volume_flat = input_seq[:, 4].flatten()  # [context_length]
        exchange_flat = input_seq[:, 5].flatten()  # [context_length]
        index_flat = input_seq[:, 6].flatten()  # [context_length]
        # 转换每个特征组
        normalized_ohl = self.ohl_pipeline.transform(
            ohl_flat.reshape(-1, 1)
        ).flatten()
        normalized_volume = self.volume_pipeline.transform(
            volume_flat.reshape(-1, 1)
        ).flatten()
        normalized_exchange = self.exchange_pipeline.transform(
            exchange_flat.reshape(-1, 1)
        ).flatten()
        normalized_index = self.index_pipeline.transform(
            index_flat.reshape(-1, 1)
        ).flatten()

        # 重塑回原始形状
        normalized[:, :4] = normalized_ohl.reshape(input_seq[:, :4].shape)
        normalized[:, 4] = normalized_volume
        normalized[:, 5] = normalized_exchange
        normalized[:, 6] = normalized_index

        return normalized

    def fit_transform(self, train_stock_info: List[Dict], index_data: np.ndarray = None) -> 'FeatureNormalizer':
        """
        拟合并返回归一化器（链式调用）

        Args:
            train_stock_info: 训练集股票信息列表

        Returns:
            self: 拟合后的归一化器
        """
        self.fit(train_stock_info, index_data)
        return self

    def save(self, path: str):
        """
        保存归一化器到文件

        Args:
            path: 保存路径（例如: './normalizer.pkl'）
        """
        if not self.is_fitted:
            raise RuntimeError("无法保存未拟合的归一化器")

        with open(path, 'wb') as f:
            pickle.dump({
                'ohl_pipeline': self.ohl_pipeline,
                'volume_pipeline': self.volume_pipeline,
                'exchange_pipeline': self.exchange_pipeline,
                'index_pipeline': self.index_pipeline,
                'is_fitted': self.is_fitted,
                'output_distribution': self.output_distribution,
                'n_quantiles': self.n_quantiles,
                'random_state': self.random_state
            }, f)

        print(f"[FeatureNormalizer] ✓ 归一化器已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """
        从文件加载归一化器

        Args:
            path: 归一化器文件路径

        Returns:
            加载的归一化器实例
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"归一化器文件不存在: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # 创建新实例
        normalizer = cls(
            output_distribution=data['output_distribution'],
            n_quantiles=data['n_quantiles'],
            random_state=data['random_state']
        )

        # 恢复状态
        normalizer.ohl_pipeline = data['ohl_pipeline']
        normalizer.volume_pipeline = data['volume_pipeline']
        normalizer.exchange_pipeline = data['exchange_pipeline']
        if 'index_pipeline' in data:
            normalizer.index_pipeline = data['index_pipeline']
        else:
            normalizer.index_pipeline = normalizer._create_pipeline()
        normalizer.is_fitted = data['is_fitted']

        print(f" ✓ 归一化器已从 {path} 加载")

        return normalizer

def load_index_data(data_dir=DataConfig.DATA_DIR):
    """
    加载上证指数数据并构建日期到涨跌幅的映射

    Args:
        data_dir: 数据目录

    Returns:
        index_changes: dict，key为time(int)，value为当日涨跌幅
                       涨跌幅 = (当日收盘 - 前一日收盘) / 前一日收盘
    """
    index_file = os.path.join(data_dir, DataConfig.INDEX_FILE)

    if not os.path.exists(index_file):
        print(f"警告：大盘数据文件不存在: {index_file}")
        return None

    try:
        df = pd.read_csv(index_file)
        df = df.sort_values('time', ascending=True).reset_index(drop=True)

        times = df['time'].values
        closes = df['end'].values

        index_changes = {}
        for i in range(1, len(times)):
            today = int(times[i])
            yesterday = int(times[i - 1])
            today_close = closes[i]
            yesterday_close = closes[i - 1]

            if yesterday_close > 0:
                change = (today_close - yesterday_close) / yesterday_close
            else:
                change = 0.0

            index_changes[today] = float(change)

        print(f"大盘涨跌幅数据已加载：{len(index_changes)} 条记录")

        return index_changes
    except Exception as e:
        print(f"加载大盘数据失败：{e}")
        return None

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
            'test_split_point': test_split_point, # 测试集起始索引，固定为最后test_days天的开始位置
            'times': times                       # 时间戳数组，用于后续分析
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
    
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f != DataConfig.INDEX_FILE]
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
            'times': stock_info['times'],
        })
        
        test_stock_info.append({
            'file_name': stock_info['file_name'],
            'data': stock_info['test_data'],
            'data_length': stock_info['data_length'],
            'test_split_point': stock_info['test_split_point'],
            'times': stock_info['times'],
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


def generate_sample_from_index(stock_info_list, stock_idx, start_idx, feature_normalizer=None, index_data=None, times=None):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引
        feature_normalizer: 可选的特征归一化器实例

    返回: dict 或 None（如果样本无效），字典包含：
        input_seq: 输入序列
        target: 标签（0或1）
        cumulative_return: 累计收益率（考虑止损）
        daily_returns: 每日收益率列表
        daily_price_changes: 每日涨跌幅列表 [day1, day2, day3]
        daily_opens: 每日开盘价列表 [t1_open, t2_open, ...]
        daily_highs: 每日最高价列表 [t1_high, t2_high, ...]
        daily_lows: 每日最低价列表 [t1_low, t2_low, ...]
    
    核心概念区分：
        【涨跌幅】用于标签生成，判断股票走势强弱
            - 基准是前一日收盘价
        【收益率】用于计算投资回报，评估模型表现，支持智能止损
            - 基准是买入价（T+1开盘价）
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    required_length = DataConfig.REQUIRED_LENGTH
    stock_times = times if times is not None else stock_info.get('times', None)

    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer,
        index_data=index_data,
        times=stock_times
    )
    
    if input_seq is None:
        return None

    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    closes = input_seq_raw[:, 3]
    prev_close = closes[-1]

    daily_opens = []
    daily_highs = []
    daily_lows = []
    daily_closes = []
    daily_price_changes = []

    for d in range(future_days):
        idx = start_idx + context_length + d
        day_open = stock_data[idx, 0]
        day_high = stock_data[idx, 1]
        day_low = stock_data[idx, 2]
        day_close = stock_data[idx, 3]

        if day_open == 0 or day_close == 0:
            return None

        daily_opens.append(day_open)
        daily_highs.append(day_high)
        daily_lows.append(day_low)
        daily_closes.append(day_close)

        base_close = prev_close if d == 0 else daily_closes[d - 1]
        daily_price_changes.append((day_close - base_close) / base_close)

    cumulative_return, daily_returns = calculate_returns(
        t1_open=daily_opens[0],
        t1_close=daily_closes[0],
        t2_open=daily_opens[1] if future_days >= 2 else None,
        t2_close=daily_closes[1] if future_days >= 2 else None,
        t3_close=daily_closes[2] if future_days >= 3 else None,
        day1_change=daily_price_changes[0],
        day2_change=daily_price_changes[1] if future_days >= 2 else None,
        day3_change=daily_price_changes[2] if future_days >= 3 else None
    )

    target = float(generate_label(
        day1_change=daily_price_changes[0],
        day2_change=daily_price_changes[1] if future_days >= 2 else 0.0,
        day3_change=daily_price_changes[2] if future_days >= 3 else 0.0
    ))

    buffer_day_open = None
    buffer_day_high = None
    buffer_day_low = None
    buffer_day_change = None
    if DataConfig.BUFFER_DAY:
        buf_idx = start_idx + context_length + future_days
        if buf_idx < len(stock_data):
            buffer_day_open = stock_data[buf_idx, 0]
            buffer_day_high = stock_data[buf_idx, 1]
            buffer_day_low = stock_data[buf_idx, 2]
            buf_close = stock_data[buf_idx, 3]
            if buffer_day_open == 0 or buf_close == 0:
                return None
            buffer_day_change = (buf_close - daily_closes[-1]) / daily_closes[-1]

    return {
        'input_seq': input_seq,
        'target': target,
        'cumulative_return': cumulative_return,
        'daily_returns': daily_returns,
        'daily_price_changes': daily_price_changes,
        'daily_opens': daily_opens,
        'daily_highs': daily_highs,
        'daily_lows': daily_lows,
        'buffer_day_open': buffer_day_open,
        'buffer_day_high': buffer_day_high,
        'buffer_day_low': buffer_day_low,
        'buffer_day_change': buffer_day_change,
    }


def generate_sample_from_index_partial(stock_info_list, stock_idx, start_idx, feature_normalizer=None, index_data=None, times=None):
    """
    生成样本，支持不完整的未来数据（用于最近几天的临时评估）

    与generate_sample_from_index的区别：
    - 由run.py使用，与模型训练阶段脚本无关
    - 允许未来数据不足 FUTURE_DAYS 天
    - 返回可用天数信息
    - 不生成标签（仅用于推理展示）

    返回: dict 或 None，字典包含：
        input_seq: 输入序列
        cumulative_return: 累计收益率（考虑止损）
        daily_returns: 每日收益率列表
        available_days: 可用的未来天数 (1 ~ FUTURE_DAYS)
        daily_price_changes: 每日涨跌幅列表（不足的为None）
        daily_opens: 每日开盘价列表（不足的为None）
        daily_highs: 每日最高价列表（不足的为None）
        daily_lows: 每日最低价列表（不足的为None）
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    data_length = len(stock_data)
    stock_times = times if times is not None else stock_info.get('times', None)

    required_length = min(DataConfig.REQUIRED_LENGTH, data_length - start_idx)
    
    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer,
        index_data=index_data,
        times=stock_times
    )
    
    if input_seq is None:
        return None

    available_days = 0
    for d in range(future_days):
        idx = start_idx + context_length + d
        if idx < data_length:
            available_days = d + 1
        else:
            break
    
    if available_days == 0:
        return None
    
    t_day_close = stock_data[start_idx + context_length - 1, 3]

    daily_opens = [None] * future_days
    daily_highs = [None] * future_days
    daily_lows = [None] * future_days
    daily_closes = [None] * future_days
    daily_price_changes = [None] * future_days

    for d in range(available_days):
        idx = start_idx + context_length + d
        day_open = stock_data[idx, 0]
        day_high = stock_data[idx, 1]
        day_low = stock_data[idx, 2]
        day_close = stock_data[idx, 3]

        if day_open == 0 or day_close == 0:
            return None

        daily_opens[d] = day_open
        daily_highs[d] = day_high
        daily_lows[d] = day_low
        daily_closes[d] = day_close

        base_close = t_day_close if d == 0 else daily_closes[d - 1]
        daily_price_changes[d] = (day_close - base_close) / base_close

    cumulative_return, daily_returns = calculate_returns(
        t1_open=daily_opens[0],
        t1_close=daily_closes[0],
        t2_open=daily_opens[1] if available_days >= 2 else None,
        t2_close=daily_closes[1] if available_days >= 2 else None,
        t3_close=daily_closes[2] if available_days >= 3 else None,
        day1_change=daily_price_changes[0],
        day2_change=daily_price_changes[1] if available_days >= 2 else None,
        day3_change=daily_price_changes[2] if available_days >= 3 else None
    )

    buffer_day_open = None
    buffer_day_high = None
    buffer_day_low = None
    buffer_day_change = None
    if DataConfig.BUFFER_DAY and available_days == future_days:
        buf_idx = start_idx + context_length + future_days
        if buf_idx < data_length:
            buffer_day_open = stock_data[buf_idx, 0]
            buffer_day_high = stock_data[buf_idx, 1]
            buffer_day_low = stock_data[buf_idx, 2]
            buf_close = stock_data[buf_idx, 3]
            if buffer_day_open != 0 and buf_close != 0 and daily_closes[future_days - 1] is not None:
                buffer_day_change = (buf_close - daily_closes[future_days - 1]) / daily_closes[future_days - 1]

    return {
        'input_seq': input_seq,
        'cumulative_return': cumulative_return,
        'daily_returns': daily_returns,
        'available_days': available_days,
        'daily_price_changes': daily_price_changes,
        'daily_opens': daily_opens,
        'daily_highs': daily_highs,
        'daily_lows': daily_lows,
        'buffer_day_open': buffer_day_open,
        'buffer_day_high': buffer_day_high,
        'buffer_day_low': buffer_day_low,
        'buffer_day_change': buffer_day_change,
    }


def sample_with_pools(sampler, stock_info_list, batch_size, batches_per_epoch, rng, feature_normalizer=None, index_data=None, times=None):
    """
    使用样本池机制采样（流式处理版）：
    1. 按时间顺序遍历样本索引
    2. 实时填充正负样本池
    3. 一旦正样本达到配额且负样本足够，立即生成Batch并清空负样本池
    4. 确保Batch之间的时间有序性，严格防止未来数据泄露到过去的Batch中
    5. 支持循环采样：数据到达末尾后自动循环回起点
    6. 动态生成索引：按需生成，直到batch数量满足要求

    Args:
        sampler: 采样器实例
        stock_info_list: 股票信息列表
        batch_size: 批次大小
        batches_per_epoch: 每个epoch的batch数量
        rng: 随机数生成器
        feature_normalizer: 可选的特征归一化器实例
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

            sample = generate_sample_from_index(stock_info_list, stock_idx, start_idx, feature_normalizer, index_data, times)
            if sample is None:
                continue

            input_seq = sample['input_seq']
            target = sample['target']
            cumulative_return = sample['cumulative_return']

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


def create_fixed_evaluation_dataset(test_stock_info, feature_normalizer=None, index_data=None, times=None):
    """
    创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）
    
    只包含完整样本（available_days == FUTURE_DAYS），用于模型评估
    
    Args:
        test_stock_info: 测试集股票信息列表
        feature_normalizer: 可选的特征归一化器实例
    
    返回: dict，包含：
        inputs: 输入序列
        targets: 标签
        cumulative_returns: 累计收益率
        day_indices: 预测日在测试集中的相对偏移量
        daily_returns: 每日收益率列表
        daily_price_changes: 每日涨跌幅列表
        daily_opens: 每日开盘价列表
        daily_highs: 每日最高价列表
        daily_lows: 每日最低价列表
        buffer_day_opens: buffer day 开盘价列表
        buffer_day_highs: buffer day 最高价列表
        buffer_day_lows: buffer day 最低价列表
        buffer_day_changes: buffer day 涨跌幅列表
    """
    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []
    eval_day_indices = []
    eval_daily_returns = []
    eval_daily_price_changes = []
    eval_daily_opens = []
    eval_daily_highs = []
    eval_daily_lows = []
    eval_buffer_day_opens = []
    eval_buffer_day_highs = []
    eval_buffer_day_lows = []
    eval_buffer_day_changes = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))

        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.REQUIRED_LENGTH
        
        if start_max < start_min:
            continue

        stock_times = times if times is not None else stock_info.get('times', None)
        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index([stock_info], 0, start_idx, feature_normalizer, index_data, stock_times)
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


def create_recent_days_dataset(test_stock_info, feature_normalizer=None, index_data=None, times=None):
    recent_inputs = []
    recent_cumulative_returns = []
    recent_day_indices = []
    recent_available_days = []

    for stock_info in test_stock_info:
        stock_data = stock_info['data']
        data_length = len(stock_data)
        test_split_point = stock_info.get('test_split_point', max(0, data_length - DataConfig.TEST_DAYS))
        
        start_min = max(1, test_split_point)
        start_max = data_length - DataConfig.CONTEXT_LENGTH - 1
        
        if start_max < start_min:
            continue

        stock_times = times if times is not None else stock_info.get('times', None)
        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index_partial([stock_info], 0, start_idx, feature_normalizer, index_data, stock_times)
            if sample is None:
                continue

            input_seq, cumulative_return, daily_returns, available_days = sample

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


def normalize_and_validate_context_window(stock_data, start_idx, context_length,
                                          check_limit_up=True, required_length=None,
                                          feature_normalizer=None,
                                          apply_fine_normalization=True,
                                          index_data=None, times=None):
    """
    统一的上下文窗口归一化和验证函数

    用于消除 run.py 和 data.py 中的代码重复。
    执行完整的数据验证和归一化流程，与 generate_sample_from_index 保持一致。

    数据处理分两阶段：
        - 粗处理：CSV → OHLE 格式（涨跌幅 -0.1~0.1，Volume 0~1，Exchange 0~1）
        - 细处理：OHLE → 标准化数据（均值≈0，方差≈1）

    Args:
        stock_data: 股票原始数据 [N, 6]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口
        feature_normalizer: 可选的特征归一化器实例，用于细处理阶段
        apply_fine_normalization: 是否应用细处理（默认 True）。设为 False 时只执行粗处理。

    Returns:
        input_seq: [context_length, 6] 归一化后的输入序列，或 None（如果验证失败）
            - 粗处理后：OHLE: -0.1~0.1, Volume: 0~1, Exchange: 0~1
            - 细处理后：均值≈0，方差≈1

    验证项：
        1. 基准日（start_idx-1）的 OHLC 和 volume 非零
        2. 上下文窗口的 close 和 volume 非零
        3. 涨停过滤：窗口内任何一天涨跌幅不超过 11%
        4. 上下文最后一天涨停过滤（可选，通过 DataConfig 控制）
        5. 归一化后无 nan/inf
    """
    if start_idx < 1:
        return None
    
    if required_length is None:
        required_length = context_length
    
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

    if check_limit_up:
        sample_window_start = start_idx - 1
        sample_window_end = start_idx + required_length
        sample_data = stock_data[sample_window_start:sample_window_end]

        limit_threshold = 0.11
        for day_idx in range(1, len(sample_data)):
            today_close = sample_data[day_idx, 3]
            yesterday_close = sample_data[day_idx - 1, 3]

            if yesterday_close == 0:
                return None
            daily_return = (today_close - yesterday_close) / yesterday_close
            if abs(daily_return) > limit_threshold:
                return None

    # 上下文最后一天涨停过滤（独立于 check_limit_up，仅受 DataConfig 控制）
    if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP:
        last_day_idx = start_idx + context_length - 1
        prev_day_idx = start_idx + context_length - 2
        prev_day_close = stock_data[prev_day_idx, 3]
        last_day_close = stock_data[last_day_idx, 3]

        if prev_day_close == 0:
            return None
        last_day_return = (last_day_close - prev_day_close) / prev_day_close
        if last_day_return >= DataConfig.LIMIT_THRESHOLD:
            return None

    input_seq = np.empty((context_length, 7), dtype=np.float32)

    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]
    
    input_seq[0, 4] = (volumes[0] - prev_volume) / prev_volume
    if context_length > 1:
        input_seq[1:, 4] = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    
    input_seq[:, 5] = input_seq_raw[:, 5] / 100.0

    if index_data is not None and times is not None:
        for i in range(context_length):
            day_time = int(times[start_idx + i])
            index_change = index_data.get(day_time, 0.0)
            input_seq[i, 6] = index_change
    else:
        input_seq[:, 6] = 0.0
    # ========== 粗处理阶段 ==========
    # OHLE: 涨跌幅，范围 -0.1 ~ 0.1
    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])
    # Volume: 变化率缩放，范围 0 ~ 1
    np.clip(input_seq[:, 4], -5.0, 5.0, out=input_seq[:, 4])
    input_seq[:, 4] = input_seq[:, 4] / 10.0 + 0.5
    np.clip(input_seq[:, 5], 0.0, 1.0, out=input_seq[:, 5])
    np.clip(input_seq[:, 6], -0.1, 0.1, out=input_seq[:, 6])

    # ========== 细处理阶段（可选）==========
    # 应用高级特征归一化，将粗处理结果转换为均值≈0、方差≈1的标准化数据
    if apply_fine_normalization and feature_normalizer is not None:
        input_seq = feature_normalizer.transform(input_seq)

    if np.any(~np.isfinite(input_seq)):
        return None

    return input_seq


def coarse_normalize_context_window(stock_data, start_idx, context_length,
                                     check_limit_up=True, required_length=None,
                                     index_data=None, times=None):
    """
    粗处理：CSV → OHLE 格式

    只执行粗处理阶段，不应用细处理（特征归一化器）。
    输出数据范围：
        - OHLE: -0.1 ~ 0.1（涨跌幅）
        - Volume: 0 ~ 1（成交量变化率）
        - Exchange: 0 ~ 1（换手率）

    Args:
        stock_data: 股票原始数据 [N, 6]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口

    Returns:
        input_seq: [context_length, 6] 粗处理后的输入序列，或 None（如果验证失败）
    """
    return normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=check_limit_up,
        required_length=required_length,
        feature_normalizer=None,
        apply_fine_normalization=False,
        index_data=index_data,
        times=times
    )


def fine_normalize_batch(input_seq, feature_normalizer):
    """
    细处理：OHLE → 标准化数据

    将粗处理后的数据送入细处理阶段，应用特征归一化器。
    输出数据特性：均值≈0，方差≈1

    Args:
        input_seq: 粗处理后的数据
            - 单个样本: [seq_len, 7]
            - 批量样本: [batch_size, seq_len, 7]
        feature_normalizer: 特征归一化器实例

    Returns:
        normalized_seq: 标准化后的数据，形状与输入相同
    """
    return feature_normalizer.transform(input_seq)


def fit_feature_normalizer(output_path='./normalizer.pkl', output_distribution='normal', n_quantiles=1000):
    """
    在训练集上拟合特征归一化器并保存到文件

    Args:
        output_path: 归一化器输出文件路径
        output_distribution: 输出分布类型 ('normal' 或 'uniform')
        n_quantiles: 分位数数量

    Returns:
        normalizer: 拟合后的 FeatureNormalizer 实例
    """
    if os.path.exists(output_path):
        print(f"归一化器文件已存在: {output_path}")
        response = input("是否重新训练？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("\n[步骤1] 加载训练集数据...")

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print(f"训练集股票数: {len(train_stock_info)}")
    print(f"测试集股票数: {len(test_stock_info)}")

    print("\n[步骤1.5] 加载大盘数据...")
    index_data = load_index_data(DataConfig.DATA_DIR)
    if index_data is None:
        print("警告：大盘数据不存在，归一化器将无法学习大盘特征")

    times_dict = {}
    for stock in train_stock_info:
        file_name = stock.get('file_name', '')
        stock_times = stock.get('times', None)
        if stock_times is not None:
            times_dict[file_name] = stock_times

    print("\n[步骤2] 创建特征归一化器...")
    print(f"  输出分布: {output_distribution}")
    print(f"  分位数数量: {n_quantiles}")

    normalizer = FeatureNormalizer(output_distribution=output_distribution,n_quantiles=n_quantiles)

    print("\n[步骤3] 在训练集上拟合归一化器...")
    normalizer.fit(train_stock_info, index_data, times_dict)

    print("\n[步骤4] 保存归一化器...")
    normalizer.save(output_path)

    return normalizer


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='数据处理模块 兼 拟合特征归一化器训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
用法示例：
  python data.py                                           # 使用默认参数拟合归一化器
  python data.py --output-distribution uniform             # 使用均匀分布拟合
  python data.py --n-quantiles 500                         # 使用500个分位数拟合
  python data.py --output ./my_normalizer.pkl              # 指定输出文件路径
        '''
    )
    parser.add_argument('--output-distribution', type=str, default='normal',choices=['normal', 'uniform'],
                        help='输出分布类型: normal (标准正态) 或 uniform (均匀分布)，默认 normal')
    parser.add_argument('--n-quantiles', type=int, default=1000,help='分位数数量（默认1000，越大越精确但越慢）')
    parser.add_argument('--output', type=str, default='./normalizer.pkl',
                        help='归一化器输出文件路径，默认 ./normalizer.pkl')

    args = parser.parse_args()
    fit_feature_normalizer(output_path=args.output,
    output_distribution=args.output_distribution,n_quantiles=args.n_quantiles)
    print(f"✓ 特征归一化器训练完成！已保存到: {args.output}")

if __name__ == "__main__":
    main()
