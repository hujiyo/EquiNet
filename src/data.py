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
        self.amount_pipeline = self._create_pipeline()
        self.exchange_pipeline = self._create_pipeline()
        self.ma_pipeline = self._create_pipeline()

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

    def _collect_training_features(self, train_stock_info: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        从训练集收集所有特征值（避免数据泄漏）

        关键：只使用每只股票的训练集部分（train_end_idx 之前）

        使用 data.py 中的 coarse_normalize_context_window() 进行粗处理，
        确保与训练时的数据处理逻辑完全一致。

        Returns:
            ohl_data: OHLC 特征 [N_values]
            volume_data: Amount 特征 [N_values]
            exchange_data: Exchange 特征 [N_values]
            ma_data: MA偏离度特征 [N_values]
        """
        from data import coarse_normalize_context_window, DataConfig

        ohl_data = []
        volume_data = []
        exchange_data = []
        ma_data = []

        context_length = DataConfig.CONTEXT_LENGTH
        max_windows = 100000
        total_windows = 0

        for stock in train_stock_info:
            if total_windows >= max_windows:
                break

            data = stock['data']
            train_start_idx = stock.get('train_start_idx', 1)
            train_end_idx = stock.get('train_end_idx', len(data))

            available = train_end_idx - context_length - train_start_idx
            if available <= 0:
                continue

            stride = max(1, available // (max_windows // max(len(train_stock_info), 1)))

            for i in range(train_start_idx, train_end_idx - context_length, stride):
                if total_windows >= max_windows:
                    break

                input_seq = coarse_normalize_context_window(
                    data, i, context_length,
                    check_limit_up=False,
                    required_length=context_length
                )

                if input_seq is None:
                    continue

                ohl_data.append(input_seq[:, :4].flatten())
                volume_data.append(input_seq[:, 4].flatten())
                exchange_data.append(input_seq[:, 5].flatten())
                ma_data.append(input_seq[:, 6:9].flatten())
                total_windows += 1

        ohl_data = np.concatenate(ohl_data) if ohl_data else np.array([])
        volume_data = np.concatenate(volume_data) if volume_data else np.array([])
        exchange_data = np.concatenate(exchange_data) if exchange_data else np.array([])
        ma_data = np.concatenate(ma_data) if ma_data else np.array([])

        print(f"[FeatureNormalizer] 收集到的训练数据 ({total_windows} 个窗口):")
        print(f"  OHLC: {len(ohl_data)} 个值")
        print(f"  Amount: {len(volume_data)} 个值")
        print(f"  Exchange: {len(exchange_data)} 个值")
        print(f"  MA: {len(ma_data)} 个值")

        return ohl_data, volume_data, exchange_data, ma_data

    def fit(self, train_stock_info: List[Dict]):
        """
        在训练集上拟合归一化器

        ⚠️ 重要：此函数必须在训练集上调用，且只能调用一次
        测试集不能调用此函数，否则会导致数据泄漏

        Args:
            train_stock_info: 训练集股票信息列表
        """
        print("\n[FeatureNormalizer] 开始拟合归一化器...")
        print(f"  输出分布: {self.output_distribution}")
        print(f"  分位数数量: {self.n_quantiles}")

        # 收集训练数据
        ohl_data, volume_data, exchange_data, ma_data = self._collect_training_features(
            train_stock_info
        )

        # 拟合每个特征组的 pipeline
        print("\n[FeatureNormalizer] 拟合 OHLC 特征...")
        self.ohl_pipeline.fit(ohl_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Amount 特征...")
        self.amount_pipeline.fit(volume_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Exchange 特征...")
        self.exchange_pipeline.fit(exchange_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 MA 特征...")
        self.ma_pipeline.fit(ma_data.reshape(-1, 1))

        self.is_fitted = True

        self._print_transform_stats(ohl_data, volume_data, exchange_data, ma_data)

        print("\n[FeatureNormalizer] ✓ 拟合完成！")

    def _print_transform_stats(self, ohl_data, volume_data, exchange_data, ma_data):
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

        # Amount
        volume_transformed = self.amount_pipeline.transform(volume_data.reshape(-1, 1)).flatten()
        print(f"  Amount:")
        print(f"    均值: {volume_transformed.mean():.6f}")
        print(f"    标准差: {volume_transformed.std():.6f}")
        print(f"    范围: [{volume_transformed.min():.6f}, {volume_transformed.max():.6f}]")

        # Exchange
        exchange_transformed = self.exchange_pipeline.transform(exchange_data.reshape(-1, 1)).flatten()
        print(f"  Exchange:")
        print(f"    均值: {exchange_transformed.mean():.6f}")
        print(f"    标准差: {exchange_transformed.std():.6f}")
        print(f"    范围: [{exchange_transformed.min():.6f}, {exchange_transformed.max():.6f}]")

        # MA
        ma_transformed = self.ma_pipeline.transform(ma_data.reshape(-1, 1)).flatten()
        print(f"  MA:")
        print(f"    均值: {ma_transformed.mean():.6f}")
        print(f"    标准差: {ma_transformed.std():.6f}")
        print(f"    范围: [{ma_transformed.min():.6f}, {ma_transformed.max():.6f}]")
    def transform(self, input_seq: np.ndarray) -> np.ndarray:
        """
        对单个样本应用归一化

        Args:
            input_seq: [context_length, 9] 原始输入序列

        Returns:
            normalized_seq: [context_length, 9] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        normalized = np.empty_like(input_seq, dtype=np.float32)

        ohl_flat = input_seq[:, :4].flatten()
        volume_flat = input_seq[:, 4].flatten()
        exchange_flat = input_seq[:, 5].flatten()
        ma_flat = input_seq[:, 6:9].flatten()

        normalized_ohl = self.ohl_pipeline.transform(
            ohl_flat.reshape(-1, 1)
        ).flatten()
        normalized_volume = self.amount_pipeline.transform(
            volume_flat.reshape(-1, 1)
        ).flatten()
        normalized_exchange = self.exchange_pipeline.transform(
            exchange_flat.reshape(-1, 1)
        ).flatten()
        normalized_ma = self.ma_pipeline.transform(
            ma_flat.reshape(-1, 1)
        ).flatten()

        normalized[:, :4] = normalized_ohl.reshape(input_seq[:, :4].shape)
        normalized[:, 4] = normalized_volume
        normalized[:, 5] = normalized_exchange
        normalized[:, 6:9] = normalized_ma.reshape(input_seq[:, 6:9].shape)

        return normalized

    def transform_batch(self, input_seqs: np.ndarray) -> np.ndarray:
        """
        批量归一化多个样本（比逐个调用transform高效10-100倍）

        内部将所有样本的特征展平后一次性送入sklearn pipeline，
        避免了逐样本调用时的重复开销（输入验证、维度检查等）。

        Args:
            input_seqs: [batch_size, context_length, 9] 原始输入序列

        Returns:
            [batch_size, context_length, 9] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        batch_size, context_length = input_seqs.shape[0], input_seqs.shape[1]
        normalized = np.empty_like(input_seqs, dtype=np.float32)

        normalized[:, :, :4] = self.ohl_pipeline.transform(
            input_seqs[:, :, :4].reshape(-1, 1)
        ).reshape(batch_size, context_length, 4)

        normalized[:, :, 4] = self.amount_pipeline.transform(
            input_seqs[:, :, 4].reshape(-1, 1)
        ).reshape(batch_size, context_length)

        normalized[:, :, 5] = self.exchange_pipeline.transform(
            input_seqs[:, :, 5].reshape(-1, 1)
        ).reshape(batch_size, context_length)

        normalized[:, :, 6:9] = self.ma_pipeline.transform(
            input_seqs[:, :, 6:9].reshape(-1, 1)
        ).reshape(batch_size, context_length, 3)

        return normalized

    def fit_transform(self, train_stock_info: List[Dict]) -> 'FeatureNormalizer':
        """
        拟合并返回归一化器（链式调用）

        Args:
            train_stock_info: 训练集股票信息列表

        Returns:
            self: 拟合后的归一化器
        """
        self.fit(train_stock_info)
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
                'amount_pipeline': self.amount_pipeline,
                'exchange_pipeline': self.exchange_pipeline,
                'ma_pipeline': self.ma_pipeline,
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
        normalizer.amount_pipeline = data['amount_pipeline']
        normalizer.exchange_pipeline = data['exchange_pipeline']
        normalizer.ma_pipeline = data.get('ma_pipeline', normalizer.ma_pipeline)
        normalizer.is_fitted = data['is_fitted']

        print(f" ✓ 归一化器已从 {path} 加载")

        return normalizer


def process_single_file(args):
    file_path, file_name, test_days, train_start_year = args
    """
    处理单个股票CSV文件，返回包含训练和测试数据的字典
    
    数据处理流程：
    1. 读取CSV并反转时间顺序
    2. 提取OHLCV数据：['open', 'high', 'low', 'close', 'amount', 'exchange']
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
                
        data = df[['open', 'high', 'low', 'close', 'amount', 'exchange', 'm5', 'm10', 'm20']].values
        times = df['date'].values
        
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

def compute_label_distance_exclusions(stock_info_list, distance=None):
    """
    计算正样本距离保护区域

    规则：如果位置i是正样本，则i-distance到i-1的负样本不参与训练（被排除）
    distance=0时不排除任何样本（等价于原始行为）
    只对训练集生效，不影响评估集

    原理：
    滑动窗口生成样本时，正样本左侧的样本特征高度重叠（29/30天相同），
    但因3天预测窗口的硬边界，标签从1翻转为0。
    排除这些"矛盾负样本"，让模型不被"前一天还不该买、后一天就该买了"的矛盾信号干扰。

    Args:
        stock_info_list: 训练集股票信息列表
        distance: 保护距离（默认使用 DataConfig.LABEL_DISTANCE）
    """
    if distance is None:
        distance = DataConfig.LABEL_DISTANCE
    if distance <= 0:
        for stock_info in stock_info_list:
            stock_info['excluded_positions'] = set()
        return

    total_positive = 0
    total_excluded = 0

    for stock_info in stock_info_list:
        data = stock_info['data']
        train_start = stock_info.get('train_start_idx', 0)
        train_end = stock_info.get('train_end_idx', len(data))
        context_length = DataConfig.CONTEXT_LENGTH
        future_days = DataConfig.FUTURE_DAYS

        # 轻量级标签计算：只用价格数据，不做完整样本验证
        # 必须与 generate_sample_from_index() 中的标签计算逻辑保持一致
        labels = {}
        use_open = DataConfig.LABEL_DAY1_USE_OPEN
        for pos in range(max(1, train_start + 1), train_end + 1):
            if pos + context_length + future_days > len(data):
                continue

            future_start = pos + context_length
            closes = data[future_start - 1 : future_start + future_days, 3]

            if any(c <= 0 for c in closes):
                continue

            if use_open:
                # day1: 开盘到收盘（对齐实际买入价）
                t1_open = data[future_start, 0]
                if t1_open <= 0:
                    continue
                day1 = (closes[1] - t1_open) / t1_open
            else:
                # day1: 收盘到收盘（原始行为）
                day1 = (closes[1] - closes[0]) / closes[0]
            day2 = (closes[2] - closes[1]) / closes[1]
            day3 = (closes[3] - closes[2]) / closes[2]
            labels[pos] = generate_label(day1, day2, day3)

        # 构建排除集：正样本左侧distance范围内的负样本
        excluded = set()
        for pos, label in labels.items():
            if label == 1:
                total_positive += 1
                for d in range(1, distance + 1):
                    prev_pos = pos - d
                    if prev_pos in labels and labels[prev_pos] == 0:
                        excluded.add(prev_pos)

        stock_info['excluded_positions'] = excluded
        total_excluded += len(excluded)

    print(f"  正样本距离保护(distance={distance}): "
          f"{total_positive}个正样本, 排除{total_excluded}个负样本")

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

        # 批量选择股票（一次调用替代N次循环，避免重复构建累积权重）
        selected_stocks = rng.choices(
            self.valid_stock_indices,
            weights=sample_weights,
            k=total_samples_to_generate
        )

        for stock_idx in selected_stocks:
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


def generate_sample_from_index(stock_info_list, stock_idx, start_idx, feature_normalizer=None):
    """
    根据预生成的索引生成单个样本（向量化优化版）

    参数:
        stock_info_list: 股票信息列表
        stock_idx: 股票索引
        start_idx: 样本起始索引
        feature_normalizer: 可选的特征归一化器实例

    返回: (input_seq, target, cumulative_return, daily_returns) 或 None（如果样本无效）
    """
    stock_info = stock_info_list[stock_idx]

    # 正样本距离保护：检查当前位置是否被排除
    excluded = stock_info.get('excluded_positions', None)
    if excluded is not None and start_idx in excluded:
        return None

    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    required_length = DataConfig.REQUIRED_LENGTH

    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer
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

        if d == 0:
            if DataConfig.LABEL_DAY1_USE_OPEN:
                # day1使用开盘到收盘的日内涨幅，对齐实际买入价（消除跳空缺口干扰）
                daily_price_changes.append((day_close - day_open) / day_open)
            else:
                # day1使用收盘到收盘涨跌幅（原始行为，包含隔夜跳空）
                daily_price_changes.append((day_close - prev_close) / prev_close)
        else:
            base_close = daily_closes[d - 1]
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

    return (input_seq, target, cumulative_return, daily_returns)


def generate_sample_from_index_partial(stock_info_list, stock_idx, start_idx, feature_normalizer=None):
    """
    生成样本，支持不完整的未来数据（用于最近几天的临时评估）

    与generate_sample_from_index的区别：
    - 由run.py使用，与模型训练阶段脚本无关
    - 允许未来数据不足 FUTURE_DAYS 天
    - 返回可用天数信息
    - 不生成标签（仅用于推理展示）

    返回: (input_seq, cumulative_return, daily_returns, available_days) 或 None
    """
    stock_info = stock_info_list[stock_idx]
    stock_data = stock_info['data']
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    data_length = len(stock_data)

    required_length = min(DataConfig.REQUIRED_LENGTH, data_length - start_idx)
    
    input_seq = normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=True, required_length=required_length,
        feature_normalizer=feature_normalizer
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

    return (input_seq, cumulative_return, daily_returns, available_days)


def create_fixed_evaluation_dataset(test_stock_info, feature_normalizer=None):
    """
    创建固定评估数据集（涨停样本已在generate_sample_from_index中过滤）
    
    只包含完整样本（available_days == FUTURE_DAYS），用于模型评估
    
    Args:
        test_stock_info: 测试集股票信息列表
        feature_normalizer: 可选的特征归一化器实例

    返回: (inputs, targets, cumulative_returns, day_indices, daily_returns)
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

        stock_times = stock_info.get('times', None)
        for start_idx in range(start_min, start_max + 1):
            # 不传归一化器，只做粗处理（后续批量细处理）
            sample = generate_sample_from_index([stock_info], 0, start_idx, None)
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

    # 批量细处理：将所有粗处理后的样本一次性归一化（比逐样本处理快10-100倍）
    eval_inputs_array = np.asarray(eval_inputs)
    if feature_normalizer is not None:
        eval_inputs_array = feature_normalizer.transform_batch(eval_inputs_array)
        # 过滤归一化后产生的NaN/Inf样本
        finite_mask = np.all(np.isfinite(eval_inputs_array.reshape(len(eval_inputs_array), -1)), axis=1)
        if not np.all(finite_mask):
            removed = np.sum(~finite_mask)
            print(f"  ⚠ 归一化后{removed}个样本包含NaN/Inf，已过滤")
            eval_inputs_array = eval_inputs_array[finite_mask]
            eval_targets = [t for t, m in zip(eval_targets, finite_mask) if m]
            eval_cumulative_returns = [r for r, m in zip(eval_cumulative_returns, finite_mask) if m]
            eval_daily_returns = [r for r, m in zip(eval_daily_returns, finite_mask) if m]
            eval_day_indices = [d for d, m in zip(eval_day_indices, finite_mask) if m]

    return (eval_inputs_array, np.asarray(eval_targets),
            np.asarray(eval_cumulative_returns), np.asarray(eval_day_indices),
            eval_daily_returns)


def create_recent_days_dataset(test_stock_info, feature_normalizer=None):
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

        stock_times = stock_info.get('times', None)
        for start_idx in range(start_min, start_max + 1):
            sample = generate_sample_from_index_partial([stock_info], 0, start_idx, feature_normalizer)
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
                                          apply_fine_normalization=True):
    """
    统一的上下文窗口归一化和验证函数

    用于消除 run.py 和 data.py 中的代码重复。
    执行完整的数据验证和归一化流程，与 generate_sample_from_index 保持一致。

    数据处理分两阶段：
        - 粗处理：CSV → OHLE 格式
            - OHLC: 日环比变化率，clip [-0.1, 0.1]
            - Volume: (amount_i - MA_N) / MA_N，MA_N 为过去 N 日均量，无 clip
            - Exchange: (exchange_i - MA_N) / MA_N，MA_N 为过去 N 日均换手率，无 clip
        - 细处理：OHLE → 标准化数据（均值≈0，方差≈1）

    Args:
        stock_data: 股票原始数据 [N, 9]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口
        feature_normalizer: 可选的特征归一化器实例，用于细处理阶段
        apply_fine_normalization: 是否应用细处理（默认 True）。设为 False 时只执行粗处理。

    Returns:
        input_seq: [context_length, 9] 归一化后的输入序列，或 None（如果验证失败）
            - 粗处理后：OHLC: [-0.1, 0.1], Amount: 相对N日均值变化率, Exchange: 相对N日均值变化率
            - 细处理后：均值≈0，方差≈1

    验证项：
        1. 基准日（start_idx-1）的 OHLC 和 amount 非零
        2. 上下文窗口的 close 和 amount 非零
        3. 最新价格过滤：上下文最后一天收盘价不超过40元
        4. 涨停过滤：窗口内任何一天涨跌幅不超过 11%
        5. 上下文最后一天涨停过滤（可选，通过 DataConfig 控制）
        6. 归一化后无 nan/inf
    """
    if start_idx < 1:
        return None
    
    if required_length is None:
        required_length = context_length
    
    input_seq_raw = stock_data[start_idx:start_idx + context_length]
    prev_day_data = stock_data[start_idx - 1]

    prev_close = prev_day_data[3]
    prev_amount = prev_day_data[4]
    if prev_close == 0 or prev_amount == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    closes = input_seq_raw[:, 3]
    amounts = input_seq_raw[:, 4]
    if np.any(closes == 0) or np.any(amounts == 0):
        return None

    if closes[-1] > 40:
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

    input_seq = np.empty((context_length, 9), dtype=np.float32)

    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]

    N = DataConfig.MA_WINDOW
    exchanges = input_seq_raw[:, 5]

    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])

    abs_indices = start_idx + np.arange(context_length)

    if start_idx >= N:
        for col, raw_vals in [(4, amounts), (5, exchanges)]:
            full_col = stock_data[:, col]
            cumsum = np.empty(len(full_col) + 1, dtype=np.float64)
            cumsum[0] = 0
            np.cumsum(full_col, out=cumsum[1:])
            ma_values = (cumsum[abs_indices] - cumsum[abs_indices - N]) / N
            if np.any(~np.isfinite(ma_values)) or np.any(ma_values <= 0):
                return None
            input_seq[:, col] = (raw_vals - ma_values) / ma_values
    else:
        left_starts = np.maximum(0, abs_indices - N)
        deficits = N - (abs_indices - left_starts)
        for col, raw_vals in [(4, amounts), (5, exchanges)]:
            full_col = stock_data[:, col]
            ma_values = np.empty(context_length, dtype=np.float32)
            for k in range(context_length):
                if deficits[k] > 0:
                    ma_values[k] = np.mean(np.concatenate([
                        full_col[left_starts[k]:abs_indices[k]],
                        full_col[abs_indices[k] + 1:abs_indices[k] + 1 + deficits[k]]
                    ]))
                else:
                    ma_values[k] = np.mean(full_col[left_starts[k]:abs_indices[k]])
            if np.any(~np.isfinite(ma_values)) or np.any(ma_values <= 0):
                return None
            input_seq[:, col] = (raw_vals - ma_values) / ma_values

    # MA偏离度特征（已在CSV中预计算为 (close-MA)/MA 比值）
    input_seq[:, 6:9] = input_seq_raw[:, 6:9]

    # ========== 细处理阶段（可选）==========
    # 应用高级特征归一化，将粗处理结果转换为均值≈0、方差≈1的标准化数据
    if apply_fine_normalization and feature_normalizer is not None:
        input_seq = feature_normalizer.transform(input_seq)

    if np.any(~np.isfinite(input_seq)):
        return None

    return input_seq


def coarse_normalize_context_window(stock_data, start_idx, context_length,
                                     check_limit_up=True, required_length=None):
    """
    粗处理：CSV → OHLE 格式

    只执行粗处理阶段，不应用细处理（特征归一化器）。
    输出数据范围：
        - OHLC: -0.1 ~ 0.1（日环比变化率）
        - Amount: 相对N日均值变化率（无固定范围，由 QuantileTransformer 统一）
        - Exchange: 相对N日均值变化率（无固定范围，由 QuantileTransformer 统一）

    Args:
        stock_data: 股票原始数据 [N, 9]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口

    Returns:
        input_seq: [context_length, 9] 粗处理后的输入序列，或 None（如果验证失败）
    """
    return normalize_and_validate_context_window(
        stock_data, start_idx, context_length,
        check_limit_up=check_limit_up,
        required_length=required_length,
        feature_normalizer=None,
        apply_fine_normalization=False
    )


def fine_normalize_batch(input_seq, feature_normalizer):
    """
    细处理：OHLE → 标准化数据

    将粗处理后的数据送入细处理阶段，应用特征归一化器。
    输出数据特性：均值≈0，方差≈1

    Args:
        input_seq: 粗处理后的数据
            - 单个样本: [seq_len, 6]
            - 批量样本: [batch_size, seq_len, 9]
        feature_normalizer: 特征归一化器实例

    Returns:
        normalized_seq: 标准化后的数据，形状与输入相同
    """
    return feature_normalizer.transform(input_seq)


def fit_feature_normalizer(output_path=None, output_distribution='normal', n_quantiles=1000):
    """
    在训练集上拟合特征归一化器并保存到文件

    Args:
        output_path: 归一化器输出文件路径（默认使用 DataConfig.NORMALIZER_PATH）
        output_distribution: 输出分布类型 ('normal' 或 'uniform')
        n_quantiles: 分位数数量

    Returns:
        normalizer: 拟合后的 FeatureNormalizer 实例
    """
    if output_path is None:
        output_path = DataConfig.NORMALIZER_PATH
    if os.path.exists(output_path):
        print(f"归一化器文件已存在: {output_path}")
        response = input("是否重新训练？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    print("\n[步骤1] 加载训练集数据...")

    train_stock_info, test_stock_info = load_and_preprocess_data()

    print(f"训练集股票数: {len(train_stock_info)}")
    print(f"测试集股票数: {len(test_stock_info)}")

    print("\n[步骤2] 创建特征归一化器...")
    print(f"  输出分布: {output_distribution}")
    print(f"  分位数数量: {n_quantiles}")

    normalizer = FeatureNormalizer(output_distribution=output_distribution,n_quantiles=n_quantiles)

    print("\n[步骤3] 在训练集上拟合归一化器...")
    normalizer.fit(train_stock_info)

    print("\n[步骤4] 保存归一化器...")
    normalizer.save(output_path)

    return normalizer


def precompute_training_pool(train_stock_info, feature_normalizer=None):
    """
    预计算所有合法训练样本，一次性完成验证+粗归一化+标签+收益率+细归一化

    训练数据在整个训练过程中不变，此函数将 sample_with_pools 中每个 epoch
    重复执行的 Python 循环（验证、归一化、标签计算）合并为
    一次预计算。后续 epoch 只需从预计算结果中索引采样。

    Returns:
        all_inputs: [N, context_length, 8] float32 归一化后的输入
        all_targets: [N] float32 标签 (0/1)
        all_returns: [N] float32 累计收益率
        pos_indices: [M] int 正样本在 all_inputs 中的索引
        neg_indices: [K] int 负样本在 all_inputs 中的索引
        sample_key_to_pool_idx: dict  (stock_idx, start_idx) -> pool_index 映射
    """
    import time
    t0 = time.time()

    all_inputs = []
    all_targets = []
    all_returns = []
    sample_key_to_pool_idx = {}

    for stock_idx, stock_info in enumerate(train_stock_info):
        train_start = max(1, stock_info.get('train_start_idx', 0) + 1)
        train_end = stock_info.get('train_end_idx', len(stock_info['data']))
        excluded = stock_info.get('excluded_positions', set())

        for start_idx in range(train_start, train_end + 1):
            if start_idx in excluded:
                continue
            sample = generate_sample_from_index(train_stock_info, stock_idx, start_idx, None)
            if sample is None:
                continue
            input_seq, target, cumulative_return, _ = sample
            pool_idx = len(all_inputs)
            sample_key_to_pool_idx[(stock_idx, start_idx)] = pool_idx
            all_inputs.append(input_seq)
            all_targets.append(target)
            all_returns.append(cumulative_return)

    if len(all_inputs) == 0:
        raise ValueError("预计算结果为空：没有有效的训练样本")

    all_inputs = np.asarray(all_inputs)
    if feature_normalizer is not None:
        chunk_size = 100_000
        for start in range(0, len(all_inputs), chunk_size):
            end = min(start + chunk_size, len(all_inputs))
            all_inputs[start:end] = feature_normalizer.transform_batch(all_inputs[start:end])
    all_targets = np.asarray(all_targets)
    all_returns = np.asarray(all_returns)

    pos_indices = np.where(all_targets >= 0.5)[0]
    neg_indices = np.where(all_targets < 0.5)[0]

    elapsed = time.time() - t0
    mem_mb = all_inputs.nbytes / 1024 / 1024
    print(f"  预计算完成: {len(all_inputs)} 个有效样本 "
          f"(正样本 {len(pos_indices)}, 负样本 {len(neg_indices)})，"
          f"耗时 {elapsed:.1f}s，占用 {mem_mb:.0f}MB")

    return all_inputs, all_targets, all_returns, pos_indices, neg_indices, sample_key_to_pool_idx


def sample_temporal_from_pool(sampler, train_stock_info,
                              all_inputs, all_targets, all_returns,
                              sample_key_to_pool_idx,
                              batch_size, batches_per_epoch):
    """
    时间顺序采样（预计算池化版）：用 TemporalSampler 的有状态指针推进生成索引，
    通过 sample_key_to_pool_idx 映射从预计算池中取数据，避免重复调用
    generate_sample_from_index。

    正负池动态攒取逻辑与旧 sample_with_pools 完全一致：
    - 按时间顺序逐个获取样本，分入正/负池
    - 正样本达到 quota 且负样本足够时，立即出 batch
    - 正样本按顺序取用，负样本从池中随机抽取后清空负池
    - 采样头到达末尾后循环回起点（由 TemporalSampler 内部处理）

    Args:
        sampler: TemporalSampler 实例（跨 epoch 保持指针状态）
        train_stock_info: 股票信息列表（用于 get_loop_stats）
        all_inputs/all_targets/all_returns: precompute_training_pool 返回的完整样本
        sample_key_to_pool_idx: dict  (stock_idx, start_idx) -> pool_index 映射
        batch_size: 批次大小
        batches_per_epoch: 每 epoch 批次数

    Returns:
        epoch_inputs: [batches_per_epoch * batch_size, context_length, 8]
        epoch_targets: [batches_per_epoch * batch_size]
        epoch_returns: [batches_per_epoch * batch_size]
    """
    positive_ratio = 0.25
    pos_quota = max(1, int(batch_size * positive_ratio))
    neg_quota = batch_size - pos_quota

    pos_pool_indices = []
    pos_pool_targets = []
    neg_pool_indices = []
    neg_pool_targets = []

    all_batch_indices = []
    all_batch_targets = []
    batches_generated = 0

    initial_rounds = 50
    total_rounds_generated = 0

    while batches_generated < batches_per_epoch:
        sample_keys = sampler.sample_batch_rounds(initial_rounds)

        if len(sample_keys) == 0:
            print(f"\n    ⚠ 警告：采样头已到达所有股票终点且无法循环，停止采样")
            break

        total_rounds_generated += initial_rounds

        for stock_idx, start_idx in sample_keys:
            if batches_generated >= batches_per_epoch:
                break

            pool_idx = sample_key_to_pool_idx.get((stock_idx, start_idx))
            if pool_idx is None:
                continue

            target = all_targets[pool_idx]

            if target >= 0.5:
                pos_pool_indices.append(pool_idx)
                pos_pool_targets.append(target)
            else:
                neg_pool_indices.append(pool_idx)
                neg_pool_targets.append(target)

            if len(pos_pool_indices) >= pos_quota and len(neg_pool_indices) >= neg_quota:
                batch_pos = pos_pool_indices[:pos_quota]
                neg_sel = random.sample(range(len(neg_pool_indices)), neg_quota)
                batch_neg = [neg_pool_indices[i] for i in neg_sel]

                batch_indices = batch_pos + batch_neg
                batch_targets = pos_pool_targets[:pos_quota] + [neg_pool_targets[i] for i in neg_sel]

                combined = list(zip(batch_indices, batch_targets))
                random.shuffle(combined)
                bi, bt = zip(*combined)

                all_batch_indices.extend(bi)
                all_batch_targets.extend(bt)
                batches_generated += 1

                pos_pool_indices = pos_pool_indices[pos_quota:]
                pos_pool_targets = pos_pool_targets[pos_quota:]
                neg_pool_indices = []
                neg_pool_targets = []

        if batches_generated < batches_per_epoch:
            remaining = batches_per_epoch - batches_generated
            if batches_generated > 0:
                estimated = max(20, int(remaining / batches_generated * total_rounds_generated * 1.2))
                initial_rounds = min(estimated, 100)
            else:
                initial_rounds = 100

    if batches_generated < batches_per_epoch:
        print(f"    ⚠ 警告：样本不足，仅生成 {batches_generated} 个Batch (目标: {batches_per_epoch})")
        if batches_generated == 0:
            raise ValueError("样本严重不足：无法生成任何Batch")

    looped_count, total_loops = sampler.get_loop_stats()
    print(f"  [循环统计] 已循环股票: {looped_count}/{len(train_stock_info)}, 总循环次数: {total_loops}")

    batch_idx = np.array(all_batch_indices)
    return all_inputs[batch_idx], np.asarray(all_batch_targets), all_returns[batch_idx]


def sample_from_pool(all_inputs, all_targets, all_returns,
                     pos_indices, neg_indices,
                     batch_size, batches_per_epoch, rng):
    """
    从预计算样本池中按 25% 正样本比例采样一个 epoch 的数据

    等价于旧 sample_with_pools 的无放回随机采样：先打乱正/负样本索引，
    再按 quota 顺序取用，保证每个 epoch 内同一样本最多出现一次。

    Args:
        all_inputs/all_targets/all_returns: precompute_training_pool 返回的完整样本
        pos_indices/neg_indices: 正/负样本索引
        batch_size: 批次大小
        batches_per_epoch: 每 epoch 批次数
        rng: random.Random 实例

    Returns:
        epoch_inputs: [batches_per_epoch * batch_size, context_length, 8]
        epoch_targets: [batches_per_epoch * batch_size]
        epoch_returns: [batches_per_epoch * batch_size]
    """
    pos_quota = max(1, int(batch_size * 0.25))
    neg_quota = batch_size - pos_quota
    total_pos_needed = pos_quota * batches_per_epoch
    total_neg_needed = neg_quota * batches_per_epoch

    pos_list = list(pos_indices)
    neg_list = list(neg_indices)

    # 不足时循环补充（等价于旧流程中采样器循环回起点）
    if len(pos_list) < total_pos_needed:
        reps = (total_pos_needed + len(pos_list) - 1) // len(pos_list)
        pos_list = pos_list * reps
    if len(neg_list) < total_neg_needed:
        reps = (total_neg_needed + len(neg_list) - 1) // len(neg_list)
        neg_list = neg_list * reps

    rng.shuffle(pos_list)
    rng.shuffle(neg_list)

    total_samples = batches_per_epoch * batch_size
    epoch_inputs = np.empty((total_samples, *all_inputs.shape[1:]), dtype=all_inputs.dtype)
    epoch_targets = np.empty(total_samples, dtype=all_targets.dtype)
    epoch_returns = np.empty(total_samples, dtype=all_returns.dtype)

    for i in range(batches_per_epoch):
        offset = i * batch_size
        p_start = i * pos_quota
        n_start = i * neg_quota
        p_idx = np.array(pos_list[p_start:p_start + pos_quota])
        n_idx = np.array(neg_list[n_start:n_start + neg_quota])
        batch_idx = np.concatenate([p_idx, n_idx])
        rng.shuffle(batch_idx)

        epoch_inputs[offset:offset + batch_size] = all_inputs[batch_idx]
        epoch_targets[offset:offset + batch_size] = all_targets[batch_idx]
        epoch_returns[offset:offset + batch_size] = all_returns[batch_idx]

    return epoch_inputs, epoch_targets, epoch_returns


def main():
    parser = argparse.ArgumentParser(
        description='数据处理模块 兼 拟合特征归一化器训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
用法示例：
  python data.py                                           # 使用默认参数拟合归一化器
  python data.py --output-distribution uniform             # 使用均匀分布拟合
  python data.py --n-quantiles 500                         # 使用500个分位数拟合
        '''
    )
    parser.add_argument('--output-distribution', type=str, default=DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION, choices=['normal', 'uniform'],
                        help=f'输出分布类型: normal (标准正态) 或 uniform (均匀分布)，默认 {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION}')
    parser.add_argument('--n-quantiles', type=int, default=DataConfig.NORMALIZER_N_QUANTILES, help=f'分位数数量（默认{DataConfig.NORMALIZER_N_QUANTILES}，越大越精确但越慢）')

    args = parser.parse_args()
    fit_feature_normalizer(
        output_distribution=args.output_distribution,
        n_quantiles=args.n_quantiles
    )
    print(f"✓ 特征归一化器训练完成！已保存到: {DataConfig.NORMALIZER_PATH}")

if __name__ == "__main__":
    main()
