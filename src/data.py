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
import pickle
import json
import numpy as np
import pandas as pd
from config import DataConfig, generate_label, calculate_returns
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from typing import Dict, List, Set, Optional

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

        # 特征组定义：(名称, 列切片)
        # 每个维度各自独立拟合分位数映射，互不污染：
        # open/close 相对昨收（对称、小幅）、high（单边≥0）、low（单边≤0）
        # m5/m10/m20 偏离度、dif/dea/macd_hist/macd_hist_diff、bb_upper/bb_lower
        # 分布形态各不相同，分别建模最安全。
        self._feature_groups = [
            ('open',          slice(0, 1)),
            ('high',          slice(1, 2)),
            ('low',           slice(2, 3)),
            ('close',         slice(3, 4)),
            ('vwap',          slice(4, 5)),
            ('amount',        slice(5, 6)),
            ('exchange',      slice(6, 7)),
            ('m5',            slice(7, 8)),
            ('m10',           slice(8, 9)),
            ('m20',           slice(9, 10)),
            ('dif',           slice(10, 11)),
            ('dea',           slice(11, 12)),
            ('macd_hist',     slice(12, 13)),
            ('macd_hist_diff',slice(13, 14)),
            ('bb_upper',      slice(14, 15)),
            ('bb_lower',      slice(15, 16)),
        ]

        # 为每个特征组创建独立的 pipeline
        self.pipelines = {name: self._create_pipeline() for name, _ in self._feature_groups}

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

    def _collect_training_features(self, train_stock_info: List[Dict]) -> Dict[str, np.ndarray]:
        """
        从训练集收集所有特征值（避免数据泄漏）
        关键：只使用每只股票的训练集部分（train_end_idx 之前）
        使用向量化批处理进行粗处理，确保与训练时的数据处理逻辑完全一致

        Returns:
            {name: flat_values} 每个特征组的展平值
        """
        buckets = {name: [] for name, _ in self._feature_groups}

        context_length = DataConfig.CONTEXT_LENGTH
        max_windows = 100000
        total_windows = 0

        for stock_idx, stock in enumerate(train_stock_info):
            if total_windows >= max_windows:
                break

            data = stock['data']
            train_start_idx = stock.get('train_start_idx', 1)
            train_end_idx = stock.get('train_end_idx', len(data))

            range_min = max(1, train_start_idx + 1)
            range_max = train_end_idx - context_length

            if range_max < range_min:
                continue

            # 计算采样步长，控制总量在 max_windows 以内
            available = range_max - range_min + 1
            remaining = max_windows - total_windows
            stride = max(1, available // max(1, remaining))

            # 向量化处理：只做粗归一化，不需要标签、收益和未来数据
            inputs, _, _, _, _, _, _ = _vectorized_process_stock(
                stock, stock_idx,
                context_length,
                DataConfig.FUTURE_DAYS,
                context_length,        # required_length=context_length（不检查未来窗口）
                DataConfig.MA_WINDOW,
                DataConfig.LIMIT_THRESHOLD,
                DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP,
                DataConfig.LABEL_DAY1_USE_OPEN,
                generate_label, calculate_returns,
                check_limit_up=False,
                require_full_future=False,
                compute_labels=False,
                compute_returns=False,
                start_min_override=range_min,
                start_max_override=range_max,
            )

            if inputs is None:
                continue

            # 按 stride 采样，控制总量
            if stride > 1 and len(inputs) > remaining:
                indices = np.arange(0, len(inputs), stride)[:remaining]
                inputs = inputs[indices]

            total_windows += len(inputs)

            for name, col_slice in self._feature_groups:
                buckets[name].append(inputs[:, :, col_slice].flatten())

        result = {}
        for name in buckets:
            result[name] = np.concatenate(buckets[name]) if buckets[name] else np.array([])

        print(f"[FeatureNormalizer] 收集到的训练数据 ({total_windows} 个窗口):")
        for name, _ in self._feature_groups:
            print(f"  {name.upper()}: {len(result[name])} 个值")

        return result

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
        feature_data = self._collect_training_features(train_stock_info)

        # 拟合每个特征组的 pipeline
        print("\n[FeatureNormalizer] 拟合每个特征的 pipeline...")
        for name, _ in self._feature_groups:
            self.pipelines[name].fit(feature_data[name].reshape(-1, 1))
        self.is_fitted = True

        self._print_transform_stats(feature_data)
        print("\n[FeatureNormalizer] ✓ 拟合完成！")

    def _print_transform_stats(self, feature_data: Dict[str, np.ndarray]):
        """打印变换后的统计信息，验证归一化效果"""
        print("\n[FeatureNormalizer] 变换后的统计信息:")

        for name, _ in self._feature_groups:
            transformed = self.pipelines[name].transform(
                feature_data[name].reshape(-1, 1)
            ).flatten()
            print(f"  {name.upper()}:")
            print(f"    均值: {transformed.mean():.6f}")
            print(f"    标准差: {transformed.std():.6f}")
            print(f"    范围: [{transformed.min():.6f}, {transformed.max():.6f}]")

    def transform(self, input_seq: np.ndarray) -> np.ndarray:
        """
        对单个样本应用归一化

        Args:
            input_seq: [context_length, 16] 原始输入序列

        Returns:
            normalized_seq: [context_length, 16] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        normalized = np.empty_like(input_seq, dtype=np.float32)

        for name, col_slice in self._feature_groups:
            flat = input_seq[:, col_slice].flatten()
            transformed = self.pipelines[name].transform(flat.reshape(-1, 1)).flatten()
            normalized[:, col_slice] = transformed.reshape(input_seq[:, col_slice].shape)

        return normalized

    def transform_batch(self, input_seqs: np.ndarray, chunk_size: int = 100000) -> np.ndarray:
        """
        批量归一化多个样本（比逐个调用transform高效10-100倍）

        内部将所有样本的特征展平后送入sklearn pipeline。为避免大评估集（如 run.py
        的 --begin 多年回测，样本数可达数百万）一次性展平导致内存爆炸（QuantileTransformer
        内部 np.interp 会再分配等大数组），按 chunk_size 分块处理，峰值内存仅与单块大小相关。

        Args:
            input_seqs: [batch_size, context_length, 16] 原始输入序列
            chunk_size: 每块处理的样本数（默认 10万，单块峰值内存约数百MB）

        Returns:
            [batch_size, context_length, 16] 归一化后的序列
        """
        if not self.is_fitted:
            raise RuntimeError("归一化器未拟合！请先调用 fit() 方法")

        total, context_length = input_seqs.shape[0], input_seqs.shape[1]
        normalized = np.empty_like(input_seqs, dtype=np.float32)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = input_seqs[start:end]
            for name, col_slice in self._feature_groups:
                orig_shape = chunk[:, :, col_slice].shape
                normalized[start:end, :, col_slice] = self.pipelines[name].transform(
                    chunk[:, :, col_slice].reshape(-1, 1)
                ).reshape(orig_shape)

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
                'pipelines': self.pipelines,
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

        # 创建新实例（__init__ 会创建空的 pipelines，直接覆盖）
        normalizer = cls(
            output_distribution=data['output_distribution'],
            n_quantiles=data['n_quantiles'],
            random_state=data['random_state']
        )

        normalizer.pipelines = data['pipelines']
        normalizer.is_fitted = data['is_fitted']

        print(f" ✓ 归一化器已从 {path} 加载")

        return normalizer

def process_single_file(args):
    stock_code, data, times, train_start_date, train_end_date, val_start_date, val_end_date, test_start_date = args
    """
    处理预加载的股票数据，返回包含训练、验证和测试数据的字典

    由 load_and_preprocess_data 预加载全部数据后调用，
    不再自行连接数据库，避免多进程重复连接开销。

    Args:
        stock_code: 股票代码
        data: 预加载的行情数据 numpy 数组 (N, 16)
        times: 预加载的日期数组 (N,)
        train_start_date: 训练集起始日期 (YYYYMMDD)
        train_end_date: 训练集截止日期 (YYYYMMDD)
        val_start_date: 验证集起始日期 (YYYYMMDD)，0表示无验证集
        val_end_date: 验证集截止日期 (YYYYMMDD)
        test_start_date: 测试集起始日期 (YYYYMMDD)

    Returns:
        dict or None: 包含股票信息的字典，数据不足时返回None
    """
    try:
        data_length = len(data)
        required_length = DataConfig.REQUIRED_LENGTH

        # 找到测试集起始位置
        test_indices = np.where(times >= test_start_date)[0]
        if len(test_indices) == 0:
            return None
        test_split_point = test_indices[0]

        # 找到验证集起始位置（如果配置了验证集）
        if val_start_date > 0:
            val_start_indices = np.where(times >= val_start_date)[0]
            val_split_point = val_start_indices[0] if len(val_start_indices) > 0 else test_split_point

            val_end_indices = np.where(times <= val_end_date)[0]
            val_end_point = val_end_indices[-1] + 1 if len(val_end_indices) > 0 else val_split_point
        else:
            val_split_point = test_split_point
            val_end_point = test_split_point

        # 找到训练集起始位置
        train_start_indices = np.where(times >= train_start_date)[0]
        if len(train_start_indices) == 0:
            train_start_idx = 0
        else:
            train_start_idx = train_start_indices[0]

        # 找到训练集截止位置（train_end_date当天最后一条数据的下一位）
        train_end_indices = np.where(times <= train_end_date)[0]
        if len(train_end_indices) == 0:
            return None
        # train_end_idx: 训练集可用数据的末尾索引（不含），需留出 REQUIRED_LENGTH 的余量
        train_end_boundary = train_end_indices[-1] + 1
        # 训练集截止不能超过验证集起始（确保训练/验证无重叠）
        train_end_idx = min(train_end_boundary, val_split_point) - required_length

        if train_start_idx >= train_end_idx:
            return None

        train_length = train_end_idx - train_start_idx
        if train_length < required_length:
            return None

        train_data = data.copy()
        test_data = data.copy()

        stock_info = {
            'file_name': stock_code,
            'data_length': data_length,
            'train_data': train_data,
            'test_data': test_data,
            'train_start_idx': train_start_idx,
            'train_end_idx': train_end_idx,
            'train_length': train_length,
            'val_split_point': val_split_point,
            'val_end_point': val_end_point,
            'test_split_point': test_split_point,
            'times': times
        }

        return stock_info
    except Exception as e:
        print(f"处理股票 {stock_code} 时出错: {e}")
        return None

def load_and_preprocess_data(db_path=DataConfig.DB_PATH,
                             train_start_date=DataConfig.TRAIN_START_DATE,
                             train_end_date=DataConfig.TRAIN_END_DATE,
                             val_start_date=DataConfig.VAL_START_DATE,
                             val_end_date=DataConfig.VAL_END_DATE,
                             test_start_date=DataConfig.TEST_START_DATE):
    """
    数据加载和预处理，使用多进程并行加载

    从 SQLite 数据库一次性加载训练池(selected)中的全部股票数据，
    再分发给 worker 做纯计算（索引切分等），避免多进程重复 DB 连接。

    采样边界设计：
    - 训练集：TRAIN_START_DATE ~ TRAIN_END_DATE
    - 验证集：VAL_START_DATE ~ VAL_END_DATE（训练时用于模型选择）
    - 测试集：TEST_START_DATE ~ 数据库最新日期（训练结束后仅评估一次）
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    query = """SELECT sd.stock_code, sd.date, sd.open, sd.high, sd.low, sd.close,
                      sd.vwap, sd.volume, sd.exchange, sd.m5, sd.m10, sd.m20,
                      sd.dif, sd.dea, sd.macd_hist, sd.macd_hist_diff, sd.bb_upper, sd.bb_lower
               FROM stock_daily sd
               JOIN stock_pool sp ON sd.stock_code = sp.stock_code
               WHERE sp.pool_type='selected' AND sp.is_active=1
               ORDER BY sd.stock_code, sd.date ASC"""
    df = pd.read_sql_query(query, conn)
    conn.close()

    cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'exchange', 'm5', 'm10', 'm20', 'dif', 'dea', 'macd_hist', 'macd_hist_diff', 'bb_upper', 'bb_lower']
    stock_codes = []
    stock_data_arrays = []
    stock_times_arrays = []
    for stock_code, group in df.groupby('stock_code', sort=False):
        stock_codes.append(stock_code)
        stock_data_arrays.append(group[cols].values)
        stock_times_arrays.append(group['date'].values)

    print(f"总共 {len(stock_codes)} 只股票 (训练池)")
    print(f"- 训练集: {train_start_date} ~ {train_end_date}")
    print(f"- 验证集: {val_start_date} ~ {val_end_date}")
    print(f"- 测试集: {test_start_date} ~ 最新")

    file_args = list(zip(stock_codes, stock_data_arrays, stock_times_arrays,
                         [train_start_date] * len(stock_codes),
                         [train_end_date] * len(stock_codes),
                         [val_start_date] * len(stock_codes),
                         [val_end_date] * len(stock_codes),
                         [test_start_date] * len(stock_codes)))
    num_workers = min(cpu_count(), 8)

    with Pool(num_workers) as pool:
        all_stock_info = [r for r in pool.map(process_single_file, file_args) if r is not None]

    discarded_count = len(stock_codes) - len(all_stock_info)
    print(f"有效股票: {len(all_stock_info)} 只，丢弃: {discarded_count} 只")

    train_stock_info = []
    val_stock_info = []
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

        # 只有当验证集有数据时才添加（val_split_point < test_split_point）
        val_sp = stock_info.get('val_split_point', stock_info['test_split_point'])
        test_sp = stock_info['test_split_point']
        if val_sp < test_sp:
            val_stock_info.append({
                'file_name': stock_info['file_name'],
                'data': stock_info['test_data'],
                'data_length': stock_info['data_length'],
                'val_split_point': val_sp,
                'test_split_point': test_sp,
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
    print(f"验证集: {len(val_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")

    return train_stock_info, val_stock_info, test_stock_info

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
        # 必须与 _vectorized_process_stock() 中的标签计算逻辑保持一致
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
    - 每只股票的指针初始位置 = train_start_idx（TRAIN_START_DATE起始位置+1，或上市第一天+1）
    - 每只股票的指针末位置 = train_end_idx（TRAIN_END_DATE边界内，留出REQUIRED_LENGTH余量）
    
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
            # 无效区间（start > max）的股票即使数据量足够也不应循环
            self.can_loop.append(start_pos <= max_pos and data_length > 600)

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
                       or self.can_loop[i]
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

def create_fixed_evaluation_dataset(test_stock_info, feature_normalizer=None,
                                     start_key='test_split_point', end_key=None,
                                     start_date=None):
    """
    创建固定评估数据集（向量化批处理）

    只包含完整样本（available_days == FUTURE_DAYS），用于模型评估

    Args:
        test_stock_info: 评估集股票信息列表（验证集或测试集）
        feature_normalizer: 可选的特征归一化器实例
        start_key: stock_info 中评估起始位置的键名（默认 'test_split_point'）
        end_key: stock_info 中评估结束位置的键名（默认 None，表示到数据末尾）
                 验证集传 'test_split_point' 表示验证样本截止到测试集之前
        start_date: 可选(int YYYYMMDD)。给定后忽略 start_key/end_key，评估范围变为
                    [start_date, 数据末尾]——即 run.py 的 --begin 全区间回测口径，
                    不再区分训练/验证/测试集。每只股票的 split_point 取其 times 中
                    首个 >= start_date 的索引。

    返回: (inputs, targets, cumulative_returns, day_indices, daily_returns)
    """
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    required_length = DataConfig.REQUIRED_LENGTH
    ma_window = DataConfig.MA_WINDOW
    limit_threshold = DataConfig.LIMIT_THRESHOLD
    filter_last_day = DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP
    label_day1_use_open = DataConfig.LABEL_DAY1_USE_OPEN

    eval_inputs = []
    eval_targets = []
    eval_cumulative_returns = []
    eval_day_indices = []
    eval_daily_returns = []
    eval_tradeable_masks = []

    for stock_idx, stock_info in enumerate(test_stock_info):
        data_length = len(stock_info['data'])

        if start_date is not None:
            # --begin 口径：评估区间 = [start_date, 数据末尾]，忽略 train/val/test 划分。
            # split_point 取 begin_idx - CONTEXT_LENGTH + 1（并夹到 >=1），
            # 使"首个预测日"恰好落在 begin 当天——模型用 begin 之前的历史做上下文，
            # 回测从 begin 开始（而非 begin+45 天后）。--begin 无训练/测试之分，无泄漏顾虑。
            times = stock_info.get('times')
            if times is None:
                continue
            mask = np.where(np.asarray(times) >= start_date)[0]
            if len(mask) == 0:
                continue
            begin_idx = int(mask[0])
            split_point = max(1, begin_idx - context_length + 1)
            start_min = max(1, split_point)
            start_max = data_length - required_length
        else:
            split_point = stock_info.get(start_key, 0)
            start_min = max(1, split_point)
            if end_key is not None:
                end_point = stock_info.get(end_key, data_length)
                start_max = min(end_point, data_length) - required_length
            else:
                start_max = data_length - required_length

        if start_max < start_min:
            continue

        inputs, targets, returns_arr, keys, _, daily_rets, tradeable_mask = _vectorized_process_stock(
            stock_info, stock_idx,
            context_length, future_days, required_length,
            ma_window, limit_threshold, filter_last_day,
            label_day1_use_open,
            generate_label, calculate_returns,
            start_min_override=start_min,
            start_max_override=start_max,
        )

        if inputs is None:
            continue

        eval_inputs.append(inputs)
        eval_targets.append(targets)
        eval_cumulative_returns.append(returns_arr)

        # 计算 day_index（predict_day - split_point）
        for _, start_idx in keys:
            eval_day_indices.append(start_idx + context_length - split_point)

        # daily_returns：从向量化批处理中收集逐样本的逐日收益
        eval_daily_returns.extend(daily_rets)

        # tradeable_mask：开盘涨停过滤（评估阶段排除不可交易样本）
        eval_tradeable_masks.append(tradeable_mask)

    if len(eval_inputs) == 0:
        raise ValueError("固定评估集为空：test_stock_info中没有可用样本")

    eval_inputs_array = np.concatenate(eval_inputs, axis=0)
    eval_targets_array = np.concatenate(eval_targets, axis=0)
    eval_returns_array = np.concatenate(eval_cumulative_returns, axis=0)
    eval_tradeable_mask_array = np.concatenate(eval_tradeable_masks, axis=0)
    # 释放分块列表（其总大小与拼接后的 array 相当），降低大评估集(--begin 多年)峰值内存
    del eval_inputs, eval_targets, eval_cumulative_returns, eval_tradeable_masks

    # 批量细处理：分块归一化（大评估集如 --begin 多年回测时避免内存爆炸）
    if feature_normalizer is not None:
        eval_inputs_array = feature_normalizer.transform_batch(eval_inputs_array)
        # 过滤归一化后产生的NaN/Inf样本（分块计算 finite_mask，避免 [N, ctx*feat] 整体展开占内存）
        n_eval = len(eval_inputs_array)
        finite_mask = np.ones(n_eval, dtype=bool)
        feat_chunk = 100000
        for s in range(0, n_eval, feat_chunk):
            e = min(s + feat_chunk, n_eval)
            finite_mask[s:e] = np.all(np.isfinite(eval_inputs_array[s:e].reshape(e - s, -1)), axis=1)
        if not np.all(finite_mask):
            removed = np.sum(~finite_mask)
            print(f"  ⚠ 归一化后{removed}个样本包含NaN/Inf，已过滤")
            eval_inputs_array = eval_inputs_array[finite_mask]
            eval_targets_array = eval_targets_array[finite_mask]
            eval_returns_array = eval_returns_array[finite_mask]
            eval_tradeable_mask_array = eval_tradeable_mask_array[finite_mask]
            eval_day_indices = [d for d, m in zip(eval_day_indices, finite_mask) if m]
            eval_daily_returns = [r for r, m in zip(eval_daily_returns, finite_mask) if m]

    return (eval_inputs_array, eval_targets_array,
            eval_returns_array, np.asarray(eval_day_indices),
            eval_daily_returns, eval_tradeable_mask_array)

def create_recent_days_dataset(test_stock_info, feature_normalizer=None, max_days=15):
    """
    创建最近几天的数据集（向量化批处理版，包含临时样本，用于展示）

    Args:
        test_stock_info: 测试集股票信息列表
        feature_normalizer: 可选的特征归一化器实例
        max_days: 只生成最近 max_days 天的样本，避免遍历整个测试期
    """
    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    required_length = DataConfig.REQUIRED_LENGTH
    ma_window = DataConfig.MA_WINDOW
    limit_threshold = DataConfig.LIMIT_THRESHOLD
    filter_last_day = DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP
    label_day1_use_open = DataConfig.LABEL_DAY1_USE_OPEN

    recent_inputs = []
    recent_cumulative_returns = []
    recent_day_indices = []
    recent_available_days = []

    for stock_idx, stock_info in enumerate(test_stock_info):
        data_length = len(stock_info['data'])
        test_split_point = stock_info.get('test_split_point', 0)

        start_max = data_length - context_length - 1
        start_min = max(1, test_split_point, start_max - max_days + 1)

        if start_max < start_min:
            continue

        # require_full_future=False: 允许不完整的未来数据（最近几天可能还没走完）
        # compute_labels=False: 不需要标签（仅用于推理展示）
        inputs, _, returns_arr, keys, avail, _, tradeable_mask = _vectorized_process_stock(
            stock_info, stock_idx,
            context_length, future_days,
            min(required_length, data_length - start_min),
            ma_window, limit_threshold, filter_last_day,
            label_day1_use_open,
            generate_label, calculate_returns,
            check_limit_up=True,
            require_full_future=False,
            compute_labels=False,
            compute_returns=True,
            start_min_override=start_min,
            start_max_override=start_max,
        )

        if inputs is None:
            continue

        # 过滤：排除T+1开盘价接近涨停的不可交易样本
        # 近期选股路径始终过滤（与评估阶段一致），不受 FILTER_LIMIT_UP_OPEN 控制。
        if not np.all(tradeable_mask):
            keep = tradeable_mask
            inputs = inputs[keep]
            returns_arr = returns_arr[keep]
            avail = avail[keep]
            keys = [k for k, m in zip(keys, keep) if m]

        # 批量细处理
        if feature_normalizer is not None:
            inputs = feature_normalizer.transform_batch(inputs)
            finite_mask = np.all(np.isfinite(inputs.reshape(len(inputs), -1)), axis=1)
            if not np.all(finite_mask):
                inputs = inputs[finite_mask]
                returns_arr = returns_arr[finite_mask]
                avail = avail[finite_mask]
                keys = [k for k, m in zip(keys, finite_mask) if m]

        recent_inputs.append(inputs)
        recent_cumulative_returns.append(returns_arr)
        recent_available_days.append(avail)

        for _, start_idx in keys:
            day_index = start_idx + context_length - test_split_point
            recent_day_indices.append(day_index)

    if len(recent_inputs) == 0:
        return None, None, None, None

    return (np.concatenate(recent_inputs),
            np.concatenate(recent_cumulative_returns),
            np.asarray(recent_day_indices),
            np.concatenate(recent_available_days))

def normalize_and_validate_context_window(stock_data, start_idx, context_length,
                                          check_limit_up=True, required_length=None,
                                          feature_normalizer=None,
                                          apply_fine_normalization=True):
    """
    统一的上下文窗口归一化和验证函数

    用于消除 run.py 和 data.py 中的代码重复。
    执行完整的数据验证和归一化流程，与 _vectorized_process_stock 保持一致。

    数据处理分两阶段：
        - 粗处理：CSV → 归一化格式
            - OHLC: open_rel/close_rel 相对前日收盘价（clip [-0.1,0.1]）；high_rel/low_rel 相对当日 open 的日内振幅（不 clip，恒 high≥0/low≤0）
            - VWAP: 相对当日收盘价偏离，clip [-0.1, 0.1]（价格类特征）
            - Amount: (amount_i - MA_N) / MA_N，MA_N 为过去 N 日均量，无 clip
            - Exchange: (exchange_i - MA_N) / MA_N，MA_N 为过去 N 日均换手率，无 clip
        - 细处理：归一化 → 标准化数据（均值≈0，方差≈1）

    Args:
        stock_data: 股票原始数据 [N, 16]
        start_idx: 上下文窗口起始索引（需要 >= 1，因为需要前一天作为基准）
        context_length: 上下文窗口长度
        check_limit_up: 是否检查涨停（默认 True）
        required_length: 完整采样窗口长度（用于涨停过滤），如果为 None 则只检查上下文窗口
        feature_normalizer: 可选的特征归一化器实例，用于细处理阶段
        apply_fine_normalization: 是否应用细处理（默认 True）。设为 False 时只执行粗处理。

    Returns:
            input_seq: [context_length, 16] 归一化后的输入序列，或 None（如果验证失败）
            - 粗处理后：open_rel/close_rel/vwap: [-0.1, 0.1], high_rel/low_rel: 日内振幅无 clip, Volume: 相对N日均值变化率, Exchange: 相对N日均值变化率
            - 细处理后：均值≈0，方差≈1

    验证项：
        1. 基准日（start_idx-1）的 OHLC、VWAP 和 volume 非零
        2. 上下文窗口的 close 和 volume 非零
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
    prev_amount = prev_day_data[5]
    if prev_close == 0 or prev_amount == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    closes = input_seq_raw[:, 3]
    vwaps = input_seq_raw[:, 4]
    amounts = input_seq_raw[:, 5]
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

    input_seq = np.empty((context_length, 16), dtype=np.float32)

    # OHLC 编码：
    #   open_rel(列0) / close_rel(列3): 相对前一日收盘价的位置，反映隔夜跳空与当日涨跌
    #   high_rel(列1) / low_rel(列2): 相对当日 open 的日内振幅，反映上/下影线强度
    #     high ≥ open → high_rel 恒 ≥ 0；low ≤ open → low_rel 恒 ≤ 0
    # 这样 candlestick 形态（实体/影线）不再淹没在"相对昨收偏离"里，横盘日也不再四列同时≈0。
    opens = input_seq_raw[:, 0]

    # open_rel / close_rel：idx0 相对 prev_close，idx1+ 相对前日收盘价
    input_seq[0, 0] = (input_seq_raw[0, 0] - prev_close) / prev_close
    input_seq[0, 3] = (input_seq_raw[0, 3] - prev_close) / prev_close
    if context_length > 1:
        prev_closes = closes[:-1]
        safe_pc = np.where(prev_closes != 0, prev_closes, 1.0)
        input_seq[1:, 0] = (input_seq_raw[1:, 0] - prev_closes) / safe_pc
        input_seq[1:, 3] = (input_seq_raw[1:, 3] - prev_closes) / safe_pc

    # high_rel / low_rel：相对当日 open，所有行统一公式（不 clip，保留极端日影线信号）
    safe_opens = np.where(opens != 0, opens, 1.0)
    input_seq[:, 1] = (input_seq_raw[:, 1] - opens) / safe_opens
    input_seq[:, 2] = (input_seq_raw[:, 2] - opens) / safe_opens

    # VWAP: 相对当日收盘价的偏离（捕捉盘中均价与收盘价的关系）
    # > 0: 均价高于收盘 → 盘中强势但尾盘回落（抛压）
    # < 0: 均价低于收盘 → 盘中弱势但尾盘拉升（买盘强）
    input_seq[:, 4] = (vwaps - closes) / closes

    N = DataConfig.MA_WINDOW
    exchanges = input_seq_raw[:, 6]

    # 只 clip 价格相对偏离类：open_rel(列0)/close_rel(列3)/vwap(列4) 仍在 ±0.1（涨跌停尺度）
    # high_rel(列1)/low_rel(列2) 为日内振幅，极端日会远超 ±0.1，保留原值交由 normalizer 压缩
    np.clip(input_seq[:, 0], -0.1, 0.1, out=input_seq[:, 0])
    np.clip(input_seq[:, 3], -0.1, 0.1, out=input_seq[:, 3])
    np.clip(input_seq[:, 4], -0.1, 0.1, out=input_seq[:, 4])

    abs_indices = start_idx + np.arange(context_length)

    # Amount(col 5) 和 Exchange(col 6): 相对N日均线变化率
    if start_idx >= N:
        for col, raw_vals in [(5, amounts), (6, exchanges)]:
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
        for col, raw_vals in [(5, amounts), (6, exchanges)]:
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

    # MA偏离度特征（已在数据库中预计算为 (close-MA)/MA 比值）
    input_seq[:, 7:10] = input_seq_raw[:, 7:10]

    # MACD特征（已在数据库中预计算为 (value)/close 比值）
    input_seq[:, 10:13] = input_seq_raw[:, 10:13]

    # MACD柱状图变化量（已在数据库中预计算为 macd_hist[t]-macd_hist[t-1]）
    input_seq[:, 13] = input_seq_raw[:, 13]

    # BB特征（已在数据库中预计算）
    input_seq[:, 14:16] = input_seq_raw[:, 14:16]

    # ========== 细处理阶段（可选）==========
    # 应用高级特征归一化，将粗处理结果转换为均值≈0、方差≈1的标准化数据
    if apply_fine_normalization and feature_normalizer is not None:
        input_seq = feature_normalizer.transform(input_seq)

    if np.any(~np.isfinite(input_seq)):
        return None

    return input_seq

def _vectorized_process_stock(stock_info, stock_idx, context_length, future_days,
                              required_length, ma_window, limit_threshold,
                              filter_last_day_limit_up, label_day1_use_open,
                              label_fn, returns_fn,
                              check_limit_up=True,
                              require_full_future=True,
                              compute_labels=True,
                              compute_returns=True,
                              start_min_override=None,
                              start_max_override=None,
                              excluded_market_dates=None):
    """
    向量化处理单只股票的所有合法窗口

    将原本 N 次 Python 函数调用合并为一次 numpy 批处理，
    消除 Python 循环调度开销，预期加速 10-20 倍。

    Args:
        stock_info: 单只股票信息字典
        stock_idx: 股票索引（用于 sample_key）
        context_length: 上下文窗口长度
        future_days: 未来天数
        required_length: 完整采样窗口长度
        ma_window: MA均线窗口
        limit_threshold: 涨停阈值
        filter_last_day_limit_up: 是否过滤上下文最后一天涨停
        label_day1_use_open: Day1是否使用开盘价
        label_fn: generate_label 函数引用
        returns_fn: calculate_returns 函数引用
        check_limit_up: 是否检查涨停过滤（默认 True）
        require_full_future: 是否要求完整的 future_days 数据（默认 True）
            False 时允许部分未来数据，仅要求至少 1 天
        compute_labels: 是否计算标签（默认 True）
        compute_returns: 是否计算累计收益（默认 True）
        start_min_override: 覆盖默认的采样起始范围（用于评估集等非训练场景）
        start_max_override: 覆盖默认的采样截止范围

    Returns:
        (inputs, targets, returns_arr, keys, available_days, daily_returns_list) 或
        (None, None, None, None, None, None)
        available_days: [Nv] int 数组，每行有多少天有效的未来数据
        daily_returns_list: list[list[float]]，每样本的逐日收益（变长 1-3）
    """
    stock_data = stock_info['data']
    T = len(stock_data)

    # 范围计算：支持 override（评估集等非训练场景不使用 train_start/train_end）
    if start_min_override is not None:
        range_min = max(1, start_min_override)
    else:
        range_min = max(1, stock_info.get('train_start_idx', 0) + 1)

    if start_max_override is not None:
        range_max = start_max_override
    else:
        range_max = stock_info.get('train_end_idx', T)

    excluded = stock_info.get('excluded_positions', set())

    max_valid_start = T - required_length
    if max_valid_start < range_min:
        return None, None, None, None, None, None, None

    all_starts = np.arange(range_min, min(range_max, max_valid_start) + 1)

    if excluded and not start_min_override:
        excl_arr = np.array(sorted(excluded), dtype=np.intp)
        mask = ~np.isin(all_starts, excl_arr)
        all_starts = all_starts[mask]

    N = len(all_starts)
    if N == 0:
        return None, None, None, None, None, None, None

    C = context_length
    offsets = np.arange(C, dtype=np.intp)
    window_idx = all_starts[:, None] + offsets[None, :]
    raw_windows = stock_data[window_idx]
    prev_days = stock_data[all_starts - 1]

    valid = np.ones(N, dtype=bool)

    valid &= prev_days[:, 3] != 0
    valid &= prev_days[:, 5] != 0
    valid &= np.all(prev_days[:, :4] != 0, axis=1)
    valid &= np.all(raw_windows[:, :, 3] != 0, axis=1)
    valid &= np.all(raw_windows[:, :, 5] != 0, axis=1)
    valid &= raw_windows[:, -1, 3] <= 40

    # 涨停过滤（可选）
    if check_limit_up:
        sample_offsets = np.arange(required_length + 1, dtype=np.intp)
        sample_idx = (all_starts[:, None] - 1) + sample_offsets[None, :]
        sample_idx = np.clip(sample_idx, 0, T - 1)

        sample_closes = stock_data[sample_idx][:, :, 3]
        prev_c = sample_closes[:, :-1]
        curr_c = sample_closes[:, 1:]
        valid &= np.all(prev_c != 0, axis=1)
        safe_prev_c = np.where(prev_c != 0, prev_c, 1.0)
        daily_rets_sample = (curr_c - prev_c) / safe_prev_c

        in_bounds = sample_idx < T
        in_bounds &= sample_idx >= 0
        in_bounds_inner = in_bounds[:, 1:]
        daily_rets_sample = np.where(in_bounds_inner, daily_rets_sample, 0.0)
        valid &= np.all(np.abs(daily_rets_sample) <= 0.11, axis=1)

    if filter_last_day_limit_up:
        last_close = raw_windows[:, -1, 3]
        prev_last_close = raw_windows[:, -2, 3]
        safe_plc = np.where(prev_last_close != 0, prev_last_close, 1.0)
        last_ret = (last_close - prev_last_close) / safe_plc
        valid &= last_ret < limit_threshold

    # 极端行情过滤：未来窗口（T+1 ~ T+future_days）落在极端日期的样本直接剔除
    if excluded_market_dates:
        times_arr = stock_info.get('times')
        if times_arr is not None:
            times_arr = np.asarray(times_arr)
            future_offsets = np.arange(context_length, context_length + future_days)
            future_idx = all_starts[:, None] + future_offsets[None, :]   # [N, future_days]
            future_idx = np.clip(future_idx, 0, len(times_arr) - 1)
            future_dates = times_arr[future_idx]                          # [N, future_days]
            excluded_arr = np.fromiter(excluded_market_dates, dtype=times_arr.dtype)
            extreme_mask = np.any(np.isin(future_dates, excluded_arr), axis=1)
            valid &= ~extreme_mask

    if not np.any(valid):
        return None, None, None, None, None, None, None

    vs = all_starts[valid]
    raw_w = raw_windows[valid]
    prev_d = prev_days[valid]
    Nv = len(vs)

    input_seqs = np.empty((Nv, C, 16), dtype=np.float32)

    closes_w = raw_w[:, :, 3]
    opens_w = raw_w[:, :, 0]
    prev_close = prev_d[:, 3:4]
    safe_prev_close = np.where(prev_close != 0, prev_close, 1.0)

    # OHLC 编码：
    #   open_rel(列0)/close_rel(列3): 相对前一日收盘价
    #   high_rel(列1)/low_rel(列2): 相对当日 open（恒 ≥0 / 恒 ≤0）
    input_seqs[:, 0, 0] = (raw_w[:, 0, 0] - prev_close[:, 0]) / safe_prev_close[:, 0]
    input_seqs[:, 0, 3] = (raw_w[:, 0, 3] - prev_close[:, 0]) / safe_prev_close[:, 0]

    if C > 1:
        closes_prev = closes_w[:, :-1]
        safe_cp = np.where(closes_prev != 0, closes_prev, 1.0)
        input_seqs[:, 1:, 0] = (raw_w[:, 1:, 0] - closes_prev) / safe_cp
        input_seqs[:, 1:, 3] = (raw_w[:, 1:, 3] - closes_prev) / safe_cp

    safe_opens = np.where(opens_w != 0, opens_w, 1.0)
    input_seqs[:, :, 1] = (raw_w[:, :, 1] - opens_w) / safe_opens
    input_seqs[:, :, 2] = (raw_w[:, :, 2] - opens_w) / safe_opens

    vwaps_w = raw_w[:, :, 4]
    safe_closes_w = np.where(closes_w != 0, closes_w, 1.0)
    input_seqs[:, :, 4] = (vwaps_w - closes_w) / safe_closes_w

    # 只 clip 价格相对偏离类：open_rel(列0)/close_rel(列3)/vwap(列4) ±0.1；high/low 日内振幅不 clip
    np.clip(input_seqs[:, :, 0], -0.1, 0.1, out=input_seqs[:, :, 0])
    np.clip(input_seqs[:, :, 3], -0.1, 0.1, out=input_seqs[:, :, 3])
    np.clip(input_seqs[:, :, 4], -0.1, 0.1, out=input_seqs[:, :, 4])

    abs_idx = vs[:, None] + offsets[None, :]

    for col in [5, 6]:
        full_col = stock_data[:, col].astype(np.float64)
        cs = np.empty(T + 1, dtype=np.float64)
        cs[0] = 0
        np.cumsum(full_col, out=cs[1:])

        left = np.maximum(abs_idx - ma_window, 0)
        deficit = ma_window - (abs_idx - left)

        # 历史区间 [left, abs_idx) 的和
        hist_sum = cs[abs_idx] - cs[left]
        # 借未来区间 [abs_idx+1, abs_idx+1+deficit) 的和，跳过当前日
        # deficit==0 时区间为空，future_sum 自动为 0，退化为纯历史均值
        future_end = np.minimum(abs_idx + 1 + deficit, T)
        future_sum = cs[future_end] - cs[abs_idx + 1]

        # 始终除以 ma_window（凑满N天的语义，与标量版一致）
        ma_vals = (hist_sum + future_sum) / ma_window

        raw_vals = raw_w[:, :, col]
        safe_ma = np.where(ma_vals > 0, ma_vals, 1.0)
        input_seqs[:, :, col] = ((raw_vals - ma_vals) / safe_ma).astype(np.float32)

        invalid_ma = ~np.isfinite(ma_vals) | (ma_vals <= 0)
        row_bad = np.any(invalid_ma, axis=1)
        input_seqs[row_bad] = np.nan

    input_seqs[:, :, 7:10] = raw_w[:, :, 7:10]

    # MACD特征（已在数据库中预计算）
    input_seqs[:, :, 10:13] = raw_w[:, :, 10:13]

    # MACD柱状图变化量（已在数据库中预计算）
    input_seqs[:, :, 13] = raw_w[:, :, 13]

    # BB特征（已在数据库中预计算）
    input_seqs[:, :, 14:16] = raw_w[:, :, 14:16]

    nan_rows = np.any(~np.isfinite(input_seqs), axis=(1, 2))
    good = ~nan_rows
    if not np.any(good):
        return None, None, None, None, None, None, None

    input_seqs = input_seqs[good]
    vs = vs[good]
    Nv = len(vs)

    # ========== 未来数据处理 ==========
    # 不需要标签和收益时，跳过未来数据验证（如归一化器训练数据收集）
    if not compute_labels and not compute_returns:
        keys = [(stock_idx, int(s)) for s in vs]
        return input_seqs, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), keys, np.empty(0, dtype=np.int32), [], np.empty(0, dtype=bool)

    future_offsets = np.arange(future_days, dtype=np.intp)
    future_idx = (vs[:, None] + C) + future_offsets[None, :]
    future_idx_clipped = np.clip(future_idx, 0, T - 1)
    future_data = stock_data[future_idx_clipped]

    # 计算每行实际可用的未来天数
    available_days = np.clip(T - (vs + C), 0, future_days).astype(np.int32)

    if require_full_future:
        # 要求全部 future_days 天存在且非零
        future_valid = ((vs[:, None] + C) + future_offsets[None, :]) < T
        future_valid &= future_data[:, :, 0] != 0
        future_valid &= future_data[:, :, 3] != 0
        fv_rows = np.all(future_valid, axis=1)
    else:
        # 允许部分未来数据：要求至少1天存在且非零
        future_valid = np.ones((Nv, future_days), dtype=bool)
        for d in range(future_days):
            day_exists = (vs + C + d) < T
            day_ok = day_exists & (future_data[:, d, 0] != 0) & (future_data[:, d, 3] != 0)
            future_valid[:, d] = day_ok
        fv_rows = np.any(future_valid, axis=1) & (available_days >= 1)

    if not np.all(fv_rows):
        input_seqs = input_seqs[fv_rows]
        vs = vs[fv_rows]
        future_data = future_data[fv_rows]
        available_days = available_days[fv_rows]
        Nv = len(vs)

    if Nv == 0:
        return None, None, None, None, None, None, None

    f_closes = future_data[:, :, 3]
    context_last_close = stock_data[vs + C - 1, 3]
    safe_ctx_close = np.where(context_last_close != 0, context_last_close, 1.0)

    # 过滤：T+1开盘价接近涨停价时标记为不可交易（评估阶段排除）
    t1_open = future_data[:, 0, 0]
    open_gap = (t1_open - context_last_close) / safe_ctx_close
    tradeable_mask = open_gap < limit_threshold

    # 计算每日涨跌幅，不可用的天填 0
    day1_valid = available_days >= 1
    safe_f_open0 = np.where(future_data[:, 0, 0] != 0, future_data[:, 0, 0], 1.0)

    day1_change = np.where(
        day1_valid,
        np.where(label_day1_use_open,
                 (f_closes[:, 0] - future_data[:, 0, 0]) / safe_f_open0,
                 (f_closes[:, 0] - context_last_close) / safe_ctx_close),
        0.0)

    day2_valid = available_days >= 2
    safe_f_closes0 = np.where(f_closes[:, 0] != 0, f_closes[:, 0], 1.0)
    day2_change = np.where(
        day2_valid,
        (f_closes[:, 1] - f_closes[:, 0]) / safe_f_closes0,
        0.0)

    day3_valid = available_days >= 3
    safe_f_closes1 = np.where(f_closes[:, 1] != 0, f_closes[:, 1], 1.0)
    day3_change = np.where(
        day3_valid,
        (f_closes[:, 2] - f_closes[:, 1]) / safe_f_closes1,
        0.0)

    # 标签计算（可选）
    targets = np.empty(0, dtype=np.float32)
    if compute_labels:
        targets = np.zeros(Nv, dtype=np.float32)
        for i in range(Nv):
            targets[i] = float(label_fn(
                day1_change=day1_change[i],
                day2_change=day2_change[i],
                day3_change=day3_change[i]
            ))

    # 收益计算（可选）
    returns_arr = np.empty(0, dtype=np.float32)
    daily_returns_list = []
    if compute_returns:
        returns_arr = np.zeros(Nv, dtype=np.float32)
        daily_returns_list = [None] * Nv
        for i in range(Nv):
            t1_open = future_data[i, 0, 0]
            t1_close = f_closes[i, 0]
            t2_open = future_data[i, 1, 0] if available_days[i] >= 2 else None
            t2_close = f_closes[i, 1] if available_days[i] >= 2 else None
            t3_close = f_closes[i, 2] if available_days[i] >= 3 else None
            cum_ret, dr = returns_fn(
                t1_open=t1_open, t1_close=t1_close,
                t2_open=t2_open, t2_close=t2_close,
                t3_close=t3_close,
                day1_change=day1_change[i],
                day2_change=day2_change[i] if available_days[i] >= 2 else None,
                day3_change=day3_change[i] if available_days[i] >= 3 else None,
            )
            returns_arr[i] = cum_ret
            daily_returns_list[i] = dr

    keys = [(stock_idx, int(s)) for s in vs]

    return input_seqs, targets, returns_arr, keys, available_days, daily_returns_list, tradeable_mask

def load_excluded_market_dates():
    """
    从 market_index.json 加载极端行情日期集合

    涨跌比（上涨家数/下跌家数）超过 EXTREME_UP_DOWN_RATIO 的日期视为极端行情日。
    未来窗口落在这些日期的样本标签由市场 beta 驱动，不属于主力运作信号，应剔除。

    Returns:
        set[int] 或 None: 极端日期集合（yyyymmdd），未启用或文件不存在时返回 None
    """
    if not DataConfig.EXCLUDE_EXTREME_MARKET:
        return None

    path = DataConfig.MARKET_BREADTH_PATH
    if not os.path.exists(path):
        print(f"  ⚠ 市场宽度数据不存在({path})，跳过极端行情过滤")
        print(f"    请先运行: python src/market_index.py")
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    threshold = DataConfig.EXTREME_UP_DOWN_RATIO
    excluded = set()
    for d in data:
        up, down = d['up_count'], d['down_count']
        ratio = up / down if down > 0 else float('inf')
        if ratio >= threshold:
            excluded.add(d['yyyymmdd'])

    if excluded:
        sorted_dates = sorted(excluded)
        print(f"  极端行情过滤: {len(excluded)} 个交易日(涨跌比≥{threshold})")
        print(f"    首个: {sorted_dates[0]}, 末个: {sorted_dates[-1]}")

    return excluded

def fit_feature_normalizer(output_path=None):
    """
    在训练集上拟合特征归一化器并保存到文件

    Args:
        output_path: 归一化器输出文件路径（默认使用 DataConfig.NORMALIZER_PATH）

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

    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()
    print(f"训练集股票数: {len(train_stock_info)}")

    print("\n[步骤2] 创建特征归一化器...")
    print(f"  输出分布: {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION}")
    print(f"  分位数数量: {DataConfig.NORMALIZER_N_QUANTILES}")

    normalizer = FeatureNormalizer(
    output_distribution=DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION,n_quantiles=DataConfig.NORMALIZER_N_QUANTILES)

    print("\n[步骤3] 在训练集上拟合归一化器...")
    normalizer.fit(train_stock_info)

    print("\n[步骤4] 保存归一化器...")
    normalizer.save(output_path)

    return normalizer

def precompute_training_pool(train_stock_info, feature_normalizer=None, max_pool_size=None):
    """
    预计算所有合法训练样本，一次性完成验证+粗归一化+标签+收益率+细归一化

    使用向量化批处理替代逐样本 Python 循环，每只股票的所有窗口一次性处理。

    Returns:
        正常模式 (max_pool_size=None):
            all_inputs: [N, context_length, 16] float32 归一化后的输入
            all_targets: [N] float32 标签 (0/1)
            all_returns: [N] float32 累计收益率
            pos_indices: [M] int 正样本在 all_inputs 中的索引
            neg_indices: [K] int 负样本在 all_inputs 中的索引
            sample_key_to_pool_idx: dict  (stock_idx, start_idx) -> pool_index 映射
        采样模式 (max_pool_size is not None):
            all_inputs: [M, 16] float32 归一化后的 K 线（展平采样，无时间维度）
            all_targets / all_returns / pos_indices / neg_indices: None
            sample_key_to_pool_idx: {} (空 dict，展平后不再对应)

    过滤：受 DataConfig.FILTER_LIMIT_UP_OPEN 控制，开启时排除 T+1 开盘涨停的
    不可交易样本（评估阶段在 create_fixed_evaluation_dataset 中始终过滤，不受此开关控制）。
    """
    import time
    t0 = time.time()

    all_inputs = []
    all_targets = []
    all_returns = []
    sample_key_to_pool_idx = {}
    total_samples = 0

    context_length = DataConfig.CONTEXT_LENGTH
    future_days = DataConfig.FUTURE_DAYS
    required_length = DataConfig.REQUIRED_LENGTH
    ma_window = DataConfig.MA_WINDOW
    limit_threshold = DataConfig.LIMIT_THRESHOLD
    filter_last_day = DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP
    label_day1_use_open = DataConfig.LABEL_DAY1_USE_OPEN
    filter_limit_up_open = DataConfig.FILTER_LIMIT_UP_OPEN

    # 加载极端行情日期，过滤 beta 驱动的噪声标签
    excluded_market_dates = load_excluded_market_dates()

    for stock_idx, stock_info in enumerate(train_stock_info):
        result = _vectorized_process_stock(
            stock_info, stock_idx,
            context_length, future_days, required_length,
            ma_window, limit_threshold, filter_last_day,
            label_day1_use_open,
            generate_label, calculate_returns,
            excluded_market_dates=excluded_market_dates,
        )
        inputs, targets, returns_arr, keys, _, _, tradeable_mask = result

        if inputs is None:
            continue

        # 过滤（可选）：排除T+1开盘涨停的不可交易样本
        # 仅训练阶段受 FILTER_LIMIT_UP_OPEN 控制；评估阶段始终过滤（在 create_fixed_evaluation_dataset 中）
        if filter_limit_up_open and tradeable_mask is not None and not np.all(tradeable_mask):
            inputs = inputs[tradeable_mask]
            targets = targets[tradeable_mask]
            returns_arr = returns_arr[tradeable_mask]
            keys = [k for k, m in zip(keys, tradeable_mask) if m]

        base_idx = total_samples
        total_samples += len(inputs)

        # 采样模式：逐股票归一化（采样后展平为2D，无法用3D的transform_batch归一化）
        if max_pool_size is not None and feature_normalizer is not None:
            inputs = feature_normalizer.transform_batch(inputs)
        all_inputs.append(inputs)
        all_targets.append(targets)
        all_returns.append(returns_arr)
        for i, key in enumerate(keys):
            sample_key_to_pool_idx[key] = base_idx + i

    if len(all_inputs) == 0:
        raise ValueError("预计算结果为空：没有有效的训练样本")

    if max_pool_size is not None:
        # 采样模式：逐股票展平采样，避免拼接完整数组导致双倍内存峰值
        # list 中分散数组(~10GB) + 连续数组(~10GB) = 峰值~20GB 会 OOM；
        # 逐股票采样后释放原数组，只拼接采样部分(~6GB)，峰值~10GB
        total_klines = sum(arr.shape[0] * arr.shape[1] for arr in all_inputs)
        if total_klines > max_pool_size:
            sampled_chunks = []
            remaining = max_pool_size
            for i in range(len(all_inputs)):
                arr = all_inputs[i]  # [N_i, 45, 16] 已归一化
                n = arr.shape[0] * arr.shape[1]  # N_i * 45
                k = int(n * max_pool_size / total_klines)  # floor: 保证 sum(k_i) <= max_pool_size，避免 round 累加超限导致尾部股票 k=0 被跳过
                k = min(k, remaining, n)
                flat = arr.reshape(-1, arr.shape[-1])  # [N_i*45, 16]
                if k < n:
                    idx = np.random.choice(n, k, replace=False)
                    sampled_chunks.append(flat[idx])
                else:
                    sampled_chunks.append(flat.copy())
                all_inputs[i] = None  # 释放原数组
                remaining -= k
            all_inputs = np.concatenate(sampled_chunks, axis=0)  # [max_pool_size, 16]
            print(f"  K线级预采样: {total_klines:,} → {len(all_inputs):,} 条")
        else:
            all_inputs = np.concatenate(all_inputs, axis=0)
            all_inputs = all_inputs.reshape(-1, all_inputs.shape[-1])
        # 展平采样后 targets/returns 不再对应，置 None
        all_targets = None
        all_returns = None
        pos_indices = None
        neg_indices = None
        sample_key_to_pool_idx = {}
    else:
        # 正常模式：拼接3D → 归一化 → 返回
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_returns = np.concatenate(all_returns, axis=0)

        if feature_normalizer is not None:
            chunk_size = 100_000
            for start in range(0, len(all_inputs), chunk_size):
                end = min(start + chunk_size, len(all_inputs))
                all_inputs[start:end] = feature_normalizer.transform_batch(all_inputs[start:end])

        pos_indices = np.where(all_targets >= 0.5)[0]
        neg_indices = np.where(all_targets < 0.5)[0]

    elapsed = time.time() - t0
    mem_mb = all_inputs.nbytes / 1024 / 1024
    if max_pool_size is not None:
        print(f"  预计算完成: {len(all_inputs):,} 条K线 (采样模式)，"
              f"耗时 {elapsed:.1f}s，占用 {mem_mb:.0f}MB")
    else:
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
    通过 sample_key_to_pool_idx 映射从预计算池中取数据。

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
    positive_ratio = DataConfig.POSITIVE_RATIO
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
    pos_quota = max(1, int(batch_size * DataConfig.POSITIVE_RATIO))
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
    """数据处理模块 兼 拟合特征归一化器训练脚本"""
    fit_feature_normalizer()
    print(f"✓ 特征归一化器训练完成！已保存到: {DataConfig.NORMALIZER_PATH}")

if __name__ == "__main__":
    main()
