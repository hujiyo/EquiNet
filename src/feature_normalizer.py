"""
特征归一化模块

核心功能：
1. 在训练集上拟合归一化器（QuantileTransformer + StandardScaler）
2. 在测试集上复用拟合结果，防止数据泄漏
3. 支持保存和加载归一化器

原理：
- QuantileTransformer: 将任意分布映射到均匀分布或正态分布
- StandardScaler: 确保最终输出均值=0，标准差=1
"""

import numpy as np
import pickle
import os
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
        normalizer.save('feature_normalizer.pkl')

        # 推理阶段
        normalizer = FeatureNormalizer.load('feature_normalizer.pkl')
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

    def _collect_training_features(self, train_stock_info: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从训练集收集所有特征值（避免数据泄漏）

        关键：只使用每只股票的训练集部分（train_end_idx 之前）
        
        使用 data.py 中的 coarse_normalize_context_window() 进行粗处理，
        确保与训练时的数据处理逻辑完全一致。

        Returns:
            ohl_data: OHLC 特征 [N_samples * 30 * 4]
            volume_data: Volume 特征 [N_samples * 30]
            exchange_data: Exchange 特征 [N_samples * 30]
        """
        from data import coarse_normalize_context_window, DataConfig
        
        ohl_data = []
        volume_data = []
        exchange_data = []
        
        context_length = DataConfig.CONTEXT_LENGTH

        for stock in train_stock_info:
            data = stock['data']
            train_end_idx = stock.get('train_end_idx', len(data))

            for i in range(1, train_end_idx - context_length):
                # 使用统一的粗处理函数
                input_seq = coarse_normalize_context_window(
                    data, i, context_length,
                    check_limit_up=False,  # 拟合归一化器时不过滤涨停，使用更多数据
                    required_length=context_length
                )
                
                if input_seq is None:
                    continue
                
                ohl_data.append(input_seq[:, :4].flatten())
                volume_data.append(input_seq[:, 4].flatten())
                exchange_data.append(input_seq[:, 5].flatten())

        ohl_data = np.concatenate(ohl_data) if ohl_data else np.array([])
        volume_data = np.concatenate(volume_data) if volume_data else np.array([])
        exchange_data = np.concatenate(exchange_data) if exchange_data else np.array([])

        print(f"[FeatureNormalizer] 收集到的训练数据:")
        print(f"  OHLC: {len(ohl_data)} 个值")
        print(f"  Volume: {len(volume_data)} 个值")
        print(f"  Exchange: {len(exchange_data)} 个值")

        return ohl_data, volume_data, exchange_data

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
        ohl_data, volume_data, exchange_data = self._collect_training_features(train_stock_info)

        # 拟合每个特征组的 pipeline
        print("\n[FeatureNormalizer] 拟合 OHLC 特征...")
        self.ohl_pipeline.fit(ohl_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Volume 特征...")
        self.volume_pipeline.fit(volume_data.reshape(-1, 1))

        print("[FeatureNormalizer] 拟合 Exchange 特征...")
        self.exchange_pipeline.fit(exchange_data.reshape(-1, 1))

        self.is_fitted = True

        # 打印变换后的统计信息
        self._print_transform_stats(ohl_data, volume_data, exchange_data)

        print("\n[FeatureNormalizer] ✓ 拟合完成！")

    def _print_transform_stats(self, ohl_data, volume_data, exchange_data):
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

        # 重塑回原始形状
        normalized[:, :4] = normalized_ohl.reshape(input_seq[:, :4].shape)
        normalized[:, 4] = normalized_volume
        normalized[:, 5] = normalized_exchange

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
            path: 保存路径（例如: './feature_normalizer.pkl'）
        """
        if not self.is_fitted:
            raise RuntimeError("无法保存未拟合的归一化器")

        with open(path, 'wb') as f:
            pickle.dump({
                'ohl_pipeline': self.ohl_pipeline,
                'volume_pipeline': self.volume_pipeline,
                'exchange_pipeline': self.exchange_pipeline,
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
        normalizer.is_fitted = data['is_fitted']

        print(f"[FeatureNormalizer] ✓ 归一化器已从 {path} 加载")

        return normalizer


def demo_usage():
    """
    演示如何使用 FeatureNormalizer
    """
    print("="*70)
    print("FeatureNormalizer 使用示例")
    print("="*70)

    # ============ 训练阶段 ============
    print("\n【训练阶段】")
    print("1. 加载训练数据...")
    # train_stock_info = load_and_preprocess_data()
    # train_stock_info, _ = train_stock_info  # 只使用训练集

    print("2. 创建并拟合归一化器...")
    # normalizer = FeatureNormalizer(output_distribution='normal')
    # normalizer.fit(train_stock_info)

    print("3. 保存归一化器...")
    # normalizer.save('./feature_normalizer.pkl')

    # ============ 推理阶段 ============
    print("\n【推理阶段】")
    print("1. 加载归一化器...")
    # normalizer = FeatureNormalizer.load('./feature_normalizer.pkl')

    print("2. 在数据预处理中使用...")
    # 在 normalize_and_validate_context_window() 中:
    # input_seq = normalizer.transform(input_seq)

    print("\n" + "="*70)
    print("✓ 演示完成")
    print("="*70)


if __name__ == "__main__":
    demo_usage()
