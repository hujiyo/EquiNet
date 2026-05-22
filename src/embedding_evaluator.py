"""
Embedding模块评估脚本

核心思路：评估Embedding层是否成功编码了6个输入维度之间的交互关系

评估对象：粗处理后的数据 → 细处理(FeatureNormalizer) → Embedding层 → d_model维输出

评估维度：
1. 跨维度交互分析：检测维度对之间的非线性交互效应（最核心）
2. 方向对比敏感性：符号翻转 vs 同向平移的输出差异比
3. 语义模式分析：原型K线模式的embedding距离关系
4. 高维连续性：沿关键维度扫值，检测平滑性和边界曲率
5. 信息保留度：从embedding能否重建原始输入特征
6. 饱和度分析：输出分布健康检查
7. 特征消融：单维度遮盖的重要性排序
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import argparse

from model import create_model
from data import load_and_preprocess_data, FeatureNormalizer
from config import DataConfig, ModelConfig, PROJECT_ROOT


class FFNEmbeddingWrapper(nn.Module):
    """
    FFN-Embedding wrapper：将 embed_proj + embed_mlp 残差封装为单一模块

    对应 StockTransformer 中的:
        x = self.embed_proj(x)
        x = x + self.embed_mlp(x)
    """
    def __init__(self, embed_proj, embed_mlp):
        super().__init__()
        self.embed_proj = embed_proj
        self.embed_mlp = embed_mlp

    def forward(self, x):
        x = self.embed_proj(x)
        x = x + self.embed_mlp(x)
        return x


class EmbeddingModule(nn.Module):
    """
    统一的 Embedding 模块

    将细处理和 Embedding 层封装为一个整体：
    - 细处理：FeatureNormalizer（训练时学到的变换）
    - Embedding 层：nn.Linear / nn.Sequential / FFNEmbeddingWrapper

    注意：粗处理（OHLE变换）在数据加载阶段完成，
    本模块接收粗处理后的数据作为输入。
    """

    def __init__(self, embedding_layer, feature_normalizer):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.feature_normalizer = feature_normalizer

    def _get_device(self):
        """获取embedding层所在的设备，兼容多种embedding结构"""
        if isinstance(self.embedding_layer, FFNEmbeddingWrapper):
            return self.embedding_layer.embed_proj.weight.device
        elif isinstance(self.embedding_layer, nn.Sequential):
            for module in self.embedding_layer:
                if hasattr(module, 'weight'):
                    return module.weight.device
        elif hasattr(self.embedding_layer, 'weight'):
            return self.embedding_layer.weight.device
        return torch.device('cpu')

    def forward(self, x):
        """
        Args:
            x: 粗处理后的数据 [batch, seq_len, 15]
               范围：OHLC [-0.1, 0.1], VWAP [-0.1, 0.1], Amount 相对MA变化率, Exchange 相对MA变化率

        Returns:
            embedded: [batch, seq_len, d_model]
        """
        if self.feature_normalizer is not None:
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            device = self._get_device()
            if x_np.ndim == 3:
                batch_size, seq_len, n_features = x_np.shape
                x_normalized = np.empty_like(x_np, dtype=np.float32)
                for b in range(batch_size):
                    x_normalized[b] = self.feature_normalizer.transform(x_np[b])
                x = torch.tensor(x_normalized, dtype=torch.float32, device=device)
            else:
                x_normalized = self.feature_normalizer.transform(x_np)
                x = torch.tensor(x_normalized, dtype=torch.float32, device=device)

        return self.embedding_layer(x)

    def transform_numpy(self, x_np):
        """
        对 numpy 数组应用细处理（不包含 Embedding 层）
        """
        if self.feature_normalizer is None:
            return x_np

        if x_np.ndim == 3:
            batch_size, seq_len, n_features = x_np.shape
            x_normalized = np.empty_like(x_np, dtype=np.float32)
            for b in range(batch_size):
                x_normalized[b] = self.feature_normalizer.transform(x_np[b])
            return x_normalized
        else:
            return self.feature_normalizer.transform(x_np)


class EmbeddingModuleAnalyzer:
    """Embedding模块分析器 - 评估维度间交互编码质量"""

    FEATURE_NAMES = ['Open', 'High', 'Low', 'Close', 'VWAP', 'Amount', 'Exchange',
                     'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD_Hist', 'BB_Upper', 'BB_Lower']

    NEUTRAL_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    DELTA_MAP = {0: 0.03, 1: 0.03, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.05, 6: 0.02,
                 7: 0.015, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01}

    KEY_PAIRS = [(0, 3), (1, 2), (3, 4), (3, 5)]
    KEY_PAIR_NAMES = ['Open-Close', 'High-Low', 'Close-VWAP', 'Close-Amount']

    # 原型交易日模式（coarse-normalized空间, 15维: OHLC + VWAP + Amount + Exchange + MA5 + MA10 + MA20 + DIF + DEA + MACD_Hist + BB_Upper + BB_Lower）
    SEMANTIC_PATTERNS = {
        'bullish_large': {
            'values': [0.01, 0.05, -0.01, 0.04, -0.01, 0.70, 0.05, 0.02, 0.015, 0.01, 0.005, 0.003, 0.004, -0.01, -0.01],
            'desc': '大阳线'
        },
        'bearish_large': {
            'values': [-0.01, 0.01, -0.05, -0.04, 0.01, 0.70, 0.05, -0.02, -0.015, -0.01, -0.005, -0.003, -0.004, -0.01, -0.01],
            'desc': '大阴线'
        },
        'doji': {
            'values': [0.0, 0.01, -0.01, 0.001, 0.0, 0.50, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'desc': '十字星'
        },
        'high_vol_bull': {
            'values': [0.01, 0.04, -0.005, 0.03, -0.01, 0.85, 0.08, 0.02, 0.015, 0.01, 0.005, 0.003, 0.004, -0.005, -0.005],
            'desc': '放量上涨'
        },
        'low_vol_bull': {
            'values': [0.01, 0.04, -0.005, 0.03, -0.005, 0.35, 0.01, 0.015, 0.01, 0.005, 0.002, 0.001, 0.002, 0.0, 0.0],
            'desc': '缩量上涨'
        },
        'divergence_bull': {
            'values': [0.005, 0.03, -0.005, 0.02, -0.005, 0.60, 0.04, 0.01, 0.005, 0.0, 0.001, 0.0, 0.002, -0.005, -0.005],
            'desc': '逆势上涨'
        },
        'following_bull': {
            'values': [0.01, 0.02, -0.005, 0.015, 0.0, 0.50, 0.03, 0.01, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'desc': '跟风上涨'
        },
        'upper_shadow': {
            'values': [0.0, 0.06, -0.005, 0.005, 0.01, 0.60, 0.04, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.01],
            'desc': '长上影线'
        },
        'lower_shadow': {
            'values': [0.0, 0.005, -0.06, 0.005, -0.01, 0.60, 0.04, -0.005, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01, 0.01],
            'desc': '长下影线'
        },
        'high_ex_limit': {
            'values': [0.02, 0.10, 0.01, 0.10, -0.02, 0.90, 0.15, 0.04, 0.03, 0.02, 0.01, 0.008, 0.004, -0.01, -0.01],
            'desc': '高换手涨停'
        }
    }

    # 语义对立的模式对（embedding应该距离远）
    EXPECTED_OPPOSITE_PAIRS = [
        ('bullish_large', 'bearish_large'),
        ('high_vol_bull', 'low_vol_bull'),
        ('upper_shadow', 'lower_shadow'),
        ('divergence_bull', 'following_bull'),
    ]

    # 语义相似的模式对（embedding应该距离近）
    EXPECTED_SIMILAR_PAIRS = [
        ('bullish_large', 'high_vol_bull'),
        ('doji', 'upper_shadow'),
        ('low_vol_bull', 'following_bull'),
    ]

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embedding_module = None
        self.feature_normalizer = None

    def load_model(self, model_path=None, feature_normalizer=None):
        """加载模型并创建 EmbeddingModule"""
        self.feature_normalizer = feature_normalizer

        if model_path:
            self.model_path = model_path
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_arch = checkpoint.get('model_arch', {})
                self.model = create_model(model_arch=model_arch).to(self.device)

                current_state = self.model.state_dict()
                loaded_state = checkpoint['state_dict']

                matched_state = {}
                for key in current_state:
                    if key in loaded_state:
                        if current_state[key].shape == loaded_state[key].shape:
                            matched_state[key] = loaded_state[key]
                        else:
                            print(f"  跳过不匹配的权重: {key}")
                    else:
                        print(f"  跳过缺失的权重: {key}")

                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型: {self.model_path}")
            elif isinstance(checkpoint, dict) and 'embed_proj_weight' in checkpoint:
                # 预训练 Embedding 格式（来自 pretrain_embedding.py）
                self.model = create_model().to(self.device)

                key_map = {
                    'embed_proj_weight': 'embed_proj.weight',
                    'embed_mlp_1_weight': 'embed_mlp.1.weight',
                }

                current_state = self.model.state_dict()
                matched_state = {}
                for src_key, dst_key in key_map.items():
                    if src_key in checkpoint and dst_key in current_state:
                        if checkpoint[src_key].shape == current_state[dst_key].shape:
                            matched_state[dst_key] = checkpoint[src_key]

                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载预训练Embedding: {self.model_path}")
                print(f"  匹配权重: {len(matched_state)} 组")
            else:
                self.model = create_model().to(self.device)
                current_state = self.model.state_dict()

                matched_state = {}
                for key in current_state:
                    if key in checkpoint:
                        if current_state[key].shape == checkpoint[key].shape:
                            matched_state[key] = checkpoint[key]

                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型(旧格式): {self.model_path}")
        else:
            self.model = create_model().to(self.device)
            print("使用随机初始化模型")
        self.model.eval()

        if hasattr(self.model, 'embed_proj') and hasattr(self.model, 'embed_mlp'):
            embedding_layer = FFNEmbeddingWrapper(self.model.embed_proj, self.model.embed_mlp)
            print("检测到FFN-Embedding结构（embed_proj + embed_mlp 残差）")
        elif hasattr(self.model, 'embedding'):
            embedding_layer = self.model.embedding
            print("检测到传统Embedding结构（单一embedding层）")
        else:
            raise AttributeError("模型不包含可识别的embedding层")

        self.embedding_module = EmbeddingModule(embedding_layer, self.feature_normalizer)

        return self.model

    # ==================== 辅助方法 ====================

    def _embed_batch(self, x_np):
        """统一的batch embedding：numpy → tensor → forward → numpy"""
        with torch.no_grad():
            x_tensor = torch.tensor(np.ascontiguousarray(x_np, dtype=np.float32),
                                    device=self.device)
            output = self.embedding_module(x_tensor)
            return output.cpu().numpy()

    # ==================== 分析方法 ====================

    def analyze_cross_dimensional_interactions(self, sample_inputs, n_samples=100):
        """
        跨维度交互分析（最核心）

        检测维度对之间的非线性交互效应：
        线性映射：f(x+δi+δj) - f(x) = [f(x+δi)-f(x)] + [f(x+δj)-f(x)]
        非线性交互：residual = f(x+δi+δj) - f(x+δi) - f(x+δj) + f(x)
        交互强度 = ||residual||

        如果embedding捕获了维度间交互（如量价关系），residual应该显著非零。
        """
        print("\n[跨维度交互分析]")
        print("  评估对象: Embedding模块（细处理 + Embedding层）")
        print("  核心问题: 维度间是否存在非线性交互效应？")

        sample_inputs = np.array(sample_inputs[:n_samples])
        n_dims = sample_inputs.shape[-1]
        interaction_sum = np.zeros((n_dims, n_dims))

        for sample_idx in range(len(sample_inputs)):
            x = sample_inputs[sample_idx:sample_idx+1]  # [1, seq_len, n_dims]

            f_base = self._embed_batch(x).flatten()

            # 单维度扰动的embedding
            f_single = {}
            for i in range(n_dims):
                x_i = x.copy()
                x_i[:, :, i] += self.DELTA_MAP[i]
                f_single[i] = self._embed_batch(x_i).flatten()

            # 双维度扰动 → 交互残差
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    x_ij = x.copy()
                    x_ij[:, :, i] += self.DELTA_MAP[i]
                    x_ij[:, :, j] += self.DELTA_MAP[j]
                    f_ij = self._embed_batch(x_ij).flatten()

                    # 交互残差 = 实际联合效应 - 各自效应之和
                    residual = f_ij - f_single[i] - f_single[j] + f_base
                    strength = np.linalg.norm(residual)

                    interaction_sum[i, j] += strength
                    interaction_sum[j, i] += strength

        interaction_matrix = interaction_sum / len(sample_inputs)

        # 关键维度对评分
        key_pair_scores = {}
        for pair, name in zip(self.KEY_PAIRS, self.KEY_PAIR_NAMES):
            key_pair_scores[name] = float(interaction_matrix[pair[0], pair[1]])

        # 最强交互 top 5
        strongest = []
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                strongest.append((i, j, interaction_matrix[i, j]))
        strongest.sort(key=lambda x: x[2], reverse=True)

        mean_abs = float(np.mean(interaction_matrix[interaction_matrix > 0]))

        print(f"  平均交互强度: {mean_abs:.6f}")
        print(f"  关键维度对交互强度:")
        for name, score in key_pair_scores.items():
            print(f"    {name}: {score:.6f}")
        print(f"  最强交互 Top 5:")
        for rank, (i, j, s) in enumerate(strongest[:5], 1):
            print(f"    {rank}. {self.FEATURE_NAMES[i]}-{self.FEATURE_NAMES[j]}: {s:.6f}")

        return {
            'interaction_matrix': interaction_matrix,
            'key_pair_scores': key_pair_scores,
            'strongest_interactions': strongest,
            'mean_abs_interaction': mean_abs
        }

    def analyze_directional_contrast(self, sample_inputs, n_samples=100, delta=0.03):
        """
        方向对比敏感性分析

        对比两种扰动方式的embedding变化：
        - 同向平移：x[j] += δ（保持方向不变，小幅移动）
        - 符号翻转：x[j] = -x[j]（方向完全反转）

        如果embedding对"方向"敏感，翻转应该产生比平移大得多的变化。
        对比率 = 翻转变化量 / 平移变化量，越大说明方向敏感性越强。
        """
        print("\n[方向对比敏感性分析]")
        print("  核心问题: 符号翻转是否比同向平移产生更大的embedding变化？")

        sample_inputs = np.array(sample_inputs[:n_samples])
        n_dims = sample_inputs.shape[-1]
        flip_mag = np.zeros(n_dims)
        shift_mag = np.zeros(n_dims)

        for sample_idx in range(len(sample_inputs)):
            x = sample_inputs[sample_idx:sample_idx+1]
            f_base = self._embed_batch(x).flatten()

            for j in range(n_dims):
                # 同向平移（正向和负向各一次，取平均）
                x_pos = x.copy()
                x_pos[:, :, j] += delta
                f_pos = self._embed_batch(x_pos).flatten()

                x_neg = x.copy()
                x_neg[:, :, j] -= delta
                f_neg = self._embed_batch(x_neg).flatten()

                shift = 0.5 * (np.linalg.norm(f_pos - f_base) + np.linalg.norm(f_neg - f_base))
                shift_mag[j] += shift

                # 符号翻转
                x_flip = x.copy()
                if j in (5, 6):  # Amount/Exchange: [0,1]范围，翻转=1-x
                    x_flip[:, :, j] = 1.0 - x_flip[:, :, j]
                else:
                    x_flip[:, :, j] = -x_flip[:, :, j]

                f_flip = self._embed_batch(x_flip).flatten()
                flip_mag[j] += np.linalg.norm(f_flip - f_base)

        flip_mag /= len(sample_inputs)
        shift_mag /= len(sample_inputs)
        contrast_ratios = flip_mag / (shift_mag + 1e-8)

        print(f"  对比率 (翻转变化 / 平移变化):")
        for j, name in enumerate(self.FEATURE_NAMES):
            ratio = contrast_ratios[j]
            tag = "强" if ratio > 3.0 else ("中" if ratio > 1.5 else "弱")
            print(f"    {name:8s}: 对比={ratio:.2f}  "
                  f"翻转={flip_mag[j]:.4f}  平移={shift_mag[j]:.4f}  ({tag})")

        return {
            'contrast_ratios': contrast_ratios,
            'flip_magnitudes': flip_mag,
            'shift_magnitudes': shift_mag
        }

    def analyze_semantic_patterns(self):
        """
        语义模式分析

        构造原型交易日（大阳线、大阴线、放量上涨、缩量上涨等），
        检测embedding是否将语义对立的模式映射到远处，将相似的模式映射到近处。

        一致性分数 = 对立模式平均距离 / 相似模式平均距离
        理想值 >> 1.0
        """
        print("\n[语义模式分析]")
        print("  核心问题: 语义对立的模式是否映射到embedding空间的对立面？")

        pattern_names = list(self.SEMANTIC_PATTERNS.keys())
        pattern_descs = [self.SEMANTIC_PATTERNS[n]['desc'] for n in pattern_names]
        n_patterns = len(pattern_names)

        # 构造单时间步输入并获取embedding
        embeddings = {}
        for name, pattern in self.SEMANTIC_PATTERNS.items():
            x = np.array(pattern['values'], dtype=np.float32).reshape(1, 1, -1)
            emb = self._embed_batch(x).flatten()
            embeddings[name] = emb

        # 两两cosine距离矩阵
        dist_matrix = np.zeros((n_patterns, n_patterns))
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                cos_dist = cosine(embeddings[pattern_names[i]], embeddings[pattern_names[j]])
                dist_matrix[i, j] = cos_dist
                dist_matrix[j, i] = cos_dist

        # 对立模式对距离
        opp_dists = []
        print(f"\n  对立模式对距离（应较远）:")
        for n1, n2 in self.EXPECTED_OPPOSITE_PAIRS:
            d1 = self.SEMANTIC_PATTERNS[n1]['desc']
            d2 = self.SEMANTIC_PATTERNS[n2]['desc']
            i, j = pattern_names.index(n1), pattern_names.index(n2)
            dist = dist_matrix[i, j]
            opp_dists.append(dist)
            print(f"    {d1} vs {d2}: {dist:.4f}")

        # 相似模式对距离
        sim_dists = []
        print(f"  相似模式对距离（应较近）:")
        for n1, n2 in self.EXPECTED_SIMILAR_PAIRS:
            d1 = self.SEMANTIC_PATTERNS[n1]['desc']
            d2 = self.SEMANTIC_PATTERNS[n2]['desc']
            i, j = pattern_names.index(n1), pattern_names.index(n2)
            dist = dist_matrix[i, j]
            sim_dists.append(dist)
            print(f"    {d1} vs {d2}: {dist:.4f}")

        mean_opp = np.mean(opp_dists) if opp_dists else 0
        mean_sim = np.mean(sim_dists) if sim_dists else 1e-8
        consistency = mean_opp / (mean_sim + 1e-8)

        tag = "STRONG" if consistency > 2.0 else ("MODERATE" if consistency > 1.0 else "WEAK")
        print(f"\n  语义一致性分数: {consistency:.2f} ({tag})")
        print(f"    对立对平均距离: {mean_opp:.4f}, 相似对平均距离: {mean_sim:.4f}")

        return {
            'pattern_names': pattern_names,
            'pattern_descs': pattern_descs,
            'embeddings': embeddings,
            'distance_matrix': dist_matrix,
            'opposite_pair_distances': opp_dists,
            'similar_pair_distances': sim_dists,
            'consistency_score': float(consistency)
        }

    def analyze_continuity(self, sample_inputs, n_steps=25):
        """
        高维连续性分析

        沿关键维度（Close, Amount）扫值，检测：
        1. embedding是否平滑（速度连续，无突变）
        2. 在关键边界处（如Close=0）是否有曲率尖峰

        好的embedding：全局平滑 + 边界处高曲率
        """
        print("\n[高维连续性分析]")
        print("  核心问题: embedding是否全局平滑但在决策边界处有高曲率？")

        base_sample = sample_inputs[0:1]  # [1, seq_len, n_dims]

        sweep_configs = {
            'Close': (3, np.linspace(-0.05, 0.05, n_steps)),
            'VWAP': (4, np.linspace(-0.05, 0.05, n_steps)),
            'Amount': (5, np.linspace(0.3, 0.7, n_steps)),
        }

        profiles = {}
        for dim_name, (dim_idx, sweep_values) in sweep_configs.items():
            embeddings = []
            for v in sweep_values:
                x = base_sample.copy()
                x[:, :, dim_idx] = v
                emb = self._embed_batch(x).flatten()
                embeddings.append(emb)
            embeddings = np.array(embeddings)

            # 速度 = 相邻embedding的距离
            velocities = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)

            # 曲率 = 二阶差分的范数
            curvatures = np.linalg.norm(
                embeddings[2:] - 2 * embeddings[1:-1] + embeddings[:-2], axis=1
            )

            # Close维度的零点曲率
            curv_at_zero = None
            if dim_idx == 3:  # Close
                zero_idx = np.argmin(np.abs(sweep_values))
                if 1 <= zero_idx <= len(curvatures):
                    curv_at_zero = float(curvatures[zero_idx - 1])

            avg_curv = float(np.mean(curvatures)) if len(curvatures) > 0 else 0
            max_curv = float(np.max(curvatures)) if len(curvatures) > 0 else 0
            max_curv_idx = int(np.argmax(curvatures)) if len(curvatures) > 0 else 0
            max_curv_value = float(sweep_values[min(max_curv_idx + 1, len(sweep_values) - 1)])

            profiles[dim_name] = {
                'dim_idx': dim_idx,
                'sweep_values': sweep_values,
                'velocities': velocities,
                'curvatures': curvatures,
                'max_curvature': max_curv,
                'max_curvature_at': max_curv_value,
                'avg_curvature': avg_curv,
                'curvature_at_zero': curv_at_zero
            }

            print(f"\n  {dim_name}维度 ({sweep_values[0]:.3f} → {sweep_values[-1]:.3f}):")
            print(f"    最大曲率: {max_curv:.6f} (在 {dim_name}={max_curv_value:.3f} 处)")
            print(f"    平均曲率: {avg_curv:.6f}")
            if curv_at_zero is not None:
                ratio = curv_at_zero / (avg_curv + 1e-8)
                print(f"    零点曲率: {curv_at_zero:.6f} (是平均值的 {ratio:.1f} 倍)")

        smoothness = float(np.mean([p['avg_curvature'] for p in profiles.values()]))

        return {
            'profiles': profiles,
            'smoothness_score': smoothness
        }

    def analyze_information_preservation(self, sample_inputs, n_samples=200):
        """
        信息保留度分析

        用Ridge回归测试：从128维embedding能否重建原始8维输入？
        如果某个特征的R²很低，说明该维度的信息在embedding中丢失了。
        """
        print("\n[信息保留度分析]")
        print("  核心问题: 从embedding能否重建原始输入特征？")

        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import cross_val_score

        sample_inputs = np.array(sample_inputs[:n_samples])

        # 获取embedding
        embeddings = self._embed_batch(sample_inputs)  # [n, seq_len, d_model]

        # 使用最后时间步
        X_emb = embeddings[:, -1, :]  # [n, d_model]

        # 目标：细处理后的输入（embedding层的直接输入）
        y_features = self.embedding_module.transform_numpy(sample_inputs)[:, -1, :]  # [n, n_dims]

        r2_values = np.zeros(y_features.shape[1])
        for j in range(y_features.shape[1]):
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            scores = cross_val_score(ridge, X_emb, y_features[:, j], cv=5, scoring='r2')
            r2_values[j] = float(np.mean(scores))

        r2_per_feature = {name: float(r2_values[j]) for j, name in enumerate(self.FEATURE_NAMES)}
        mean_r2 = float(np.mean(r2_values))

        print(f"  各特征重建R² (Ridge回归, 5折CV):")
        for name, r2 in r2_per_feature.items():
            tag = "GOOD" if r2 > 0.8 else ("MODERATE" if r2 > 0.5 else "POOR")
            print(f"    {name:8s}: R²={r2:.4f} ({tag})")
        print(f"  平均R²: {mean_r2:.4f}")

        worst_idx = int(np.argmin(r2_values))
        best_idx = int(np.argmax(r2_values))

        return {
            'r2_per_feature': r2_per_feature,
            'r2_values': r2_values,
            'mean_r2': mean_r2,
            'worst_feature': (self.FEATURE_NAMES[worst_idx], float(r2_values[worst_idx])),
            'best_feature': (self.FEATURE_NAMES[best_idx], float(r2_values[best_idx]))
        }

    def analyze_saturation(self, sample_inputs, n_samples=100):
        """饱和度分析: 输出分布健康检查"""
        print("\n[饱和度分析]")

        sample_inputs = np.array(sample_inputs[:n_samples])

        with torch.no_grad():
            x_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            hidden = self.embedding_module(x_tensor)

            hidden_flat = hidden.cpu().numpy().flatten()
            saturation_ratio = float(np.mean(np.abs(hidden_flat) > 3))
            dead_ratio = float(np.mean(np.abs(hidden_flat) < 0.01))

        print(f"  Embedding模块输出:")
        print(f"    均值: {np.mean(hidden_flat):.4f}")
        print(f"    标准差: {np.std(hidden_flat):.4f}")
        print(f"    范围: [{np.min(hidden_flat):.4f}, {np.max(hidden_flat):.4f}]")
        print(f"    饱和比例(|x|>3): {saturation_ratio*100:.2f}%")
        print(f"    死神经元比例(|x|<0.01): {dead_ratio*100:.2f}%")

        return {
            'hidden_mean': float(np.mean(hidden_flat)),
            'hidden_std': float(np.std(hidden_flat)),
            'hidden_min': float(np.min(hidden_flat)),
            'hidden_max': float(np.max(hidden_flat)),
            'saturation_ratio': saturation_ratio,
            'dead_neuron_ratio': dead_ratio
        }

    def analyze_feature_ablation(self, sample_inputs, n_samples=100):
        """特征消融分析: 逐一遮盖各特征，测量输出变化"""
        print("\n[特征消融分析]")

        sample_inputs = np.array(sample_inputs[:n_samples])

        with torch.no_grad():
            sample_inputs_tensor = torch.tensor(sample_inputs, dtype=torch.float32, device=self.device)
            base_output = self.embedding_module(sample_inputs_tensor)
            base_norm = torch.norm(base_output).item()

            importance_scores = []

            for j, name in enumerate(self.FEATURE_NAMES):
                masked_input = sample_inputs.copy()
                masked_input[:, :, j] = self.NEUTRAL_VALUES[j]

                masked_tensor = torch.tensor(masked_input, dtype=torch.float32, device=self.device)
                masked_output = self.embedding_module(masked_tensor)
                masked_norm = torch.norm(masked_output).item()

                relative_change = abs(masked_norm - base_norm) / (base_norm + 1e-6)
                importance_scores.append(relative_change)

        sorted_indices = np.argsort(importance_scores)[::-1]
        print(f"  特征重要性排序:")
        for rank, idx in enumerate(sorted_indices, 1):
            print(f"    {rank}. {self.FEATURE_NAMES[idx]:8s}: {importance_scores[idx]*100:.2f}%")

        return {
            'feature_names': self.FEATURE_NAMES,
            'importance_scores': [float(s) for s in importance_scores],
            'sorted_indices': sorted_indices.tolist()
        }

    # ==================== 可视化 ====================

    def visualize_results(self, sample_inputs, results, save_dir=None):
        """生成2x3可视化图表"""
        if save_dir is None:
            save_dir = os.path.join(PROJECT_ROOT, 'out_eval_results')

        os.makedirs(save_dir, exist_ok=True)
        sample_inputs = np.array(sample_inputs[:500])

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle('Embedding评估：维度间交互编码质量', fontsize=14, fontweight='bold')

        # ---- [0,0] 跨维度交互热力图 ----
        ax = axes[0, 0]
        if 'cross_interaction' in results:
            matrix = results['cross_interaction']['interaction_matrix']
            n = matrix.shape[0]
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(self.FEATURE_NAMES, rotation=45, ha='right')
            ax.set_yticklabels(self.FEATURE_NAMES)
            ax.set_title('跨维度交互强度')
            for i in range(n):
                for j in range(n):
                    if i != j:
                        ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center',
                                fontsize=7, color='black' if matrix[i,j] < matrix.max()*0.7 else 'white')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- [0,1] 方向对比柱状图 ----
        ax = axes[0, 1]
        if 'directional_contrast' in results:
            dc = results['directional_contrast']
            ratios = dc['contrast_ratios']
            colors = ['#2ecc71' if r > 3.0 else '#f39c12' if r > 1.5 else '#e74c3c' for r in ratios]
            bars = ax.bar(self.FEATURE_NAMES, ratios, color=colors, alpha=0.85)
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='ratio=1.0')
            ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.5, label='ratio=3.0')
            ax.set_ylabel('对比率 (翻转/平移)')
            ax.set_title('方向对比敏感性')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            for bar, r in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{r:.1f}', ha='center', va='bottom', fontsize=8)

        # ---- [0,2] 语义模式距离矩阵 ----
        ax = axes[0, 2]
        if 'semantic_patterns' in results:
            sp = results['semantic_patterns']
            dist_mat = sp['distance_matrix']
            descs = sp['pattern_descs']
            im = ax.imshow(dist_mat, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(descs)))
            ax.set_yticks(range(len(descs)))
            ax.set_xticklabels(descs, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(descs, fontsize=7)
            ax.set_title(f'语义模式距离 (一致性={sp["consistency_score"]:.2f})')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # ---- [1,0] Close轴连续性曲线 ----
        ax = axes[1, 0]
        if 'continuity' in results:
            profiles = results['continuity']['profiles']
            if 'Close' in profiles:
                p = profiles['Close']
                sweep = p['sweep_values']
                curvatures = p['curvatures']
                velocities = p['velocities']

                # 曲率曲线（归一化到0-1以便叠加显示）
                curv_x = sweep[1:-1]
                curv_norm = curvatures / (curvatures.max() + 1e-8)
                ax.plot(curv_x, curv_norm, 'r-o', markersize=3, label='曲率 (归一化)')

                # 速度曲线
                vel_x = sweep[1:]
                vel_norm = velocities / (velocities.max() + 1e-8)
                ax.plot(vel_x, vel_norm, 'b-o', markersize=3, label='速度 (归一化)')

                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Close=0')
                ax.set_xlabel('Close值')
                ax.set_ylabel('归一化强度')
                ax.set_title('Close轴连续性（曲率+速度）')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        # ---- [1,1] 信息保留度R² ----
        ax = axes[1, 1]
        if 'info_preservation' in results:
            ip = results['info_preservation']
            r2_vals = ip['r2_values']
            colors = ['#2ecc71' if r > 0.8 else '#f39c12' if r > 0.5 else '#e74c3c' for r in r2_vals]
            bars = ax.bar(self.FEATURE_NAMES, r2_vals, color=colors, alpha=0.85)
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='R²=0.5')
            ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='R²=0.8')
            ax.set_ylabel('R²')
            ax.set_title(f'信息保留度 (均值={ip["mean_r2"]:.3f})')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            for bar, r in zip(bars, r2_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{r:.2f}', ha='center', va='bottom', fontsize=8)

        # ---- [1,2] PCA散点图（按量价关系着色）----
        ax = axes[1, 2]
        n_pca = min(500, len(sample_inputs))
        pca_inputs = sample_inputs[:n_pca]
        pca_outputs = self._embed_batch(pca_inputs).reshape(n_pca, -1)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(pca_outputs)

        # 用最后时间步的Close和VWAP着色
        close_vals = pca_inputs[:, -1, 3]
        vol_vals = pca_inputs[:, -1, 4]
        categories = []
        for c, v in zip(close_vals, vol_vals):
            if c > 0.01 and v > 0.5:
                categories.append(0)  # 放量上涨
            elif c > 0.01:
                categories.append(1)  # 缩量上涨
            elif c < -0.01 and v > 0.5:
                categories.append(2)  # 放量下跌
            elif c < -0.01:
                categories.append(3)  # 缩量下跌
            else:
                categories.append(4)  # 横盘
        categories = np.array(categories)

        cat_names = ['放量上涨', '缩量上涨', '放量下跌', '缩量下跌', '横盘']
        cat_colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#95a5a6']
        for cat_id in range(5):
            mask = categories == cat_id
            if mask.any():
                ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1], c=cat_colors[cat_id],
                           label=cat_names[cat_id], alpha=0.5, s=10)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('Embedding PCA（按量价关系着色）')
        ax.legend(fontsize=7, markerscale=2)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'embedding_module_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n  可视化图表已保存到: {save_path}")

    # ==================== 总结 ====================

    def print_summary(self, results):
        """生成分析总结，给出PASS/WEAK评定"""
        print("\n" + "=" * 70)
        print("Embedding评估总结")
        print("=" * 70)

        issues = []
        recommendations = []

        # 1. 跨维度交互
        if 'cross_interaction' in results:
            ci = results['cross_interaction']
            mean_int = ci['mean_abs_interaction']
            status = "STRONG" if mean_int > 0.1 else ("MODERATE" if mean_int > 0.01 else "WEAK")
            print(f"\n[1] 跨维度交互: {status}")
            print(f"    平均交互强度: {mean_int:.6f}")
            for name, score in ci['key_pair_scores'].items():
                print(f"    {name}: {score:.6f}")

            if status == "WEAK":
                issues.append("维度间几乎无非线性交互，embedding接近线性映射")
                recommendations.append("增加embedding层的非线性能力（更深/更宽的MLP，或使用更强的激活函数）")
            # 检查Close-Amount是否在top 5
            top5 = ci['strongest_interactions'][:5]
            top5_pairs = {(s[0], s[1]) for s in top5}
            if (3, 5) not in top5_pairs:
                issues.append("Close-Amount交互不在Top5，量价关系未被重点编码")
                recommendations.append("考虑显式构造Close×Amount交互特征")

        # 2. 方向对比
        if 'directional_contrast' in results:
            dc = results['directional_contrast']
            print(f"\n[2] 方向对比敏感性:")
            for j, name in enumerate(self.FEATURE_NAMES):
                r = dc['contrast_ratios'][j]
                tag = "强" if r > 3.0 else ("中" if r > 1.5 else "弱")
                print(f"    {name:8s}: {tag} (对比率={r:.2f})")

            close_ratio = dc['contrast_ratios'][3]
            if close_ratio < 1.5:
                issues.append(f"Close方向对比弱 (ratio={close_ratio:.2f})，涨跌方向区分不足")
                recommendations.append("确保Close维度的正负值映射到embedding的不同区域")

        # 3. 语义一致性
        if 'semantic_patterns' in results:
            sp = results['semantic_patterns']
            cs = sp['consistency_score']
            tag = "STRONG" if cs > 2.0 else ("MODERATE" if cs > 1.0 else "WEAK")
            print(f"\n[3] 语义一致性: {tag} (分数={cs:.2f})")

            if cs < 1.0:
                issues.append(f"语义一致性<1.0，对立模式反而比相似模式更近")
                recommendations.append("embedding的语义空间混乱，需要重新设计或增加训练")

        # 4. 连续性
        if 'continuity' in results:
            profiles = results['continuity']['profiles']
            print(f"\n[4] 高维连续性:")
            for dim_name, p in profiles.items():
                print(f"    {dim_name}: 最大曲率={p['max_curvature']:.6f} "
                      f"(在 {dim_name}={p['max_curvature_at']:.3f} 处)", end="")
                if p['curvature_at_zero'] is not None:
                    ratio = p['curvature_at_zero'] / (p['avg_curvature'] + 1e-8)
                    print(f", 零点曲率倍数={ratio:.1f}x")
                else:
                    print()

            if 'Close' in profiles:
                p = profiles['Close']
                if p['curvature_at_zero'] is not None:
                    ratio = p['curvature_at_zero'] / (p['avg_curvature'] + 1e-8)
                    if ratio < 1.5:
                        issues.append(f"Close零点曲率无尖峰 (仅平均值的{ratio:.1f}倍)")
                        recommendations.append("期望在Close=0处有曲率突增，表明涨跌方向有清晰的决策边界")

        # 5. 信息保留
        if 'info_preservation' in results:
            ip = results['info_preservation']
            print(f"\n[5] 信息保留度: 平均R²={ip['mean_r2']:.4f}")
            for name, r2 in ip['r2_per_feature'].items():
                print(f"    {name:8s}: R²={r2:.4f}")

            lost = [name for name, r2 in ip['r2_per_feature'].items() if r2 < 0.5]
            if lost:
                issues.append(f"以下特征信息丢失风险高 (R²<0.5): {', '.join(lost)}")
                recommendations.append(f"检查embedding是否充分保留了 {', '.join(lost)} 的信息")

        # 6. 饱和度
        if 'saturation' in results:
            sat = results['saturation']
            print(f"\n[6] 饱和度:")
            print(f"    均值={sat['hidden_mean']:.4f}, 标准差={sat['hidden_std']:.4f}")
            print(f"    饱和={sat['saturation_ratio']*100:.1f}%, 死神经元={sat['dead_neuron_ratio']*100:.1f}%")

            if sat['dead_neuron_ratio'] > 0.1:
                issues.append(f"死神经元比例 {sat['dead_neuron_ratio']*100:.1f}% > 10%")
                recommendations.append("调整初始化或增加学习率，避免embedding输出坍缩到0附近")
            if sat['saturation_ratio'] > 0.1:
                issues.append(f"饱和比例 {sat['saturation_ratio']*100:.1f}% > 10%")
                recommendations.append("考虑添加LayerNorm或降低权重初始化增益")

        # 7. 特征消融
        if 'feature_ablation' in results:
            fa = results['feature_ablation']
            print(f"\n[7] 特征重要性排序:")
            for rank, idx in enumerate(fa['sorted_indices'], 1):
                print(f"    {rank}. {fa['feature_names'][idx]:8s}: {fa['importance_scores'][idx]*100:.2f}%")

        # 总结
        if issues:
            print(f"\n{'='*70}")
            print("发现的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\n优化建议:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print(f"\n{'='*70}")
            print("所有指标表现良好，embedding层编码质量合格。")

        return {
            'issues': issues,
            'recommendations': recommendations
        }


def main():
    parser = argparse.ArgumentParser(description='Embedding模块评估')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要分析的模型文件路径')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用的模型文件并退出')
    args = parser.parse_args()

    print("Embedding模块评估...")
    print("评估方向: 维度间交互编码质量")

    out_dir = DataConfig.OUTPUT_DIR
    model_files = []
    # 搜索 out/ 和 out/embedding_pretrain/ 下的模型文件
    search_dirs = [out_dir]
    from config import EmbeddingConfig
    if hasattr(EmbeddingConfig, 'OUTPUT_DIR'):
        search_dirs.append(EmbeddingConfig.OUTPUT_DIR)
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for f in os.listdir(search_dir):
                if f.endswith('.pth'):
                    model_files.append(os.path.join(search_dir, f))

    if args.list_models:
        print("\n可用的模型文件:")
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, mf in enumerate(model_files, 1):
                mtime = datetime.fromtimestamp(os.path.getmtime(mf)).strftime('%Y-%m-%d %H:%M:%S')
                size = os.path.getsize(mf) / (1024 * 1024)
                print(f"  {i}. {os.path.basename(mf)} ({mtime}, {size:.2f} MB)")
        else:
            print("  未找到任何模型文件")
        return None, None

    if args.model:
        if os.path.sep not in args.model and '/' not in args.model:
            potential_path = os.path.join(out_dir, args.model)
            if os.path.exists(potential_path):
                args.model = potential_path

        if not os.path.exists(args.model):
            print(f"\n错误: 指定的模型文件不存在: {args.model}")
            return None, None
        model_path = args.model
        print(f"\n使用指定的模型: {os.path.basename(model_path)}")
    else:
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"\n找到 {len(model_files)} 个模型文件，使用最新的: {os.path.basename(model_files[0])}")
            model_path = model_files[0]
        else:
            print("\n未找到模型文件，将使用随机初始化模型")
            model_path = None

    save_dir = os.path.join(PROJECT_ROOT, 'out_eval_results')
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n未找到归一化器文件: {DataConfig.NORMALIZER_PATH}")
        return None, None

    print("\n" + "=" * 60)
    print("[步骤1] 加载数据...")
    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    print("\n[步骤2] 准备测试样本...")
    from data import coarse_normalize_context_window
    eval_ctx = DataConfig.CONTEXT_LENGTH
    eval_req = DataConfig.REQUIRED_LENGTH
    all_inputs = []
    for stock in test_stock_info[:50]:
        data = stock['data']
        test_split = stock['test_split_point']
        for i in range(test_split, min(test_split + 10, len(data) - eval_req)):
            input_seq = coarse_normalize_context_window(
                data, i, eval_ctx, check_limit_up=False, required_length=eval_req
            )
            if input_seq is not None:
                all_inputs.append(input_seq)

    sample_inputs = np.array(all_inputs[:500])
    print(f"  准备了 {len(sample_inputs)} 个测试样本")

    analyzer = EmbeddingModuleAnalyzer(model_path=model_path)
    analyzer.load_model(feature_normalizer=feature_normalizer)

    print("\n[步骤3] 执行分析...")
    results = {}

    results['cross_interaction'] = analyzer.analyze_cross_dimensional_interactions(
        sample_inputs, n_samples=100)
    results['directional_contrast'] = analyzer.analyze_directional_contrast(
        sample_inputs, n_samples=100)
    results['semantic_patterns'] = analyzer.analyze_semantic_patterns()
    results['continuity'] = analyzer.analyze_continuity(
        sample_inputs, n_steps=25)
    results['info_preservation'] = analyzer.analyze_information_preservation(
        sample_inputs, n_samples=200)
    results['saturation'] = analyzer.analyze_saturation(
        sample_inputs, n_samples=100)
    results['feature_ablation'] = analyzer.analyze_feature_ablation(
        sample_inputs, n_samples=100)

    print("\n[步骤4] 生成可视化...")
    analyzer.visualize_results(sample_inputs, results, save_dir)

    print("\n[步骤5] 生成总结...")
    analyzer.print_summary(results)

    print(f"\n分析完成！可视化图表已保存到: {save_dir}/embedding_module_analysis.png")

    return results


if __name__ == "__main__":
    results = main()
