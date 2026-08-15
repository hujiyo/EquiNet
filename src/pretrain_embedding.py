"""
EquiNet Embedding层训练脚本

通过 VISReg 几何正则 + 重建约束，预训练 FFN-Embedding 层，
使其成为一个固定的、具有几何保证的K线特征提取器：

1. VISReg 几何正则 (Wu, Balestriero & Levine, 2026; arXiv:2606.02572)：
   约束嵌入分布趋向各向同性高斯 N(0, target_std²)
2. 线性解码器重建损失：确保嵌入向量足以恢复原始16维 + 11个衍生特征
   刻意不用非线性解码器——非线性层自己就能完成乘积/绝对值/除法等跨维运算，
   会把"逼融合"的压力吞掉，embedding 逐维线性拷贝也能过关；
   线性解码器算不了任何非线性运算，跨维结构必须由 embedding 内部预先算好
   并编码成可线性读出的方向。此时重建损失 = 线性探针误差，
   O=/D= 日志直接衡量 embedding 中可线性读出的原始/衍生信息量。

用法：
  python src/pretrain_embedding.py                        # 使用默认参数
  python src/pretrain_embedding.py --epochs 300           # 自定义轮数
  python src/pretrain_embedding.py --test                 # 测试已保存权重的输出std
"""

import os
import sys
import math
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


from config import (DataConfig, EmbeddingConfig, DeviceConfig,ModelConfig,
                     TrainingConfig)
from data import (load_and_preprocess_data, FeatureNormalizer,
                  precompute_training_pool)
from training_utils import _get_amp_context
from visreg import VISRegLoss

# ==================== 衍生特征 ====================
def compute_derived_features(x, eps=None, chunk_size=4_000_000, rel_ohlc=None):
    """
    从归一化16维特征计算11个衍生特征 [N, 11] + 语义掩码 [N, 11]

    设计原则：全部为"跨维度非线性函数"——
    单维度信息 decoder 用线性层就能从原始16维算出，逼不出 embedding 融合；
    只有跨维度的非线性关系（abs/sign/乘积/除法）才能强迫 embedding
    在128维里编码"维度间关系"，而非逐维独立存储。

    输入 x 为 QuantileTransformer 细处理后的特征（各维均值0方差1），
    embedding 也只见过这个分布。衍生目标分两个空间计算：
    - 归一化空间（符号族/breakthrough/均线/布林）：与 embedding 输入同分布，直接可推；
    - 未归一化相对空间（body_ratio/upper_shadow/lower_shadow 及 span_turnover 的
      振幅项，经 rel_ohlc 逆变换）：物理语义纯净，embedding 需自行学会分位数逆映射
      后才能推出，任务更硬。

    rel_ohlc: [N, 4] 未归一化（粗处理后、分位数变换前）的相对特征
              [open_rel, high_rel, low_rel, close_rel]（由 feature_normalizer
              逆变换得到）。注意这不是原始价格：
              - open_rel/close_rel = (O或C - 昨收)/昨收，参考系=昨收(列0/3被clip±0.1)
              - high_rel/low_rel   = (H或L - 当日开盘)/开盘，参考系=当日开盘
              K线形态特征计算前必须先统一参考系（除以 1+open_rel 换算到开盘系），
              否则跳空日影线为负、body_ratio 有 ±10% 畸变（见 _compute_derived_block）。
              统一后 K线序关系(high≥max(open,close)≥...≥low)在该空间严格成立，
              100%有效无需掩码。None 时退化为归一化空间计算+掩码剔除
              （归一化空间 ~65% 样本影线为负，纯为分位数变换伪影）。

    大数据集分块计算：衍生特征+掩码+中间临时变量峰值约 20·N·4 bytes，
    1.6亿条需~12GB 会 OOM；分块后每块峰值仅 ~20·chunk·4 bytes (chunk=4M→~320MB)。

    16维索引:
      0:open_rel 1:high_rel(≥0) 2:low_rel(≤0) 3:close_rel 4:vwap_rel
      5:amount 6:exchange 7:m5 8:m10 9:m20 10:dif 11:dea
      12:macd_hist 13:macd_hist_diff 14:bb_upper 15:bb_lower
    """
    if eps is None:
        eps = EmbeddingConfig.DERIVED_EPS

    n = x.shape[0]
    if n > chunk_size:
        derived_chunks = []
        mask_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            ohlc_block = rel_ohlc[start:end] if rel_ohlc is not None else None
            d, m = _compute_derived_block(x[start:end], eps, ohlc_block)
            derived_chunks.append(d)
            mask_chunks.append(m)
        # 两个 concat 必须分开并显式释放 chunks：若写进同一 return 顺序求值，
        # chunks 列表与两份结果全程并存，大池下双倍峰值会 OOM
        derived = np.concatenate(derived_chunks, axis=0)
        del derived_chunks
        masks = np.concatenate(mask_chunks, axis=0)
        del mask_chunks
        return derived, masks
    return _compute_derived_block(x, eps, rel_ohlc)


def _compute_derived_block(x, eps, rel_ohlc=None):
    """单块衍生特征计算（compute_derived_features 的分块内核）"""
    open_r, high_r, low_r, close_r = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    vwap = x[:, 4]
    amount, exchange = x[:, 5], x[:, 6]
    m5, m10, m20 = x[:, 7], x[:, 8], x[:, 9]
    macd_h, macd_hd = x[:, 12], x[:, 13]
    bb_u, bb_l = x[:, 14], x[:, 15]

    n = x.shape[0]
    ones = np.ones(n, dtype=np.uint8)

    # ---- A. 符号监督族（修 raw MSE 的符号盲区：±1 目标梯度 O(1)，恒定，
    #        与样本离零点多近无关；raw 重建在零值附近梯度 O(0.001)，无压力区分符号）----
    # 注意：符号均在归一化空间定义——逐列分位数变换不保跨列差符号、不保物理零点，
    # 故 direction 是"收/开两列的相对强弱"而非物理 C-O 符号，sign(macd_h) 是
    # "相对本列分布中心"而非物理 MACD 正负；但仍是确定性的跨维非线性目标，监督有效
    direction = np.sign(close_r - open_r)                # 收/开列相对强弱 {-1,0,1}
    vol_price_align = direction * amount                 # 量价配合: 放量走强>0 / 放量走弱<0
    momentum_accel = np.sign(macd_h) * np.sign(macd_hd)  # 柱体与变化相对各自分布中心: 同向+1/反向-1
    vwap_side = np.sign(close_r - vwap)                  # 收盘列 vs 均价列 相对强弱

    # 归一化空间的 body/span（用于 breakthrough；span 仅退化路径使用）
    body_norm = np.abs(close_r - open_r)
    span_norm = high_r - low_r

    # ---- B. K线形态域（依赖跨维序关系 high≥max(open,close)≥min(open,close)≥low）----
    if rel_ohlc is not None:
        # 未归一化相对空间计算：物理约束完好，100%有效无需掩码。
        # 注意逆变换得到的是混合参考系的相对特征，必须先统一到当日开盘系：
        #   o=(O-昨收)/昨收, c=(C-昨收)/昨收 （昨收系, 被clip±0.1）
        #   h=(H-O)/O,      l=(L-O)/O      （开盘系, 恒 h≥0≥l）
        # 换算：开盘系下 (C-O)/O = (c-o)/(1+o)（1+o=O/昨收∈[0.9,1.1]）。
        # 若直接用 o/c 与 h/l 相减（昨收系混开盘系），跳空高开日 upper_shadow<0、
        # 跳空低开日 lower_shadow<0，伪影未消除只是换了样本；body_ratio 还带
        # ±10% 的 O/昨收 畸变。统一参考系后序关系严格恢复：
        #   span=(H-L)/O≥0, |C-O|≤H-L → body_ratio∈[0,1], 影线恒≥0。
        o, h, l, c = rel_ohlc[:, 0], rel_ohlc[:, 1], rel_ohlc[:, 2], rel_ohlc[:, 3]
        safe_scale = 1.0 + o                              # = O/昨收 ∈ [0.9, 1.1]
        body_open = (c - o) / safe_scale                  # (C-O)/O 开盘系实体(带符号)
        span_open = h - l                                 # (H-L)/O ≥ 0
        body_ratio = np.clip(np.abs(body_open) / (span_open + eps), 0.0, 1.0)
        upper_shadow = h - np.maximum(0.0, body_open)     # (H-max(O,C))/O ≥ 0
        lower_shadow = np.minimum(0.0, body_open) - l     # (min(O,C)-L)/O ≥ 0
        mask_body, mask_upper, mask_lower = ones, ones, ones
    else:
        # 退化：归一化空间计算 + 掩码剔除伪影
        body_ratio = np.clip(body_norm / (span_norm + eps), 0.0, 1.0)
        upper_shadow = high_r - np.maximum(open_r, close_r)
        lower_shadow = np.minimum(open_r, close_r) - low_r
        mask_body = (span_norm > 0).astype(np.uint8)
        mask_upper = (upper_shadow >= 0).astype(np.uint8)
        mask_lower = (lower_shadow >= 0).astype(np.uint8)

    # ---- C. 量价关系域 ----
    breakthrough = body_norm * amount                    # 实体×量: 突破强度 (归一化空间)
    if rel_ohlc is not None:
        # 物理振幅 (H-L)/O 恒≥0；归一化空间 high_r-low_r 在小振幅日可为负
        # （逐列变换不保跨列序，伪影，见 mask 注释），故振幅项不用归一化空间
        span_turnover = span_open * exchange             # 振幅×换手: 波动×资金参与
    else:
        span_turnover = span_norm * exchange             # 退化: 归一化空间(含符号伪影, 剪尾+z-score缓解)

    # ---- D. 均线域 (绝对值线性化，不用平方：平方使极端离散日主导损失) ----
    ma_spread = np.abs(m5 - m10) + np.abs(m10 - m20)     # 均线离散度

    # ---- E. 布林带域 (绝对值线性化，不用平方) ----
    bb_width = np.abs(bb_u - bb_l)                       # 布林带宽: 波动率

    derived = np.stack([
        direction, vol_price_align, momentum_accel, vwap_side,
        body_ratio, upper_shadow, lower_shadow,
        breakthrough, span_turnover,
        ma_spread, bb_width,
    ], axis=1).astype(np.float32)

    # 掩码直接以 uint8 构建（float32 掩码在大池上多占 4 倍内存且 concat 时双倍峰值）
    masks = np.stack([
        ones, ones, ones, ones,
        mask_body, mask_upper, mask_lower,
        ones, ones,
        ones, ones,
    ], axis=1)
    return derived, masks


def _standardize_derived(derived, masks, winsorize_pct=None):
    """
    剪尾 + z-score 标准化（防"极端值霸凌" + 统一量级）

    平方类目标有重尾（极端行情日误差平方后主导整个 loss），
    绝对值线性化后仍可能有产品类重尾（两个归一化因子的积），
    统一按分位数剪尾后再 z-score：剪掉的是极少数极端行情样本，
    其余样本的分布和量级与原始16维（已归一化，方差1）对齐，等权 MSE 才成立。

    统计量（分位数/均值/标准差）只在掩码有效样本上计算；
    掩码样本标准化后置 0（损失端会被掩码剔除，0 仅为占位）。
    """
    if winsorize_pct is None:
        winsorize_pct = EmbeddingConfig.DERIVED_WINSORIZE_PCT
    lo, hi = winsorize_pct
    n_feat = derived.shape[1]
    for i in range(n_feat):
        col = derived[:, i]
        m = masks[:, i] > 0
        valid = col[m]
        if valid.size == 0:
            continue
        if lo > 0.0 or hi < 100.0:
            p_lo, p_hi = np.percentile(valid, [lo, hi])
            np.clip(col, p_lo, p_hi, out=col)
        col_mean = col[m].mean()
        col_std = col[m].std() + 1e-6
        col[:] = (col - col_mean) / col_std
        col[~m] = 0.0
    return derived


DERIVED_FEATURE_NAMES = [
    'direction', 'vol_price_align', 'momentum_accel', 'vwap_side',
    'body_ratio', 'upper_shadow', 'lower_shadow',
    'breakthrough', 'span_turnover',
    'ma_spread', 'bb_width',
]


def build_derived_targets(kline_data, feature_normalizer, chunk_size=4_000_000):
    """
    从归一化K线构建衍生训练目标：OHLC 逆变换 + 衍生特征 + 剪尾 z-score

    只对实际参与训练/探测的子集调用，而非全池：
    衍生目标与掩码按行独立计算，子集统计（分位数/均值/标准差）与全池
    在百万级样本下已收敛等价，但内存从 全池×(16+11+11) 列 降到 子集×27 列，
    避免大池上衍生数组的双倍 concat 峰值 OOM。

    Returns:
        derived_data: [M, n_derived] float32 剪尾+z-score 后的衍生目标
        derived_mask: [M, n_derived] uint8 语义掩码 (1=有效, 0=归一化伪影样本)
    """
    # 逆变换 open/high/low/close 到未归一化相对空间（粗处理后的 open_rel 等，
    # 非原始价格，参考系混合见 compute_derived_features 的 rel_ohlc 说明），
    # 用于计算依赖 K线序关系的形态特征(body_ratio/影线)与物理振幅 span：
    # 逐维分位数变换不保跨维序关系(high≥max(open,close)≥...≥low)，
    # 归一化空间中 ~65% 样本影线为负(伪影)；逆变换后统一参考系计算，物理约束完好
    if feature_normalizer is not None:
        rel_ohlc = np.empty((len(kline_data), 4), dtype=np.float32)
        for start in range(0, len(kline_data), chunk_size):
            end = min(start + chunk_size, len(kline_data))
            for col, name in enumerate(['open', 'high', 'low', 'close']):
                rel_ohlc[start:end, col] = feature_normalizer.pipelines[name].inverse_transform(
                    kline_data[start:end, col:col+1]).flatten()
        print(f"  OHLC 逆变换完成 ({len(kline_data):,} 条, 未归一化相对空间计算形态/振幅)")
    else:
        rel_ohlc = None

    derived_data, derived_mask = compute_derived_features(
        kline_data, rel_ohlc=rel_ohlc, chunk_size=chunk_size)
    print(f"  衍生特征: {derived_data.shape[1]}维 "
          f"(跨域非线性; 形态/振幅域在未归一化相对空间100%有效, "
          f"无 rel_ohlc 时退化为归一化空间+掩码剔除, 见 compute_derived_features)")

    # 剪尾 + z-score 标准化：
    # - 剪尾 (分位数 clip) 防"极端值霸凌"：极少数极端行情日(涨停放巨量等)误差平方后
    #   会主导整个衍生 loss，按分位数剪掉尾部后再参与统计，其余样本才不被带偏
    # - z-score 防"量级失衡"：各衍生特征量级差异大
    #   (direction∈{-1,0,1} vs breakthrough=body*amount 可能>10)，
    #   直接拼接做等权 MSE 会让大量级特征主导损失、小量级特征被忽略；
    #   标准化后所有衍生特征方差=1，与原始16维(已归一化、方差1)量级一致，等权 MSE 才合理
    derived_data = _standardize_derived(derived_data, derived_mask)
    print(f"  衍生特征已剪尾+z-score 标准化 (μ→0, σ→1, 掩码样本剔除后置0)")
    print(f"  衍生特征分布 (有效样本):")
    for i, name in enumerate(DERIVED_FEATURE_NAMES):
        col = derived_data[:, i]
        m = derived_mask[:, i] > 0
        valid = col[m]
        print(f"    {name:>16s}: 有效={valid.size:,} ({m.mean()*100:.1f}%)  "
              f"μ={valid.mean():.3f}  σ={valid.std():.3f}  "
              f"[{valid.min():.3f}, {valid.max():.3f}]")
    return derived_data, derived_mask


# ==================== 模型定义 ====================

class KLineEmbedding(nn.Module):
    """
    单日K线嵌入模块（结构与 StockTransformer 的 FFN-Embedding 完全一致）

    MLP(input_dim→128→256→GELU→128)
    网络足够浅（3层），无需残差连接，纯 MLP 即可充分学习非线性特征交互。
    """

    def __init__(self, input_dim=ModelConfig.INPUT_DIM, d_model=128, expand_ratio=2):
        super().__init__()
        hidden_dim = d_model * expand_ratio
        self.embed_proj = nn.Linear(input_dim, d_model, bias=False)
        self.embed_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False)
        )
        # 初始化（非逐层调参，从架构推导）
        # MLP分支增益=1（σ1·σ2·0.588·√(d·h) = 1），输出std = embed_proj输出std
        # 合计std≈0.141（接近VISReg目标std=0.2，训练中VISReg会微调至精确目标）
        nn.init.normal_(self.embed_proj.weight,
                        std=0.2 / (math.sqrt(2) * math.sqrt(input_dim)))
        nn.init.normal_(self.embed_mlp[0].weight,
                        std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.embed_mlp[2].weight,
                        std=1.0 / (0.588 * math.sqrt(hidden_dim)))

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] 归一化K线特征
        Returns:
            z: [batch, d_model] 嵌入向量（保留方向+幅度）
        """
        h = self.embed_proj(x)
        h = self.embed_mlp(h)
        return h


class PretrainModel(nn.Module):
    """
    Embedding层训练模型 = KLineEmbedding + 线性解码器

    X → Embedding → S → Linear → Y
                      ↑              ↑
                  VISReg(S)   MSE(Y, [X, derived(X)])

    解码器刻意退化为单层线性映射（线性探针）：
    - 非线性解码器（如旧版 MLP 128→256→GELU→128）自己就能近似乘积/绝对值/
      除法等跨维运算，从"逐维线性拷贝"的 embedding 中算出全部衍生目标，
      把逼融合的压力全部吞掉 → embedding 依然可以逐维独立存储而不被惩罚。
    - 线性解码器算不了任何非线性运算：sign/比值/乘积/影线这些结构
      必须由 embedding 内部预先算好、存成可线性读出的方向，
      这才是"逼 embedding 编码维度间关系"的严格版本。
    - 附带收益：训练中的重建损失就是线性探针误差，
      O=/D= 日志直接衡量 embedding 中可线性读出的原始/衍生信息量，
      不需要事后单独跑探针实验验证机制是否成立。
    - 下游 Transformer（6层 FFN 128→512→128）的解码能力远超线性读出器，
      只要线性层能读出的信息，backbone 一定能提取。
    解码器在预训练完成后丢弃，只保留 embedding 权重。
    """

    def __init__(self, input_dim=ModelConfig.INPUT_DIM, d_model=128,
                 expand_ratio=2, n_derived=0):
        super().__init__()
        self.embedding = KLineEmbedding(input_dim, d_model, expand_ratio)
        self.input_dim = input_dim
        self.n_derived = n_derived

        out_dim = input_dim + n_derived
        self.decoder = nn.Linear(d_model, out_dim, bias=True)

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim]
        Returns:
            z: [batch, d_model] 嵌入向量 S（用于 VISReg）
            recon: [batch, input_dim + n_derived] 重建=[原始16维, 衍生特征]
        """
        z = self.embedding(x)
        recon = self.decoder(z)
        return z, recon


# ==================== 数据收集 ====================

def collect_kline_data(train_stock_info, feature_normalizer=None, pool_cap=None):
    """
    从训练集中提取逐日K线特征向量（归一化后的2D池，不去重）

    衍生目标不在此处计算：调用方（pretrain/probe）先确定实际使用的子集，
    再用 build_derived_targets 对子集构建衍生目标，避免全池上
    衍生数组(4.4GB+1.1GB)与双倍 concat 峰值叠加导致 OOM。

    Args:
        train_stock_info: 训练集股票信息列表
        feature_normalizer: 特征归一化器
        pool_cap: 池大小上限（None=默认 MAX_SAMPLES×EPOCHS×5；
                  probe 等小规模用途可传小值，避免构建全池）

    Returns:
        kline_data: [M, input_dim] numpy array (完整池)
    """
    print("\n[数据收集] 提取逐日K线向量...")

    max_pool_size = (pool_cap if pool_cap is not None
                     else EmbeddingConfig.MAX_SAMPLES * EmbeddingConfig.EPOCHS * 5)

    # 传入 max_pool_size：precompute_training_pool 逐股票采样，
    # 避免拼接完整 [N,45,16] 数组(~10GB)+新数组(~10GB)=峰值~20GB OOM；
    # 采样后直接返回 [M,16] 2D数组，全在内存中完成，不写磁盘
    kline_data, _, _, _, _, _ = precompute_training_pool(
        train_stock_info, feature_normalizer, max_pool_size=max_pool_size
    )

    print(f"  K线池大小: {len(kline_data):,}")

    valid_mask = np.all(np.isfinite(kline_data), axis=1)
    kline_data = kline_data[valid_mask]
    print(f"  有效K线数: {len(kline_data):,}")

    print(f"  池大小: {len(kline_data):,} (不去重)")
    print(f"  原始特征范围:")
    for i, name in enumerate(['Open', 'High', 'Low', 'Close',
                               'VWAP', 'Volume', 'Exchange',
                               'MA5', 'MA10', 'MA20',
                               'DIF', 'DEA', 'MACD_Hist',
                               'MACD_Hist_Diff',
                               'BB_Upper', 'BB_Lower']):
        col = kline_data[:, i]
        print(f"    {name:>8s}: [{col.min():.4f}, {col.max():.4f}]  "
              f"μ={col.mean():.4f}  σ={col.std():.4f}")

    return kline_data


def sample_diverse_batch(pool, batch_size, precision,
                         oversample_factor=EmbeddingConfig.BATCH_DEDUP_OVERSAMPLE):
    """
    从池中采样一个多样性有保证的 batch

    1. 过采样 oversample_factor × batch_size 条 K 线
    2. round 到 precision 位小数后去重
    3. 取前 batch_size 条（保留原始精度数值）
    """
    n_oversample = int(batch_size * oversample_factor)
    n_oversample = min(n_oversample, len(pool))

    # replace=True: 后续会去重，允许少量重复索引
    # replace=False 在大池子上会触发全排列，极慢
    indices = np.random.choice(len(pool), n_oversample, replace=True)
    candidates = pool[indices]

    rounded = np.round(candidates, precision)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    deduped = candidates[unique_idx]

    n_deduped = len(deduped)
    if n_deduped >= batch_size:
        return deduped[:batch_size], n_oversample, n_deduped
    else:
        # 极端情况：去重后不够，补采凑满
        extra_needed = batch_size - n_deduped
        extra_idx = np.random.choice(len(pool), extra_needed, replace=len(pool) < extra_needed)
        return np.vstack([deduped, pool[extra_idx]])[:batch_size], n_oversample, n_deduped


# ==================== 学习率调度器 ====================

class WarmupCosineScheduler:
    """Warmup + 余弦退火调度器"""

    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
        # 第一个epoch使用warmup起始lr，避免随机初始化+全LR的不稳定更新
        warmup_start_lr = self.base_lrs[0] / max(1, warmup_epochs)
        for group in optimizer.param_groups:
            group['lr'] = warmup_start_lr

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups,
                                         self.base_lrs):
            param_group['lr'] = lr

    def _get_lr(self):
        if self.current_epoch < self.warmup_epochs:
            return self.base_lrs[0] * (self.current_epoch + 1) / self.warmup_epochs
        progress = (self.current_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs)
        return self.eta_min + 0.5 * (self.base_lrs[0] - self.eta_min) * \
               (1 + math.cos(math.pi * progress))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ==================== 保存函数 ====================

def save_pretrained_embedding(embedding, path, metrics=None, decoder=None):
    """
    保存预训练 embedding 权重（及可选解码器）

    格式与 StockTransformer 的 embed_proj / embed_mlp 直接兼容。
    传入 decoder 时同时保存解码器权重，供重建可视化等用途。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'embed_proj_weight': embedding.embed_proj.weight.data.cpu(),
        'embed_mlp_0_weight': embedding.embed_mlp[0].weight.data.cpu(),
        'embed_mlp_2_weight': embedding.embed_mlp[2].weight.data.cpu(),
        'input_dim': embedding.embed_proj.in_features,
        'd_model': embedding.embed_proj.out_features,
        'expand_ratio': embedding.embed_mlp[0].out_features // embedding.embed_proj.out_features,
        'config': {
            'epochs': EmbeddingConfig.EPOCHS,
            'batch_size': EmbeddingConfig.BATCH_SIZE,
            'learning_rate': EmbeddingConfig.LEARNING_RATE,
            'visreg_weight': EmbeddingConfig.VISREG_WEIGHT,
        },
    }
    if metrics:
        checkpoint['metrics'] = metrics
    if decoder is not None:
        # MLP解码器: 保存所有层权重
        checkpoint['decoder_state_dict'] = decoder.state_dict()

    torch.save(checkpoint, path)
    print(f"  Embedding权重已保存: {path}"
          f"{' (含解码器)' if decoder is not None else ''}")


def measure_embedding_std(checkpoint_path, kline_pool, device, n_probe=10000):
    """
    测量已保存的 embedding 权重在真实归一化K线上的输出std

    从磁盘加载权重（save/load round-trip 验证），用真实K线池抽样探测，
    反映 embedding 在真实数据分布上是否达成 VISReg 的 TARGET_STD 目标。

    Args:
        checkpoint_path: .pth 权重文件路径
        kline_pool: [M, input_dim] 真实归一化K线池（numpy）
        device: 计算设备
        n_probe: 探测样本数（默认 10000）

    Returns:
        std: 输出标准差（float）；文件不存在时返回 None
    """
    if not os.path.exists(checkpoint_path):
        print(f"  ✗ 权重文件不存在: {checkpoint_path}")
        return None

    print(f"\n[输出std验证] 加载权重: {checkpoint_path}")
    with torch.no_grad():
        # 从池中抽 n_probe 条真实归一化K线作为探测输入
        # （比合成 N(0,1) 噪声更贴近推理时的输入分布：真实特征间高度相关）
        n = min(n_probe, len(kline_pool))
        probe_idx = np.random.choice(len(kline_pool), n, replace=False)
        test_input = torch.tensor(kline_pool[probe_idx], dtype=torch.float32).to(device)

        # 从磁盘加载权重，新建模型灌入（验证落盘文件可正确还原）
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        tmp = KLineEmbedding(ModelConfig.INPUT_DIM, ModelConfig.D_MODEL).to(device)
        tmp.embed_proj.weight.data.copy_(ckpt['embed_proj_weight'])
        tmp.embed_mlp[0].weight.data.copy_(ckpt['embed_mlp_0_weight'])
        tmp.embed_mlp[2].weight.data.copy_(ckpt['embed_mlp_2_weight'])
        tmp.eval()
        z = tmp(test_input)
        std = z.std().item()
        target = EmbeddingConfig.TARGET_STD
        print(f"  {os.path.basename(checkpoint_path)}: 输出std = {std:.4f}  "
              f"(目标 {target}, 基于真实K线 {n} 条)")
    return std


# ==================== 主训练函数 ====================

def pretrain(train_stock_info, feature_normalizer=None, device=None,
             epochs=None, batch_size=None, lr=None):
    """
    Embedding层训练-主函数
    """
    epochs = epochs or EmbeddingConfig.EPOCHS
    batch_size = batch_size if batch_size is not None else EmbeddingConfig.BATCH_SIZE
    lr = lr if lr is not None else EmbeddingConfig.LEARNING_RATE

    # 1. 收集K线数据（不去重，保留完整池）
    kline_pool = collect_kline_data(train_stock_info, feature_normalizer)
    pool_size = len(kline_pool)

    # 2. 预采样 MAX_SAMPLES * EPOCHS 条，供所有 epoch 使用
    samples_per_epoch = EmbeddingConfig.MAX_SAMPLES
    total_needed = samples_per_epoch * epochs
    precision = EmbeddingConfig.DEDUP_PRECISION
    oversample = int(EmbeddingConfig.BATCH_DEDUP_OVERSAMPLE)
    loader_batch_size = batch_size * oversample

    if pool_size >= total_needed * oversample:
        indices = np.random.choice(pool_size, total_needed * oversample, replace=False)
        print(f"\n[数据] 池中有 {pool_size:,} 条K线，"
              f"无重复采样 {total_needed * oversample:,} 条")
    else:
        indices = np.random.choice(pool_size, total_needed * oversample, replace=True)
        repeat_ratio = total_needed * oversample / pool_size
        print(f"\n[数据] 池中有 {pool_size:,} 条K线，"
              f"需 {total_needed * oversample:,} 条（平均重复 {repeat_ratio:.1f} 次）")

    pre_sampled = kline_pool[indices]
    # 释放全池(~6.4GB)：衍生目标只对实际训练的预采样子集按行独立计算，
    # 子集统计与全池在千万级样本下已收敛等价，省去全池衍生数组的常驻与concat峰值
    del kline_pool

    # 衍生目标（预采样后构建，与 pre_sampled 行对齐）
    pre_sampled_derived, pre_sampled_mask = build_derived_targets(
        pre_sampled, feature_normalizer)

    epoch_data = pre_sampled.reshape(epochs, samples_per_epoch * oversample, -1)
    epoch_derived = pre_sampled_derived.reshape(epochs, samples_per_epoch * oversample, -1)
    epoch_mask = pre_sampled_mask.reshape(epochs, samples_per_epoch * oversample, -1)
    print(f"  分配: {epochs} 个 epoch × {samples_per_epoch * oversample:,} 条/epoch")
    print(f"  batch 内去重: precision={precision}, "
          f"oversample={oversample}x → DataLoader batch={loader_batch_size}")

    # 3. 创建模型
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES
    model = PretrainModel(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        n_derived=n_derived,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embedding.parameters())
    decoder_params = total_params - embed_params
    print(f"\n[模型] 参数量: Embedding={embed_params:,}  "
          f"Decoder={decoder_params:,}  总计={total_params:,}")

    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=EmbeddingConfig.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=EmbeddingConfig.WARMUP_EPOCHS,
        total_epochs=epochs,
        eta_min=EmbeddingConfig.COSINE_ETA_MIN
    )

    # 损失权重 (论文式9/7): loss = (1-λ)·L_pred + λ·L_reg
    # L_reg = w_scale·L_scale + w_shape·L_shape + w_center·L_center
    # 三分量均为均值形式、天然 O(1)，与 Recon MSE 同量级，无需 SIGReg 时代的 EMA 归一化。
    visreg_weight = EmbeddingConfig.VISREG_WEIGHT
    target_std = EmbeddingConfig.TARGET_STD
    derived_weight = EmbeddingConfig.DERIVED_WEIGHT  # 衍生重建权重(标准化后等权=1.0)
    input_dim = ModelConfig.INPUT_DIM                # 原始16维，recon前 input_dim 列为原始重建
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES

    # VISReg 几何正则：尺度/形状(SWD)/中心化 三项解耦
    visreg_loss_fn = VISRegLoss(
        num_slices=EmbeddingConfig.VISREG_NUM_SLICES,
        w_scale=EmbeddingConfig.VISREG_W_SCALE,
        w_shape=EmbeddingConfig.VISREG_W_SHAPE,
        w_center=EmbeddingConfig.VISREG_W_CENTER,
    ).to(device)

    # 5. 训练循环
    print(f"\n{'='*60}")
    amp_str = "BF16混合精度" if TrainingConfig.USE_AMP and device.type == 'cuda' else "FP32精度"
    print(f"开始 Embedding 预训练")
    print(f"  轮数={epochs}  batch={batch_size}  lr={lr}")
    print(f"  精度={amp_str}  设备={device}")
    print(f"  损失公式: {visreg_weight:.2f}·VISReg + {1 - visreg_weight:.2f}·Recon  (论文原版)")
    print(f"  解码器=线性探针, Recon=原始16维 + {derived_weight:.2f}·{n_derived}衍生(掩码MSE)")
    print(f"{'='*60}")

    output_dir = EmbeddingConfig.OUTPUT_DIR

    amp_ctx = _get_amp_context(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_visreg_w = 0.0  # λ·VISReg 加权贡献
        epoch_recon_w = 0.0   # (1-λ)·Recon 加权贡献
        epoch_visreg_raw = 0.0  # VISReg 原始值
        epoch_recon_raw = 0.0   # Recon 原始值(原始重建+衍生重建之和)
        epoch_recon_orig_raw = 0.0    # 原始16维重建原始值
        epoch_recon_derived_raw = 0.0 # 衍生重建原始值(掩码后逐特征均值)
        epoch_dedup_total = 0   # batch内去重去掉的条数
        n_batches = 0

        t0 = time.time()

        dataset = TensorDataset(
            torch.tensor(epoch_data[epoch - 1], dtype=torch.float32),
            torch.tensor(epoch_derived[epoch - 1], dtype=torch.float32),
            torch.tensor(epoch_mask[epoch - 1], dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

        for (raw_batch, derived_batch, mask_batch) in loader:
            # batch 内去重：round → unique → 取前 batch_size
            # 去重基于原始K线，衍生特征/掩码按相同索引同步保留
            raw_np = raw_batch.numpy()
            derived_np = derived_batch.numpy()
            mask_np = mask_batch.numpy()
            rounded = np.round(raw_np, precision)
            _, unique_idx = np.unique(rounded, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            deduped = raw_np[unique_idx]
            deduped_derived = derived_np[unique_idx]
            deduped_mask = mask_np[unique_idx]
            epoch_dedup_total += len(raw_np) - len(deduped)

            # 取 batch_size 条（不够则全部用上）
            batch_x = torch.tensor(
                deduped[:batch_size], dtype=torch.float32).to(device)
            batch_d = torch.tensor(
                deduped_derived[:batch_size], dtype=torch.float32).to(device)
            batch_m = torch.tensor(
                deduped_mask[:batch_size], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            with amp_ctx:
                z, recon = model(batch_x)

                # VISReg: 缩放到目标 std，尺度项 target std=1，形状项 SWD 对齐高斯
                loss_visreg = visreg_loss_fn(z / target_std)

                # 重建损失: 分离原始重建与衍生重建
                # 衍生项逼 embedding 编码跨维度关系，而非逐维独立存储；
                # 解码器是线性层，衍生目标(非线性函数)无法由解码器自行计算，
                # 结构必须由 embedding 内部算好 → 重建损失即线性探针误差
                # 衍生目标已剪尾+z-score 标准化，与原始16维量级一致，等权合理；
                # derived_weight 可调，防止辅助任务(衍生)主导主任务(原始重建)
                loss_recon_orig = F.mse_loss(recon[:, :input_dim], batch_x)
                # 掩码 MSE：仅有效样本参与，逐特征归一(每个衍生目标等权)，
                # 掩码剔除归一化伪影样本(span≤0、影线<0)，不学伪影当信号
                diff2 = (recon[:, input_dim:] - batch_d) ** 2
                num = (diff2 * batch_m).sum(dim=0)
                den = batch_m.sum(dim=0) + 1e-8
                loss_recon_derived = (num / den).mean()
                loss_recon = loss_recon_orig + derived_weight * loss_recon_derived

                # 论文原版加权: (1-λ)·Recon + λ·VISReg
                loss = (1 - visreg_weight) * loss_recon + visreg_weight * loss_visreg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 统计
            raw_visreg = loss_visreg.item()
            raw_recon = loss_recon.item()
            visreg_w = visreg_weight * raw_visreg
            recon_w = (1 - visreg_weight) * raw_recon
            epoch_loss += visreg_w + recon_w
            epoch_visreg_w += visreg_w
            epoch_recon_w += recon_w
            epoch_visreg_raw += raw_visreg
            epoch_recon_raw += raw_recon
            epoch_recon_orig_raw += loss_recon_orig.item()
            epoch_recon_derived_raw += loss_recon_derived.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_visreg_w = epoch_visreg_w / n_batches
        avg_recon_w = epoch_recon_w / n_batches
        avg_visreg_raw = epoch_visreg_raw / n_batches
        avg_recon_raw = epoch_recon_raw / n_batches
        avg_recon_orig = epoch_recon_orig_raw / n_batches
        avg_recon_derived = epoch_recon_derived_raw / n_batches
        current_lr = scheduler.get_lr()

        # 打印日志 （O=/D= 分别为原始16维/衍生重建原始值，D= 即线性探针误差，
        # 直接衡量 embedding 中可线性读出的跨维结构量，观察衍生目标是否被学到）
        avg_dedup = epoch_dedup_total / max(1, n_batches)
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f} ({avg_visreg_w:.4f}·VISReg + {avg_recon_w:.4f}·Recon)  "
              f"VISReg={avg_visreg_raw:.4f}  Recon={avg_recon_raw:.4f} "
              f"(O={avg_recon_orig:.4f}/D={avg_recon_derived:.4f})  "
              f"LR={current_lr:.6f}  "
              f"Dedup={avg_dedup:.0f}/batch  "
              f"Time={elapsed:.1f}s")

    # 训练结束后保存最后一轮模型到磁盘
    # （不按 loss 挑选：预训练接近纯约束收敛，各轮 loss 差异小，
    #   直接保留最后一轮收敛后的权重）
    last_metrics = {
        'epoch': epochs,
        'loss': avg_loss,
        'visreg_weighted': avg_visreg_w,
        'recon_weighted': avg_recon_w,
    }
    best_path = os.path.join(output_dir, 'best_embedding.pth')
    save_pretrained_embedding(model.embedding, best_path,
                              metrics=last_metrics, decoder=model.decoder)

    # 测量保存文件的输出std（pre_sampled 为池的随机子集，分布等价）
    measure_embedding_std(best_path, pre_sampled, device)

    print(f"\n预训练完成！最后一轮 Loss={avg_loss:.4f} (第{epochs}轮)")
    print(f"权重保存位置: {output_dir}")

    return model.embedding


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='EquiNet Embedding层训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'预训练轮数 (默认 {EmbeddingConfig.EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'批大小 (默认 {EmbeddingConfig.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'学习率 (默认 {EmbeddingConfig.LEARNING_RATE})')
    parser.add_argument('--visreg-weight', type=float, default=None,
                        help=f'VISReg损失权重λ (默认 {EmbeddingConfig.VISREG_WEIGHT})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'输出目录')
    parser.add_argument('--test', action='store_true',
                        help='只测试已保存 embedding 权重的输出std，不训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'--test 模式加载的权重文件 (默认 <output-dir>/best_embedding.pth)')

    args = parser.parse_args()

    # 覆盖配置
    if args.epochs:
        EmbeddingConfig.EPOCHS = args.epochs
    if args.batch_size:
        EmbeddingConfig.BATCH_SIZE = args.batch_size
    if args.visreg_weight is not None:
        EmbeddingConfig.VISREG_WEIGHT = args.visreg_weight
    if args.output_dir:
        EmbeddingConfig.OUTPUT_DIR = args.output_dir

    device = DeviceConfig.get_device()

    # 加载数据
    print("[步骤1] 加载训练数据...")
    train_stock_info, _, _ = load_and_preprocess_data()

    print("\n[步骤2] 加载特征归一化器...")
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print("  归一化器不存在，先运行 python src/data.py 创建")
        sys.exit(1)

    # 测试模式：只测已保存权重的输出std，跳过训练
    if args.test:
        ckpt_path = (args.checkpoint
                     or os.path.join(EmbeddingConfig.OUTPUT_DIR, 'best_embedding.pth'))
        print(f"\n[测试模式] 测量已训练 embedding 的输出std")
        kline_pool = collect_kline_data(train_stock_info, feature_normalizer,
                                        pool_cap=1_000_000)
        measure_embedding_std(ckpt_path, kline_pool, device)
        return

    # 预训练
    print("\n[步骤3] 开始 Embedding 预训练...")
    pretrain(
        train_stock_info=train_stock_info,
        feature_normalizer=feature_normalizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
