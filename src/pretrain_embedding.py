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
3. 掩码对比损失（几何轴，SCARF式）：同一条K线的两个不同掩码视图经
   projector 投影后互为正对、batch内其他K线为负对（InfoNCE）。
   1/2 管的是"每条K线编码了什么信息"，3 管的是"样本之间怎么排布"——
   下游 attention 消费的是 z 的点积相似度，该几何必须显式训练。
   对比施加在 projector 输出上（SimCLR/DINOv2 路线）：projector 吸收
   "收紧"压力，z 保住 O/D 线性可读性的同时继承光滑相似度结构；
   projector 训练后丢弃。CONTRASTIVE_ENABLED=False 可退回纯探针模式。
4. 分类头（粗粒度类别监督，Kronos 分层监督的"粗"端）：符号类衍生目标
   本质是类别 {-1,0,1}，被 MSE 当连续数训时存在"猜 0.7 也算对"的盲区，
   模型没有动力让同类K线在 embedding 空间靠拢；交叉熵没有"差不多"，
   必须选边站，逼 embedding 出现可线性读出的类别可分结构
   （同类聚拢/异类分开）——下游 attention 消费点积相似度，此几何直接可用。
   三类=跌/平/涨，"平"为 |v|≤ε 的区间（ε 由池上分位数预计算，CLS_FLAT_PCT），
   非"恰好=0"（连续分布上测度为零、平类会退化）。
   头为单层线性（与线性解码器同哲学：非线性头会把分类压力吞掉），
   标签由归一化输入即时计算（与符号族衍生特征同定义、符号严格一致、无噪声），
   训练后与 decoder/projector 一起丢弃。CLS_ENABLED=False 可退回纯回归。

用法：
  python src/pretrain_embedding.py
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


# ==================== 分类头（粗粒度类别监督） ====================

CLS_HEAD_SPEC = {
    # name: n_classes —— 符号类衍生目标的分类版本（Kronos 分层监督的"粗"端）。
    # 与 compute_derived_features 符号族同列索引、同归一化空间；平带外标签
    # 符号与该族衍生特征严格一致（sign(a·b)≡sign(a)·sign(b)，含零点恒等），
    # 平带内（|v|≤ε，约 CLS_FLAT_PCT% 样本）标签为"平"≠衍生符号±1——
    # 刻意的粗粒度化：分类管"粗"（类别可分几何），MSE 回归管"细"（幅度）。
    # vol_price_align 的 v = sign(close-open)·amount 与衍生特征数值恒等，
    # 平带 ε 恰落在衍生特征自身近零集（|amount| 小的弱量日）上；
    # 不用 (close-open)·amount 并非符号原因（由上述恒等式，amount<0 时
    # 两者符号同步翻转，无矛盾），而是幅度原因：后者混入 |close-open|，
    # 平带会漂移到"小实体日"，与衍生特征近零集错位。
    # 标签由归一化输入即时计算：输入的可确定函数 → 无噪声，
    # 与训练 batch 天然对齐 → 零存储、零数据管线改动。
    # 三类 = 跌/平/涨，其中"平"不是"恰好=0"而是 |v|≤ε 的区间
    # （ε 由 compute_cls_stats 在池上按分位数预计算，见 CLS_FLAT_PCT）。
    'direction':       3,   # v = close-open                 收/开相对强弱
    'vol_price_align': 3,   # v = sign(close-open)·amount    量价配合
    'momentum_accel':  3,   # v = macd_h·macd_hd           MACD柱同向/反向
    'vwap_side':       3,   # v = close-vwap                 收在均价上/下
}

CLS_SHORT_NAMES = {
    'direction': 'dir', 'vol_price_align': 'vpa',
    'momentum_accel': 'mom', 'vwap_side': 'vws',
}


def _cls_values(x):
    """
    各分类头的原始判别值 v：分类标签 = sign(v)（±ε 平带），四个头的共性入口。
    归一化空间、与 compute_derived_features 符号族同列索引（16维索引见该函数）；
    平带外 sign(v) 与该族衍生特征符号严格一致（见 CLS_HEAD_SPEC 注释）。
    numpy 与 torch 张量均可（切片+算术+sign 均为两者共有算子）。

    逐头惰性生成（yield）：池级调用（40M 行）任一时刻只物化一个头的数组，
    避免 4 头同时常驻（~640MB）——本项目 OOM 纪律，同 compute_derived_features
    分块哲学；batch 级调用（数千行）开销可忽略，同一入口两用。

    Yields:
        (name, v)：v 为 [N] 判别值，以 0 为对称中心
    """
    sign = torch.sign if torch.is_tensor(x) else np.sign
    yield 'direction', x[:, 3] - x[:, 0]
    yield 'vol_price_align', sign(x[:, 3] - x[:, 0]) * x[:, 5]
    yield 'momentum_accel', x[:, 12] * x[:, 13]
    yield 'vwap_side', x[:, 3] - x[:, 4]


def compute_cls_stats(pool, pct):
    """
    在池上预计算每个分类头的"平"阈值 ε 与三类占比（numpy，零 torch 开销）

    ε = |v| 的 pct 分位数：|v| ≤ ε 判"平"，保证三类都非空；
    逐头独立 → 不引入人工量纲（各 v 的量级差异大：
    direction≈O(0.1~1) 而 vol_price_align=sign(close-open)·amount 可达 O(几)）。
    池用实际参与训练的 pre_sampled 子集，与 batch 分布一致、确定性。

    Args:
        pool: [M, input_dim] 归一化K线池（numpy）
        pct: 平类占比（百分数，如 20.0）
    Returns:
        dict[name -> (eps, (跌占比, 平占比, 涨占比))]
    """
    out = {}
    for name, v in _cls_values(pool):
        abs_v = np.abs(v)  # 复用：分位数与平类计数共用一份 |v|（省一次全量拷贝）
        eps = float(np.percentile(abs_v, pct))
        n_flat = int((abs_v <= eps).sum())
        n_neg = int((v < -eps).sum())
        n_pos = int((v > eps).sum())
        total = n_flat + n_neg + n_pos
        out[name] = (eps, (n_neg / total, n_flat / total, n_pos / total))
    return out


def compute_cls_targets(x, cls_eps):
    """
    从归一化K线即时计算分类标签 {0:跌, 1:平, 2:涨}

    平类判据为区间 |v| ≤ ε（ε 来自池上分位数 compute_cls_stats），
    与"恰好=0"的 sign 不同：三类非空，模型无法偷懒只学二分类。
    全 torch 算子，batch 内一步得出，无 numpy 往返。

    Args:
        x: [B, input_dim] 归一化K线（与训练 batch 相同）
        cls_eps: dict[name -> ε 阈值]（compute_cls_stats 输出）
    Returns:
        dict[name -> [B] torch.long, 值域 {0,1,2}]
    """
    neg, flat, pos = (torch.tensor(i, device=x.device) for i in (0, 1, 2))
    targets = {}
    for name, v in _cls_values(x):
        eps = cls_eps[name]
        targets[name] = torch.where(
            v > eps, pos, torch.where(v < -eps, neg, flat))
    return targets


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


# ==================== 掩码对比学习（几何轴） ====================
def make_masked_view(x, k_min, k_max):
    """
    SCARF式特征掩码视图：每行随机掩 k∈[k_min, k_max] 个特征，
    被掩位置的值替换为 batch 内随机样本的同列值（分布内重采样）。

    为什么重采样而非置零：置零=均值插补（归一化后均值0），"被掩"痕迹
    可被模型轻易识破，视图退化为轻度噪声，对比任务过简单；
    重采样值在分布内合理，模型必须依赖跨维相关性才能区分真值与伪装
    ——这正是要 embedding 学的东西。两视图掩码独立采样，被掩子集不同，
    InfoNCE 拉近两视图 = 强迫从剩余特征推断被掩特征（度量形式的掩码预测）。

    batch 内去重（training loop, precision=1）保证无重复K线，
    否则相同样本互为负对会强迫同输入映射到远点，毒化几何。

    Args:
        x: [B, D] tensor（device 上）
    Returns:
        [B, D] 掩码+重采样后的视图
    """
    B, D = x.shape
    # 每行独立的随机 k
    k = torch.randint(k_min, k_max + 1, (B,), device=x.device)
    # 每行恰好 k 个 True：阈值取该行第 k 大的随机值（升序 sort 用 D-k 索引），
    # r >= 第k大 → 恰好 k 个位置入选（随机子集，等价于均匀抽 k 列）
    r = torch.rand(B, D, device=x.device)
    thresh = r.sort(dim=1).values.gather(1, (D - k).unsqueeze(1))
    mask = r >= thresh
    # 被掩位置的替换值来自 batch 内同列随机样本（分布内重采样）。
    # 用 roll(1) 而非 randperm：roll 严格无固定点（randperm 有 1/B 概率
    # 抽到自己→整行替换成原值、该样本视图失效）；batch 本身经 DataLoader
    # shuffle，前驱行即均匀随机样本，统计上等价
    resample = torch.roll(x, shifts=1, dims=0)
    return torch.where(mask, resample, x)


def info_nce_loss(p1, p2, tau):
    """
    对称 NT-Xent InfoNCE：p1/p2 为同 batch 两个视图的 L2 归一化投影 [B, d]。

    正对 = 同一样本的两个视图 (i ↔ i+B)，负对 = 2B-2 个其他样本。
    logits 必须在 fp32 计算（τ=0.2 除法后 bf16 精度不足）。

    同时返回 Wang & Isola (ICML 2020) 几何度量用于监控：
    - alignment: 正对欧氏距离²（越小说明正对拉得越近）
    - uniformity: log E[exp(-2·dist²)]（越小说明样本在超球面上铺得越均匀）
    两者同时下降 = 几何在改善（只降一个可能是坍塌前兆）。

    Returns:
        (loss, alignment, uniformity) 后两者为 python float
    """
    B = p1.shape[0]
    z = torch.cat([p1, p2], dim=0)                      # [2B, d]
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(float('-inf'))                   # 排除自身
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)                   # i ↔ i+B 互为正对
    loss = F.cross_entropy(sim, targets)
    with torch.no_grad():
        alignment = (p1 - p2).pow(2).sum(dim=1).mean()
        # uniformity 复用已算好的 sim，不用 torch.pdist：
        # 输入已 L2 归一化 ⇒ d² = 2 - 2·dot = 2 - 2τ·sim
        # ⇒ exp(-2d²) = exp(-4 + 4τ·sim)，逐对等价（数值验证 diff=0）
        # pdist 的 CUDA kernel 对 N 个向量需逐对算 N(N-1)/2 个距离
        # （B=512 时 ~50 万对），实测 ~300ms/batch（曾占整个 epoch
        # 耗时的 ~90%，纯监控指标不值得）；对角 exp(-inf)=0 自动剔除，
        # 除以 N(N-1) 与 pdist 的 i<j 无序对均值严格一致
        # （对称矩阵，有序对均值相同）
        N = z.shape[0]
        s = sim.mul(4 * tau).exp_().sum()
        uniformity = torch.log(
            s / (N * (N - 1)) * math.exp(-4.0) + 1e-12)
    return loss, alignment.item(), uniformity.item()


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
    Embedding层训练模型 = KLineEmbedding + 线性解码器 + 对比 projector + 分类头

    X ──────────→ Embedding → S ─┬─ Linear → Y            (线性重建探针)
                                 ├─ Linear×4 → 类别 (分类头, 粗粒度, 训练后丢弃)
                                 ├─ VISReg(S)
                                 └─→ (丢弃)
    X 掩码视图1 ┐
               ├→ Embedding → S → projector g(S) → L2归一化 ─┐
    X 掩码视图2 ┘                                              ┴─ InfoNCE

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

    分类头（粗粒度类别监督，Kronos 分层监督的"粗"端）：
    - 符号类衍生目标本质是类别 {-1,0,1}，被 MSE 当连续数训时"猜 0.7 也算对"，
      模型没有动力让同类K线在 embedding 空间靠拢；
    - 交叉熵没有"差不多"，必须选边站，逼 embedding 出现可线性读出的
      类别可分结构（同类聚拢/异类分开）——下游 attention 消费点积相似度，
      此几何直接可用；
    - 头必须单层线性（同解码器哲学：MLP 头自己就把类别算出来了，
      压力被吞掉）；标签由归一化输入即时计算（compute_cls_targets），无噪声；
    - 注意与 VISReg 相克：可分性在 z 分布上产生多峰，高斯形状项拉单峰，
      CLS_WEIGHT 别开大；训练后丢弃，与 decoder/projector 相同。

    projector（对比分支，SimCLR/DINOv2 路线）：InfoNCE 不直接作用在 S 上
    （会把"对区分样本身份无贡献"的方差压掉，冲掉 O/D 线性可读结构），
    而是作用在 g(S) 上——projector 吸收几何收紧压力，S 继承温和的部分。
    decoder、projector 与分类头均在预训练完成后丢弃，只保留 embedding 权重。
    """

    def __init__(self, input_dim=ModelConfig.INPUT_DIM, d_model=128,
                 expand_ratio=2, n_derived=0, cls_heads=None):
        super().__init__()
        self.embedding = KLineEmbedding(input_dim, d_model, expand_ratio)
        self.input_dim = input_dim
        self.n_derived = n_derived

        out_dim = input_dim + n_derived
        self.decoder = nn.Linear(d_model, out_dim, bias=True)

        # 分类头（线性探针）：每头一个 Linear(d_model, n_classes)，
        # bias=0 初始化使初始 logits≈0 → 初始 CE≈ln(C)，与回归头互不干扰
        self.cls_heads = None
        if cls_heads:
            self.cls_heads = nn.ModuleDict({
                name: nn.Linear(d_model, n_classes, bias=True)
                for name, n_classes in cls_heads.items()
            })
            for head in self.cls_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

        # 对比 projector: 128→256→GELU→128（与 embed_mlp 同构，初始化同推导）
        # MLP分支增益=1（σ1·σ2·0.588·√(d·h) = 1），输出 L2 归一化后进入 InfoNCE
        hidden_dim = d_model * expand_ratio
        self.projector = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False),
        )
        nn.init.normal_(self.projector[0].weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.projector[2].weight,
                        std=1.0 / (0.588 * math.sqrt(hidden_dim)))

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

    def cls_forward(self, z):
        """
        分类头前向（与 forward 分离：探针脚本只调 forward 不受影响）。

        Args:
            z: [batch, d_model]（fp32；AMP 下 bf16 需先 .float()）
        Returns:
            dict[name -> [batch, n_classes] logits]
        """
        if self.cls_heads is None:
            raise RuntimeError("PretrainModel 未启用分类头 (cls_heads=None)")
        return {name: head(z) for name, head in self.cls_heads.items()}


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

def pretrain(train_stock_info, feature_normalizer=None, device=None):
    """
    Embedding层训练-主函数（超参统一走 EmbeddingConfig，CLI 不再提供覆盖入口）
    """
    epochs = EmbeddingConfig.EPOCHS
    batch_size = EmbeddingConfig.BATCH_SIZE
    lr = EmbeddingConfig.LEARNING_RATE

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

    # 分类头（粗粒度类别监督）开关与权重：须在模型构造之前定义。
    # w_cls 独立于 (1-w_c) 预算（置零即退回纯回归监督）。
    use_cls = EmbeddingConfig.CLS_ENABLED and len(CLS_HEAD_SPEC) > 0
    w_cls = EmbeddingConfig.CLS_WEIGHT if use_cls else 0.0
    # "平"类阈值：池上分位数预计算（确定性、与 batch 分布一致）
    cls_eps = None
    cls_stats = None
    if use_cls:
        cls_stats = compute_cls_stats(pre_sampled, EmbeddingConfig.CLS_FLAT_PCT)
        cls_eps = {name: eps for name, (eps, _) in cls_stats.items()}

    # 3. 创建模型
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES
    model = PretrainModel(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        n_derived=n_derived,
        cls_heads=CLS_HEAD_SPEC if use_cls else None,
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

    # 掩码对比（几何轴）：w_c=InfoNCE 权重，剩余 (1-w_c) 按 λ 分配给 VISReg/Recon，
    # w_c=0 或 CONTRASTIVE_ENABLED=False 时公式严格退回旧版 (1-λ)·Recon + λ·VISReg
    use_contrastive = EmbeddingConfig.CONTRASTIVE_ENABLED
    w_c = EmbeddingConfig.CONTRASTIVE_WEIGHT if use_contrastive else 0.0
    w_v = (1.0 - w_c) * visreg_weight
    w_r = (1.0 - w_c) * (1.0 - visreg_weight)
    ncl_tau = EmbeddingConfig.CONTRASTIVE_TAU
    mask_k_min = EmbeddingConfig.MASK_K_MIN
    mask_k_max = EmbeddingConfig.MASK_K_MAX

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
    if use_contrastive:
        formula = (f"{w_v:.2f}·VISReg + {w_r:.2f}·Recon + {w_c:.2f}·InfoNCE"
                   f"(掩码对比, τ={ncl_tau}, k∈[{mask_k_min},{mask_k_max}], projector头)")
    else:
        formula = (f"{visreg_weight:.2f}·VISReg + {1 - visreg_weight:.2f}·Recon  (论文原版)")
    if use_cls:
        formula += f" + {w_cls:.2f}·CLS({','.join(CLS_SHORT_NAMES.values())})"
    print(f"  损失公式: {formula}")
    print(f"  解码器=线性探针, Recon=原始16维 + {derived_weight:.2f}·{n_derived}衍生(掩码MSE)")
    if use_cls:
        print(f"  分类头=线性探针(粗粒度, 跌/平/涨):")
        for name in CLS_HEAD_SPEC:
            eps, shares = cls_stats[name]
            print(f"    {name:>16s}: ε={eps:.4f}  "
                  f"池上占比 跌{shares[0]*100:.0f}%/平{shares[1]*100:.0f}%/涨{shares[2]*100:.0f}%")
        print(f"  注: 分类可分性在 z 分布上产生多峰, 与 VISReg 高斯形状项相克, "
              f"w_cls={w_cls} 不宜开大; 标签由归一化输入即时计算, 无噪声")
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
        epoch_ncl_raw = 0.0      # InfoNCE 原始值
        epoch_ncl_w = 0.0        # w_c·InfoNCE 加权贡献
        epoch_align = 0.0        # alignment: 正对距离²（越低越好）
        epoch_uniform = 0.0      # uniformity: 超球面均匀度（越低越好）
        epoch_cls_raw = 0.0      # CLS 原始值（各头 CE 平均）
        epoch_cls_w = 0.0        # w_cls·CLS 加权贡献
        epoch_cls_head_raw = [0.0] * len(CLS_HEAD_SPEC)  # 逐头 CE
        epoch_cls_head_acc = [0.0] * len(CLS_HEAD_SPEC)  # 逐头准确率
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
            # 掩码视图（几何轴）：两视图独立采样不同掩码子集，
            # 前向走 AMP（小 MLP 开销可忽略），InfoNCE 的 logits 在 AMP 外算 fp32
            if use_contrastive:
                v1 = make_masked_view(batch_x, mask_k_min, mask_k_max)
                v2 = make_masked_view(batch_x, mask_k_min, mask_k_max)

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

                # 对比分支前向在 AMP 内（小 MLP），投影输出转 fp32 后归一化
                if use_contrastive:
                    p1 = model.projector(model.embedding(v1))
                    p2 = model.projector(model.embedding(v2))
                # 加权合成（InfoNCE 的 fp32 计算放 AMP 外，避免 bf16 logits 精度不足）
                loss = w_v * loss_visreg + w_r * loss_recon

            if use_contrastive:
                p1 = F.normalize(p1.float(), dim=1)
                p2 = F.normalize(p2.float(), dim=1)
                loss_ncl, align_val, uniform_val = info_nce_loss(p1, p2, ncl_tau)
                loss = loss + w_c * loss_ncl

            # 分类头（粗粒度监督）：fp32 计算（CE 在 bf16 下精度不足，同 InfoNCE）。
            # 标签由归一化输入即时计算（compute_cls_targets），与 batch 天然对齐；
            # 梯度经线性头回流 embedding，逼 z 出现可线性读出的类别可分结构
            if use_cls:
                cls_logits = model.cls_forward(z.float())
                cls_targets = compute_cls_targets(batch_x, cls_eps)
                cls_losses = {
                    name: F.cross_entropy(cls_logits[name], cls_targets[name])
                    for name in CLS_HEAD_SPEC
                }
                loss_cls = torch.stack(list(cls_losses.values())).mean()
                loss = loss + w_cls * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 统计
            raw_visreg = loss_visreg.item()
            raw_recon = loss_recon.item()
            visreg_w = w_v * raw_visreg
            recon_w = w_r * raw_recon
            epoch_loss += visreg_w + recon_w
            epoch_visreg_w += visreg_w
            epoch_recon_w += recon_w
            epoch_visreg_raw += raw_visreg
            epoch_recon_raw += raw_recon
            epoch_recon_orig_raw += loss_recon_orig.item()
            epoch_recon_derived_raw += loss_recon_derived.item()
            if use_contrastive:
                ncl_w = w_c * loss_ncl.item()
                epoch_loss += ncl_w
                epoch_ncl_w += ncl_w
                epoch_ncl_raw += loss_ncl.item()
                epoch_align += align_val
                epoch_uniform += uniform_val
            if use_cls:
                cls_w = w_cls * loss_cls.item()
                epoch_loss += cls_w
                epoch_cls_w += cls_w
                epoch_cls_raw += loss_cls.item()
                for i, name in enumerate(CLS_HEAD_SPEC):
                    epoch_cls_head_raw[i] += cls_losses[name].item()
                    acc = (cls_logits[name].argmax(dim=1) == cls_targets[name])
                    epoch_cls_head_acc[i] += acc.float().mean().item()
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
        # 直接衡量 embedding 中可线性读出的跨维结构量，观察衍生目标是否被学到；
        # NCL=InfoNCE 原始值，A/U=alignment/uniformity 几何度量，两者同降=几何改善；
        # CLS=各头 CE 平均，逐头格式 CE/Acc——CE 从初始 ln(3)≈1.1 下降 + Acc 上升
        # = 类别可分结构正在被嵌入；Acc 受 label 噪声/可分离度上限约束，无需追满）
        avg_dedup = epoch_dedup_total / max(1, n_batches)
        weighted_str = f"{avg_visreg_w:.4f}·VISReg + {avg_recon_w:.4f}·Recon"
        ncl_str = ""
        if use_contrastive:
            weighted_str += f" + {epoch_ncl_w / n_batches:.4f}·NCL"
            ncl_str = (f"NCL={epoch_ncl_raw / n_batches:.4f} "
                       f"(A={epoch_align / n_batches:.3f}/U={epoch_uniform / n_batches:.3f})  ")
        cls_str = ""
        if use_cls:
            weighted_str += f" + {epoch_cls_w / n_batches:.4f}·CLS"
            cls_parts = []
            for i, name in enumerate(CLS_HEAD_SPEC):
                ce = epoch_cls_head_raw[i] / n_batches
                acc = epoch_cls_head_acc[i] / n_batches
                cls_parts.append(f"{CLS_SHORT_NAMES[name]}={ce:.3f}/{acc*100:.0f}%")
            cls_str = (f"CLS={epoch_cls_raw / n_batches:.4f} "
                       f"({' '.join(cls_parts)})  ")
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f} ({weighted_str})  "
              f"VISReg={avg_visreg_raw:.4f}  Recon={avg_recon_raw:.4f} "
              f"(O={avg_recon_orig:.4f}/D={avg_recon_derived:.4f})  "
              f"{ncl_str}"
              f"{cls_str}"
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
    if use_contrastive:
        last_metrics.update({
            'contrastive_weighted': epoch_ncl_w / n_batches,
            'contrastive_raw': epoch_ncl_raw / n_batches,
            'alignment': epoch_align / n_batches,
            'uniformity': epoch_uniform / n_batches,
        })
    if use_cls:
        last_metrics.update({
            'cls_weighted': epoch_cls_w / n_batches,
            'cls_raw': epoch_cls_raw / n_batches,
        })
        for i, name in enumerate(CLS_HEAD_SPEC):
            last_metrics[f'cls_ce_{name}'] = epoch_cls_head_raw[i] / n_batches
            last_metrics[f'cls_acc_{name}'] = epoch_cls_head_acc[i] / n_batches
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
    parser.add_argument('--test', action='store_true',
                        help='只测试已保存 embedding 权重的输出std，不训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'--test 模式加载的权重文件 (默认 <output-dir>/best_embedding.pth)')

    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
