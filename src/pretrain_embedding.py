"""
EquiNet Embedding层训练脚本

通过 VISReg 几何正则 + 重建约束，预训练 FFN-Embedding 层，
使其成为一个固定的、具有几何保证的K线特征提取器：

1. VISReg 几何正则 (Wu, Balestriero & Levine, 2026; arXiv:2606.02572)：
   约束嵌入分布趋向各向同性高斯 N(0, target_std²)
2. 线性解码器重建损失：确保嵌入向量足以恢复原始19维 + 11个衍生特征
   刻意不用非线性解码器——非线性层自己就能完成乘积/绝对值/除法等跨维运算，
   会把"逼融合"的压力吞掉，embedding 逐维线性拷贝也能过关；
   线性解码器算不了任何非线性运算，跨维结构必须由 embedding 内部预先算好
   并编码成可线性读出的方向。此时重建损失 = 线性探针误差，
   O=/D= 日志直接衡量 embedding 中可线性读出的原始/衍生信息量。
3. 核加权类感知对比损失（几何轴主损失，SupCon；Khosla et al.,
   NeurIPS 2020）：正对权重 = 多头一致度核 K_ij（全粗+细分头中标签
   一致的比例 ∈ {0, 1/H, ..., 1}），施加在 projector 输出 g(z) 上。
   1/2 管的是"每条K线编码了什么信息"，3 管的是"样本之间怎么排布"——
   下游 attention 消费的是 z 的点积相似度，该几何必须显式训练。
   锚点设计：单标签（如仅 vpa 3类）锚点贫乏——同类聚拢无终点（簇内
   坍缩）、与锚点弱相关的语义被挤出几何（vws 退化实测）；多头一致度核
   把锚点增殖为头组合胞元（~百级），全头一致全力度聚拢、部分一致分级
   聚拢、全不一致斥远——样本按丰富语义在空间中分级"沉淀"，
   拉力与语义一致度成正比。正对核=CKA 目标核（同一数学对象），
   损失由 batch 内全部成对点积构成，梯度触及 z 的每个方向。
   掩码视图 InfoNCE（SCARF式）已删除：其前提"被掩列可从剩余列恢复"
   对本特征表示不成立——技术列依赖历史（不在当日输入）、逐列分位数
   归一化破坏跨列代数关系，掩技术列 → 模型学会忽略它们（不变性捷径，
   alignment→0 任务空转）；掩源列 → 语义真空（伪方向视图，全列掩码下
   ~39% 视图伪 direction）。SCARF 是无标签时代的拐杖，有确定性标签时
   直接用标签教几何。
   projector 吸收"收紧"压力，z 保住 O/D 线性可读性；z 不坍缩由
   VISReg（各向同性高斯散度）+ Recon（细粒度信息）保证。projector
   训练后丢弃。SUPCON_ENABLED=False 可退回纯探针模式。
4. 分类头（类别监督，Kronos 粗细分层监督的判别式移植）：符号类衍生目标
   本质是类别 {-1,0,1}，被 MSE 当连续数训时存在"猜 0.7 也算对"的盲区，
   模型没有动力让同类K线在 embedding 空间靠拢；交叉熵没有"差不多"，
   必须选边站，逼 embedding 出现可线性读出的类别可分结构
   （同类聚拢/异类分开）——下游 attention 消费点积相似度，此几何直接可用。
   粗头（CLS_ENABLED）：四族 dir/vpa/drv/vws 各三类=负/中性/正，"中性"
   为 |v|≤ε 的区间（ε 由池上分位数预计算，CLS_FLAT_PCT），非"恰好=0"
   （连续分布上测度为零、平类会退化）。drv=macd_drive 取 MACD柱差分
   本身：hd>0 多头边际加强（红柱变长/绿柱变短）、hd<0 空头边际加强——
   替换原 momentum_accel 乘积锚点（sign(h)·sign(hd) 销毁多空方向，
   把多头加强和空头加强混进同类，与交易语义相悖）。
   细头（CLS_FINE_ENABLED）：仅 dir/vpa 两族，同一判别值等频分
   CLS_FINE_BUCKETS 桶——粗头只有"分得开"的压力、类内几何是平的
   （涨0.5%与涨9.8%同为"涨"），细头在桶粒度继续聚拢，给类内几何加
   分辨率；vws 细版与 dir/vpa 强相关属冗余不设。
   头为单层线性（与线性解码器同哲学：非线性头会把分类压力吞掉），
   标签由归一化输入即时计算（与符号族衍生特征同定义、平带外符号一致、
   无噪声），训练后与 decoder/projector 一起丢弃。
   CKA 监控（CKA_LOG_ENABLED）：逐 epoch 计算 CKA(z, 全头一致度核)
   （Kornblith et al. 2019）——z 的成对相似度结构与标签核的中心化相关，
   几何轴指标（与 probe B/C 的信息轴正交）：高=类别几何饱和，低=有空间。

用法：
  python src/pretrain_embedding.py
  python src/pretrain_embedding.py --test                 # 测试已保存权重的输出std
"""

import os
import sys
import math
import re
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
def compute_derived_features(x, eps=None, chunk_size=4_000_000, coarse=None):
    """
    从归一化19维特征计算11个衍生特征 [N, 11] + 语义掩码 [N, 11]

    设计原则：全部为"跨维度非线性函数"——
    单维度信息 decoder 用线性层就能从原始19维算出，逼不出 embedding 融合；
    只有跨维度的非线性关系（abs/sign/乘积/除法）才能强迫 embedding
    在128维里编码"维度间关系"，而非逐维独立存储。

    输入 x 为 QuantileTransformer 细处理后的特征（各维均值0方差1），
    embedding 也只见过这个分布。衍生目标分两个空间计算：
    - 归一化空间（符号族/breakthrough/均线/布林）：与输入同分布，直接可推；
    - 粗处理空间（body_ratio/upper_shadow/lower_shadow + 物理振幅，经 coarse
      逆变换）：教科书 K线解剖量，embedding 需自行学会分位数逆映射后才能
      推出，任务更硬。

    coarse: None 或 (w_up, w_dn, high_raw, low_raw)——列16/17/1/2 逆变换回
      粗处理空间的原始量（构建见 build_derived_targets）：
        - w_up/w_dn: [N] 上/下影线占比 ∈[0,1]
        - high_raw/low_raw: [N] 距开盘运动总量 (高−开)/开、(低−开)/开，两者之差
          即物理振幅 (H−L)/O ≥ 0（high/low 列保留后振幅可精确重建）
      提供时：
        - K线形态域走占比口径：upper_shadow=w_up、lower_shadow=w_dn、
          body_ratio=1−w_up−w_dn（K线解剖恒等式：三段占满全幅），三者天然
          ∈[0,1]、无归一化伪影，掩码恒有效
        - 量价关系域恢复物理振幅：span_turnover = (high_raw−low_raw)×exchange
      None 时两域退化为归一化空间近似+掩码剔除（仅函数完整性保留，
      kline-shape-v1 下 build_derived_targets 恒提供 coarse）。

    大数据集分块计算：衍生特征+掩码+中间临时变量峰值约 20·N·4 bytes，
    1.6亿条需~12GB 会 OOM；分块后每块峰值仅 ~20·chunk·4 bytes (chunk=4M→~320MB)。

    19维索引:
      0:open_rel 1:high_rel 2:low_rel 3:close_rel 4:vwap_rel
      5:amount 6:exchange 7:m5 8:m10 9:m20 10:dif 11:dea
      12:macd_hist 13:macd_hist_diff 14:bb_upper 15:bb_lower
      16:wick_up(∈[0,1]) 17:wick_dn(∈[0,1]) 18:body_ratio(∈[0,1])
    """
    if eps is None:
        eps = EmbeddingConfig.DERIVED_EPS

    n = x.shape[0]
    if n > chunk_size:
        derived_chunks = []
        mask_chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            coarse_block = None
            if coarse is not None:
                coarse_block = tuple(c[start:end] for c in coarse)
            d, m = _compute_derived_block(x[start:end], eps, coarse_block)
            derived_chunks.append(d)
            mask_chunks.append(m)
        # 两个 concat 必须分开并显式释放 chunks：若写进同一 return 顺序求值，
        # chunks 列表与两份结果全程并存，大池下双倍峰值会 OOM
        derived = np.concatenate(derived_chunks, axis=0)
        del derived_chunks
        masks = np.concatenate(mask_chunks, axis=0)
        del mask_chunks
        return derived, masks
    return _compute_derived_block(x, eps, coarse)


def _compute_derived_block(x, eps, coarse=None):
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

    # 归一化空间的 body（用于 breakthrough）；span_norm 仅 coarse 缺失时兜底
    body_norm = np.abs(close_r - open_r)
    span_norm = high_r - low_r

    # ---- B. K线形态域（占比口径）----
    if coarse is not None:
        # 占比口径：w_up/w_dn 由归一化列逆变换而来的教科书影线占比 ∈[0,1]，
        # K线解剖恒等式 上影+实体+下影 = 全幅 ⇒ 实体占比 = 1−w_up−w_dn。
        # 三者天然有界、无归一化伪影，掩码恒有效。
        w_up, w_dn, _, _ = coarse
        upper_shadow = np.clip(w_up, 0.0, 1.0)
        lower_shadow = np.clip(w_dn, 0.0, 1.0)
        body_ratio = np.clip(1.0 - upper_shadow - lower_shadow, 0.0, 1.0)
        mask_body, mask_upper, mask_lower = ones, ones, ones
    else:
        # 无占比信息时的退化：归一化空间近似 + 掩码剔除伪影
        # （kline-shape-v1 下 build_derived_targets 恒提供 coarse，
        #   此分支仅为函数完整性保留）
        body_ratio = np.clip(body_norm / (span_norm + eps), 0.0, 1.0)
        upper_shadow = high_r - np.maximum(open_r, close_r)
        lower_shadow = np.minimum(open_r, close_r) - low_r
        mask_body = (span_norm > 0).astype(np.uint8)
        mask_upper = (upper_shadow >= 0).astype(np.uint8)
        mask_lower = (lower_shadow >= 0).astype(np.uint8)

    # ---- C. 量价关系域 ----
    breakthrough = body_norm * amount                    # 实体×量: 突破强度 (归一化空间)
    if coarse is not None:
        # 物理振幅 (H−L)/O 精确重建：high/low 列保留后，
        # high_raw−low_raw = (高−开)/开 − (低−开)/开 = (高−低)/开 ≥ 0，
        # 不再需要归一化空间近似。
        _, _, high_raw, low_raw = coarse
        span_turnover = (high_raw - low_raw) * exchange  # 物理振幅×换手: 波动×资金参与
    else:
        # 无 coarse 时退化：归一化空间 |span|×exchange（函数完整性保留）。
        # abs() 消除逐列变换的符号伪影，但 rank 变换不保跨列差的绝对序，
        # 保序性受损，仅作软指标。
        span_turnover = np.abs(span_norm) * exchange     # |振幅|×换手

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
    其余样本的分布和量级与原始19维（已归一化，方差1）对齐，等权 MSE 才成立。

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


# ==================== 分类头（类别监督：粗头+细头） ====================

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
    # macd_drive 的语义（替换原 momentum_accel 分类锚点）：交易语义中多空
    # 攻守由柱体变化方向决定——红柱变长(h>0,hd>0)与绿柱变短(h<0,hd>0)同为
    # 多头边际加强（hd>0），红柱变短与绿柱变长同为空头边际加强（hd<0）；
    # 原乘积 sign(h)·sign(hd) 恰好销毁该方向信息（把多头加强和空头加强各
    # 拆一半混进"同向"类）。macd_drive 直接取 hd 本身：v>ε 多头边际加强 /
    # |v|≤ε 盘整 / v<−ε 空头边际加强。momentum_accel（动能加速）仍保留在
    # MSE 衍生重建 11 维中——作为回归目标语义无碍，仅不再作分类锚点。
    'direction':       3,   # v = close-open                 收/开相对强弱
    'vol_price_align': 3,   # v = sign(close-open)·amount    量价配合
    'macd_drive':      3,   # v = macd_hd                    多头/空头 边际加强
    'vwap_side':       3,   # v = close-vwap                 收在均价上/下
}

CLS_SHORT_NAMES = {
    'direction': 'dir', 'vol_price_align': 'vpa',
    'macd_drive': 'drv', 'vwap_side': 'vws',
}

# 有细头（等频分桶）的基头：仅方向/量价两族。细头的价值在类内幅度分辨率
# （大涨vs小涨、放量程度分级）；vws 与 dir/vpa 强相关（细版冗余，run3 中
# 退化最重），macd_drive 为新锚点先粗粒度验证。结构性事实写死于代码
# （git 管理），不做运行时配置。
FINE_HEAD_BASES = ('direction', 'vol_price_align')


def _cls_values(x):
    """
    各分类头的原始判别值 v：分类标签 = sign(v)（±ε 平带），四个头的共性入口。
    归一化空间、与 compute_derived_features 符号族同列索引（19维索引见该函数）；
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
    yield 'macd_drive', x[:, 13]
    yield 'vwap_side', x[:, 3] - x[:, 4]


def fine_head_name(name):
    """细头名：direction -> direction_f5（桶数取自 CLS_FINE_BUCKETS）"""
    return f"{name}_f{EmbeddingConfig.CLS_FINE_BUCKETS}"


def build_cls_head_spec():
    """
    构建参与 CE 损失的完整头规格（粗+细），成员由 EmbeddingConfig 决定。

    Returns:
        dict[head_name -> n_classes]；空 dict = 分类监督整体关闭
    """
    heads = {}
    if EmbeddingConfig.CLS_ENABLED:
        heads.update(CLS_HEAD_SPEC)
    if EmbeddingConfig.CLS_FINE_ENABLED:
        heads.update({fine_head_name(n): EmbeddingConfig.CLS_FINE_BUCKETS
                      for n in CLS_HEAD_SPEC if n in FINE_HEAD_BASES})
    return heads


def _bucket_shares(v, boundaries):
    """按 bucketize 口径统计各类占比（numpy 侧与 torch.bucketize 严格同口径）"""
    labels = np.searchsorted(boundaries, v, side='left')
    counts = np.bincount(labels, minlength=len(boundaries) + 1)
    return tuple(counts / len(v))


def compute_cls_stats(pool, coarse, fine):
    """
    在池上预计算每个分类头的桶边界与类别占比（numpy，零 torch 开销）

    统一边界表示：每头一个升序边界数组 b（长度 n_classes-1），
    标签 = bucketize(v, b)（numpy 侧 searchsorted side='left' 与
    torch.bucketize 默认 right=False 严格一致）：
    - 粗头（3类）：b = [-ε, +ε]，ε = |v| 的 CLS_FLAT_PCT 分位数（平带）；
      逐头独立 → 不引入人工量纲（各 v 量级差异大）
    - 细头（n桶）：b = v 的等频分位数 i/n·100（i=1..n-1），天然类平衡

    逐头惰性（_cls_values 生成器），40M 池上任一时刻只物化一个头的数组
    （峰值 ~400MB 而非 4 头齐物化的 ~1.1GB，本项目 OOM 纪律）。
    池用实际参与训练的 pre_sampled 子集，与 batch 分布一致、确定性。

    Args:
        pool: [M, input_dim] 归一化K线池（numpy）
        coarse: 是否计算粗头边界（SupCon/CKA 消费粗头标签，
                即使粗头 CE 关闭也可能需要，由调用方决定）
        fine: 是否计算细头边界
    Returns:
        dict[head_name -> (boundaries: np.ndarray[n_classes-1] float32 升序,
                           shares: 各类占比 tuple，顺序=标签值升序)]
    """
    out = {}
    for name, v in _cls_values(pool):
        if coarse and name in CLS_HEAD_SPEC:
            eps = float(np.percentile(np.abs(v), EmbeddingConfig.CLS_FLAT_PCT))
            b = np.array([-eps, eps], dtype=np.float32)
            out[name] = (b, _bucket_shares(v, b))
        if fine and name in FINE_HEAD_BASES:
            b = np.quantile(v, np.arange(1, EmbeddingConfig.CLS_FINE_BUCKETS)
                            / EmbeddingConfig.CLS_FINE_BUCKETS).astype(np.float32)
            out[fine_head_name(name)] = (b, _bucket_shares(v, b))
    return out


def compute_cls_targets(x, cls_boundaries):
    """
    从归一化K线即时计算分类标签（bucketize，粗/细头统一入口）

    全 torch 算子，batch 内一步得出，无 numpy 往返；
    边界为池上预计算的升序 tensor（compute_cls_stats 输出转 torch 后传入）。
    粗头三类语义 {0:跌, 1:平, 2:涨}——平类判据为区间 |v|≤ε 而非"恰好=0"
    （连续分布上测度为零、平类退化），模型无法偷懒只学二分类；
    细头 n 桶按判别值大小升序编号。

    Args:
        x: [B, input_dim] 归一化K线（与训练 batch 相同）
        cls_boundaries: dict[head_name -> 边界 tensor(升序, x.device)]
    Returns:
        dict[head_name -> [B] torch.long, 值域 {0..n_classes-1}]
    """
    targets = {}
    for name, v in _cls_values(x):
        if name in cls_boundaries:                      # 粗头
            targets[name] = torch.bucketize(v, cls_boundaries[name])
        fine = fine_head_name(name)                     # 细头
        if fine in cls_boundaries:
            targets[fine] = torch.bucketize(v, cls_boundaries[fine])
    return targets


def build_derived_targets(kline_data, feature_normalizer, chunk_size=4_000_000):
    """
    从归一化K线构建衍生训练目标：影线占比逆变换 + 衍生特征 + 剪尾 z-score

    只对实际参与训练/探测的子集调用，而非全池：
    衍生目标与掩码按行独立计算，子集统计（分位数/均值/标准差）与全池
    在百万级样本下已收敛等价，但内存从 全池×(19+11+11) 列 降到 子集×30 列，
    避免大池上衍生数组的双倍 concat 峰值 OOM。

    Returns:
        derived_data: [M, n_derived] float32 剪尾+z-score 后的衍生目标
        derived_mask: [M, n_derived] uint8 语义掩码 (1=有效, 0=归一化伪影样本)
    """
    # 逆变换 wick_up/wick_dn 回粗处理占比空间（[0,1] 教科书影线占比），
    # 供 K线形态域使用；实体占比由
    # 恒等式 1−w_up−w_dn 直接得到（上影+实体+下影恒等全幅）。
    # 同时逆变换 high/low 回「距开盘运动总量」原始量 (高−开)/开、(低−开)/开，
    # 两者之差即物理振幅 (H−L)/O——high/low 列保留后振幅可精确重建，
    # span_turnover 不再需要归一化空间近似。
    if feature_normalizer is not None:
        # 列号取 slice.start（与 _feature_groups 的真实列定义对齐），
        # 不用分组序号——后者仅在"每 group 恰为单列 slice 且按列序排列"
        # 时才碰巧等价，属隐式脆弱耦合
        group_slice = dict(feature_normalizer._feature_groups)
        ups, dns, highs, lows = [], [], [], []
        for start in range(0, len(kline_data), chunk_size):
            end = min(start + chunk_size, len(kline_data))
            block = kline_data[start:end]
            for name, bucket in (('wick_up', ups), ('wick_dn', dns),
                                 ('high', highs), ('low', lows)):
                c = group_slice[name].start
                bucket.append(feature_normalizer.pipelines[name]
                              .inverse_transform(block[:, c:c + 1]).flatten())
        coarse = (
            np.concatenate(ups).astype(np.float32),
            np.concatenate(dns).astype(np.float32),
            np.concatenate(highs).astype(np.float32),
            np.concatenate(lows).astype(np.float32),
        )
        # 振幅范围用 min(high)−max(low) / max(high)−min(low) 表达，
        # 避免物化整条 (high−low) 临时数组（大池下省 ~N×4B 峰值）
        amp_lo = float(np.min(coarse[2]) - np.max(coarse[3]))
        amp_hi = float(np.max(coarse[2]) - np.min(coarse[3]))
        print(f"  K线形态域逆变换完成 ({len(kline_data):,} 条, "
              f"w_up∈[{coarse[0].min():.3f},{coarse[0].max():.3f}], "
              f"w_dn∈[{coarse[1].min():.3f},{coarse[1].max():.3f}], "
              f"振幅 (H−L)/O∈[{amp_lo:.3f},{amp_hi:.3f}])")
    else:
        coarse = None

    derived_data, derived_mask = compute_derived_features(
        kline_data, coarse=coarse, chunk_size=chunk_size)
    print(f"  衍生特征: {derived_data.shape[1]}维 "
          f"(跨域非线性; 形态域走占比口径无伪影)")

    # 剪尾 + z-score 标准化：
    # - 剪尾 (分位数 clip) 防"极端值霸凌"：极少数极端行情日(涨停放巨量等)误差平方后
    #   会主导整个衍生 loss，按分位数剪掉尾部后再参与统计，其余样本才不被带偏
    # - z-score 防"量级失衡"：各衍生特征量级差异大
    #   (direction∈{-1,0,1} vs breakthrough=body*amount 可能>10)，
    #   直接拼接做等权 MSE 会让大量级特征主导损失、小量级特征被忽略；
    #   标准化后所有衍生特征方差=1，与原始19维(已归一化、方差1)量级一致，等权 MSE 才合理
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


# ==================== 类感知对比学习（几何轴） ====================


def supcon_loss(p, k_pos, tau):
    """
    核加权类感知对比损失（SupCon "out" 形式；Khosla et al., NeurIPS 2020）

    正对权重 = 多头一致度核 k_pos_ij ∈ [0,1]（label_kernel_from_targets）：
    全头一致 → 全力度聚拢，部分一致 → 分级聚拢，全不一致 → 斥远。
    锚点从单标签 3 簇增殖为头组合胞元（~百级）——样本按丰富语义在空间中
    分级"沉淀"，拉力与语义一致度成正比（soft nearest neighbor 的对比版）。
    正对核 = CKA 目标核：SupCon 把 CKA 要测的"语义成对几何一致性"直接
    变成损失在优化；损失由 batch 内全部成对点积构成，梯度触及 z 的
    每个方向（CE 只塑形头权重张成的 ≤8 个方向）。

    单锚点（如仅 vpa 硬标签）的反面教训：同类聚拢无终点（同=0.968 簇内
    坍缩）、与锚点弱相关的语义（vws）被挤出几何——锚点贫乏是病根，
    聚拢本身不是。核密度较高（任意两样本大概率至少共享一头一致）属预期，
    分级权重保持相对拉力结构；若过软可阈值化（K≥0.5 才计正对）。

    batch 内去重（training loop, precision=1）保证无重复K线，
    否则相同样本互为正对会退化为恒等任务（无监督信号）。
    fp32 计算（τ 除法后 bf16 精度不足）。

    Args:
        p: [B, d] L2 归一化投影（fp32）
        k_pos: [B, B] 一致度核（对称，对角线任意值——内部强制排除自身）
        tau: 温度
    Returns:
        标量损失
    """
    n = p.size(0)
    sim = (p @ p.t()) / tau                              # [B, B]
    mask_self = torch.eye(n, dtype=torch.bool, device=p.device)
    # 每行对 a≠i 做 log-softmax（自身位置 -inf 排除出分母）
    sim = sim.masked_fill(mask_self, float('-inf'))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # 自身位置 log_prob=-inf，置 0 防止与权重相乘产生 nan
    log_prob = log_prob.masked_fill(mask_self, 0.0)
    # 核权重（排除自身）；与任何样本都无一致头的行（实践中概率≈0，
    # 8 头×2560 样本下不存在）安全跳过，避免 0/0 噪声放大
    w = k_pos.float().masked_fill(mask_self, 0.0)
    w_sum = w.sum(1)
    valid = w_sum > 1e-6
    mean_log_prob_pos = (w * log_prob).sum(1)[valid] / w_sum[valid]
    return -mean_log_prob_pos.mean()


def p_geometry_stats(p, k_pos):
    """
    类感知对比的几何监控（uniformity + 锚点分离度 + 核相关）

    - uniformity: log E_{i≠j}[exp(-2‖p_i−p_j‖²)]，p 已 L2 归一化 ⇒
      ‖·‖²=2−2·dot ⇒ exp(-2d²)=exp(4·dot−4)（不用 torch.pdist：
      CUDA kernel 逐对算 N(N-1)/2 个距离，实测 ~300ms/batch，
      纯监控指标不值得；对角 exp(-inf)=0 自动剔除，除以 N(N-1)
      与无序对均值严格一致）
    - sim_k1 / sim_k0: 全头一致(K=1)/全头不一致(K=0)样本对的平均余弦
      ——两者之差是"锚点胞元"分离度的直接读数
    - corr: 非对角 sim 与 K 的 Pearson 相关——p 几何与语义核的整体
      一致性（"沉淀"是否发生的总指标，CKA 在 p 空间的线性版）

    纯监控，调用方需 no_grad + detach。

    Args:
        p: [B, d] L2 归一化投影（fp32）
        k_pos: [B, B] 一致度核（对称）
    Returns:
        (uniformity, sim_k1, sim_k0, corr) python float 四元组
    """
    n = p.size(0)
    sim = p @ p.t()
    mask_self = torch.eye(n, dtype=torch.bool, device=p.device)
    off = ~mask_self
    s = (sim.mul(4.0) - 4.0).masked_fill(mask_self, float('-inf')).exp().sum()
    uniformity = torch.log(s / (n * (n - 1)) + 1e-12)
    k1 = off & (k_pos > 1.0 - 1e-6)
    k0 = off & (k_pos < 1e-6)
    sim_k1 = sim[k1].mean().item() if k1.any() else 0.0
    sim_k0 = sim[k0].mean().item() if k0.any() else 0.0
    # Pearson = 中心化向量的余弦（norm 形式避免 std/n 除法）
    v_s = sim[off]
    v_k = k_pos[off].float()
    v_s = v_s - v_s.mean()
    v_k = v_k - v_k.mean()
    corr = ((v_s * v_k).sum() / (v_s.norm() * v_k.norm() + 1e-12)).item()
    return (uniformity.item(), sim_k1, sim_k0, corr)


def label_kernel_from_targets(cls_targets, names):
    """
    多头一致度核：L_ij = mean_h 1[y_h(i)==y_h(j)] ∈ {0, 1/H, ..., 1}

    双用途：SupCon 的正对权重核（锚点=头组合胞元，分级聚拢/斥远）
    与 CKA 的目标核（同一数学对象，SupCon 直接优化 CKA 要测的东西）。
    L_ij=1 表示所有头同类，=0 表示所有头异类；各头示性核均为 PSD
    （类指示向量外积之和），均值仍 PSD，可作 CKA 的目标核。
    头集合（粗+细）由调用方传入——细头参与使核更细粒度
    （同一 v 上粗/细两级一致度平均，等效逐数量分级权重）。

    Args:
        cls_targets: dict[head_name -> [B] long]（compute_cls_targets 输出）
        names: 参与核构建的头名列表（训练中=cls_heads_spec 全部头）
    Returns:
        [B, B] float tensor（对称，对角线=1）
    """
    y = torch.stack([cls_targets[n] for n in names], dim=1)       # [B, H]
    return (y[:, None, :] == y[None, :, :]).float().mean(dim=2)  # [B, B]


def compute_label_cka(z, label_kernel):
    """
    线性 CKA（Kornblith et al., ICML 2019）：z 的 Gram 与标签核的中心化相关

    K = z_c z_cᵀ（z 列中心化 ⇒ 等价于核的双中心化 HXH），
    L_c = H L H（标签核显式双中心化），
    CKA = ⟨K, L_c⟩_F / (‖K‖_F·‖L_c‖_F) ∈ [-1, 1]。

    衡量"z 里挨得近的样本对，标签上是否也同类"——成对几何轴指标，
    与 probe B/C（单样本线性读出，信息轴）正交。纯监控，调用方需 no_grad。

    Args:
        z: [B, d] fp32（建议 detach 后传入）
        label_kernel: [B, B] 对称标签核（label_kernel_from_targets 输出）
    Returns:
        python float
    """
    zc = z - z.mean(dim=0, keepdim=True)
    k = zc @ zc.t()
    lc = (label_kernel - label_kernel.mean(dim=0, keepdim=True)
          - label_kernel.mean(dim=1, keepdim=True) + label_kernel.mean())
    return ((k * lc).sum() / (k.norm() * lc.norm() + 1e-12)).item()


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
                                 ├─ Linear×6 → 类别 (分类头, dir/vpa粗+细 + drv/vws粗, 训练后丢弃)
                                 ├─ VISReg(S)
                                 └─ projector g(S) → L2归一化 → SupCon(多头一致度核)

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

    分类头（类别监督，粗3类+细5桶两档粒度）：
    - 符号类衍生目标本质是类别 {-1,0,1}，被 MSE 当连续数训时"猜 0.7 也算对"，
      模型没有动力让同类K线在 embedding 空间靠拢；
    - 交叉熵没有"差不多"，必须选边站，逼 embedding 出现可线性读出的
      类别可分结构（同类聚拢/异类分开）——下游 attention 消费点积相似度，
      此几何直接可用；
    - 细头（等频分桶）在桶粒度继续聚拢，补粗头类内几何的分辨率盲区
      （涨0.5%与涨9.8%粗标签同为"涨"）；
    - 头必须单层线性（同解码器哲学：MLP 头自己就把类别算出来了，
      压力被吞掉）；标签由归一化输入即时计算（compute_cls_targets），无噪声；
    - 注意与 VISReg 相克：可分性在 z 分布上产生多峰，高斯形状项拉单峰，
      CLS_WEIGHT/CLS_FINE_WEIGHT 别开大；训练后丢弃，与 decoder/projector 相同。

    projector（对比分支，SimCLR/DINOv2 路线）：SupCon 不直接作用在 S 上
    （对比的收紧压力会把"对类别无贡献"的方差压掉，冲掉 O/D 线性可读结构），
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
        # MLP分支增益=1（σ1·σ2·0.588·√(d·h) = 1），输出 L2 归一化后进入 SupCon
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
            recon: [batch, input_dim + n_derived] 重建=[原始19维, 衍生特征]
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
    # 避免拼接完整 [N,45,19] 数组(~10GB)+新数组(~10GB)=峰值~20GB OOM；
    # 采样后直接返回 [M,19] 2D数组，全在内存中完成，不写磁盘
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

def save_pretrained_embedding(embedding, path, metrics=None, decoder=None, tag=None):
    """
    保存预训练 embedding 权重（及可选解码器）

    格式与 StockTransformer 的 embed_proj / embed_mlp 直接兼容。
    传入 decoder 时同时保存解码器权重，供重建可视化等用途。
    checkpoint['config'] 落盘全部损失权重与实验臂标签——任何 .pth
    均可独立追溯其训练配方，多臂对比产物不混淆。
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
            # 损失权重全量快照（实验臂自描述）
            'visreg_w_scale': EmbeddingConfig.VISREG_W_SCALE,
            'visreg_w_shape': EmbeddingConfig.VISREG_W_SHAPE,
            'visreg_w_center': EmbeddingConfig.VISREG_W_CENTER,
            'target_std': EmbeddingConfig.TARGET_STD,
            'derived_weight': EmbeddingConfig.DERIVED_WEIGHT,
            'supcon_weight': EmbeddingConfig.SUPCON_WEIGHT,
            'supcon_tau': EmbeddingConfig.SUPCON_TAU,
            'cls_weight': EmbeddingConfig.CLS_WEIGHT,
            'cls_fine_weight': EmbeddingConfig.CLS_FINE_WEIGHT,
            'tag': tag,
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

def pretrain(train_stock_info, feature_normalizer=None, device=None, tag=None):
    """
    Embedding层训练-主函数（超参统一走 EmbeddingConfig，CLI 不再提供覆盖入口）

    tag: 实验臂标签（非 None 时产物存为 best_embedding_<tag>.pth，
         并写入 checkpoint['config']['tag']；None=常规产物）
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

    # ---- 分类头（粗+细）/ SupCon / CKA 开关与预计算（须在模型构造前定义） ----
    # 粗头(3类平带) + 细头(等频分桶)；SupCon/CKA 消费粗头标签，
    # 故 CLS_ENABLED=False 时粗头边界仍需计算（need_coarse）。
    # w_cls/w_fine 独立于 (1-w_c) 预算（置零即退回纯回归监督）。
    use_cls_coarse = EmbeddingConfig.CLS_ENABLED
    use_cls_fine = EmbeddingConfig.CLS_FINE_ENABLED
    use_supcon = EmbeddingConfig.SUPCON_ENABLED
    use_cka = EmbeddingConfig.CKA_LOG_ENABLED
    need_coarse = use_cls_coarse or use_supcon or use_cka
    w_cls = EmbeddingConfig.CLS_WEIGHT if use_cls_coarse else 0.0
    w_fine = EmbeddingConfig.CLS_FINE_WEIGHT if use_cls_fine else 0.0

    # 头规格（模型侧：只含参与 CE 损失的头；SupCon/CKA 不需要头只要标签）
    cls_heads_spec = build_cls_head_spec()
    use_cls = len(cls_heads_spec) > 0

    # 桶边界：池上分位数预计算（确定性、与 batch 分布一致），转 torch 上 device
    cls_stats = None
    cls_boundaries = None
    if need_coarse or use_cls_fine:
        cls_stats = compute_cls_stats(pre_sampled,
                                      coarse=need_coarse, fine=use_cls_fine)
        cls_boundaries = {head: torch.tensor(b, dtype=torch.float32, device=device)
                          for head, (b, _) in cls_stats.items()}

    # 3. 创建模型
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES
    model = PretrainModel(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        n_derived=n_derived,
        cls_heads=cls_heads_spec if use_cls else None,
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
    input_dim = ModelConfig.INPUT_DIM                # 原始19维，recon前 input_dim 列为原始重建
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES

    # 类感知对比（几何轴主损失）：w_c=SupCon 权重，剩余 (1-w_c) 按 λ 分配给
    # VISReg/Recon；w_c=0 或 SUPCON_ENABLED=False 时公式严格退回旧版
    # (1-λ)·Recon + λ·VISReg（use_supcon 在上方分类头块定义）
    w_c = EmbeddingConfig.SUPCON_WEIGHT if use_supcon else 0.0
    w_v = (1.0 - w_c) * visreg_weight
    w_r = (1.0 - w_c) * (1.0 - visreg_weight)
    supcon_tau = EmbeddingConfig.SUPCON_TAU

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
    if use_supcon:
        formula = (f"{w_v:.2f}·VISReg + {w_r:.2f}·Recon + "
                   f"{w_c:.2f}·SupCon(核加权, {len(cls_heads_spec)}头一致度核, "
                   f"τ={supcon_tau}, projector头)")
    else:
        formula = (f"{visreg_weight:.2f}·VISReg + {1 - visreg_weight:.2f}·Recon  (论文原版)")
    if use_cls_coarse:
        formula += f" + {w_cls:.2f}·CLS({','.join(CLS_SHORT_NAMES.values())})"
    if use_cls_fine:
        formula += (f" + {w_fine:.2f}·FINE(f{EmbeddingConfig.CLS_FINE_BUCKETS})")
    print(f"  损失公式: {formula}")
    print(f"  解码器=线性探针, Recon=原始19维 + {derived_weight:.2f}·{n_derived}衍生(掩码MSE)")
    if use_cls_coarse:
        print(f"  分类头(粗,3类 跌/平/涨)=线性探针:")
        for name in CLS_HEAD_SPEC:
            b, shares = cls_stats[name]
            print(f"    {name:>16s}: ε={-b[0]:.4f}  "
                  f"池上占比 跌{shares[0]*100:.0f}%/平{shares[1]*100:.0f}%/涨{shares[2]*100:.0f}%")
    if use_cls_fine:
        nb = EmbeddingConfig.CLS_FINE_BUCKETS
        print(f"  分类头(细,{nb}桶 等频, 仅{'+'.join(FINE_HEAD_BASES)})=线性探针:")
        for name in FINE_HEAD_BASES:
            b, shares = cls_stats[fine_head_name(name)]
            b_str = ' '.join(f"{x:+.3f}" for x in b)
            s_str = '/'.join(f"{s*100:.0f}%" for s in shares)
            print(f"    {name:>16s}: 边界[{b_str}]  池上占比 {s_str}")
    if use_supcon:
        print(f"  SupCon(核加权, 几何轴主损失): 正对权重=全{len(cls_heads_spec)}头"
              f"(dir/vpa各粗+细, drv/vws粗)标签一致度 "
              f"K∈{{0,1/{len(cls_heads_spec)},...,1}}, "
              f"τ={supcon_tau}; 锚点从单标签3簇增殖为头组合胞元(~十级), "
              f"样本按语义一致度分级沉淀; 正对核=CKA目标核")
    if use_cka:
        print(f"  CKA监控: CKA(z, 全头一致度核) 逐epoch日志"
              f" (几何轴指标, 与probe B/C信息轴正交; 与SupCon正对核同一对象)")
    if use_cls_coarse or use_cls_fine:
        print(f"  注: 分类可分性在 z 分布上产生多峰, 与 VISReg 高斯形状项相克, "
              f"权重别开大; 标签由归一化输入即时计算, 无噪声")
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
        epoch_recon_orig_raw = 0.0    # 原始19维重建原始值
        epoch_recon_derived_raw = 0.0 # 衍生重建原始值(掩码后逐特征均值)
        epoch_cls_raw = 0.0      # 粗头 CE 平均（原始值）
        epoch_cls_w = 0.0        # w_cls·粗头CE 加权贡献
        epoch_cls_fine_raw = 0.0 # 细头 CE 平均（原始值）
        epoch_cls_fine_w = 0.0   # w_fine·细头CE 加权贡献
        epoch_cls_head_raw = {}  # 逐头 CE（粗+细，按头名）
        epoch_cls_head_acc = {}  # 逐头准确率（粗+细，按头名）
        epoch_sup_raw = 0.0      # SupCon 原始值
        epoch_sup_w = 0.0        # w_c·SupCon 加权贡献
        epoch_sup_hfloor = 0.0   # SUP 理论下界(核行熵均值): softmax 完美匹配正对核时的损失值
        # 语义几何指标一律只在 z 本体测（下游消费 z 而非 g(z)；p 是训练后丢弃的
        # 脚手架，其绝对几何无消费者——p 空间 U/K1/K0/r 已移除，历史教训：
        # p 的全局收缩是设计内现象（VISReg 反压拦截在 z 端），看 p 只会误报）
        epoch_sim_k1_z = 0.0     # z 上全头一致(K=1)对平均余弦
        epoch_sim_k0_z = 0.0     # z 上全头不一致(K=0)对平均余弦
        epoch_sim_corr_z = 0.0   # z 上 sim~K Pearson 相关
        epoch_cka = 0.0          # CKA(z, 标签核)（batch 级平均）
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

            # 分类标签：由归一化输入即时计算（CE/SupCon/CKA 三方消费，
            # 与 batch 天然对齐、零存储），先于任何前向算好
            cls_targets = (compute_cls_targets(batch_x, cls_boundaries)
                           if cls_boundaries is not None else None)

            optimizer.zero_grad()
            # 前向走 AMP（小 MLP 开销可忽略），对比 logits 在 AMP 外算 fp32
            with amp_ctx:
                z, recon = model(batch_x)

                # VISReg: 缩放到目标 std，尺度项 target std=1，形状项 SWD 对齐高斯
                loss_visreg = visreg_loss_fn(z / target_std)

                # 重建损失: 分离原始重建与衍生重建
                # 衍生项逼 embedding 编码跨维度关系，而非逐维独立存储；
                # 解码器是线性层，衍生目标(非线性函数)无法由解码器自行计算，
                # 结构必须由 embedding 内部算好 → 重建损失即线性探针误差
                # 衍生目标已剪尾+z-score 标准化，与原始19维量级一致，等权合理；
                # derived_weight 可调，防止辅助任务(衍生)主导主任务(原始重建)
                loss_recon_orig = F.mse_loss(recon[:, :input_dim], batch_x)
                # 掩码 MSE：仅有效样本参与，逐特征归一(每个衍生目标等权)，
                # 掩码剔除归一化伪影样本(span≤0、影线<0)，不学伪影当信号
                diff2 = (recon[:, input_dim:] - batch_d) ** 2
                num = (diff2 * batch_m).sum(dim=0)
                den = batch_m.sum(dim=0) + 1e-8
                loss_recon_derived = (num / den).mean()
                loss_recon = loss_recon_orig + derived_weight * loss_recon_derived

                # 对比分支前向在 AMP 内（小 MLP），复用主前向的 z
                # （省两次 embedding 前向，梯度也更直接）
                if use_supcon:
                    p_clean = model.projector(z)
                # 加权合成（对比项的 fp32 计算放 AMP 外，避免 bf16 logits 精度不足）
                loss = w_v * loss_visreg + w_r * loss_recon

            # 类感知对比（几何轴主损失）：干净 x 的投影 + 多头一致度核。
            # K_ij = 各头标签一致比例——锚点从单标签 3 簇增殖为头组合
            # 胞元（~百级），正对权重=语义一致度：全头一致全力度聚拢、
            # 部分一致分级聚拢、全不一致斥远，样本按丰富语义分级"沉淀"。
            # 正对核=CKA 目标核（同一 K 两处共用）。无视图身份游戏，
            # "几乎相同的两条K线被拉近"是 feature：下游要语义相似度
            # 而非身份识别
            if use_supcon:
                p_n = F.normalize(p_clean.float(), dim=1)
                k_sem = label_kernel_from_targets(cls_targets,
                                                  list(cls_heads_spec))
                loss_sup = supcon_loss(p_n, k_sem, supcon_tau)
                loss = loss + w_c * loss_sup
                # 纯监控：语义几何只在 z 本体测（下游消费 z；p 是丢弃的脚手架，
                # 其全局收缩是设计内现象——VISReg 反压把它拦截在 g 内部，
                # 历史 p 空间 U/K1/K0/r 指标已移除以免误读）
                with torch.no_grad():
                    z_n = F.normalize(z.detach().float(), dim=1)
                    _, sim_k1_z, sim_k0_z, sim_corr_z = p_geometry_stats(z_n, k_sem)
                    # SUP 理论下界 Hf：若 softmax 完美匹配归一化正对核 q̃∝K，
                    # 损失恰为核行熵的平均——核越稠密下界越高（稠密核下 ~5-7）。
                    # SUP 绝对值无信息，有效读数 = SUP − Hf（超下界余量）
                    k_off = k_sem.float().clone()
                    k_off.fill_diagonal_(0.0)
                    q_row = k_off / k_off.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    h_floor = -(q_row * torch.log(q_row.clamp_min(1e-12))).sum(dim=1).mean()

            # 分类头（粗+细）：fp32 计算（CE 在 bf16 下精度不足，同 SupCon）。
            # 标签已由 compute_cls_targets 预算好；梯度经线性头回流 embedding，
            # 逼 z 出现可线性读出的类别可分结构
            if use_cls:
                cls_logits = model.cls_forward(z.float())
                cls_losses = {
                    name: F.cross_entropy(cls_logits[name], cls_targets[name])
                    for name in cls_heads_spec
                }
                # 粗/细分组（键在 CLS_HEAD_SPEC 中=粗头，其余=细头），各自等权平均
                coarse_l = [cls_losses[n] for n in cls_heads_spec
                            if n in CLS_HEAD_SPEC]
                fine_l = [cls_losses[n] for n in cls_heads_spec
                          if n not in CLS_HEAD_SPEC]
                loss_cls = torch.stack(coarse_l).mean() if coarse_l else None
                loss_cls_fine = torch.stack(fine_l).mean() if fine_l else None
                if loss_cls is not None:
                    loss = loss + w_cls * loss_cls
                if loss_cls_fine is not None:
                    loss = loss + w_fine * loss_cls_fine

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
            if use_supcon:
                sup_w = w_c * loss_sup.item()
                epoch_loss += sup_w
                epoch_sup_w += sup_w
                epoch_sup_raw += loss_sup.item()
                epoch_sim_k1_z += sim_k1_z
                epoch_sim_k0_z += sim_k0_z
                epoch_sim_corr_z += sim_corr_z
                epoch_sup_hfloor += h_floor.item()
            if use_cls:
                if loss_cls is not None:
                    cls_w = w_cls * loss_cls.item()
                    epoch_loss += cls_w
                    epoch_cls_w += cls_w
                    epoch_cls_raw += loss_cls.item()
                if loss_cls_fine is not None:
                    fine_w = w_fine * loss_cls_fine.item()
                    epoch_loss += fine_w
                    epoch_cls_fine_w += fine_w
                    epoch_cls_fine_raw += loss_cls_fine.item()
                for name in cls_heads_spec:
                    epoch_cls_head_raw[name] = (epoch_cls_head_raw.get(name, 0.0)
                                                + cls_losses[name].item())
                    acc = (cls_logits[name].argmax(dim=1) == cls_targets[name])
                    epoch_cls_head_acc[name] = (epoch_cls_head_acc.get(name, 0.0)
                                                + acc.float().mean().item())
            if use_cka:
                # 纯监控：z 已反传完毕，detach+no_grad 零额外图开销。
                # 目标核与 SupCon 正对核共用（use_supcon 时直接复用 k_sem）
                with torch.no_grad():
                    k_l = (k_sem if use_supcon else
                           label_kernel_from_targets(cls_targets,
                                                     list(cls_heads_spec)))
                    epoch_cka += compute_label_cka(z.detach().float(), k_l)
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

        # 打印日志 （O=/D= 分别为原始19维/衍生重建原始值，D= 即线性探针误差，
        # 直接衡量 embedding 中可线性读出的跨维结构量，观察衍生目标是否被学到；
        # SUP=核加权 SupCon 原始值（对 B≈2560 候选的平均负对数概率，
        # 绝对值无信息——完美匹配时损失也只降到核行熵）；
        # Hf=SUP 理论下界(核行熵均值)，SUPexc=SUP−Hf 才是有效读数：
        # 越接近 0 = 通道越接近其目标允许的极限；
        # zK1/zK0/zr=语义几何指标，只在 z 本体测（L2 归一化后与语义核比）——
        # 下游消费 z 而非 g(z)，p 是训练后丢弃的脚手架，其全局收缩是设计内
        # 现象（VISReg 反压拦截在 g 内部），p 空间 U/K1/K0/r 指标已移除；
        # CLS/FINE=粗/细头 CE 平均，逐头格式 CE/Acc——CLS 从 ln(3)≈1.1、
        # FINE 从 ln(5)≈1.61 下降 + Acc 上升 = 类别可分结构正在被嵌入
        # （Acc 受 label 噪声/可分离度上限约束，无需追满）；
        # CKA=z 与标签核的中心化相关（几何轴，与 SupCon 正对核同一对象），
        # 上升=类别成对几何与标签一致性提高——VISReg 单峰压力会限制其
        # 上限，看趋势不看绝对值）
        avg_dedup = epoch_dedup_total / max(1, n_batches)
        weighted_str = f"{avg_visreg_w:.4f}·VISReg + {avg_recon_w:.4f}·Recon"
        sup_str = ""
        if use_supcon:
            weighted_str += f" + {epoch_sup_w / n_batches:.4f}·SUP"
            sup_str = (f"SUP={epoch_sup_raw / n_batches:.4f} "
                       f"(zK1={epoch_sim_k1_z / n_batches:.3f}/"
                       f"zK0={epoch_sim_k0_z / n_batches:.3f}/"
                       f"zr={epoch_sim_corr_z / n_batches:.3f}/"
                       f"Hf={epoch_sup_hfloor / n_batches:.3f})  "
                       f"SUPexc={epoch_sup_raw / n_batches - epoch_sup_hfloor / n_batches:.3f}  ")
        cls_str = ""
        if use_cls_coarse:
            weighted_str += f" + {epoch_cls_w / n_batches:.4f}·CLS"
            cls_parts = []
            for name in CLS_HEAD_SPEC:
                ce = epoch_cls_head_raw[name] / n_batches
                acc = epoch_cls_head_acc[name] / n_batches
                cls_parts.append(f"{CLS_SHORT_NAMES[name]}={ce:.3f}/{acc*100:.0f}%")
            cls_str = (f"CLS={epoch_cls_raw / n_batches:.4f} "
                       f"({' '.join(cls_parts)})  ")
        fine_str = ""
        if use_cls_fine:
            weighted_str += f" + {epoch_cls_fine_w / n_batches:.4f}·FINE"
            nb = EmbeddingConfig.CLS_FINE_BUCKETS
            fine_parts = []
            for name in FINE_HEAD_BASES:
                head = fine_head_name(name)
                ce = epoch_cls_head_raw[head] / n_batches
                acc = epoch_cls_head_acc[head] / n_batches
                fine_parts.append(
                    f"{CLS_SHORT_NAMES[name]}{nb}={ce:.3f}/{acc*100:.0f}%")
            fine_str = (f"FINE={epoch_cls_fine_raw / n_batches:.4f} "
                        f"({' '.join(fine_parts)})  ")
        cka_str = ""
        if use_cka:
            cka_str = f"CKA={epoch_cka / n_batches:.4f}  "
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f} ({weighted_str})  "
              f"VISReg={avg_visreg_raw:.4f}  Recon={avg_recon_raw:.4f} "
              f"(O={avg_recon_orig:.4f}/D={avg_recon_derived:.4f})  "
              f"{sup_str}"
              f"{cls_str}"
              f"{fine_str}"
              f"{cka_str}"
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
    if use_cls_coarse:
        last_metrics.update({
            'cls_weighted': epoch_cls_w / n_batches,
            'cls_raw': epoch_cls_raw / n_batches,
        })
    if use_cls_fine:
        last_metrics.update({
            'cls_fine_weighted': epoch_cls_fine_w / n_batches,
            'cls_fine_raw': epoch_cls_fine_raw / n_batches,
        })
    for name in cls_heads_spec:
        last_metrics[f'cls_ce_{name}'] = epoch_cls_head_raw[name] / n_batches
        last_metrics[f'cls_acc_{name}'] = epoch_cls_head_acc[name] / n_batches
    if use_supcon:
        last_metrics.update({
            'supcon_weighted': epoch_sup_w / n_batches,
            'supcon_raw': epoch_sup_raw / n_batches,
            'sim_k1_z': epoch_sim_k1_z / n_batches,
            'sim_k0_z': epoch_sim_k0_z / n_batches,
            'sim_kernel_corr_z': epoch_sim_corr_z / n_batches,
            'supcon_h_floor': epoch_sup_hfloor / n_batches,
        })
    if use_cka:
        last_metrics['cka'] = epoch_cka / n_batches
    ckpt_name = f'best_embedding{("_" + tag) if tag else ""}.pth'
    best_path = os.path.join(output_dir, ckpt_name)
    save_pretrained_embedding(model.embedding, best_path,
                              metrics=last_metrics, decoder=model.decoder,
                              tag=tag)

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
    parser.add_argument('--tag', type=str, default=None,
                        help='实验臂标签: 产物存为 best_embedding_<tag>.pth 并写入'
                             ' checkpoint（多臂对比时区分产物，不影响 train.py 默认加载路径）')

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
    if args.tag is not None and not re.fullmatch(r'[\w\-]+', args.tag):
        parser.error(f"--tag 仅允许字母/数字/下划线/连字符: {args.tag!r}")

    print("\n[步骤3] 开始 Embedding 预训练...")
    pretrain(
        train_stock_info=train_stock_info,
        feature_normalizer=feature_normalizer,
        device=device,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
