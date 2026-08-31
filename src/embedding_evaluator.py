"""
Embedding 训后诊断脚本（v2 完全重写，2026-08）

定位
====
embedding 预训练完成后的**离线深度诊断**。只回答训练期指标覆盖不了的
问题——与训练日志的分工（避免重复造轮子）：

    训练期已覆盖（本脚本不再重复）：
      - 信息可解码度        → 线性探针 D=（衍生重建误差）
      - 输出分布健康度      → TARGET_STD 契约 + measure_embedding_std
      - 语义几何是否沉淀    → SupCon 核 + CKA / zK1 / zK0 / zr
    本脚本独有（训练期看不到的离线问题）：
      [1] 方向对比          —— 符号翻转 vs 同幅平移的响应比，
          检验输入维的"方向"是否被放大编码
      [2] 跨维交互残差矩阵   —— 维度对之间的非线性二阶响应：
          embedding 是否只做逐维加性映射，还是真的编码了
          "量价配合""影线形态"这类跨维结构
      [3] 连续性扫值         —— 沿每个输入维扫值时输出轨迹是否平滑、
          在语义边界（=该列历史中位水平）处是否有曲率峰
      [4] 特征消融           —— 把某一维替换为中性值后的输出位移，
          单维重要性排序（与预训练分类头的压力来源相互印证）

设计三原则（旧版死于违反它们）
==============================
P1 一切基准量从测试池实测校准：中性点=池中位数、扰动幅度=各列 IQR×
   固定比例、扫值区间=[q02,q98]，启动时打印校准表供人工核查。禁止
   手调绝对值——2026-06/07 两轮特征工程改动后，旧版手写的
   NEUTRAL_VALUES / DELTA_MAP / SEMANTIC_PATTERNS 全部失真且不报错，
   这是旧版被判不可信的直接原因。
P2 扰动在细处理空间进行（=embedding 真实输入空间）。细处理是逐列
   QuantileTransformer→StandardScaler（见 data.py FeatureNormalizer），
   输出近似标准正态：扰动幅度跨列可比、不会被子列尾部饱和吞掉。
   注意：z=0 在该空间表示"该列处于历史中位水平"，对相对涨跌/变化率
   列即"相对基准不变"，语义边界与粗空间的 0 对应。
P3 特征名运行时取自 FeatureNormalizer._feature_groups 并校验维度，
   不维护抄写副本。特征增删改名时本脚本自动跟随或显式报错。

输入合法性（继承并强化旧版的严格校验）
======================================
- 全模型 checkpoint：state_dict 与当前架构严格匹配（缺键/多键/shape
  不符一律报错），不接受静默 partial load；
- 预训练 checkpoint：按 input_dim/d_model/expand_ratio 元数据重建
  KLineEmbedding，与当前配置不符直接报错而不是产出垃圾结果。

解读纪律
========
本脚本输出的是"观察"，不是"判卷"。阈值（强/中/弱）是启发式参考线；
数值含义依赖校准表和当前特征口径，请结合训练期指标一起读。
"""

import argparse
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans',
                                   'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch
import torch.nn as nn

from config import DataConfig, EmbeddingConfig, ModelConfig, PROJECT_ROOT
from data import (FeatureNormalizer, load_and_preprocess_data,
                  normalize_and_validate_context_window)
from model import create_model
from pretrain_embedding import KLineEmbedding


# ==================== 可调参数 ====================

DEFAULT_SAMPLES = 64     # 参与分析的单日样本数（从测试集窗口抽取）
DEFAULT_STEPS = 25       # 连续性扫值的步数
SHIFT_IQR_FRAC = 0.5     # 方向对比/交互分析的扰动幅度 = 各列 IQR × 此系数
SWEEP_QLO = 0.02         # 扫值下界分位数
SWEEP_QHI = 0.98         # 扫值上界分位数
TOP_K_INTERACT = 8       # 打印的最强交互对数量
RATIO_STRONG = 3.0       # 方向对比"强"参考线（启发式，非硬标准）
RATIO_MODERATE = 1.5     # 方向对比"中"参考线
EDGE_MASS_WARN_PCT = 2.0  # 粗空间端点焊死比例≥此值 → 判为边界平台列

# 语义上预期存在跨维配合的特征对（仅用于在交互矩阵里高亮对照，
# 不参与任何阈值判定——哪些交互"应该"强属于业务判断，脚本不越权）
KEY_PAIR_NAMES = [('close', 'amount'), ('close', 'vwap'),
                  ('open', 'close'), ('wick_up', 'wick_dn')]


# ==================== 权重加载 ====================

def _validate_state_dict_match(current_state, loaded_state, source_path):
    """严格校验 loaded_state 与 current_state 完全匹配，不一致直接报错"""
    missing_keys = set(current_state.keys()) - set(loaded_state.keys())
    extra_keys = set(loaded_state.keys()) - set(current_state.keys())
    shape_mismatches = []
    for key in current_state:
        if key in loaded_state and current_state[key].shape != loaded_state[key].shape:
            shape_mismatches.append(
                f"  {key}: checkpoint {list(loaded_state[key].shape)} "
                f"vs 模型 {list(current_state[key].shape)}")
    if missing_keys or extra_keys or shape_mismatches:
        parts = [f"权重与 checkpoint 不匹配: {source_path}"]
        if missing_keys:
            parts.append(f"  缺少 ({len(missing_keys)}): {sorted(missing_keys)[:8]} ...")
        if extra_keys:
            parts.append(f"  多余 ({len(extra_keys)}): {sorted(extra_keys)[:8]} ...")
        if shape_mismatches:
            parts.append(f"  shape 不一致 ({len(shape_mismatches)}):")
            parts.extend(shape_mismatches)
        raise ValueError('\n'.join(parts))


class FFNEmbedding(nn.Module):
    """StockTransformer 的 embedding 前段：embed_proj → embed_mlp"""

    def __init__(self, embed_proj, embed_mlp):
        super().__init__()
        self.embed_proj = embed_proj
        self.embed_mlp = embed_mlp

    def forward(self, x):
        return self.embed_mlp(self.embed_proj(x))


def _embed_numpy(embed_fn, x_np, device):
    """
    统一前向入口：numpy → 强制 float32 → forward → numpy

    为什么强制 float32：校准量来自 np.percentile（恒返回 float64），
    任何参与运算的数组都会被提升为 double，而模型权重是 float32，
    混合输入会直接 RuntimeError（double != float）。所有 embed_fn
    调用必须经过这里，禁止各分析函数自建 tensor。
    """
    with torch.no_grad():
        t = torch.tensor(np.ascontiguousarray(x_np, dtype=np.float32),
                         device=device)
        return embed_fn(t).cpu().numpy()


def list_model_files():
    """收集 out/embedding_pretrain 与 out/ 下的 .pth，按修改时间降序"""
    candidates = []
    for d in {EmbeddingConfig.OUTPUT_DIR, DataConfig.OUTPUT_DIR}:
        if os.path.isdir(d):
            candidates += [os.path.join(d, f) for f in os.listdir(d)
                           if f.endswith('.pth')]
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def load_embedding_fn(model_path, device):
    """
    加载 .pth，返回 (embed_fn, recipe_str)

    embed_fn: callable，[N, input_dim] 细处理向量 tensor → [N, d_model]
              即单日K线 embedding 的前向映射。
    支持两种格式，均带严格校验：
      A) 预训练格式（pretrain_embedding.save_pretrained_embedding 落盘）：
         键 embed_proj_weight / embed_mlp_0_weight / embed_mlp_2_weight
         + input_dim/d_model/expand_ratio 元数据 + config 配方快照
      B) 完整模型格式：state_dict + model_arch，加载 StockTransformer
         后取其 embedding 前段
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(f"无法识别的 checkpoint 格式: {model_path}")

    # ---- A) 预训练 embedding 格式 ----
    if 'embed_proj_weight' in ckpt:
        input_dim = int(ckpt.get('input_dim', ModelConfig.INPUT_DIM))
        d_model = int(ckpt.get('d_model', ModelConfig.D_MODEL))
        expand_ratio = int(ckpt.get('expand_ratio', 2))
        if input_dim != ModelConfig.INPUT_DIM or d_model != ModelConfig.D_MODEL:
            raise ValueError(
                f"checkpoint 维度 (input={input_dim}, d_model={d_model}) "
                f"与当前配置 (INPUT_DIM={ModelConfig.INPUT_DIM}, "
                f"D_MODEL={ModelConfig.D_MODEL}) 不一致——产物来自另一代"
                f"特征/模型口径，拒绝评估以免输出无意义结果")

        emb = KLineEmbedding(input_dim=input_dim, d_model=d_model,
                             expand_ratio=expand_ratio).to(device)
        key_map = {'embed_proj_weight': 'embed_proj.weight',
                   'embed_mlp_0_weight': 'embed_mlp.0.weight',
                   'embed_mlp_2_weight': 'embed_mlp.2.weight'}
        missing = [src for src in key_map if src not in ckpt]
        if missing:
            raise ValueError(
                f"预训练 Embedding 缺少必需的权重 {missing}: {model_path}")
        pretrain_state = {dst: ckpt[src] for src, dst in key_map.items()}
        _validate_state_dict_match(emb.state_dict(), pretrain_state, model_path)
        emb.load_state_dict(pretrain_state)
        emb.eval()

        cfg = ckpt.get('config') or {}
        recipe = (f"预训练格式 | 训练配方(checkpoint内嵌): tag={cfg.get('tag')}"
                  f", w_shape={cfg.get('visreg_w_shape')}"
                  f", w_scale={cfg.get('visreg_w_scale')}"
                  f", lam={cfg.get('visreg_weight')}"
                  f", epochs={cfg.get('epochs')}")

        return (lambda x: emb(x)), recipe

    # ---- B) 完整模型格式 ----
    if 'state_dict' in ckpt:
        if 'model_arch' not in ckpt:
            raise ValueError(f"checkpoint 缺少 'model_arch' 元数据: {model_path}")
        model = create_model(model_arch=ckpt['model_arch']).to(device)
        _validate_state_dict_match(model.state_dict(), ckpt['state_dict'], model_path)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        if not (hasattr(model, 'embed_proj') and hasattr(model, 'embed_mlp')):
            raise ValueError("模型不含 embed_proj/embed_mlp，评估器不适用于此架构")
        ffn = FFNEmbedding(model.embed_proj, model.embed_mlp)
        return (lambda x: ffn(x)), f"完整模型格式 | arch={ckpt['model_arch']}"

    raise ValueError(f"无法识别的 checkpoint 格式: {model_path}")


# ==================== 测试池构建与校准 ====================

def resolve_feature_names(feature_normalizer, expected_dim):
    """特征名取自 normalizer 的权威定义并校验维度（原则 P3）"""
    names = [n for n, _ in feature_normalizer._feature_groups]
    if len(names) != expected_dim:
        raise ValueError(
            f"FeatureNormalizer 定义了 {len(names)} 列特征 {names}，但当前模型"
            f"输入维度为 {expected_dim}——normalizer.pkl 与权重不是同一代口径，"
            f"拒绝继续以免产出无意义结果")
    return names


def build_day_pool(n_max_days):
    """
    从测试集抽取单日K线（粗处理后 19 维向量）

    复用 run/pretrain 同一条 normalize_and_validate_context_window 管线，
    取每个上下文窗口最后一日；窗口起点大步进以降低相邻日的相关性。
    KLineEmbedding 逐时间步独立作用（不含跨时间混合），因此分析
    "单日19维 → d_model" 映射即可完整刻画 embedding 行为。
    """
    _, _, test_stock_info = load_and_preprocess_data()
    ctx = DataConfig.CONTEXT_LENGTH
    req = DataConfig.REQUIRED_LENGTH
    stride = max(ctx * 2, req)
    days, used_stocks = [], 0
    for stock in test_stock_info[:80]:
        data = stock['data']
        split = stock['test_split_point']
        end = min(split + req * 3, len(data) - req)
        got_this = False
        for i in range(split, max(split + 1, end), stride):
            seq = normalize_and_validate_context_window(
                data, i, ctx, check_limit_up=False, required_length=req,
                feature_normalizer=None, apply_fine_normalization=False)
            if seq is not None:
                days.append(seq[-1])
                got_this = True
            if len(days) >= n_max_days:
                break
        used_stocks += 1 if got_this else 0
        if len(days) >= n_max_days:
            break
    if len(days) < max(8, DEFAULT_SAMPLES // 4):
        raise RuntimeError(f"测试池只抽到 {len(days)} 个有效日样本，不足以诊断")
    pool = np.asarray(days, dtype=np.float32)
    # 收集顺序=股票顺序，不打乱的话 [:args.samples] 截取会使诊断样本
    # 集中在列表最前的股票；固定种子打乱保证可复现（校准用全池，
    # 分位数/端点统计与顺序无关，打乱只影响子采样代表性）
    np.random.default_rng(0).shuffle(pool)
    print(f"  测试池: {pool.shape[0]} 个单日样本（粗处理空间），"
          f"来自 {used_stocks} 只测试股票")
    return pool


def calibrate(coarse_pool, feature_normalizer, names):
    """
    细处理空间校准（原则 P1 的落点），返回后续分析共用的基准量。

    打印的表格供人工核查：mean≈0/std≈1 是 QuantileTransformer+
    StandardScaler 的契约，明显偏离说明 normalizer 与数据口径不符。
    """
    fine = np.asarray(feature_normalizer.transform(coarse_pool), dtype=np.float32)

    def _f32(a):
        # np.percentile 恒返回 float64；不显式收敛的话，后续任何参与运算的
        # 数组都会被提升为 double，而模型权重是 float32，混合即 RuntimeError
        return np.asarray(a, dtype=np.float32)

    q02, q50, q98 = map(_f32, np.percentile(
        fine, [100 * SWEEP_QLO, 50, 100 * SWEEP_QHI], axis=0))
    q25, q75 = map(_f32, np.percentile(fine, [25, 75], axis=0))

    # 边界焊死检测（诊断 high 影线列失真后新增，2026-08）：粗空间里大量
    # 样本恰好等于该列最小/最大值（一字板把影线钉死在 0），rank 变换把
    # 这些并列值映射为同一常数，形成无梯度平台。此类列 z≈0 不再是语义
    # 中心——连续性参考点自动改用池中位数，结论降权阅读。
    edge_pct = np.zeros(len(names), dtype=np.float32)
    for j in range(len(names)):
        col = coarse_pool[:, j]
        vmin, vmax = float(col.min()), float(col.max())
        edge_pct[j] = 100 * max(float(np.mean(col == vmin)),
                                float(np.mean(col == vmax)))
    plateau = edge_pct >= EDGE_MASS_WARN_PCT
    ref_z = np.where(plateau, q50, np.float32(0.0)).astype(np.float32)

    stats = {'fine_pool': fine, 'q02': q02, 'q50': q50, 'q98': q98,
             'iqr': q75 - q25,
             'anchor': q50.copy(),              # 基线日 = 池中位数向量
             'edge_pct': edge_pct, 'plateau': plateau, 'ref_z': ref_z}

    print("\n[校准表] 细处理空间（embedding 直接输入空间）实测统计：")
    print(f"  {'特征':<14s}{'mean':>8s}{'std':>8s}{'q02':>8s}"
          f"{'q50':>8s}{'q98':>8s}{'IQR':>8s}{'端点焊死%':>9s}")
    for j, name in enumerate(names):
        col = fine[:, j]
        print(f"  {name:<14s}{col.mean():>8.3f}{col.std():>8.3f}"
              f"{stats['q02'][j]:>8.3f}{q50[j]:>8.3f}{q98[j]:>8.3f}"
              f"{stats['iqr'][j]:>8.3f}{edge_pct[j]:>9.1f}")
    bad = [(n, float(c.mean()), float(c.std()))
           for n, c in zip(names, fine.T)
           if abs(float(c.mean())) > 0.15 or abs(float(c.std()) - 1.0) > 0.3]
    if bad:
        print("  ⚠ 以下列偏离 mean≈0/std≈1 契约较远，相关结论需打折阅读:")
        for n, m, s in bad:
            print(f"    {n}: mean={m:.3f}, std={s:.3f}")
    if plateau.any():
        pl = ", ".join(f"{names[j]}({edge_pct[j]:.0f}%)" for j in np.where(plateau)[0])
        print(f"  ⚠ 边界平台列: {pl} —— 大量样本焊死在支持域端点（rank 变换产物），"
              f"其连续性/翻转读数自动以池中位数为参考并整体降权")
    return stats


# ==================== 分析一：方向对比 ====================

def analyze_flip_contrast(pool, stats, embed_fn, names, device):
    """
    符号翻转 vs 同幅平移的响应比（逐列独立扰动，其余列保持原值）。

    翻转定义为绕该列池中位数镜像 x'=2·q50−x：对标准化后的相对量列，
    即把'高于历史典型水平'翻成同等程度的'低于'。此定义对所有列类型
    语义正确（旧版对变化率列用 1-x 属于 bug）。比值大 = 该维的方向
    信息被重点编码。
    """
    print("\n[1] 方向对比敏感性（翻转 vs 平移，逐列独立扰动）")
    n, c = pool.shape
    med = stats['q50']
    delta = stats['iqr'] * SHIFT_IQR_FRAC

    rows = np.repeat(np.arange(n), c)           # 样本优先展开
    cols = np.tile(np.arange(c), n)
    seq = np.arange(n * c)

    base_rep = np.repeat(pool, c, axis=0)       # [N*C, C]，第 r 行 = 样本 r//c
    plus = base_rep.copy()
    minus = base_rep.copy()
    flip = base_rep.copy()
    plus[seq, cols] += delta[cols]
    minus[seq, cols] -= delta[cols]
    flip[seq, cols] = 2.0 * med[cols] - pool[rows, cols]

    with torch.no_grad():
        b = _embed_numpy(embed_fn, pool, device)                    # [N,d]
        p_plus = _embed_numpy(embed_fn, plus, device).reshape(n, c, -1)
        p_minus = _embed_numpy(embed_fn, minus, device).reshape(n, c, -1)
        p_flip = _embed_numpy(embed_fn, flip, device).reshape(n, c, -1)

    # 只允许被扰动的列产生差异；对样本维取均值得到每列一个标量
    d_plus = np.linalg.norm(p_plus - b[:, None, :], axis=2).mean(axis=0)    # [C]
    d_minus = np.linalg.norm(p_minus - b[:, None, :], axis=2).mean(axis=0)
    flip_mag = np.linalg.norm(p_flip - b[:, None, :], axis=2).mean(axis=0)
    shift_mag = 0.5 * (d_plus + d_minus)
    ratios = flip_mag / (shift_mag + 1e-8)

    print(f"  扰动幅度=各列IQR×{SHIFT_IQR_FRAC}；比值=翻转位移/平移平均位移")
    print(f"  {'特征':<14s}{'ratio':>8s}{'翻转位移':>10s}{'平移位移':>10s}   参考")
    for j, name in enumerate(names):
        tag = ("强" if ratios[j] > RATIO_STRONG else
               "中" if ratios[j] > RATIO_MODERATE else "弱")
        print(f"  {name:<14s}{ratios[j]:>8.2f}{flip_mag[j]:>10.4f}"
              f"{shift_mag[j]:>10.4f}   {tag}")
    return {'ratios': ratios, 'flip_mag': flip_mag, 'shift_mag': shift_mag}


# ==================== 分析二：跨维交互残差 ====================

def analyze_interaction(pool, stats, embed_fn, names, device):
    """
    二阶交互残差：residual = f(x+δi+δj) − f(x+δi) − f(x+δj) + f(x)

    若 embedding 近似逐维加性映射，残差≈0；显著非零说明存在跨维非
    线性配合。除绝对强度外另报告 主效应归一比 = ‖residual‖ /
    ((‖Δ_i‖+‖Δ_j‖)/2)，无量纲、跨权重文件可比。
    """
    print("\n[2] 跨维交互残差矩阵")
    n, c = pool.shape
    delta = stats['iqr'] * SHIFT_IQR_FRAC
    pairs = [(i, j) for i in range(c) for j in range(i + 1, c)]
    p_cnt = len(pairs)

    # 构造批：base(N) + singles(N×C) + pairs(N×P)，全部一次 forward
    blocks = [pool.reshape(n, 1, c)]
    singles = np.repeat(pool, c, axis=0)             # [N*C, C] 样本优先
    col_of_row = np.tile(np.arange(c), n)
    singles[np.arange(n * c), col_of_row] += delta[col_of_row]
    blocks.append(singles.reshape(n, c, c))          # [N][第k行=第k列扰动]
    pair_blocks = np.empty((p_cnt, n, c), dtype=np.float32)
    for p, (i, j) in enumerate(pairs):
        blk = pool.copy()
        blk[:, i] += delta[i]
        blk[:, j] += delta[j]
        pair_blocks[p] = blk
    blocks.append(pair_blocks.transpose(1, 0, 2))    # [N, P, C]

    X = np.concatenate(blocks, axis=1)               # [N, 1+C+P, C]
    with torch.no_grad():
        Y = _embed_numpy(embed_fn, X.reshape(-1, c), device) \
            .reshape(n, 1 + c + p_cnt, -1)

    base = Y[:, 0, :]                                # [N,d]
    single = Y[:, 1:1 + c, :]                        # [N,C,d]
    pair = Y[:, 1 + c:, :]                           # [N,P,d]

    single_norm = np.linalg.norm(single - base[:, None, :], axis=2)  # [N,C]
    main_effect = single_norm.mean(axis=0)                            # [C]

    # 交互残差（向量化到配对轴）
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    res = np.linalg.norm(pair - single[:, ii, :] - single[:, jj, :]
                         + base[:, None, :], axis=2)  # [N,P]
    inter_abs = res.mean(axis=0)                     # [P]
    denom = 0.5 * (main_effect[ii] + main_effect[jj]) + 1e-8
    inter_rel = inter_abs / denom                    # 主效应归一比

    mat = np.zeros((c, c))
    for p, (i, j) in enumerate(pairs):
        mat[i, j] = mat[j, i] = inter_abs[p]

    order = np.argsort(-inter_rel)
    print(f"  扰动幅度=各列IQR×{SHIFT_IQR_FRAC}；rel=残差/两侧主效应均值（无量纲）")
    print(f"  平均绝对残差={inter_abs.mean():.4f}   中位 rel={np.median(inter_rel):.3f}")
    print(f"  最强交互 Top{min(TOP_K_INTERACT, p_cnt)}:")
    for rank, idx in enumerate(order[:TOP_K_INTERACT], 1):
        i, j = pairs[idx]
        print(f"    {rank:>2d}. {names[i]:<12s}-{names[j]:<12s}"
              f"abs={inter_abs[idx]:.4f}  rel={inter_rel[idx]:.3f}")

    name_idx = {nm: k for k, nm in enumerate(names)}
    print("  关键语义对照对:")
    for a, b in KEY_PAIR_NAMES:
        if a in name_idx and b in name_idx:
            i, j = sorted((name_idx[a], name_idx[b]))
            p = pairs.index((i, j))
            print(f"    {a}-{b}: abs={inter_abs[p]:.4f}  rel={inter_rel[p]:.3f}")

    return {'matrix': mat, 'pairs': pairs, 'inter_abs': inter_abs,
            'inter_rel': inter_rel, 'main_effect': main_effect}


# ==================== 分析三：连续性扫值 ====================

def analyze_continuity(stats, embed_fn, names, device, n_steps):
    """
    以池中位数为基线日，沿每列在 [q02,q98] 扫值，考察：
      平滑性     —— 输出轨迹的速度/曲率是否均匀
      边界尖锐度 —— 最大曲率位置是否落在 z≈0（历史中位水平）
    """
    print("\n[3] 连续性扫值")
    c = len(names)
    anchor = stats['anchor'][None, :].astype(np.float32)      # [1,C]
    q02, q98 = stats['q02'], stats['q98']

    grids = [np.linspace(q02[j], q98[j], n_steps) for j in range(c)]
    X = np.tile(anchor, (c * n_steps, 1))                     # [C*S, C]
    for j in range(c):
        X[j * n_steps:(j + 1) * n_steps, j] = grids[j]

    with torch.no_grad():
        Y = _embed_numpy(embed_fn, X, device)        # [C*S,d]

    results = {}
    print(f"  基线日=池中位数向量；每列扫 [{SWEEP_QLO:.2f},{SWEEP_QHI:.2f}] 分位区间")
    print(f"  {'特征':<14s}{'锋利度':>8s}{'峰值@':>9s}{'参考z':>7s}{'参考倍数':>9s}")
    for j, name in enumerate(names):
        seg = Y[j * n_steps:(j + 1) * n_steps]                # [S,d]
        curv = np.linalg.norm(seg[2:] - 2 * seg[1:-1] + seg[:-2], axis=1)  # [S-2]
        grid_mid = grids[j][1:-1]                             # 曲率对应的输入值
        sharp = float(curv.max() / (curv.mean() + 1e-8))
        peak_at = float(grid_mid[int(np.argmax(curv))])
        # 参考点自适应：符号对称列看 z=0（方向分界），边界平台列（影线族）
        # 看池中位数——后者是它们唯一有语义的中心
        ref = float(stats['ref_z'][j])
        ref_ratio = None
        lo, hi = grids[j][0], grids[j][-1]
        if lo <= ref <= hi:
            k = int(np.argmin(np.abs(grid_mid - ref)))
            ref_ratio = float(curv[k] / (curv.mean() + 1e-8))
        results[name] = {'sharpness': sharp, 'peak_at': peak_at,
                         'ref_z': ref, 'ref_ratio': ref_ratio,
                         'grid': grids[j], 'curv_norm':
                             curv / (curv.max() + 1e-8)}
        rr_str = f"{ref_ratio:>6.1f}x" if ref_ratio is not None else "      —"
        print(f"  {name:<14s}{sharp:>8.1f}{peak_at:>9.2f}{ref:>+7.2f}{rr_str}")

    ranked = sorted(results.items(), key=lambda kv: -kv[1]['sharpness'])
    print(f"  边界最陡的 3 列: {', '.join(nm for nm, _ in ranked[:3])}"
          f"（曲率峰越贴近各自参考点越好：符号对称列=方向分界，平台列=典型水平）")
    return results


# ==================== 分析四：特征消融 ====================

def analyze_ablation(pool, stats, embed_fn, names, device):
    """
    将某列替换为中性值（池中位数），测输出位移。

    与旧版的差别：中性点从手调常数改为实测中位数；度量从"输出范数
    变化"改为"输出位移的 L2 距离"——后者直接衡量信息通道强度，不被
    输出尺度契约混淆。
    """
    print("\n[4] 特征消融（替换为池中位数）")
    n, c = pool.shape
    med = stats['q50']

    masked = np.repeat(pool[:, None, :], c, axis=1)      # [N,C,C]
    col_idx = np.arange(c)
    for j in range(c):
        masked[:, j, j] = med[j]

    with torch.no_grad():
        Y = _embed_numpy(embed_fn, pool, device)
        Ym = _embed_numpy(embed_fn, masked.reshape(n * c, c), device) \
            .reshape(n, c, -1)

    disp = np.linalg.norm(Ym - Y[:, None, :], axis=2).mean(axis=0)   # [C]
    share = disp / (disp.sum() + 1e-8)

    order = np.argsort(-disp)
    print(f"  {'排名':<4s}{'特征':<14s}{'位移':>10s}{'占比':>8s}")
    for rank, j in enumerate(order, 1):
        print(f"  {rank:<4d}{names[j]:<14s}{disp[j]:>10.4f}{share[j]*100:>7.1f}%")
    return {'disp': disp, 'share': share}


# ==================== 总结与可视化 ====================

def summarize(names, results):
    print("\n" + "=" * 70)
    print("观察总结（非判卷；结合训练期指标阅读）")
    print("=" * 70)
    fc = results['flip_contrast']
    strong = [names[j] for j in range(len(names))
              if fc['ratios'][j] > RATIO_STRONG]
    weak = [names[j] for j in range(len(names))
            if fc['ratios'][j] <= RATIO_MODERATE]
    print(f"\n[方向] 编码了明确方向的维度: {', '.join(strong) or '（无）'}；"
          f"方向感弱的维度: {', '.join(weak) or '（无）'}")

    it = results['interaction']
    top_rel = np.argsort(-it['inter_rel'])[:3]
    txt = ', '.join(f"{names[it['pairs'][k][0]]}-{names[it['pairs'][k][1]]}"
                    f"(rel={it['inter_rel'][k]:.2f})" for k in top_rel)
    print(f"[交互] 非线性配合最强的三维对: {txt}")
    name_idx = {nm: k for k, nm in enumerate(names)}
    hints = []
    for a, b in KEY_PAIR_NAMES:
        if a in name_idx and b in name_idx:
            i, j = sorted((name_idx[a], name_idx[b]))
            k = it['pairs'].index((i, j))
            hints.append(f"{a}-{b} rel={it['inter_rel'][k]:.2f}")
    print(f"[交互] 关键语义对照对读数: {'; '.join(hints)}"
          f"（业务上预期这些应有可感知的非线性配合）")

    ct = results['continuity']
    peak_at_ref = [nm for nm, r in ct.items()
                   if r['ref_ratio'] is not None and r['ref_ratio'] > 1.5]
    sharpest = sorted(ct.items(), key=lambda kv: -kv[1]['sharpness'])[:3]
    sharp_txt = ', '.join("{}({:.1f})".format(nm, r['sharpness'])
                          for nm, r in sharpest)
    print(f"[连续性] 全局轨迹锋利度 Top3: {sharp_txt}；"
          f"在各自语义参考点（符号0/典型水平）出现曲率峰的维度: "
          f"{', '.join(peak_at_ref) or '（无）'}")

    ab = results['ablation']
    top_ab = np.argsort(-ab['disp'])[:3]
    ab_txt = ', '.join("{}({:.0f}%)".format(names[j], ab['share'][j] * 100)
                       for j in top_ab)
    print(f"[消融] 信息通道占比 Top3: {ab_txt}")


def visualize(names, results, save_dir, model_tag):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f'Embedding 训后诊断（{model_tag}）', fontsize=14,
                 fontweight='bold')

    # [0,0] 交互热力图
    ax = axes[0, 0]
    mat = results['interaction']['matrix']
    im = ax.imshow(mat, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title('跨维交互绝对残差')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # [0,1] 方向对比柱状图
    ax = axes[0, 1]
    ratios = results['flip_contrast']['ratios']
    colors = ['#2ecc71' if r > RATIO_STRONG else
              '#f39c12' if r > RATIO_MODERATE else '#e74c3c' for r in ratios]
    ax.bar(names, ratios, color=colors, alpha=0.85)
    ax.axhline(RATIO_STRONG, color='green', ls='--', alpha=0.5)
    ax.axhline(RATIO_MODERATE, color='orange', ls='--', alpha=0.5)
    ax.set_ylabel('翻转/平移 位移比')
    ax.set_title('方向对比敏感性')
    ax.tick_params(axis='x', rotation=60, labelsize=8)
    ax.grid(axis='y', alpha=0.3)

    # [1,0] 连续性曲率曲线（Top4 锋利度）
    ax = axes[1, 0]
    ct = results['continuity']
    top4 = [nm for nm, _ in
            sorted(ct.items(), key=lambda kv: -kv[1]['sharpness'])[:4]]
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for k, nm in enumerate(top4):
        r = ct[nm]
        ax.plot(r['grid'][1:-1], r['curv_norm'], '-o', markersize=2,
                color=palette[k % 4], label=f"{nm}(峰@{r['peak_at']:.2f})")
        ax.axvline(0, color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_xlabel('细处理空间扫值 z')
    ax.set_ylabel('曲率（按列内最大值归一）')
    ax.set_title('连续性扫值（锋利度Top4）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # [1,1] 消融占比水平条形图
    ax = axes[1, 1]
    ab = results['ablation']
    order = np.argsort(ab['share'])
    ax.barh([names[j] for j in order], ab['share'][order] * 100,
            color='#3498db', alpha=0.85)
    ax.set_xlabel('消融位移占比 (%)')
    ax.set_title('特征消融重要性')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"embedding_eval_{model_tag}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化已保存: {path}")
    return path


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description='Embedding 训后诊断 v2')
    parser.add_argument('--model', type=str, default=None,
                        help='待评估的 .pth 路径（默认取最新的预训练产物）')
    parser.add_argument('--list-models', action='store_true',
                        help='列出候选 .pth 并退出')
    parser.add_argument('--samples', type=int, default=DEFAULT_SAMPLES,
                        help=f'单日样本数（默认 {DEFAULT_SAMPLES}）')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS,
                        help=f'连续性扫值步数（默认 {DEFAULT_STEPS}）')
    parser.add_argument('--out', type=str, default=None,
                        help='可视化输出目录')
    parser.add_argument('--no-plot', action='store_true',
                        help='不生成图（仅终端读数）')
    args = parser.parse_args()

    if args.list_models:
        files = list_model_files()
        print("候选 .pth（按修改时间）：")
        for k, f in enumerate(files, 1):
            mb = os.path.getsize(f) / 1024 / 1024
            t = datetime.fromtimestamp(os.path.getmtime(f))
            print(f"  {k}. {os.path.basename(f)}  {t:%Y-%m-%d %H:%M}  {mb:.2f}MB")
        if not files:
            print("  （未找到任何 .pth）")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 权重
    if args.model:
        model_path = args.model
        if not os.path.exists(model_path):
            print(f"错误: 文件不存在 {model_path}")
            return
    else:
        files = list_model_files()
        if not files:
            print("错误: 未找到任何 .pth，用 --model 显式指定路径")
            return
        model_path = files[0]
    print(f"[步骤1] 加载权重: {os.path.basename(model_path)}")
    embed_fn, recipe = load_embedding_fn(model_path, device)
    print(f"  {recipe}")

    # 归一化器（评估的坐标定义，必须存在）
    if not os.path.exists(DataConfig.NORMALIZER_PATH):
        print(f"错误: 归一化器不存在 {DataConfig.NORMALIZER_PATH}，"
              f"先训练/重建它再评估")
        return
    normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)

    print("\n[步骤2] 构建测试池并校准...")
    coarse_pool = build_day_pool(max(args.samples * 2, DEFAULT_SAMPLES))
    names = resolve_feature_names(normalizer, ModelConfig.INPUT_DIM)
    stats = calibrate(coarse_pool, normalizer, names)
    pool = stats['fine_pool'][:args.samples]         # 分析统一在细处理空间

    print("\n[步骤3] 执行四项诊断...")
    results = {
        'flip_contrast': analyze_flip_contrast(pool, stats, embed_fn, names, device),
        'interaction': analyze_interaction(pool, stats, embed_fn, names, device),
        'continuity': analyze_continuity(stats, embed_fn, names, device, args.steps),
        'ablation': analyze_ablation(pool, stats, embed_fn, names, device),
    }

    summarize(names, results)

    if not args.no_plot:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        stem = ''.join(ch if (ch.isalnum() or ch in '_-') else '_'
                       for ch in stem)[:40]
        save_dir = args.out or os.path.join(PROJECT_ROOT, 'out_eval_results')
        try:
            visualize(names, results, save_dir, stem)
        except Exception as exc:                      # 图表失败不影响读数
            print(f"⚠ 可视化跳过: {exc}")


if __name__ == "__main__":
    main()
