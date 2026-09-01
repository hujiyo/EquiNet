"""
EquiNet Embedding层训练——成对余弦相似度回归（SIM）想法验证脚本
（想法来源：code.html 可视化演示）

核心想法（演示结论）：
  以"输入空间成对余弦相似度"为几何目标，让 z 空间的成对余弦
  cos(z_i,z_j) 直接回归 cos(x_i,x_j)（MSE，batch 内全部样本对）：
  输入空间相似的样本对在 z 中拉近、不相似的推远，拉力连续、无温度/
  锐化超参。演示中三种目标相似度（余弦 / z-score内积 / 原始点积）里
  余弦最稳定，且学到的 z 图案与初始分布高度相似——"保真几何"：
  不引入外部锚点（标签/簇），只要求 z 忠实保持输入已有的样本间
  相似度结构（下游 attention 消费的恰是 z 的成对点积）。

与 pretrain_embedding.py 主脚本的关系（单一变量实验设计）：
  SupCon 用"多头标签一致度核"塑形 z 的成对几何（语义沉淀），
  SIM 用"输入相似度核"塑形成对几何（分布保真）——同一损失家族
  （核回归），只换目标核。本臂与主配方的唯一差异 = 删 SupCon 换 SIM，
  其余（VISReg/Recon/CE粗/CE细）权重逐位相同：
  - 单一变量原则：下游对照的任何差异可干净归因于"输入核 vs 标签核"；
  - 额外删减（去 CE/去 Recon）会引入第二变量，赢了不知归因谁，
    输了说不清是思路差还是砍多了；
  - CE（信息轴类别监督）保留：与几何轴正交，且是"几何监督挤语义"
    （vws 退化教训）的防线；Recon 的 D 是幅度/细粒度信息唯一守门人
    （SIM 对幅度不变）；VISReg 管幅度契约（下游消费带幅度的点积）。

损失公式（默认臂，对照主配方"0.2·SupCon + 0.8·[λ·VISReg+(1-λ)·Recon]
+ 0.1·CE + 0.1·FINE"，W_AUX=0.8 使辅助预算与主配方逐位相同）：
  loss = W_SIM·SIM + W_AUX·[λ·VISReg + (1-λ)·Recon(O + DERIVED_WEIGHT·D)]
         + 0.1·CE(粗,4头) + 0.1·FINE(细,f5桶)
  其中 λ=EmbeddingConfig.VISREG_WEIGHT=0.6。
  W_SIM=1.0 独立于预算公式（SIM 原始值 O(0.02) vs SupCon 的 ln(B)≈7.8，
  量级不可比；用梯度实测校准：SIM≈0.43/Recon≈1.85/VISReg≈1.23，
  1.0 使 SIM 占总梯度 ~25-30%，几何主损失地位确立且 30 轮实测压不坏 O/D）。
  --pure 复刻演示的纯 SIM 模式（W_AUX=0 且无 CE）：观察无 VISReg 时
  z std 的漂移（幅度失控的直接读数），仅诊断用。

验收读数（逐 epoch）：
  - r：样本对上 cos(x) vs cos(z) 的 Pearson 相关（演示的 stR 指标，
    本脚本主指标；train=去重 batch 平均，eval=held-out 固定子集）
  - SIM：(cos_z − cos_x)² 均值
  - O=/D=：线性解码器重建误差（信息轴，验证 SIM 不破坏可读性）
  - CLS=/FINE=：粗/细头 CE/Acc（信息轴类别监督，与主配方同读数）
  - CKA：z 与标签核的中心化相关（几何轴交叉读数——输入核教的几何
    是否顺带对齐语义；主配方 SupCon 直接优化此对象，SIM 臂看它
    能否"免费"涨起来，是下游裁决前最有信息量的中间指标）
  - zstd：z 输出标准差（幅度契约，目标 TARGET_STD；--pure 下看漂移）

注意基线陷阱（合成数据实测）：
  随机初始化的 KLineEmbedding 近似线性投影，本身已保余弦几何
  （JL 引理，19维→128维），基线 r 可高达 ~0.95——"训练前基线"打印
  是想法增量的第一读数：真实数据上 r0 低（几何被初始化/其它损失
  破坏）→ SIM 有提升空间；r0 已高 → 想法的天花板有限，
  看末轮 r 与 O/D 的权衡是否仍优于现配方。

用法：
  python src/pretrain_embedding_test.py
  python src/pretrain_embedding_test.py --pure            # 纯SIM复刻演示
  python src/pretrain_embedding_test.py --epochs 50 --w-sim 2.0
产物: out/best_embedding_simtest.pth（tag=simtest，不覆盖主产物）
"""

import os
import re
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (DataConfig, EmbeddingConfig, DeviceConfig, ModelConfig,
                    TrainingConfig)
from data import load_and_preprocess_data, FeatureNormalizer
from training_utils import _get_amp_context
from visreg import VISRegLoss
from pretrain_embedding import (KLineEmbedding, collect_kline_data,
                                build_derived_targets, WarmupCosineScheduler,
                                save_pretrained_embedding, measure_embedding_std,
                                build_cls_head_spec, compute_cls_stats,
                                compute_cls_targets, label_kernel_from_targets,
                                compute_label_cka, CLS_HEAD_SPEC,
                                CLS_SHORT_NAMES, FINE_HEAD_BASES, fine_head_name)

# ==================== 测试臂超参（本地常量，勿改 EmbeddingConfig） ====================
# 轮数/warmup 对齐主脚本配方（损失配方不同，训练预算对齐才可比）；
# 池上限小于主脚本（验证想法优先，控制内存）

TEST_EPOCHS = 100                           # 轮数：对齐主脚本 EPOCHS=100（30轮实测
                                            # 末期 D/O/SIM 仍同步下降未收敛，LR 退火
                                            # 压慢尾段；CLI --epochs 可覆盖）
POOL_CAP = 2_000_000                        # K线池上限（测试够用，控制内存）
EVAL_N = 4096                               # held-out 评估样本数（16.7M 对，r 统计充足）
W_SIM = 1.0                                 # SIM 权重：原始值 O(0.02~0.1)（余弦差平方），
                                            # 梯度实测(随机初始化 batch=2560):
                                            # SIM≈0.43 / Recon≈1.85 / VISReg≈1.23，
                                            # 1.0 起步使 SIM 占总梯度 ~40%
W_AUX = 0.8                                 # 辅助损失(VISReg+Recon)总权重 =主配方
                                            # (1-w_c)=0.8：辅助预算与主配方逐位
                                            # 相同，下游对照差异干净归因于核的替换
WARMUP_EPOCHS = 10                          # 预热轮数（对齐主脚本 WARMUP_EPOCHS=10）


# ==================== SIM 损失 ====================

def pairwise_sim_stats(z, x):
    """
    成对余弦相似度回归：SIM = E_{i≠j}[(cos_z_ij − cos_x_ij)²]

    目标核 t = x_n·x_nᵀ（输入空间余弦；x 是数据非模型输出，天然无梯度），
    预测核 s = z_n·z_nᵀ（可微）。对角线（自身对恒为 1）排除。
    复用点积矩阵，不用 torch.pdist（项目纪律：成对量走 matmul）。
    fp32 计算（AMP 下 z 为 bf16，先 .float()）。

    Returns:
        (sim_loss, r)：r = 非对角 (t, s) 的 Pearson 相关，no_grad 纯监控
    """
    z_n = F.normalize(z.float(), dim=1)
    x_n = F.normalize(x.float(), dim=1)
    s = z_n @ z_n.t()                                   # 预测：z 空间余弦核
    t = x_n @ x_n.t()                                   # 目标：输入空间余弦核
    n = s.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=s.device)
    loss = ((s - t) ** 2)[off].mean()
    with torch.no_grad():
        a = s[off].detach()
        b = t[off]
        a = a - a.mean()
        b = b - b.mean()
        r = ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()
    return loss, r


@torch.no_grad()
def evaluate_similarity(embedding, eval_x):
    """
    held-out 评估：r / SIM / z std（演示 stR 主指标的泛化版）

    eval_x 为固定去重子集（与训练分布一致但样本不重合于任一 batch 的
    去重纪律 guarantee），训练全程复用同一批样本 → r 的逐 epoch 变化
    纯粹反映模型进步，无采样噪声。
    """
    was_training = embedding.training
    embedding.eval()
    z = embedding(eval_x)
    if was_training:
        embedding.train()
    loss, r = pairwise_sim_stats(z, eval_x)
    return r, loss.item(), z.float().std().item()


# ==================== 数据采样 ====================

def sample_batch_idx(pool, pool_derived, pool_mask, batch_size, precision,
                     oversample):
    """
    index 版 batch 采样（对齐主脚本 sample_diverse_batch 的去重纪律）：

    1. 过采样 oversample × batch_size 条池行索引
    2. round 到 precision 位后按行去重（假负样本防护：重复K线互为
       "相似对"是恒等任务，毒化几何信号——与 SupCon 同一教训）
    3. 取前 batch_size 行，衍生目标/掩码按相同池索引同步取出

    与 sample_diverse_batch 的差别：返回行数据的同时保留衍生/掩码
    行对齐（主脚本靠 epoch 级预采样对齐，本脚本逐 batch 采样更贴近
    演示的 minibatch 形态）。
    """
    n_over = min(int(batch_size * oversample), len(pool))
    idx = np.random.choice(len(pool), n_over, replace=True)
    rounded = np.round(pool[idx], precision)
    _, unique_pos = np.unique(rounded, axis=0, return_index=True)
    unique_pos = np.sort(unique_pos)
    sel = idx[unique_pos]
    if len(sel) < batch_size:                       # 极端情况：去重后不够，补采凑满
        extra_needed = batch_size - len(sel)
        sel = np.concatenate([sel, np.random.choice(
            len(pool), extra_needed, replace=len(pool) < extra_needed)])
    sel = sel[:batch_size]
    return pool[sel], pool_derived[sel], pool_mask[sel]


# ==================== 主训练函数 ====================

def pretrain_sim(train_stock_info, feature_normalizer=None, device=None,
                 tag='simtest', epochs=TEST_EPOCHS, w_sim=W_SIM, w_aux=W_AUX):
    """
    SIM 想法验证训练主函数

    w_aux=0 即纯 SIM 模式（--pure）：无 VISReg/Recon，仅监控 z std 漂移；
    w_aux>0 为组合臂：SIM 管"样本间相似度结构保真"，VISReg 管幅度契约，
    Recon(O/D) 管单样本信息可线性读出——三条轴各自的验收读数都在日志里。
    """
    pure = (w_aux == 0.0)
    batch_size = EmbeddingConfig.BATCH_SIZE
    precision = EmbeddingConfig.DEDUP_PRECISION
    oversample = int(EmbeddingConfig.BATCH_DEDUP_OVERSAMPLE)
    steps_per_epoch = max(1, EmbeddingConfig.MAX_SAMPLES // batch_size)
    input_dim = ModelConfig.INPUT_DIM
    n_derived = EmbeddingConfig.N_DERIVED_FEATURES
    target_std = EmbeddingConfig.TARGET_STD
    derived_weight = EmbeddingConfig.DERIVED_WEIGHT
    lam = EmbeddingConfig.VISREG_WEIGHT              # 辅助损失内部 λ 分配（沿用主脚本语义）
    w_cls = EmbeddingConfig.CLS_WEIGHT               # CE 粗头权重（0.1，同主配方）
    w_fine = EmbeddingConfig.CLS_FINE_WEIGHT         # CE 细头权重（0.1，同主配方）

    # 1. 收集K线池 + 衍生目标（行对齐）
    kline_pool = collect_kline_data(train_stock_info, feature_normalizer,
                                    pool_cap=POOL_CAP)
    print(f"\n[数据] 构建衍生目标（与 {len(kline_pool):,} 行池逐行对齐）...")
    pool_derived, pool_mask = build_derived_targets(kline_pool, feature_normalizer)

    # held-out 评估子集：过采样+去重取前 EVAL_N（与训练 batch 同去重口径）
    eval_x_np, _, _ = sample_batch_idx(kline_pool, pool_derived, pool_mask,
                                       EVAL_N, precision, oversample)
    eval_x = torch.tensor(eval_x_np, dtype=torch.float32, device=device)
    print(f"  held-out 评估集: {EVAL_N:,} 条 ({EVAL_N * (EVAL_N - 1) // 2:,} 对)")

    # ---- 分类头（粗+细，同主配方）/ CKA 交叉监控 ----
    # CE 是信息轴类别监督（与几何轴正交），保留 = 与主配方逐位对齐
    # （单一变量：删 SupCon 换 SIM 是本臂唯一差异）；
    # CKA 监控恒开（纯模式除外）：输入核教的几何是否顺带对齐语义，
    # 需要粗头标签 → 边界总是计算
    cls_heads_spec = build_cls_head_spec() if not pure else {}
    use_cls = len(cls_heads_spec) > 0
    cls_boundaries = None
    if not pure:
        cls_stats = compute_cls_stats(kline_pool, coarse=True,
                                      fine=EmbeddingConfig.CLS_FINE_ENABLED)
        cls_boundaries = {head: torch.tensor(b, dtype=torch.float32, device=device)
                          for head, (b, _) in cls_stats.items()}
        print(f"  分类头边界已预计算（池上分位数, {len(cls_boundaries)} 头）")

    # 2. 模型：真实 KLineEmbedding + 线性解码器（O/D 探针）+ 线性分类头
    embedding = KLineEmbedding(input_dim, ModelConfig.D_MODEL).to(device)
    params = list(embedding.parameters())
    decoder = None
    cls_heads = None
    if not pure:
        decoder = nn.Linear(ModelConfig.D_MODEL, input_dim + n_derived,
                            bias=True).to(device)
        params += list(decoder.parameters())
        if use_cls:
            # 同主脚本 PretrainModel：单层线性头，bias=0 初始化使初始
            # logits≈0 → 初始 CE≈ln(C)，与回归头互不干扰
            cls_heads = nn.ModuleDict({
                name: nn.Linear(ModelConfig.D_MODEL, n_classes, bias=True)
                for name, n_classes in cls_heads_spec.items()
            }).to(device)
            for head in cls_heads.values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            params += list(cls_heads.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"\n[模型] KLineEmbedding({input_dim}→{ModelConfig.D_MODEL})  "
          f"+ 线性解码器{'' if decoder is not None else '(纯SIM模式: 无)'}"
          f"  + 分类头×{len(cls_heads_spec) if cls_heads is not None else 0}  "
          f"参数量={n_params:,}")

    # 3. 优化器/调度器/损失
    optimizer = torch.optim.AdamW(params, lr=EmbeddingConfig.LEARNING_RATE,
                                  weight_decay=EmbeddingConfig.WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=min(WARMUP_EPOCHS, max(1, epochs // 6)),
        total_epochs=epochs,
        eta_min=EmbeddingConfig.COSINE_ETA_MIN)
    visreg_loss_fn = None
    if not pure:
        visreg_loss_fn = VISRegLoss(
            num_slices=EmbeddingConfig.VISREG_NUM_SLICES,
            w_scale=EmbeddingConfig.VISREG_W_SCALE,
            w_shape=EmbeddingConfig.VISREG_W_SHAPE,
            w_center=EmbeddingConfig.VISREG_W_CENTER,
        ).to(device)

    # 4. 训练循环
    print(f"\n{'='*60}")
    print(f"开始 SIM 想法验证训练 (成对余弦相似度回归)")
    print(f"  轮数={epochs}  steps/epoch={steps_per_epoch}  batch={batch_size}  "
          f"lr={EmbeddingConfig.LEARNING_RATE}  设备={device}")
    if pure:
        print(f"  损失公式: {w_sim:.2f}·SIM  (纯模式, 复刻演示)")
    else:
        print(f"  损失公式: {w_sim:.2f}·SIM + {w_aux:.2f}·"
              f"[{lam:.2f}·VISReg + {1 - lam:.2f}·Recon(O+{derived_weight:.1f}·D)]"
              + (f" + {w_cls:.2f}·CLS({','.join(CLS_SHORT_NAMES[n] for n in CLS_HEAD_SPEC)})"
                 f" + {w_fine:.2f}·FINE(f{EmbeddingConfig.CLS_FINE_BUCKETS})"
                 if use_cls else ""))
        print(f"  (对照主配方 0.2·SupCon + 0.8·[0.6·VISReg+0.4·Recon] + "
              f"0.1·CLS + 0.1·FINE：辅助预算逐位相同，唯一差异=几何核)")
    print(f"  SIM = E[(cos_z − cos_x)²], batch 全对去对角, fp32, 直接作用 z 本体"
          f" (无 projector, 演示验证可行; O/D 同步监控验证可读性)")
    print(f"  主指标 r = cos(x) vs cos(z) 的 Pearson 相关"
          f" (train=batch平均 / eval=held-out)")
    print(f"{'='*60}")

    r0, sim0, std0 = evaluate_similarity(embedding, eval_x)
    print(f"  [训练前基线] r={r0:.4f}  SIM={sim0:.4f}  zstd={std0:.4f} "
          f"(目标std={target_std})")

    amp_ctx = _get_amp_context(device)
    history = []

    for epoch in range(1, epochs + 1):
        embedding.train()
        if decoder is not None:
            decoder.train()
        if cls_heads is not None:
            cls_heads.train()
        t0 = time.time()
        ep_loss = 0.0          # 总损失（加权后）
        ep_sim_w = 0.0         # w_sim·SIM 加权贡献
        ep_sim_raw = 0.0       # SIM 原始值
        ep_r_train = 0.0       # 训练 batch 上 r 平均
        ep_visreg_raw = 0.0
        ep_recon_raw = 0.0
        ep_recon_orig = 0.0
        ep_recon_derived = 0.0
        ep_cls_raw = 0.0       # 粗头 CE 平均（原始值）
        ep_cls_w = 0.0         # w_cls·粗头CE 加权贡献
        ep_cls_fine_raw = 0.0  # 细头 CE 平均（原始值）
        ep_cls_fine_w = 0.0    # w_fine·细头CE 加权贡献
        ep_cls_head_raw = {}   # 逐头 CE（粗+细，按头名）
        ep_cls_head_acc = {}   # 逐头准确率（粗+细，按头名）
        ep_cka = 0.0           # CKA(z, 标签核)（batch 级平均）
        n_steps = 0

        for _ in range(steps_per_epoch):
            x_np, d_np, m_np = sample_batch_idx(
                kline_pool, pool_derived, pool_mask, batch_size, precision,
                oversample)
            batch_x = torch.tensor(x_np, dtype=torch.float32).to(device)
            batch_d = torch.tensor(d_np, dtype=torch.float32).to(device)
            batch_m = torch.tensor(m_np, dtype=torch.float32).to(device)

            # 分类标签：由归一化输入即时计算（CE/CKA 共消费，零存储）
            cls_targets = (compute_cls_targets(batch_x, cls_boundaries)
                           if cls_boundaries is not None else None)

            optimizer.zero_grad()
            # 前向走 AMP（小 MLP），SIM 损失在 AMP 外算 fp32
            # （bf16 下余弦核差的平方精度不足，同 SupCon logits 的教训）
            with amp_ctx:
                z = embedding(batch_x)
                if not pure:
                    recon = decoder(z)
                    loss_visreg = visreg_loss_fn(z / target_std)
                    loss_recon_orig = F.mse_loss(recon[:, :input_dim], batch_x)
                    diff2 = (recon[:, input_dim:] - batch_d) ** 2
                    num = (diff2 * batch_m).sum(dim=0)
                    den = batch_m.sum(dim=0) + 1e-8
                    loss_recon_derived = (num / den).mean()
                    loss_recon = (loss_recon_orig
                                  + derived_weight * loss_recon_derived)

            loss_sim, r_b = pairwise_sim_stats(z, batch_x)
            if pure:
                loss = w_sim * loss_sim
            else:
                loss = w_sim * loss_sim + w_aux * (lam * loss_visreg
                                                   + (1 - lam) * loss_recon)

            # 分类头（粗+细）：fp32 计算（CE 在 bf16 下精度不足，同 SupCon）。
            # 梯度经线性头回流 embedding，逼 z 出现可线性读出的类别可分结构
            if use_cls:
                cls_logits = {name: head(z.float())
                              for name, head in cls_heads.items()}
                cls_losses = {
                    name: F.cross_entropy(cls_logits[name], cls_targets[name])
                    for name in cls_heads_spec
                }
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
            torch.nn.utils.clip_grad_norm_(params,
                                           TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            ep_loss += loss.item()
            ep_sim_w += w_sim * loss_sim.item()
            ep_sim_raw += loss_sim.item()
            ep_r_train += r_b
            if not pure:
                ep_visreg_raw += loss_visreg.item()
                ep_recon_raw += loss_recon.item()
                ep_recon_orig += loss_recon_orig.item()
                ep_recon_derived += loss_recon_derived.item()
            if use_cls:
                if loss_cls is not None:
                    ep_cls_w += w_cls * loss_cls.item()
                    ep_cls_raw += loss_cls.item()
                if loss_cls_fine is not None:
                    ep_cls_fine_w += w_fine * loss_cls_fine.item()
                    ep_cls_fine_raw += loss_cls_fine.item()
                for name in cls_heads_spec:
                    ep_cls_head_raw[name] = (ep_cls_head_raw.get(name, 0.0)
                                             + cls_losses[name].item())
                    acc = (cls_logits[name].argmax(dim=1) == cls_targets[name])
                    ep_cls_head_acc[name] = (ep_cls_head_acc.get(name, 0.0)
                                             + acc.float().mean().item())
                # CKA 交叉监控：z 与标签核的中心化相关——输入核教的几何
                # 是否顺带对齐语义（主配方 SupCon 直接优化此对象）
                with torch.no_grad():
                    k_l = label_kernel_from_targets(cls_targets,
                                                    list(cls_heads_spec))
                    ep_cka += compute_label_cka(z.detach().float(), k_l)
            n_steps += 1

        scheduler.step()
        r_eval, sim_eval, std_eval = evaluate_similarity(embedding, eval_x)
        history.append({'epoch': epoch, 'r_eval': r_eval, 'sim_eval': sim_eval,
                        'zstd': std_eval})

        # 日志：r 是主指标（train/eval 双读数），O=/D= 信息轴探针，
        # CLS/FINE=类别监督读数（格式同主脚本），CKA=几何轴交叉读数
        # （输入核几何 vs 语义核），zstd 幅度契约
        aux_str = ""
        if not pure:
            aux_str = (f"VISReg={ep_visreg_raw / n_steps:.4f}  "
                       f"Recon={ep_recon_raw / n_steps:.4f} "
                       f"(O={ep_recon_orig / n_steps:.4f}/"
                       f"D={ep_recon_derived / n_steps:.4f})  ")
        cls_str = ""
        if use_cls:
            cls_parts = []
            for name in CLS_HEAD_SPEC:
                ce = ep_cls_head_raw[name] / n_steps
                acc = ep_cls_head_acc[name] / n_steps
                cls_parts.append(f"{CLS_SHORT_NAMES[name]}={ce:.3f}/{acc*100:.0f}%")
            if ep_cls_raw > 0:
                cls_str += f"CLS={ep_cls_raw / n_steps:.4f} ({' '.join(cls_parts)})  "
            fine_parts = []
            for name in FINE_HEAD_BASES:
                head = fine_head_name(name)
                ce = ep_cls_head_raw[head] / n_steps
                acc = ep_cls_head_acc[head] / n_steps
                fine_parts.append(
                    f"{CLS_SHORT_NAMES[name]}{EmbeddingConfig.CLS_FINE_BUCKETS}="
                    f"{ce:.3f}/{acc*100:.0f}%")
            if ep_cls_fine_raw > 0:
                cls_str += (f"FINE={ep_cls_fine_raw / n_steps:.4f} "
                            f"({' '.join(fine_parts)})  ")
        cka_str = f"CKA={ep_cka / n_steps:.4f}  " if use_cls else ""
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={ep_loss / n_steps:.4f} "
              f"({ep_sim_w / n_steps:.4f}·SIM)  "
              f"SIM={ep_sim_raw / n_steps:.4f}  "
              f"r={ep_r_train / n_steps:.3f}/{r_eval:.3f}  "
              f"{aux_str}"
              f"{cls_str}"
              f"{cka_str}"
              f"zstd={std_eval:.3f}  "
              f"LR={scheduler.get_lr():.2e}  "
              f"Time={time.time() - t0:.1f}s")

    # 5. 保存（tag 隔离产物，不覆盖主脚本 best_embedding.pth）
    last_metrics = {
        'epoch': epochs,
        'loss': ep_loss / n_steps,
        'w_sim': w_sim,
        'w_aux': w_aux,
        'pure': pure,
        'r_eval': r_eval,
        'sim_eval': sim_eval,
        'sim_train': ep_sim_raw / n_steps,
        'zstd': std_eval,
        'r_baseline': r0,
    }
    if not pure:
        last_metrics.update({
            'visreg_raw': ep_visreg_raw / n_steps,
            'recon_orig': ep_recon_orig / n_steps,
            'recon_derived': ep_recon_derived / n_steps,
        })
    if use_cls:
        last_metrics.update({
            'cls_weighted': ep_cls_w / n_steps,
            'cls_raw': ep_cls_raw / n_steps,
            'cls_fine_weighted': ep_cls_fine_w / n_steps,
            'cls_fine_raw': ep_cls_fine_raw / n_steps,
            'cka': ep_cka / n_steps,
        })
        for name in cls_heads_spec:
            last_metrics[f'cls_ce_{name}'] = ep_cls_head_raw[name] / n_steps
            last_metrics[f'cls_acc_{name}'] = ep_cls_head_acc[name] / n_steps
    ckpt_name = f'best_embedding{("_" + tag) if tag else ""}.pth'
    best_path = os.path.join(EmbeddingConfig.OUTPUT_DIR, ckpt_name)
    save_pretrained_embedding(embedding, best_path, metrics=last_metrics,
                              decoder=decoder, tag=tag)

    # 6. 落盘 round-trip 验证（save/load 后在真实池上测输出std）
    measure_embedding_std(best_path, kline_pool, device)

    print(f"\n{'='*60}")
    print(f"SIM 想法验证完成")
    print(f"  r: {r0:.4f} (基线) → {r_eval:.4f} (末轮, held-out)")
    print(f"  zstd: {std_eval:.4f} (目标 {target_std}, 幅度契约)")
    if use_cls:
        print(f"  CKA: {ep_cka / n_steps:.4f} (输入核几何 vs 语义核, "
              f"主配方 SupCon 直接优化此对象——对照读数)")
    if pure:
        print(f"  [纯SIM] 幅度无契约——zstd 漂移即'余弦对幅度不变'的实证，"
              f"组合臂(W_AUX>0)由 VISReg 管住")
    print(f"  产物: {best_path}")
    print(f"{'='*60}")

    return embedding


# ==================== 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='EquiNet Embedding SIM想法验证脚本 (成对余弦相似度回归)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--epochs', type=int, default=TEST_EPOCHS,
                        help=f'训练轮数 (默认 {TEST_EPOCHS})')
    parser.add_argument('--pure', action='store_true',
                        help='纯SIM模式：复刻演示，无 VISReg/Recon/CE 辅助损失，'
                             '监控 z std 漂移（仅诊断用，产物不可插下游）')
    parser.add_argument('--w-sim', type=float, default=W_SIM,
                        help=f'SIM 损失权重 (默认 {W_SIM})')
    parser.add_argument('--w-aux', type=float, default=W_AUX,
                        help=f'辅助损失(VISReg+Recon)总权重，0 等价 --pure (默认 {W_AUX})')
    parser.add_argument('--tag', type=str, default='simtest',
                        help='实验臂标签：产物存为 best_embedding_<tag>.pth '
                             '(默认 simtest，不覆盖主产物)')

    args = parser.parse_args()
    if args.tag is not None and not re.fullmatch(r'[\w\-]+', args.tag):
        parser.error(f"--tag 仅允许字母/数字/下划线/连字符: {args.tag!r}")
    if args.w_sim <= 0:
        parser.error("--w-sim 必须为正数")
    if args.w_aux < 0:
        parser.error("--w-aux 不能为负（0 等价 --pure）")
    if args.pure:
        args.w_aux = 0.0

    # 固定种子：评估子集/采样/初始化可复现（r 逐 epoch 可比）
    np.random.seed(DataConfig.RANDOM_SEED)
    torch.manual_seed(DataConfig.RANDOM_SEED)

    device = DeviceConfig.get_device()

    print("[步骤1] 加载训练数据...")
    train_stock_info, _, _ = load_and_preprocess_data()

    print("\n[步骤2] 加载特征归一化器...")
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print("  归一化器不存在，先运行 python src/data.py 创建")
        sys.exit(1)

    print("\n[步骤3] 开始 SIM 想法验证训练...")
    pretrain_sim(
        train_stock_info=train_stock_info,
        feature_normalizer=feature_normalizer,
        device=device,
        tag=args.tag,
        epochs=args.epochs,
        w_sim=args.w_sim,
        w_aux=args.w_aux,
    )


if __name__ == "__main__":
    main()
