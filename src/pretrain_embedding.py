"""
EquiNet Embedding层训练脚本

通过 SIGReg 几何正则 + 重建约束，预训练 FFN-Embedding 层，
使其成为一个固定的、具有几何保证的K线特征提取器：

1. SIGReg 几何正则 (Balestriero & LeCun, 2025)：约束嵌入分布趋向各向同性高斯
   通过 Cramér-Wold + Epps-Pulley 检验，数学上保证无维度/子空间/聚类坍塌
2. MLP 解码器重建损失：确保嵌入向量足以恢复原始15维特征
   非线性解码器允许 embedding 自由学习特征融合，而不仅限于线性可编码的表示

用法：
  python src/pretrain_embedding.py                        # 使用默认参数
  python src/pretrain_embedding.py --epochs 300           # 自定义轮数
"""

import os
import sys
import math
import argparse
import time
import copy
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
from sigreg import SIGRegLoss


# ==================== 模型定义 ====================

class KLineEmbedding(nn.Module):
    """
    单日K线嵌入模块（结构与 StockTransformer 的 FFN-Embedding 完全一致）

    Linear(15→128) → GELU → Linear(128→128) + 残差连接
    保留完整向量信息（方向 + 幅度）。
    """

    def __init__(self, input_dim=15, d_model=128):
        super().__init__()
        self.embed_proj = nn.Linear(input_dim, d_model, bias=False)
        self.embed_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        # 残差等贡献初始化（非逐层调参，从架构推导）
        # 原理：残差分支输出std = 主路径输出std，确保128维空间充分覆盖
        # GELU有效增益≈0.588 → MLP层σ = 1/(√d·0.588) ≈ 0.150
        # 两条路径各贡献std≈0.141 → 合计std≈0.2（匹配SIGReg目标）
        nn.init.normal_(self.embed_mlp[1].weight,
                        std=1.0 / (math.sqrt(d_model) * 0.588))
        nn.init.normal_(self.embed_proj.weight,
                        std=0.2 / (math.sqrt(2) * math.sqrt(input_dim)))

    def forward(self, x):
        """
        Args:
            x: [batch, 15] 归一化K线特征
        Returns:
            z: [batch, d_model] 嵌入向量（保留方向+幅度）
        """
        h = self.embed_proj(x)
        h = h + self.embed_mlp(h)
        return h


class PretrainModel(nn.Module):
    """
    Embedding层训练模型 = KLineEmbedding + MLP解码器

    X → Embedding → S → MLP → Y
                      ↑           ↑
                  SIGReg(S)   MSE(Y, X)

    非线性解码器允许 embedding 自由学习特征融合表示，
    不强制要求信息线性可编码。
    下游 Transformer（6层 FFN 128→512→128）的解码能力远超此解码器，
    只要解码器能恢复的信息，backbone 一定能提取。
    解码器在预训练完成后丢弃，只保留 embedding 权重。
    """

    def __init__(self, input_dim=15, d_model=128):
        super().__init__()
        self.embedding = KLineEmbedding(input_dim, d_model)
        self.input_dim = input_dim

        # MLP解码器: Linear → GELU → Linear（与 embedding 对称）
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, input_dim, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 15]
        Returns:
            z: [batch, d_model] 嵌入向量 S（用于 SIGReg）
            recon: [batch, 15] 重建的原始输入 Y
        """
        z = self.embedding(x)
        recon = self.decoder(z)
        return z, recon


# ==================== 数据收集 ====================

def collect_kline_data(train_stock_info, feature_normalizer=None):
    """
    从训练集中提取逐日K线特征向量，返回完整池（不去重）

    去重由 sample_diverse_batch() 在每个 batch 采样时执行，
    保证 batch 内多样性，同时池子保留最大信息量。

    Args:
        train_stock_info: 训练集股票信息列表
        feature_normalizer: 特征归一化器

    Returns:
        kline_data: [M, 15] numpy array (完整池)
    """
    print("\n[数据收集] 提取逐日K线向量...")

    pool_inputs, _, _, _, _, _ = precompute_training_pool(
        train_stock_info, feature_normalizer
    )

    n_samples, seq_len, feat_dim = pool_inputs.shape
    kline_data = pool_inputs.reshape(-1, feat_dim)
    print(f"  展平后总K线数: {len(kline_data):,}")

    valid_mask = np.all(np.isfinite(kline_data), axis=1)
    kline_data = kline_data[valid_mask]
    print(f"  有效K线数: {len(kline_data):,}")

    # 数据量过大时预采样（仅受内存限制，不做去重）
    max_pool_size = EmbeddingConfig.MAX_SAMPLES * EmbeddingConfig.EPOCHS * 5
    if len(kline_data) > max_pool_size:
        print(f"  预采样: {len(kline_data):,} → {max_pool_size:,} 条")
        idx = np.random.choice(len(kline_data), max_pool_size, replace=False)
        kline_data = kline_data[idx]

    print(f"  池大小: {len(kline_data):,} (不去重)")
    print(f"  特征范围:")
    for i, name in enumerate(['Open', 'High', 'Low', 'Close',
                               'VWAP', 'Volume', 'Exchange',
                               'MA5', 'MA10', 'MA20',
                               'DIF', 'DEA', 'MACD_Hist',
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
        'embed_mlp_1_weight': embedding.embed_mlp[1].weight.data.cpu(),
        'input_dim': embedding.embed_proj.in_features,
        'd_model': embedding.embed_proj.out_features,
        'config': {
            'epochs': EmbeddingConfig.EPOCHS,
            'batch_size': EmbeddingConfig.BATCH_SIZE,
            'learning_rate': EmbeddingConfig.LEARNING_RATE,
            'sigreg_weight': EmbeddingConfig.SIGREG_WEIGHT,
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
    epoch_data = pre_sampled.reshape(epochs, samples_per_epoch * oversample, -1)
    print(f"  分配: {epochs} 个 epoch × {samples_per_epoch * oversample:,} 条/epoch")
    print(f"  batch 内去重: precision={precision}, "
          f"oversample={oversample}x → DataLoader batch={loader_batch_size}")

    # 3. 创建模型
    model = PretrainModel(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
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

    # 损失权重 (凸组合: λ·SIGReg + (1-λ)·Recon)
    sigreg_weight = EmbeddingConfig.SIGREG_WEIGHT
    target_std = EmbeddingConfig.TARGET_STD

    # 自适应归一化: EMA 跟踪各损失量级，归一化至 O(1) 后再做凸组合
    sigreg_ema = None
    recon_ema = None
    ema_momentum = 0.9

    # SIGReg 几何正则
    sigreg_loss_fn = SIGRegLoss(
        d_model=ModelConfig.D_MODEL,
        num_slices=EmbeddingConfig.SIGREG_NUM_SLICES,
        t_max=EmbeddingConfig.SIGREG_T_MAX,
        n_points=EmbeddingConfig.SIGREG_N_POINTS,
    ).to(device)

    # 5. 训练循环
    print(f"\n{'='*60}")
    amp_str = "BF16混合精度" if TrainingConfig.USE_AMP and device.type == 'cuda' else "FP32精度"
    print(f"开始 Embedding 预训练")
    print(f"  轮数={epochs}  batch={batch_size}  lr={lr}")
    print(f"  精度={amp_str}  设备={device}")
    print(f"  损失公式: {sigreg_weight:.0%}·(SIGReg/ema) + {1 - sigreg_weight:.0%}·(Recon/ema)")
    print(f"{'='*60}")

    best_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_metrics = None
    output_dir = EmbeddingConfig.OUTPUT_DIR

    amp_ctx = _get_amp_context(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_sigreg_w = 0.0  # SIGReg 加权贡献 (λ·sigreg/ema)
        epoch_recon_w = 0.0   # Recon 加权贡献 ((1-λ)·recon/ema)
        epoch_sigreg_raw = 0.0  # SIGReg 原始值
        epoch_recon_raw = 0.0   # Recon 原始值
        epoch_dedup_total = 0   # batch内去重去掉的条数
        n_batches = 0

        t0 = time.time()

        dataset = TensorDataset(
            torch.tensor(epoch_data[epoch - 1], dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=loader_batch_size, shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

        for (raw_batch,) in loader:
            # batch 内去重：round → unique → 取前 batch_size
            raw_np = raw_batch.numpy()
            rounded = np.round(raw_np, precision)
            _, unique_idx = np.unique(rounded, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            deduped = raw_np[unique_idx]
            epoch_dedup_total += len(raw_np) - len(deduped)

            # 取 batch_size 条（不够则全部用上）
            batch = torch.tensor(
                deduped[:batch_size], dtype=torch.float32).to(device)

            optimizer.zero_grad()
            with amp_ctx:
                z, recon = model(batch)

                # SIGReg: 缩放到 N(0,1) 目标，EP 检验各向同性高斯
                loss_sigreg = sigreg_loss_fn(z / target_std)

                # 重建损失
                loss_recon = F.mse_loss(recon, batch)

                # 首次迭代用原始值初始化 EMA
                if sigreg_ema is None:
                    sigreg_ema = loss_sigreg.item()
                    recon_ema = loss_recon.item()

                # 自适应归一化凸组合 (归一化后各 ≈O(1)，λ 真实反映占比)
                loss = (sigreg_weight * (loss_sigreg / sigreg_ema)
                        + (1 - sigreg_weight) * (loss_recon / recon_ema))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            EmbeddingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 统计
            raw_sigreg = loss_sigreg.item()
            raw_recon = loss_recon.item()
            sigreg_w = sigreg_weight * raw_sigreg / sigreg_ema
            recon_w = (1 - sigreg_weight) * raw_recon / recon_ema
            epoch_loss += sigreg_w + recon_w
            epoch_sigreg_w += sigreg_w
            epoch_recon_w += recon_w
            epoch_sigreg_raw += raw_sigreg
            epoch_recon_raw += raw_recon
            n_batches += 1

            # 更新归一化基准
            sigreg_ema = ema_momentum * sigreg_ema + (1 - ema_momentum) * raw_sigreg
            recon_ema = ema_momentum * recon_ema + (1 - ema_momentum) * raw_recon

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_sigreg_w = epoch_sigreg_w / n_batches
        avg_recon_w = epoch_recon_w / n_batches
        avg_sigreg_raw = epoch_sigreg_raw / n_batches
        avg_recon_raw = epoch_recon_raw / n_batches
        current_lr = scheduler.get_lr()

        # 打印日志
        avg_dedup = epoch_dedup_total / max(1, n_batches)
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f} ({avg_sigreg_w:.4f}·SIGReg + {avg_recon_w:.4f}·Recon)  "
              f"SIGReg={avg_sigreg_raw:.4f}  Recon={avg_recon_raw:.4f}  "
              f"LR={current_lr:.6f}  "
              f"Dedup={avg_dedup:.0f}/batch  "
              f"Time={elapsed:.1f}s")

        # 记录最佳模型（仅存内存，不写磁盘）
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'epoch': epoch,
                'loss': avg_loss,
                'sigreg_weighted': avg_sigreg_w,
                'recon_weighted': avg_recon_w,
            }

    # 训练结束后保存最佳模型到磁盘
    best_path = None
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        best_path = os.path.join(output_dir, 'best_embedding.pth')
        save_pretrained_embedding(model.embedding, best_path,
                                  metrics=best_metrics, decoder=model.decoder)

    # 测量保存文件的输出std
    if best_path is not None:
        print("\n[输出std验证]")
        model.eval()
        with torch.no_grad():
            test_input = torch.randn(10000, ModelConfig.INPUT_DIM, device=device)
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
            tmp = KLineEmbedding(ModelConfig.INPUT_DIM, ModelConfig.D_MODEL).to(device)
            tmp.embed_proj.weight.data.copy_(ckpt['embed_proj_weight'])
            tmp.embed_mlp[1].weight.data.copy_(ckpt['embed_mlp_1_weight'])
            z = tmp(test_input)
            print(f"  {os.path.basename(best_path)}: 输出std = {z.std().item():.4f}")

    print(f"\n预训练完成！最佳 Loss={best_loss:.4f} (第{best_epoch}轮)")
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
    parser.add_argument('--sigreg-weight', type=float, default=None,
                        help=f'SIGReg损失权重λ (默认 {EmbeddingConfig.SIGREG_WEIGHT})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'输出目录')

    args = parser.parse_args()

    # 覆盖配置
    if args.epochs:
        EmbeddingConfig.EPOCHS = args.epochs
    if args.batch_size:
        EmbeddingConfig.BATCH_SIZE = args.batch_size
    if args.sigreg_weight is not None:
        EmbeddingConfig.SIGREG_WEIGHT = args.sigreg_weight
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
