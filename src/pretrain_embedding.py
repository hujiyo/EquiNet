"""
EquiNet Embedding层训练脚本

通过 SIGReg 几何正则 + 重建约束，预训练 FFN-Embedding 层，
使其成为一个固定的、具有几何保证的K线特征提取器：

1. SIGReg 几何正则 (Balestriero & LeCun, 2025)：约束嵌入分布趋向各向同性高斯
   通过 Cramér-Wold + Epps-Pulley 检验，数学上保证无维度/子空间/聚类坍塌
2. MLP 解码器重建损失：确保嵌入向量足以恢复原始10维特征

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

from config import (ModelConfig, DataConfig, EmbeddingConfig, DeviceConfig,
                     TrainingConfig)
from data import (load_and_preprocess_data, FeatureNormalizer,
                  precompute_training_pool)
from training_utils import _get_amp_context
from sigreg import SIGRegLoss


# ==================== 模型定义 ====================

class KLineEmbedding(nn.Module):
    """
    单日K线嵌入模块（结构与 StockTransformer 的 FFN-Embedding 完全一致）

    Linear(10→128) → GELU → Linear(128→128) + 残差连接
    保留完整向量信息（方向 + 幅度）。
    """

    def __init__(self, input_dim=10, d_model=128):
        super().__init__()
        self.embed_proj = nn.Linear(input_dim, d_model)
        self.embed_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 与 StockTransformer 一致的初始化
        nn.init.xavier_uniform_(self.embed_proj.weight,
                                gain=ModelConfig.EMBEDDING_INIT_GAIN)
        nn.init.zeros_(self.embed_proj.bias)
        nn.init.xavier_uniform_(self.embed_mlp[1].weight,
                                gain=ModelConfig.FFN_INIT_GAIN)
        nn.init.zeros_(self.embed_mlp[1].bias)

    def forward(self, x):
        """
        Args:
            x: [batch, 10] 归一化K线特征
        Returns:
            z: [batch, d_model] 嵌入向量（保留方向+幅度）
        """
        h = self.embed_proj(x)
        h = h + self.embed_mlp(h)
        return h


class PretrainModel(nn.Module):
    """
    Embedding层训练模型 = KLineEmbedding + 线性解码器

    X → Embedding → S → Linear → Y
                      ↑            ↑
                  SIGReg(S)    MSE(Y, X)

    线性解码器迫使 embedding 将信息编码在128维空间的线性可提取方向上，
    不给 embedding 通过非线性查表偷懒的空间。
    解码器在预训练完成后丢弃，只保留 embedding 权重。
    """

    def __init__(self, input_dim=10, d_model=128):
        super().__init__()
        self.embedding = KLineEmbedding(input_dim, d_model)
        self.input_dim = input_dim

        # 线性解码器: x̂ = Wz + b, W ∈ R^(10×128)
        self.decoder = nn.Linear(d_model, input_dim)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        Args:
            x: [batch, 10]
        Returns:
            z: [batch, d_model] 嵌入向量 S（用于 SIGReg）
            recon: [batch, 10] 重建的原始输入 Y
        """
        z = self.embedding(x)
        recon = self.decoder(z)
        return z, recon


# ==================== 数据收集 ====================

def _deduplicate_klines(data, precision=3):
    """
    对K线向量去重

    将特征量化到指定精度后，移除完全相同的行，
    保证后续采样的样本多样性。
    """
    n_before = len(data)
    rounded = np.round(data, precision)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    data = data[unique_idx]
    n_after = len(data)
    print(f"  去重: {n_before:,} → {n_after:,} "
          f"(去除 {n_before - n_after:,} 条重复, "
          f"保留 {n_after / n_before * 100:.1f}%)")
    return data


def collect_kline_data(train_stock_info, feature_normalizer=None):
    """
    从训练集中提取逐日K线特征向量，去重后返回完整池

    利用 precompute_training_pool 获取所有合法的 [N, 45, 10] 样本，
    展平为 [N*45, 10] 逐日向量，去重后返回。

    Args:
        train_stock_info: 训练集股票信息列表
        feature_normalizer: 特征归一化器

    Returns:
        kline_data: [M, 10] numpy array (去重后的完整池)
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

    # 数据量过大时先随机预采样，避免 np.unique 内存爆炸
    max_dedup_size = EmbeddingConfig.MAX_SAMPLES * EmbeddingConfig.EPOCHS * 3
    if len(kline_data) > max_dedup_size:
        print(f"  预采样: {len(kline_data):,} → {max_dedup_size:,} 条")
        idx = np.random.choice(len(kline_data), max_dedup_size, replace=False)
        kline_data = kline_data[idx]

    kline_data = _deduplicate_klines(kline_data, EmbeddingConfig.DEDUP_PRECISION)

    print(f"  特征范围:")
    for i, name in enumerate(['Open', 'High', 'Low', 'Close',
                               'VWAP', 'Volume', 'Exchange',
                               'MA5', 'MA10', 'MA20']):
        col = kline_data[:, i]
        print(f"    {name:>8s}: [{col.min():.4f}, {col.max():.4f}]  "
              f"μ={col.mean():.4f}  σ={col.std():.4f}")

    return kline_data


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

def save_pretrained_embedding(embedding, path, metrics=None):
    """
    保存预训练 embedding 权重

    格式与 StockTransformer 的 embed_proj / embed_mlp 直接兼容。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        'embed_proj_weight': embedding.embed_proj.weight.data.cpu(),
        'embed_proj_bias': embedding.embed_proj.bias.data.cpu(),
        'embed_mlp_1_weight': embedding.embed_mlp[1].weight.data.cpu(),
        'embed_mlp_1_bias': embedding.embed_mlp[1].bias.data.cpu(),
        'input_dim': embedding.embed_proj.in_features,
        'd_model': embedding.embed_proj.out_features,
        'config': {
            'epochs': EmbeddingConfig.EPOCHS,
            'batch_size': EmbeddingConfig.BATCH_SIZE,
            'learning_rate': EmbeddingConfig.LEARNING_RATE,
            'beta': EmbeddingConfig.BETA,
            'sigreg_weight': EmbeddingConfig.SIGREG_WEIGHT,
        },
    }
    if metrics:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"  Embedding权重已保存: {path}")


# ==================== 主训练函数 ====================

def pretrain(train_stock_info, feature_normalizer=None, device=None,
             epochs=None, batch_size=None, lr=None):
    """
    Embedding层训练-主函数
    """
    epochs = epochs or EmbeddingConfig.EPOCHS
    batch_size = batch_size if batch_size is not None else EmbeddingConfig.BATCH_SIZE
    lr = lr if lr is not None else EmbeddingConfig.LEARNING_RATE

    # 1. 收集并去重K线数据（返回完整池）
    kline_pool = collect_kline_data(train_stock_info, feature_normalizer)
    pool_size = len(kline_pool)

    # 2. 预采样 MAX_SAMPLES * EPOCHS 条，供所有 epoch 使用
    samples_per_epoch = EmbeddingConfig.MAX_SAMPLES
    total_needed = samples_per_epoch * epochs

    if pool_size >= total_needed:
        indices = np.random.choice(pool_size, total_needed, replace=False)
        print(f"\n[数据] 池中有 {pool_size:,} 条唯一K线，"
              f"无重复采样 {total_needed:,} 条")
    else:
        indices = np.random.choice(pool_size, total_needed, replace=True)
        repeat_ratio = total_needed / pool_size
        print(f"\n[数据] 池中有 {pool_size:,} 条唯一K线，"
              f"需 {total_needed:,} 条（平均重复 {repeat_ratio:.1f} 次）")

    pre_sampled = kline_pool[indices]
    epoch_data = pre_sampled.reshape(epochs, samples_per_epoch, -1)
    print(f"  分配: {epochs} 个 epoch × {samples_per_epoch:,} 条/epoch")

    # 3. 创建模型
    model = PretrainModel(
        input_dim=EmbeddingConfig.INPUT_DIM,
        d_model=EmbeddingConfig.D_MODEL,
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

    # 损失权重
    beta = EmbeddingConfig.BETA
    sigreg_weight = EmbeddingConfig.SIGREG_WEIGHT
    target_std = EmbeddingConfig.TARGET_STD

    # SIGReg 几何正则
    sigreg_loss_fn = SIGRegLoss(
        d_model=EmbeddingConfig.D_MODEL,
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
    print(f"  损失权重: β(重建)={beta}  λ(SIGReg)={sigreg_weight}")
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
        epoch_recon = 0.0
        epoch_sigreg = 0.0
        n_batches = 0

        t0 = time.time()

        dataset = TensorDataset(
            torch.tensor(epoch_data[epoch - 1], dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

        for (batch,) in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            with amp_ctx:
                z, recon = model(batch)

                # SIGReg: 缩放到 N(0,1) 目标，EP 检验各向同性高斯
                loss_sigreg = sigreg_loss_fn(z / target_std)

                # 重建损失
                loss_recon = F.mse_loss(recon, batch)

                loss = beta * loss_recon + sigreg_weight * loss_sigreg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            EmbeddingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_recon += loss_recon.item()
            epoch_sigreg += loss_sigreg.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_sigreg = epoch_sigreg / n_batches
        current_lr = scheduler.get_lr()

        # 打印日志
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f}  "
              f"Recon={avg_recon:.4f}  "
              f"SIGReg={avg_sigreg:.4f}  "
              f"LR={current_lr:.6f}  "
              f"Time={elapsed:.1f}s")

        # 记录最佳模型（仅存内存，不写磁盘）
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {
                'epoch': epoch,
                'loss': avg_loss,
                'recon_loss': avg_recon,
                'sigreg_loss': avg_sigreg,
            }

    # 训练结束后一次性保存到磁盘
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        best_path = os.path.join(output_dir, 'best_embedding.pth')
        save_pretrained_embedding(model.embedding, best_path,
                                  metrics=best_metrics)

    final_path = os.path.join(output_dir, 'pretrained_embedding.pth')
    save_pretrained_embedding(model.embedding, final_path,
                              metrics={
                                  'epoch': epochs,
                                  'loss': avg_loss,
                                  'best_loss': best_loss,
                              })

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
    parser.add_argument('--beta', type=float, default=None,
                        help=f'重建损失权重 (默认 {EmbeddingConfig.BETA})')
    parser.add_argument('--sigreg-weight', type=float, default=None,
                        help=f'SIGReg损失权重 (默认 {EmbeddingConfig.SIGREG_WEIGHT})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'输出目录')

    args = parser.parse_args()

    # 覆盖配置
    if args.epochs:
        EmbeddingConfig.EPOCHS = args.epochs
    if args.batch_size:
        EmbeddingConfig.BATCH_SIZE = args.batch_size
    if args.beta is not None:
        EmbeddingConfig.BETA = args.beta
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
