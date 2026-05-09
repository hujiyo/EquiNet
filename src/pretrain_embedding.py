"""
EquiNet Embedding层训练脚本

通过对比学习 + 重建约束 + 球面均匀性，预训练 FFN-Embedding 层，
使其成为一个固定的、具有几何保证的K线特征提取器：

1. InfoNCE 对比损失：相似K线→嵌入距离近，相异K线→嵌入距离远
2. MLP 解码器重建损失：确保方向信息足以恢复原始10维特征
3. 均匀性损失：嵌入在单位超球面上均匀分布

用法：
  python src/pretrain_embedding.py                        # 使用默认参数
  python src/pretrain_embedding.py --epochs 300           # 自定义轮数
  python src/pretrain_embedding.py --temperature 0.05     # 自定义温度
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

from config import (ModelConfig, DataConfig, EmbeddingConfig, DeviceConfig,
                     TrainingConfig)
from data import (load_and_preprocess_data, FeatureNormalizer,
                  precompute_training_pool)
from training_utils import _get_amp_context


# ==================== 数据增强 ====================

class KLineAugmentation:
    """
    K线数据增强器，为对比学习生成正样本对

    对单条K线向量 [batch, 10] 应用增强：
    - OHLC (col 0-3): 高斯噪声，clip回 [-0.1, 0.1]
    - VWAP (col 4): 高斯噪声，clip回 [-0.1, 0.1]
    - Volume/Exchange (col 5-6): 随机缩放
    - MA偏离度 (col 7-9): 高斯噪声

    每次调用至少激活一种增强（不会是恒等变换）。
    """

    def __init__(self, noise_std=0.02, mask_prob=0.1,
                 vol_scale_range=(0.8, 1.2)):
        self.noise_std = noise_std
        self.mask_prob = mask_prob
        self.vol_scale_low = vol_scale_range[0]
        self.vol_scale_high = vol_scale_range[1]

    def __call__(self, x):
        """
        Args:
            x: [batch, 10] 归一化后的K线特征
        Returns:
            augmented: [batch, 10] 增强后的K线特征
        """
        augmented = x.clone()
        batch_size = x.size(0)

        # OHLC (col 0-3): 高斯噪声
        noise_ohl = torch.randn_like(augmented[:, :4]) * self.noise_std
        augmented[:, :4] = augmented[:, :4] + noise_ohl

        # VWAP (col 4): 高斯噪声
        noise_vwap = torch.randn_like(augmented[:, 4:5]) * self.noise_std
        augmented[:, 4:5] = augmented[:, 4:5] + noise_vwap

        # Volume/Exchange (col 5-6): 随机缩放
        scale = (torch.rand(batch_size, 2, device=x.device)
                 * (self.vol_scale_high - self.vol_scale_low)
                 + self.vol_scale_low)
        augmented[:, 5:7] = augmented[:, 5:7] * scale

        # MA偏离度 (col 7-9): 高斯噪声
        noise_ma = torch.randn_like(augmented[:, 7:10]) * self.noise_std
        augmented[:, 7:10] = augmented[:, 7:10] + noise_ma

        # 随机特征 masking（以一定概率将单个特征维度置零）
        if self.mask_prob > 0:
            mask = torch.rand_like(augmented) < self.mask_prob
            augmented = augmented.masked_fill(mask, 0.0)

        return augmented


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
    Embedding层训练模型 = KLineEmbedding + LayerNorm + MLP解码器

    LayerNorm 模拟下游 StockTransformer 的 Pre-Norm 行为：
    下游 Transformer 的第一层会对 embedding 输出做 LayerNorm，
    因此预训练的对比损失和重建损失也应作用在 LayerNorm 之后。

    使用 elementwise_affine=False（无可训练 γ/β），
    因为下游模型有自己的 LayerNorm 参数。

    解码器在预训练完成后丢弃，只保留 embedding 权重。
    """

    def __init__(self, input_dim=10, d_model=128, decoder_hidden_dim=512,
                 decoder_layers=2):
        super().__init__()
        self.embedding = KLineEmbedding(input_dim, d_model)
        self.input_dim = input_dim

        # 模拟下游 Pre-Norm：纯标准化，无可训练参数
        self.embed_norm = nn.LayerNorm(d_model, elementwise_affine=False)

        # MLP 解码器：逐日向量重建，不需要序列级 transformer
        layers = []
        in_dim = d_model
        for _ in range(decoder_layers - 1):
            layers.append(nn.Linear(in_dim, decoder_hidden_dim))
            layers.append(nn.GELU())
            in_dim = decoder_hidden_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

        # 解码器初始化
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [batch, 10]
        Returns:
            z_normed: [batch, d_model] LayerNorm 后的嵌入（模拟下游 Transformer 看到的）
            x_recon: [batch, 10] 重建的原始输入
        """
        z = self.embedding(x)
        z_normed = self.embed_norm(z)  # 模拟下游 Pre-Norm
        x_recon = self.decoder(z_normed)
        return z_normed, x_recon


# ==================== 损失函数 ====================

def info_nce_loss(z1, z2, temperature=0.07):
    """
    InfoNCE 对比损失

    正样本对 = 同一K线的两种增强视角
    负样本对 = batch内所有不同K线

    Args:
        z1: [batch, d] 视角1的嵌入向量
        z2: [batch, d] 视角2的嵌入向量
        temperature: softmax温度
    """
    batch_size = z1.size(0)
    # 拼接两个视角: [2*batch, d]
    z = torch.cat([z1, z2], dim=0)

    # 余弦相似度矩阵（输入已归一化，点积=余弦相似度）
    sim = torch.mm(z, z.t()) / temperature

    # 遮蔽对角线（自身不作为正/负样本）
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # 标签：z1[i] 的正样本是 z2[i]，即位置 i+batch
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device)
    ])

    return F.cross_entropy(sim, labels)


def uniformity_loss(z, t=None):
    """
    超球面均匀性损失 (Wang & Isola, 2020)

    最小化高斯核势能 → 等价于 Thomson 问题的连续松弛。
    值越小（越负），分布越均匀。

    Args:
        z: [batch, d] 嵌入向量
        t: 温度参数（默认使用 EmbeddingConfig.UNIFORMITY_T）
    """
    if t is None:
        t = EmbeddingConfig.UNIFORMITY_T
    pw_dists = torch.pdist(z, p=2)
    return pw_dists.pow(2).mul(-t).exp().mean().log()


def entropy_regularization_loss(z, inv_temperature=1.0):
    """
    熵正则化损失：鼓励嵌入空间各维度均匀利用

    对 batch 内每个维度计算 soft sign 分布（正/负比例），
    最大化二值熵 → 每个维度在 batch 内正负均衡。

    受 Kronos 的 Binary Spherical Quantization 启发，
    但适配连续嵌入场景，使用 sigmoid soft quantization。

    Args:
        z: [batch, d_model] 嵌入向量
        inv_temperature: 控制soft sign的锐度
    Returns:
        loss: 标量，范围 [-1, 0]，0 = 完全均匀利用
    """
    scale = math.sqrt(z.shape[-1]) * inv_temperature
    p = torch.sigmoid(z * scale)
    avg_p = p.mean(dim=0)
    entropy = -(avg_p * torch.log(avg_p + 1e-8) +
                (1 - avg_p) * torch.log(1 - avg_p + 1e-8))
    max_entropy = math.log(2) * z.shape[-1]
    return -entropy.sum() / max_entropy


def scale_regularization(z, target_std):
    """
    标准差正则化：约束 embedding 原始输出的标准差接近目标值

    embedding 输出和位置编码在主模型中直接相加，
    两者的标准差必须匹配（≈0.2），否则信号比失衡。

    Args:
        z: [batch, d_model] embedding 原始输出（LayerNorm 之前）
        target_std: 目标标准差（应与位置编码的 std 一致）
    Returns:
        loss: 标量，0 = 完美匹配
    """
    actual_std = z.std()
    return ((actual_std - target_std) / target_std) ** 2


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

    def step(self):
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups,
                                         self.base_lrs):
            param_group['lr'] = lr

    def _get_lr(self):
        if self.current_epoch <= self.warmup_epochs:
            # 线性预热
            return self.base_lrs[0] * self.current_epoch / max(1, self.warmup_epochs)
        # 余弦退火
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
            'temperature': EmbeddingConfig.TEMPERATURE,
            'alpha': EmbeddingConfig.ALPHA,
            'beta': EmbeddingConfig.BETA,
            'gamma': EmbeddingConfig.GAMMA,
            'entropy_weight': EmbeddingConfig.ENTROPY_WEIGHT,
        },
    }
    if metrics:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, path)
    print(f"  Embedding权重已保存: {path}")


# ==================== 主训练函数 ====================

def pretrain(train_stock_info, feature_normalizer=None, device=None,
             epochs=None, batch_size=None, lr=None, temperature=None):
    """
    Embedding层训练-主函数
    """
    epochs = epochs or EmbeddingConfig.EPOCHS
    batch_size = batch_size if batch_size is not None else EmbeddingConfig.BATCH_SIZE
    lr = lr if lr is not None else EmbeddingConfig.LEARNING_RATE
    temperature = temperature if temperature is not None else EmbeddingConfig.TEMPERATURE

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
        decoder_hidden_dim=EmbeddingConfig.DECODER_HIDDEN_DIM,
        decoder_layers=EmbeddingConfig.DECODER_LAYERS,
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

    # 5. 数据增强
    augmentation = KLineAugmentation(
        noise_std=EmbeddingConfig.NOISE_STD,
        mask_prob=EmbeddingConfig.FEATURE_MASK_PROB,
        vol_scale_range=EmbeddingConfig.VOLUME_SCALE_RANGE,
    )

    # 损失权重
    alpha = EmbeddingConfig.ALPHA
    beta = EmbeddingConfig.BETA
    gamma = EmbeddingConfig.GAMMA
    delta = EmbeddingConfig.ENTROPY_WEIGHT

    # 6. 训练循环
    print(f"\n{'='*60}")
    amp_str = "BF16混合精度" if TrainingConfig.USE_AMP and device.type == 'cuda' else "FP32精度"
    print(f"开始 Embedding 预训练")
    print(f"  轮数={epochs}  batch={batch_size}  lr={lr}  τ={temperature}")
    print(f"  精度={amp_str}  设备={device}")
    print(f"  损失权重: α(对比)={alpha}  β(重建)={beta}  "
          f"γ(均匀)={gamma}  δ(熵)={delta}")
    print(f"{'='*60}")

    best_loss = float('inf')
    output_dir = EmbeddingConfig.OUTPUT_DIR

    amp_ctx = _get_amp_context(device)
    entropy_inv_temp = EmbeddingConfig.ENTROPY_INV_TEMPERATURE

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_contrast = 0.0
        epoch_recon = 0.0
        epoch_uniform = 0.0
        epoch_entropy = 0.0
        epoch_scale = 0.0
        n_batches = 0

        t0 = time.time()

        dataset = TensorDataset(
            torch.tensor(epoch_data[epoch - 1], dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            drop_last=True, num_workers=0, pin_memory=True)

        for (batch,) in loader:
            batch = batch.to(device)

            # 生成两个增强视角
            view1 = augmentation(batch)
            view2 = augmentation(batch)

            # 前向传播 + 损失计算（AMP混合精度）
            optimizer.zero_grad()
            with amp_ctx:
                # 原始 embedding 输出（LayerNorm 之前）用于 std 控制
                z1_raw = model.embedding(view1)
                z2_raw = model.embedding(view2)

                # LayerNorm 后的向量用于对比/重建/均匀/熵损失
                z1 = model.embed_norm(z1_raw)
                z2 = model.embed_norm(z2_raw)
                recon1 = model.decoder(z1)
                recon2 = model.decoder(z2)

                z1_cos = F.normalize(z1, p=2, dim=-1)
                z2_cos = F.normalize(z2, p=2, dim=-1)
                loss_contrast = info_nce_loss(z1_cos, z2_cos, temperature)
                loss_recon = (F.mse_loss(recon1, batch) +
                             F.mse_loss(recon2, batch)) / 2
                loss_uniform = (uniformity_loss(z1) + uniformity_loss(z2)) / 2
                loss_entropy = entropy_regularization_loss(
                    torch.cat([z1, z2], dim=0), entropy_inv_temp)

                # 标准差正则化：原始输出 std 必须与位置编码匹配
                target_std = EmbeddingConfig.TARGET_STD
                loss_scale = (scale_regularization(z1_raw, target_std) +
                             scale_regularization(z2_raw, target_std)) / 2

                epsilon = EmbeddingConfig.SCALE_WEIGHT
                loss = (alpha * loss_contrast +
                        beta * loss_recon +
                        gamma * loss_uniform +
                        delta * loss_entropy +
                        epsilon * loss_scale)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            EmbeddingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_contrast += loss_contrast.item()
            epoch_recon += loss_recon.item()
            epoch_uniform += loss_uniform.item()
            epoch_entropy += loss_entropy.item()
            epoch_scale += loss_scale.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = epoch_loss / n_batches
        avg_contrast = epoch_contrast / n_batches
        avg_recon = epoch_recon / n_batches
        avg_uniform = epoch_uniform / n_batches
        avg_entropy = epoch_entropy / n_batches
        avg_scale = epoch_scale / n_batches
        current_lr = scheduler.get_lr()

        # 打印日志
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"Loss={avg_loss:.4f}  "
              f"Contrast={avg_contrast:.4f}  "
              f"Recon={avg_recon:.4f}  "
              f"Uniform={avg_uniform:.4f}  "
              f"Entropy={avg_entropy:.4f}  "
              f"Scale={avg_scale:.4f}  "
              f"LR={current_lr:.6f}  "
              f"Time={elapsed:.1f}s")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(output_dir, 'best_embedding.pth')
            save_pretrained_embedding(
                model.embedding, best_path,
                metrics={
                    'epoch': epoch,
                    'loss': avg_loss,
                    'contrast_loss': avg_contrast,
                    'recon_loss': avg_recon,
                    'uniform_loss': avg_uniform,
                    'entropy_loss': avg_entropy,
                }
            )


    # 保存最终模型
    final_path = os.path.join(output_dir, 'pretrained_embedding.pth')
    save_pretrained_embedding(model.embedding, final_path,
                              metrics={
                                  'epoch': epochs,
                                  'loss': avg_loss,
                                  'best_loss': best_loss,
                              })

    print(f"\n预训练完成！最佳 Loss={best_loss:.4f}")
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
    parser.add_argument('--temperature', type=float, default=None,
                        help=f'InfoNCE温度 (默认 {EmbeddingConfig.TEMPERATURE})')
    parser.add_argument('--alpha', type=float, default=None,
                        help=f'对比损失权重 (默认 {EmbeddingConfig.ALPHA})')
    parser.add_argument('--beta', type=float, default=None,
                        help=f'重建损失权重 (默认 {EmbeddingConfig.BETA})')
    parser.add_argument('--gamma', type=float, default=None,
                        help=f'均匀性损失权重 (默认 {EmbeddingConfig.GAMMA})')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'输出目录')

    args = parser.parse_args()

    # 覆盖配置
    if args.epochs:
        EmbeddingConfig.EPOCHS = args.epochs
    if args.batch_size:
        EmbeddingConfig.BATCH_SIZE = args.batch_size
    if args.alpha is not None:
        EmbeddingConfig.ALPHA = args.alpha
    if args.beta is not None:
        EmbeddingConfig.BETA = args.beta
    if args.gamma is not None:
        EmbeddingConfig.GAMMA = args.gamma
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
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
