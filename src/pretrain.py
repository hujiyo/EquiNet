"""
EquiNet 自监督预训练脚本

GPT式自回归预训练：模型在K线序列上进行因果预测，
学习时间序列的动态规律，预测下一根K线的 Open(60%)、Close(25%)、Amount(15%)。

用法：
  python src/pretrain.py                        # 使用默认参数
  python src/pretrain.py --epochs 100           # 自定义轮数
"""

import os
import sys
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (ModelConfig, DataConfig, PretrainConfig, EmbeddingConfig, DeviceConfig)
from model import create_model, generate_causal_mask
from data import (load_and_preprocess_data, FeatureNormalizer,
                  precompute_pretrain_pool, sample_pretrain_batch)
from training_utils import (
    _get_amp_context,
    create_optimizer_from_config_for_params,
    create_scheduler_from_config,
)


def pretrain(model, train_pool, val_pool,
             epochs=PretrainConfig.EPOCHS,
             learning_rate=PretrainConfig.LEARNING_RATE,
             batch_size=PretrainConfig.BATCH_SIZE,
             batches_per_epoch=PretrainConfig.BATCHES_PER_EPOCH,
             device=None):
    """
    自回归预训练主循环

    Args:
        model: StockTransformer(mode='pretrain')
        train_pool: 训练数据池
        val_pool: 验证数据池
    """
    print("\n" + "=" * 60)
    print("自回归预训练 (GPT)")
    print("=" * 60)
    print(f"序列长度: {PretrainConfig.SEQ_LEN}")
    print(f"预测目标: Open({PretrainConfig.LOSS_WEIGHTS[0]*100:.0f}%), "
          f"Close({PretrainConfig.LOSS_WEIGHTS[1]*100:.0f}%), "
          f"Amount({PretrainConfig.LOSS_WEIGHTS[2]*100:.0f}%)")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}, 每轮批次: {batches_per_epoch}")
    print("=" * 60 + "\n")

    # 损失权重
    loss_weights = torch.tensor(PretrainConfig.LOSS_WEIGHTS, device=device)

    # 临时覆盖 TrainingConfig 以复用 optimizer/scheduler 工厂函数
    import config as cfg
    orig_warmup = cfg.TrainingConfig.WARMUP_EPOCHS
    orig_anneal = cfg.TrainingConfig.COSINE_ANNEAL_EPOCHS
    orig_eta_min = cfg.TrainingConfig.COSINE_ETA_MIN
    cfg.TrainingConfig.WARMUP_EPOCHS = PretrainConfig.WARMUP_EPOCHS
    cfg.TrainingConfig.COSINE_ANNEAL_EPOCHS = PretrainConfig.COSINE_ANNEAL_EPOCHS
    cfg.TrainingConfig.COSINE_ETA_MIN = PretrainConfig.COSINE_ETA_MIN

    # 优化器（embedding 已冻结，只训练可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
    optimizer = create_optimizer_from_config_for_params(trainable_params, lr=learning_rate)

    # 学习率调度器
    warmup_scheduler, main_scheduler, warmup_epochs = create_scheduler_from_config(
        optimizer, epochs=epochs, lr=learning_rate,
        eta_min=PretrainConfig.COSINE_ETA_MIN,
        warmup_start_lr=PretrainConfig.WARMUP_START_LR,
    )

    # 因果掩码（固定大小，整个预训练过程复用）
    causal_mask = generate_causal_mask(PretrainConfig.SEQ_LEN, device)

    # 随机数生成器
    train_rng = np.random.RandomState(DataConfig.RANDOM_SEED)
    val_rng = np.random.RandomState(DataConfig.RANDOM_SEED + 1)

    # 最佳模型追踪
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    # CSV日志
    epoch_logs = []

    for epoch in range(epochs):
        model.train()

        # 学习率更新
        if warmup_scheduler.is_warmup_phase():
            current_lr = warmup_scheduler.step(epoch)
            lr_status = f"预热 ({epoch + 1}/{warmup_epochs})"
        else:
            current_lr = main_scheduler.get_last_lr()[0]
            lr_status = "正常训练"

        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status})')

        total_loss = 0
        total_samples = 0

        t_start = time.time()

        for step in range(batches_per_epoch):
            # 采样
            inputs, targets = sample_pretrain_batch(
                train_pool, PretrainConfig.SEQ_LEN, PretrainConfig.PREDICT_DIMS,
                batch_size, rng=train_rng
            )

            batch_inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            batch_targets = torch.tensor(targets, dtype=torch.float32).to(device)

            # 前向 + 反向
            amp_ctx = _get_amp_context(device)
            optimizer.zero_grad()

            with amp_ctx:
                output = model(batch_inputs, causal_mask=causal_mask)
                # output: [batch, seq_len, 3], target: [batch, seq_len, 3]
                # 加权 MSE
                squared_error = (output - batch_targets) ** 2
                weighted_loss = (squared_error * loss_weights).mean()

            loss_val = weighted_loss.item()
            total_loss += loss_val * batch_size
            total_samples += batch_size

            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=PretrainConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            # 进度
            progress = (step + 1) / batches_per_epoch * 100
            avg_loss = total_loss / total_samples
            print(f'\r  训练进度: {progress:.1f}%, Loss: {avg_loss:.6f}', end='', flush=True)

        print()

        # 更新学习率
        if not warmup_scheduler.is_warmup_phase():
            main_scheduler.step()

        avg_train_loss = total_loss / total_samples if total_samples > 0 else 0

        # 验证
        val_loss = evaluate_pretrain(model, val_pool, causal_mask, loss_weights, device, batch_size)

        elapsed = time.time() - t_start
        print(f'  训练Loss: {avg_train_loss:.6f}, 验证Loss: {val_loss:.6f} ({elapsed:.1f}s)')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            print(f'  ✓ 新最佳模型！验证Loss: {best_val_loss:.6f} (第{best_epoch}轮)')

        print("-" * 60)

        epoch_logs.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
        })

    # 保存最佳模型
    print("\n" + "=" * 60)
    print(f"预训练完成！")
    print(f"最佳模型: 第{best_epoch}轮, 验证Loss: {best_val_loss:.6f}")

    # 保存
    os.makedirs(PretrainConfig.OUTPUT_DIR, exist_ok=True)
    save_path = PretrainConfig.BEST_PRETRAIN_PATH

    torch.save({
        'state_dict': best_model_state,
        'model_arch': {
            'input_dim': ModelConfig.INPUT_DIM,
            'd_model': ModelConfig.D_MODEL,
            'nhead': ModelConfig.NHEAD,
            'num_layers': ModelConfig.NUM_LAYERS,
            'output_dim': ModelConfig.OUTPUT_DIM,
            'context_length': PretrainConfig.SEQ_LEN,
            'mode': 'pretrain',
        },
        'train_params': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'optimizer': PretrainConfig.OPTIMIZER_TYPE,
        },
        'eval_stats': {
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        },
    }, save_path)

    print(f"✓ 预训练模型已保存: {save_path}")
    print("=" * 60)

    # 保存训练日志
    import csv
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_path = os.path.join(PretrainConfig.OUTPUT_DIR, f"pretrain_log_{timestamp}.csv")

    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr'])
        writer.writeheader()
        for log in epoch_logs:
            writer.writerow(log)

    print(f"✓ 训练日志已保存: {os.path.basename(log_path)}")
    return best_val_loss


def evaluate_pretrain(model, val_pool, causal_mask, loss_weights, device, batch_size):
    """验证集评估"""
    model.eval()

    val_rng = np.random.RandomState(42)
    total_loss = 0
    total_samples = 0
    val_batches = max(1, min(50, sum(s['length'] for s in val_pool) // (PretrainConfig.SEQ_LEN + 1) // batch_size))

    with torch.no_grad():
        for _ in range(val_batches):
            inputs, targets = sample_pretrain_batch(
                val_pool, PretrainConfig.SEQ_LEN, PretrainConfig.PREDICT_DIMS,
                batch_size, rng=val_rng
            )

            batch_inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            batch_targets = torch.tensor(targets, dtype=torch.float32).to(device)

            amp_ctx = _get_amp_context(device)
            with amp_ctx:
                output = model(batch_inputs, causal_mask=causal_mask)
                squared_error = (output - batch_targets) ** 2
                loss = (squared_error * loss_weights).mean()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples if total_samples > 0 else float('inf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EquiNet 自监督预训练')
    parser.add_argument('--epochs', type=int, default=PretrainConfig.EPOCHS)
    parser.add_argument('--batch-size', type=int, default=PretrainConfig.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=PretrainConfig.LEARNING_RATE)
    args = parser.parse_args()

    # 设备
    device = DeviceConfig.get_device()

    # 创建输出目录
    os.makedirs(PretrainConfig.OUTPUT_DIR, exist_ok=True)

    # 加载归一化器
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"错误: 归一化器不存在: {DataConfig.NORMALIZER_PATH}")
        print("请先运行: python src/data.py")
        sys.exit(1)

    # 加载数据
    train_stock_info, _, _ = load_and_preprocess_data()

    # 创建模型（pretrain 模式）
    print(f"\n正在创建预训练模型...")
    model = create_model(mode='pretrain', seq_len=PretrainConfig.SEQ_LEN).to(device)

    # 加载预训练 Embedding（冻结）
    embedding_path = EmbeddingConfig.BEST_EMBEDDING_PATH
    if os.path.exists(embedding_path):
        print(f"加载预训练 Embedding: {embedding_path}")
        model.load_pretrained_embedding(embedding_path)
        model.freeze_embedding(True)
    else:
        print(f"错误: 预训练 Embedding 不存在: {embedding_path}")
        print("请先运行: python src/pretrain_embedding.py")
        sys.exit(1)

    # 预计算数据池
    print("\n预计算预训练数据池...")
    train_pool, val_pool = precompute_pretrain_pool(
        train_stock_info, feature_normalizer, val_ratio=PretrainConfig.VAL_RATIO
    )

    # 开始预训练
    best_val_loss = pretrain(
        model, train_pool, val_pool,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\n最终结果: 最佳验证Loss = {best_val_loss:.6f}")
