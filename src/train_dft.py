'''
DFT模型训练脚本

核心思想：
- 从out/目录加载已有模型权重
- 使用自引导DFT（Self-Guided Direct Fine-Tuning）机制继续微调
- 只训练这一个模型（B链路）

自引导DFT权重机制（基于B自身的预测排名分位数）：
- 将B对batch内所有样本的预测值排序，得到每个样本的分位数 rank ∈ [0, 1]
- B预测排名最高的样本（rank→1）：B已经很确定是好的 → 低权值（已学会，无需多学）
- B预测排名最低的样本（rank→0）：B已经很确定是差的 → 低权值（已学会，无需多学）
- B预测排名在中间的样本（rank≈0.5）：B还没分清楚的 → 高权值（最有学习价值）
- 权重公式：w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
  这是一个开口朝下的抛物线，在 rank=0.5 处取最大值 w_max，在 rank=0/1 处取最小值 w_min
- 权重随训练动态演化：随着B学习进步，"不确定"的样本会变化，权重自然跟着调整
'''

import os, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, numpy as np
import argparse
import copy
import random
import csv
from datetime import datetime
from config import (ModelConfig, TrainingConfig, DataConfig,
                   DeviceConfig, ModelSaveConfig,
                   print_config_summary, LossConfig)

from model import create_model

from data import (
    load_and_preprocess_data,
    TemporalSampler, sample_with_pools,
    create_fixed_evaluation_dataset
)

from train import (
    WarmupScheduler,
    evaluate_model,
    save_model_with_metadata,
    EarlyStopping,
    calculate_test_loss,
    DynamicWeightedBCE
)


def train_dft_model(model, train_stock_info, test_stock_info,
                    epochs=TrainingConfig.EPOCHS,
                    learning_rate=TrainingConfig.LEARNING_RATE,
                    device=None,
                    batch_size=TrainingConfig.BATCH_SIZE,
                    batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
                    dft_w_min=0.1,
                    dft_w_max=1.0,
                    seed=DataConfig.RANDOM_SEED):
    """
    DFT模型训练函数（自引导模式）

    训练策略：
    - 加载已有模型，使用自引导DFT继续微调
    - 样本权重基于模型自身的预测排名分位数：中间排名高权值，头尾低权值
    - w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
    """
    print("\n" + "="*60)
    print("DFT自引导微调训练")
    print("="*60)
    print(f"训练策略：")
    print(f"  - 加载已有模型，使用自引导DFT继续微调")
    print(f"  - 样本权重基于自身预测排名分位数")
    print(f"  - 权重范围: [{dft_w_min}, {dft_w_max}]")
    print(f"  - 中间排名(rank≈0.5)权重最高，头尾(rank→0/1)权重最低")
    print("="*60 + "\n")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = create_fixed_evaluation_dataset(test_stock_info)

    stats_init = evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="初始模型", eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns)
    print(f"初始模型评估: AUC={stats_init['auc']:.4f}, Top1%收益={stats_init['top_return']*100:+.2f}%")
    if stats_init['realistic_stats'] is not None:
        rs = stats_init['realistic_stats']
        print(f"              【实战收益率】平均: {rs['avg_realistic_return']*100:.1f}%")

    dft_lr = learning_rate * 0.2
    print(f"DFT学习率: {dft_lr:.6f} (原学习率的20%)")

    if TrainingConfig.USE_MANO:
        from optimizers import create_optimizer
        optimizer = create_optimizer(
            model,
            optimizer_type='mano',
            lr=dft_lr,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            betas=TrainingConfig.MANO_ADAMW_BETAS
        )
    elif TrainingConfig.USE_ADAMW:
        optimizer = optim.AdamW(model.parameters(), lr=dft_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=dft_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)

    total_main_epochs = epochs
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_main_epochs,
        eta_min=dft_lr * 0.01
    )

    if LossConfig.use_dynamic_bce():
        print("损失函数: DynamicWeightedBCE (正样本权重4.0，负样本动态调整)")
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        
        # 测试集权重：开局算一次，整个训练过程复用
        test_targets = np.array(eval_targets)
        test_pos_count = np.sum(test_targets >= 0.5)
        test_neg_count = np.sum(test_targets < 0.5)
        if test_pos_count > 0 and test_neg_count > 0:
            test_neg_weight = LossConfig.POS_WEIGHT * (test_pos_count / test_neg_count)
        elif test_pos_count == 0:
            test_neg_weight = float(LossConfig.POS_WEIGHT)
        else:
            test_neg_weight = 0.1
        eval_criterion.weight_0_0.fill_(test_neg_weight)
        print(f"测试集权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={test_neg_weight:.4f} (正负比例={test_pos_count}:{test_neg_count})")
    else:
        print("损失函数: 简单BCE (BCEWithLogitsLoss)")
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def weighted_bce_with_logits(inputs, targets, weights):
        """DFT加权损失，输入为logits，权重在batch内动态调节。"""
        inputs = inputs.squeeze(-1)
        targets = targets.squeeze()
        weights = weights.squeeze()

        loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction='none')
        weights = weights.to(dtype=loss.dtype)
        return (loss * weights).mean()

    def compute_dft_weights(pred, w_min=dft_w_min, w_max=dft_w_max):
        """
        根据预测值在batch内的排名分位数计算样本权重。
        rank=0.5(中间排名)权重最高，rank=0/1(头尾)权重最低。
        抛物线公式：w = w_min + (w_max - w_min) * 4 * rank * (1 - rank)
        """
        pred_squeezed = pred.squeeze().detach()
        n = pred_squeezed.shape[0]
        ranks = pred_squeezed.argsort().argsort().float() / (n - 1) if n > 1 else torch.full_like(pred_squeezed, 0.5)
        weights = w_min + (w_max - w_min) * 4.0 * ranks * (1.0 - ranks)
        return weights

    best_return = stats_init['top_return']
    best_auc = stats_init['auc']
    best_threshold = stats_init['top_threshold']
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    sampler = TemporalSampler(train_stock_info)
    train_rng = random.Random(seed)

    epoch_returns = []

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_samples = 0

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} (DFT微调)')

        epoch_inputs, epoch_targets = sample_with_pools(
            sampler, train_stock_info, batch_size, batches_per_epoch, train_rng
        )

        looped_count, total_loops = sampler.get_loop_stats()
        print(f"  [循环统计] 已循环股票: {looped_count}/{len(train_stock_info)}, 总循环次数: {total_loops}")

        count_positive = np.sum(epoch_targets >= 0.9)
        count_boundary = np.sum((epoch_targets > 0.1) & (epoch_targets < 0.9))
        count_negative = np.sum(epoch_targets <= 0.1)
        total_count = len(epoch_targets)
        print(f'  标签分布: 上涨={count_positive}({count_positive/total_count:.1%}), 边界={count_boundary}({count_boundary/total_count:.1%}), 不涨={count_negative}({count_negative/total_count:.1%})')

        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.bfloat16).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.bfloat16).to(device)

        actual_batches = len(epoch_inputs_tensor) // batch_size
        if actual_batches < batches_per_epoch:
            print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

        for step in range(actual_batches):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size

            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]

            optimizer.zero_grad()

            output = model(batch_inputs)

            with torch.no_grad():
                pred_prob = torch.sigmoid(output)
                dft_weights = compute_dft_weights(pred_prob)

            loss = weighted_bce_with_logits(output, batch_targets, dft_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TrainingConfig.GRADIENT_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item() * (end_idx - start_idx)
            total_samples += (end_idx - start_idx)

            progress = (step + 1) / actual_batches * 100
            avg_loss = total_loss / total_samples
            print(f'\r  训练进度: {progress:.1f}%, Loss(DFT): {avg_loss:.4f}', end='', flush=True)

        print()
        print()

        del epoch_inputs_tensor, epoch_targets_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        main_scheduler.step()

        stats = evaluate_model(model, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="DFT", eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device)

        print(f'  [DFT模型] 训练损失: {avg_loss:.4f}, 测试损失: {test_loss:.4f}, AUC: {stats["auc"]:.4f}')
        print(f'            预测均值: {stats["pred_mean"]:.3f}, 高置信(>0.7): {stats["high_conf_count"]}, 低置信(<0.2): {stats["low_conf_count"]}')
        print(f'            Top{DataConfig.TOP_PERCENT}%收益: {stats["top_return"]*100:+.2f}%')
        
        if stats['realistic_stats'] is not None:
            rs = stats['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else "全局阈值"
            print(f'            【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'            【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')
        
        if stats.get('smart_exit_stats') is not None:
            se = stats['smart_exit_stats']
            print(f'            【智能止损】收益率: {se["avg_realistic_return"]*100:.1f}%, Day1止损: {se["stop_loss_day1_count"]}次, 累计止损: {se["stop_loss_cum_count"]}次, 止盈: {se["take_profit_count"]}次')

        epoch_return = {
            'turn': epoch + 1,
            'return': stats['top_return'] * 100,
            'train_loss': avg_loss,
            'test_loss': test_loss
        }
        epoch_returns.append(epoch_return)

        improved, improve_reason = early_stopping.check_improve(
            avg_loss=test_loss,
            top_return=stats['top_return'],
            auc=stats['auc'],
            threshold=stats['top_threshold']
        )

        if improved:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'            ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
        else:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'            ⚠ 无改善 ({no_improve_count}/{patience_limit})')

        if stats['top_return'] > best_return:
            best_return = stats['top_return']
            best_auc = stats['auc']
            best_threshold = stats['top_threshold']
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            print(f'            ✓ 新最佳模型！Top1%收益: {best_return*100:+.2f}% (第{best_epoch}轮)')

        print("-" * 60)

        if early_stopping.should_stop():
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break

    print("\n" + "=" * 60)
    print(f"训练完成！")
    print(f"最佳模型: 第{best_epoch}轮, Top1%收益: {best_return*100:+.2f}%, AUC: {best_auc:.4f}")

    timestamp_csv = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"dft_epoch_returns_{timestamp_csv}.csv")
    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['turn', 'return', 'train_loss', 'test_loss'])
        writer.writeheader()

        for epoch_return in epoch_returns:
            row = {
                'turn': epoch_return['turn'],
                'return': f"{epoch_return['return']:.2f}",
                'train_loss': f"{epoch_return['train_loss']:.4f}",
                'test_loss': f"{epoch_return['test_loss']:.4f}"
            }
            writer.writerow(row)

    print(f"✓ 每轮收益率已保存: {os.path.basename(returns_csv_path)}")
    print(f"  共记录 {len(epoch_returns)} 轮训练数据")

    save_path = save_model_with_metadata(
        best_model_state,
        best_return,
        best_threshold,
        best_auc,
        best_epoch,
        model_prefix="modelB_dft",
        output_dir=DataConfig.OUTPUT_DIR
    )
    filename = os.path.basename(save_path)
    print(f"✓ DFT模型已保存: {filename}")
    print(f"  Top1%阈值: {best_threshold:.4f}")
    print("=" * 60)

    return best_return, best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFT自引导微调：从已有模型加载，使用自引导权重继续微调')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='要微调的模型权重路径')
    parser.add_argument('--epochs', '-e', type=int, default=TrainingConfig.EPOCHS,
                        help=f'训练轮数（默认: {TrainingConfig.EPOCHS}）')
    parser.add_argument('--w_min', type=float, default=0.1,
                        help='DFT最小权重（默认: 0.1）')
    parser.add_argument('--w_max', type=float, default=1.0,
                        help='DFT最大权重（默认: 1.0）')
    parser.add_argument('--seed', type=int, default=DataConfig.RANDOM_SEED,
                        help=f'随机种子（默认: {DataConfig.RANDOM_SEED}）')
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在: {args.model}")
        exit(1)

    print_config_summary()

    device = DeviceConfig.print_device_info()

    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)

    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()

    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    print("="*60)

    print(f"\n正在加载模型: {args.model}")
    model = create_model().to(device)
    model = model.to(dtype=torch.bfloat16)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数: {total_params:,}")

    print("\n开始DFT自引导微调训练...")
    best_return, best_auc = train_dft_model(
        model, train_stock_info, test_stock_info,
        device=device,
        epochs=args.epochs,
        dft_w_min=args.w_min,
        dft_w_max=args.w_max,
        seed=args.seed
    )

    print(f"\n最终结果:")
    print(f"  最佳Top1%收益: {best_return*100:+.2f}%")
    print(f"  最佳AUC: {best_auc:.4f}")
