'''
EquiNet 模型训练脚本

训练策略：
- 使用 DynamicWeightedBCE / PairwiseWeightedBCE 损失函数
- 两阶段学习率调度：Warmup + Cosine Annealing
- 训练时在验证集上评估，按验证集 loss 和实战收益率选择最佳模型
- 训练结束后用最佳模型在测试集上做最终评估
'''

import os, sys, torch, torch.nn as nn, numpy as np
import copy
import random
import csv
from datetime import datetime
from config import (TrainingConfig,DataConfig,DeviceConfig,ModelConfig,print_config_summary,LossConfig,EmbeddingConfig,PretrainConfig)

from model import create_model

from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,FeatureNormalizer,
    compute_label_distance_exclusions,
    precompute_training_pool,
    sample_from_pool,
    sample_temporal_from_pool,
    TemporalSampler
)

from training_utils import (
    WarmupScheduler,
    evaluate_model,
    save_model_with_metadata,
    DynamicWeightedBCE,
    PairwiseWeightedBCE,
    EarlyStopping,
    print_dispersion_sparkline,
    create_optimizer_from_config_for_params,
    create_scheduler_from_config,
    training_step
)

def train(model, train_stock_info, val_stock_info, test_stock_info,
          epochs=TrainingConfig.EPOCHS,
          learning_rate=None,
          device=None,
          batch_size=TrainingConfig.BATCH_SIZE,
          batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
          feature_normalizer=None):
    """
    模型训练函数

    Args:
        model: 待训练的模型实例
        train_stock_info: 训练集股票信息
        val_stock_info: 验证集股票信息（训练时用于模型选择）
        test_stock_info: 测试集股票信息（训练结束后仅评估一次）
        epochs: 训练轮数
        learning_rate: 学习率（None时自动使用当前优化器默认值）
        device: 训练设备
        batch_size: 批大小
        batches_per_epoch: 每轮批次数
        feature_normalizer: 特征归一化器实例
    """
    print("\n" + "="*60)
    print("模型训练")
    print("="*60)
    print("="*60 + "\n")

    # 设置随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)

    # 解析学习率（None时自动使用当前优化器的默认值）
    if learning_rate is None:
        learning_rate = TrainingConfig.get_base_lr()

    # 创建验证集评估数据集（训练时用于模型选择）
    has_val = len(val_stock_info) > 0
    if has_val:
        print("创建验证集评估数据集...")
        eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = create_fixed_evaluation_dataset(
            val_stock_info, feature_normalizer,
            start_key='val_split_point', end_key='test_split_point'
        )
        print(f"  验证集样本数: {len(eval_inputs)}")
    else:
        print("⚠ 无验证集数据，使用测试集进行模型选择（不推荐）")
        eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = create_fixed_evaluation_dataset(
            test_stock_info, feature_normalizer
        )

    # 创建测试集评估数据集（训练结束后仅评估一次）
    test_eval_inputs, test_eval_targets, test_eval_cumulative_returns, test_eval_day_indices, test_eval_daily_returns = create_fixed_evaluation_dataset(
        test_stock_info, feature_normalizer
    )

    # 创建优化器（embedding 已冻结，只传可训练参数）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer_from_config_for_params(trainable_params, lr=learning_rate)

    # 创建学习率调度器
    warmup_scheduler, main_scheduler, warmup_epochs = create_scheduler_from_config(
        optimizer,
        epochs=epochs,
        lr=learning_rate
    )

    # 损失函数选择
    if LossConfig.LOSS_TYPE.lower() == 'pairwise_bce':
        print(f"损失函数: PairwiseWeightedBCE (BCE权重{LossConfig.POS_WEIGHT}, Pairwise权重{LossConfig.PAIRWISE_WEIGHT}, Top{LossConfig.PAIRWISE_TOP_K*100:.0f}%)")
        print(f"  Pairwise: Top{LossConfig.PAIRWISE_TOP_K*100:.0f}%区域, {LossConfig.PAIRWISE_NUM_NEG}个负样本/正样本, warmup {LossConfig.PAIRWISE_WARMUP_EPOCHS}轮")
        criterion = PairwiseWeightedBCE(
            pos_weight=LossConfig.POS_WEIGHT,
            reduction='mean',
            pairwise_weight=LossConfig.PAIRWISE_WEIGHT,
            pairwise_top_k=LossConfig.PAIRWISE_TOP_K,
            pairwise_pos_weight=LossConfig.PAIRWISE_POS_WEIGHT,
            warmup_epochs=LossConfig.PAIRWISE_WARMUP_EPOCHS,
            sigma=LossConfig.PAIRWISE_SIGMA,
            num_neg=LossConfig.PAIRWISE_NUM_NEG
        )
        # 评估损失始终使用纯BCE（Pairwise仅用于训练）
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
    elif LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        print("损失函数: DynamicWeightedBCE (正样本权重4.0，负样本动态调整)")
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
    else:
        print("损失函数: 简单BCE (BCEWithLogitsLoss)")
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    # 设置验证集评估损失权重（BCE类损失共享此逻辑）
    if isinstance(eval_criterion, DynamicWeightedBCE):
        val_targets = np.array(eval_targets)
        val_pos_count = np.sum(val_targets >= 0.5)
        val_neg_count = np.sum(val_targets < 0.5)
        if val_pos_count > 0 and val_neg_count > 0:
            val_neg_weight = LossConfig.POS_WEIGHT * (val_pos_count / val_neg_count)
        elif val_pos_count == 0:
            val_neg_weight = float(LossConfig.POS_WEIGHT)
        else:
            val_neg_weight = 0.1
        eval_criterion.weight_0_0.fill_(val_neg_weight)
        eval_set_name = "验证集" if has_val else "测试集"
        print(f"{eval_set_name}权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={val_neg_weight:.4f} (正负比例={val_pos_count}:{val_neg_count})")

    # 按loss保存的最佳模型（条件：预热结束后, 实战收益率>=1.4%, 收益率>0.8%, AUC>65%）
    best_loss = float('inf')
    best_loss_epoch = 0
    best_model_by_loss = None
    best_return_at_best_loss = 0.0
    best_auc_at_best_loss = 0.0
    best_threshold_at_best_loss = 0.0
    best_realistic_return_at_best_loss = 0.0

    # 按实战收益率保存的最佳模型（预热结束后）
    best_realistic_return = -float('inf')
    best_realistic_return_epoch = 0
    best_model_by_realistic_return = None
    best_return_at_best_realistic = 0.0
    best_auc_at_best_realistic = 0.0
    best_threshold_at_best_realistic = 0.0
    best_realistic_return_value_at_best = 0.0

    # 早停机制（patience = EPOCHS * 0.25）
    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    # 预计算所有训练样本（验证+归一化+标签+收益率只做一次，后续epoch只做数组索引）
    train_rng = random.Random(DataConfig.RANDOM_SEED)
    pool_inputs, pool_targets, pool_returns, pos_indices, neg_indices, sample_key_map = precompute_training_pool(
        train_stock_info, feature_normalizer
    )

    # 根据采样策略初始化采样器
    use_temporal = DataConfig.SAMPLING_STRATEGY == 'temporal'
    if use_temporal:
        temporal_sampler = TemporalSampler(train_stock_info)
        print("  使用时间顺序采样策略（预计算池化版）")

    # 记录每轮收益率
    epoch_returns = []

    for epoch in range(epochs):
        model.train()

        # 更新损失函数的epoch信息（控制Pairwise warmup）
        if hasattr(criterion, 'set_epoch'):
            was_active = criterion.pairwise_active
            criterion.set_epoch(epoch)
            if not was_active and criterion.pairwise_active:
                print(f'  ⚙ Pairwise排序损失已激活 (第{epoch+1}轮)')
                print()

        total_loss = 0

        # 学习率更新
        if warmup_scheduler.is_warmup_phase():
            current_lr = warmup_scheduler.step(epoch)
            lr_status = f"预热阶段 ({epoch + 1}/{warmup_epochs})"
        else:
            current_lr = main_scheduler.get_last_lr()[0]
            lr_status = "正常训练"

        print(f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f} ({lr_status})')

        # 从预计算池中采样
        if use_temporal:
            epoch_inputs, epoch_targets, epoch_cum_returns = sample_temporal_from_pool(
                temporal_sampler, train_stock_info,
                pool_inputs, pool_targets, pool_returns, sample_key_map,
                batch_size, batches_per_epoch
            )
        else:
            epoch_inputs, epoch_targets, epoch_cum_returns = sample_from_pool(
                pool_inputs, pool_targets, pool_returns, pos_indices, neg_indices,
                batch_size, batches_per_epoch, train_rng
            )

        # 打印标签分布
        count_positive = np.sum(epoch_targets >= 0.9)
        count_boundary = np.sum((epoch_targets > 0.1) & (epoch_targets < 0.9))
        count_negative = np.sum(epoch_targets <= 0.1)
        total_count = len(epoch_targets)
        print(f'  标签分布: 上涨={count_positive}({count_positive/total_count:.1%}), 边界={count_boundary}({count_boundary/total_count:.1%}), 不涨={count_negative}({count_negative/total_count:.1%})')

        # 转换为tensor
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.float32).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.float32).to(device)
        epoch_returns_tensor = torch.tensor(epoch_cum_returns, dtype=torch.float32).to(device)

        # 计算实际可用的batch数量
        actual_batches = len(epoch_inputs_tensor) // batch_size
        if actual_batches < batches_per_epoch:
            print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

        for step in range(actual_batches):
            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size

            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]
            batch_returns = epoch_returns_tensor[start_idx:end_idx]

            def _loss_fn():
                output = model(batch_inputs)
                if hasattr(criterion, 'update_weights'):
                    criterion.update_weights(batch_targets)
                loss = criterion(output.squeeze(-1), batch_targets)
                return loss, output

            loss_val, _ = training_step(model, optimizer, _loss_fn)
            total_loss += loss_val * (end_idx - start_idx)

            # 进度显示
            progress = (step + 1) / actual_batches * 100
            processed_samples = (step + 1) * batch_size
            avg_loss = total_loss / processed_samples
            print(f'\r  训练进度: {progress:.1f}%, Loss: {avg_loss:.4f}', end='', flush=True)

        print()
        print()

        # 清理内存
        del epoch_inputs_tensor, epoch_targets_tensor, epoch_returns_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 更新学习率
        if not warmup_scheduler.is_warmup_phase():
            main_scheduler.step()

        # 评估模型（在验证集上）
        stats = evaluate_model(
            model, eval_inputs, eval_targets, eval_cumulative_returns,
            device,
            eval_day_indices=eval_day_indices,
            eval_daily_returns=eval_daily_returns,
            criterion=eval_criterion
        )

        # 计算训练集平均损失
        total_samples = len(epoch_inputs)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        val_loss = stats['test_loss']
        eval_label = "验证集" if has_val else "测试集"

        print(f'  [模型] 训练损失: {avg_loss:.4f}, {eval_label}损失: {val_loss:.4f}, AUC: {stats["auc"]:.4f}, Prec@10%: {stats["precision_top10"]:.3f}, Prec@5%: {stats["precision_top5"]:.3f}, Prec@3%: {stats["precision_top3"]:.3f} (基线: {stats["base_positive_rate"]:.3f})')
        print(f'         预测均值: {stats["pred_mean"]:.3f}, 高置信(>0.7): {stats["high_conf_count"]}, 低置信(<0.2): {stats["low_conf_count"]}')
        daily_str = f', 日Top{DataConfig.TOP_K}%收益: {stats["daily_top_return"]*100:+.2f}%' if stats.get("daily_top_return") is not None else ''
        print(f'         Top{DataConfig.TOP_K}%收益: {stats["top_return"]*100:+.2f}%{daily_str}')

        if stats['realistic_stats'] is not None:
            rs = stats['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
            print(f'         【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'         【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')

        epoch_return = {
            'turn': epoch + 1,
            'return': stats['top_return'] * 100,
            'daily_return': stats.get('daily_top_return'),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'auc': stats['auc'],
            'precision_top10': stats['precision_top10'],
            'precision_top5': stats['precision_top5'],
            'precision_top3': stats['precision_top3'],
            'avg_realistic_return': stats['realistic_stats']['avg_realistic_return'] if stats.get('realistic_stats') else None,
            'dispersion_std': stats.get('dispersion_std', 0),
            'dispersion_range': stats.get('dispersion_range', 0),
            'dispersion_iqr': stats.get('dispersion_iqr', 0),
            'pos_ratio': stats.get('pred_mean', 0),
            'high_conf_ratio': stats.get('high_conf_count', 0) / len(eval_targets) if eval_targets is not None else 0,
        }
        epoch_returns.append(epoch_return)

        print_dispersion_sparkline(stats.get('all_preds', []), epoch_returns, all_targets=stats.get('all_targets'))

        # 早停检测（基于验证集指标）
        improved, improve_reason = early_stopping.check_improve(
            avg_loss=val_loss,
            top_return=stats['top_return'],
            auc=stats['auc'],
            threshold=stats['top_threshold']
        )

        if improved:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'         ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
        else:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'         ⚠ 无改善 ({no_improve_count}/{patience_limit})')

        # 按loss保存最佳模型（基于验证集指标）
        realistic_return = stats['realistic_stats']['avg_realistic_return'] if stats.get('realistic_stats') else 0.0
        if (epoch + 1) > warmup_epochs and realistic_return >= 0.014 and stats['top_return'] > 0.008 and stats['auc'] > 0.65:
            if val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch + 1
                best_model_by_loss = copy.deepcopy(model.state_dict())
                best_return_at_best_loss = stats['top_return']
                best_auc_at_best_loss = stats['auc']
                best_threshold_at_best_loss = stats['top_threshold']
                best_realistic_return_at_best_loss = realistic_return
                print(f'         ✓ 新最佳模型（{eval_label} loss）！Loss: {best_loss:.4f}, 实战收益率: {best_realistic_return_at_best_loss*100:.1f}% (第{best_loss_epoch}轮)')

        # 按实战收益率保存最佳模型（基于验证集指标）
        if (epoch + 1) > warmup_epochs:
            if realistic_return > best_realistic_return:
                best_realistic_return = realistic_return
                best_realistic_return_epoch = epoch + 1
                best_model_by_realistic_return = copy.deepcopy(model.state_dict())
                best_return_at_best_realistic = stats['top_return']
                best_auc_at_best_realistic = stats['auc']
                best_threshold_at_best_realistic = stats['top_threshold']
                best_realistic_return_value_at_best = realistic_return
                print(f'         ✓ 新最佳模型（{eval_label}实战收益率）！实战: {best_realistic_return*100:.1f}%, Top1%: {best_return_at_best_realistic*100:+.2f}% (第{best_realistic_return_epoch}轮)')

        print("-" * 60)

        # 早停检查
        if early_stopping.should_stop() and TrainingConfig.OPEN_EARLY_STOPPING:
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break

    # ========== 最终测试集评估 → 保存模型 ==========
    print("\n" + "=" * 60)
    print(f"训练完成！")
    eval_label = "验证集" if has_val else "测试集"
    print(f"最佳模型（按{eval_label} loss）: 第{best_loss_epoch}轮, Loss: {best_loss:.4f}, 实战收益率: {best_realistic_return_at_best_loss*100:.1f}%")
    print(f"最佳模型（按{eval_label}实战收益率）: 第{best_realistic_return_epoch}轮, 实战收益率: {best_realistic_return_value_at_best*100:.1f}%, Top1%: {best_return_at_best_realistic*100:+.2f}%")

    # 测试集评估（用测试集指标保存到模型文件中）
    test_return = 0.0
    test_realistic_return = 0.0

    # 用于保存的测试集指标（默认用验证集指标，有测试集时覆盖）
    save_loss_return = best_return_at_best_loss
    save_loss_threshold = best_threshold_at_best_loss
    save_loss_auc = best_auc_at_best_loss
    save_realistic_return = best_return_at_best_realistic
    save_realistic_threshold = best_threshold_at_best_realistic
    save_realistic_auc = best_auc_at_best_realistic

    if len(test_eval_inputs) > 0:
        print("\n" + "=" * 60)
        print("测试集最终评估（即将保存的模型）")
        print("=" * 60)

        # 创建测试集评估损失
        if isinstance(eval_criterion, DynamicWeightedBCE):
            test_eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
            t_targets = np.array(test_eval_targets)
            t_pos = np.sum(t_targets >= 0.5)
            t_neg = np.sum(t_targets < 0.5)
            if t_pos > 0 and t_neg > 0:
                t_neg_w = LossConfig.POS_WEIGHT * (t_pos / t_neg)
            elif t_pos == 0:
                t_neg_w = float(LossConfig.POS_WEIGHT)
            else:
                t_neg_w = 0.1
            test_eval_criterion.weight_0_0.fill_(t_neg_w)
        else:
            test_eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        # 评估按loss选出的最佳模型
        if best_model_by_loss is not None:
            model.load_state_dict(best_model_by_loss)
            test_stats_loss = evaluate_model(
                model, test_eval_inputs, test_eval_targets, test_eval_cumulative_returns,
                device,
                eval_day_indices=test_eval_day_indices,
                eval_daily_returns=test_eval_daily_returns,
                criterion=test_eval_criterion
            )
            print(f"\n  [测试集] 模型(loss): "
                  f"Loss: {test_stats_loss['test_loss']:.4f}, "
                  f"AUC: {test_stats_loss['auc']:.4f}, "
                  f"Top{DataConfig.TOP_K}%收益: {test_stats_loss['top_return']*100:+.2f}%")
            if test_stats_loss.get('realistic_stats'):
                rs = test_stats_loss['realistic_stats']
                print(f"           实战收益率: {rs['avg_realistic_return']*100:.1f}%")
            # 用测试集指标覆盖保存指标
            save_loss_return = test_stats_loss['top_return']
            save_loss_threshold = test_stats_loss['top_threshold']
            save_loss_auc = test_stats_loss['auc']
            test_return = test_stats_loss['top_return']

        # 评估按实战收益率选出的最佳模型
        if best_model_by_realistic_return is not None:
            model.load_state_dict(best_model_by_realistic_return)
            test_stats_realistic = evaluate_model(
                model, test_eval_inputs, test_eval_targets, test_eval_cumulative_returns,
                device,
                eval_day_indices=test_eval_day_indices,
                eval_daily_returns=test_eval_daily_returns,
                criterion=test_eval_criterion
            )
            print(f"\n  [测试集] 模型(realistic): "
                  f"Loss: {test_stats_realistic['test_loss']:.4f}, "
                  f"AUC: {test_stats_realistic['auc']:.4f}, "
                  f"Top{DataConfig.TOP_K}%收益: {test_stats_realistic['top_return']*100:+.2f}%")
            if test_stats_realistic.get('realistic_stats'):
                rs = test_stats_realistic['realistic_stats']
                print(f"           实战收益率: {rs['avg_realistic_return']*100:.1f}%")
            # 用测试集指标覆盖保存指标
            save_realistic_return = test_stats_realistic['top_return']
            save_realistic_threshold = test_stats_realistic['top_threshold']
            save_realistic_auc = test_stats_realistic['auc']
            test_realistic_return = rs['avg_realistic_return'] if test_stats_realistic.get('realistic_stats') else 0.0

        print("=" * 60)

    # 保存模型（嵌入测试集评估指标）
    if best_model_by_loss is not None:
        save_path = save_model_with_metadata(
            best_model_by_loss,
            save_loss_return, save_loss_threshold, save_loss_auc,
            best_loss_epoch,
            model_prefix="model_loss",
            output_dir=DataConfig.OUTPUT_DIR,
        )
        print(f"✓ 模型(loss)已保存: {os.path.basename(save_path)}")
        print(f"  测试集 Top1%收益: {save_loss_return*100:+.2f}%, AUC: {save_loss_auc:.4f}")

    if best_model_by_realistic_return is not None:
        save_path_realistic = save_model_with_metadata(
            best_model_by_realistic_return,
            save_realistic_return, save_realistic_threshold, save_realistic_auc,
            best_realistic_return_epoch,
            model_prefix="model_realistic",
            output_dir=DataConfig.OUTPUT_DIR,
        )
        print(f"✓ 模型(realistic)已保存: {os.path.basename(save_path_realistic)}")
        print(f"  测试集 Top1%收益: {save_realistic_return*100:+.2f}%, AUC: {save_realistic_auc:.4f}")

    # 保存每轮收益率到CSV
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"epoch_returns_{timestamp}.csv")

    fieldnames = ['turn', 'top_return', 'daily_return', 'train_loss', 'val_loss', 'auc', 'prec_top10', 'prec_top5', 'prec_top3', 'avg_realistic_return']

    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for er in epoch_returns:
            row = {
                'turn': er['turn'],
                'top_return': f"{er['return']:.2f}",
                'daily_return': f"{er['daily_return']*100:.2f}" if er.get('daily_return') is not None else "",
                'train_loss': f"{er['train_loss']:.4f}" if er.get('train_loss') is not None else "",
                'val_loss': f"{er['val_loss']:.4f}" if er.get('val_loss') is not None else "",
                'auc': f"{er['auc']:.4f}" if er.get('auc') is not None else "",
                'prec_top10': f"{er['precision_top10']:.4f}" if er.get('precision_top10') is not None else "",
                'prec_top5': f"{er['precision_top5']:.4f}" if er.get('precision_top5') is not None else "",
                'prec_top3': f"{er['precision_top3']:.4f}" if er.get('precision_top3') is not None else "",
                'avg_realistic_return': f"{er['avg_realistic_return']*100:.1f}" if er.get('avg_realistic_return') is not None else "",
            }
            writer.writerow(row)

    print(f"✓ 训练日志已保存: {os.path.basename(returns_csv_path)}")
    return test_return, test_realistic_return


if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary()

    # 获取设备
    device = DeviceConfig.get_device()

    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)

    # ========== 特征归一化器配置 ==========
    print("\n" + "="*60)
    print("特征归一化器配置")
    print(f" 归一化器路径: {DataConfig.NORMALIZER_PATH}")
    print(f" 输出分布: {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION}")
    print(f" 分位数数量: {DataConfig.NORMALIZER_N_QUANTILES}")

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n⚠ 归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")
        print("请先运行以下命令创建归一化器：")
        print(f"  python data.py --output-distribution {DataConfig.NORMALIZER_OUTPUT_DISTRIBUTION} --n-quantiles {DataConfig.NORMALIZER_N_QUANTILES}")
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    print("="*60)

    # 加载数据
    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    # 正样本距离保护
    compute_label_distance_exclusions(train_stock_info)

    # 打印数据集统计
    print("\n" + "="*60)
    print("数据集统计")
    print(f" 训练集: {len(train_stock_info)} 只股票")
    print(f" 验证集: {len(val_stock_info)} 只股票")
    print(f" 测试集: {len(test_stock_info)} 只股票")

    # 创建模型（微调模式）
    amp_str = "BF16混合精度" if TrainingConfig.USE_AMP else "FP32精度"
    print(f"\n正在创建模型 ({amp_str})...")
    model = create_model(mode='finetune', seq_len=PretrainConfig.SEQ_LEN).to(device)

    # 加载预训练 Embedding
    embedding_path = EmbeddingConfig.BEST_EMBEDDING_PATH
    if os.path.exists(embedding_path):
        print(f"加载预训练 Embedding: {embedding_path}")
        model.load_pretrained_embedding(embedding_path)
        model.freeze_embedding(True)
    else:
        print(f"错误: 预训练 Embedding 不存在: {embedding_path}")
        print("请先运行: python src/pretrain_embedding.py")
        sys.exit(1)

    # 加载预训练 Backbone（冻结，解冻最后N层Transformer）
    pretrain_path = PretrainConfig.BEST_PRETRAIN_PATH
    if os.path.exists(pretrain_path):
        print(f"加载预训练 Backbone: {pretrain_path}")
        model.load_pretrained_backbone(pretrain_path)
        model.freeze_backbone(unfreeze_last_n=PretrainConfig.FINETUNE_UNFREEZE_LAYERS)
    else:
        print(f"⚠ 预训练 Backbone 不存在: {pretrain_path}")
        print("将从头训练（无预训练权重）")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" 模型参数数: {total_params:,}, 可训练: {trainable_params:,}")

    # 开始训练
    print("\n开始模型训练...")
    best_return, best_realistic_return = train(
        model, train_stock_info, val_stock_info, test_stock_info,
        device=device,
        feature_normalizer=feature_normalizer
    )

    print(f"\n最终结果:")
    print(f"  最佳Top1%收益={best_return*100:+.2f}%")
    print(f"  最佳实战收益率={best_realistic_return*100:.1f}%")
