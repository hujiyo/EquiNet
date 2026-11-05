'''
EquiNet 模型训练脚本

训练策略：
- 单阶段端到端训练：FFN-Embedding 与 Transformer 主干一起优化，不冻结、不加载预训练权重
- VISReg 作为 Embedding 的几何正则：每个 step 独立采样单日 K 线，额外计算
  尺度/形状/中心化三项损失，只约束表征分布，不参与任务判决
- 使用 DynamicWeightedBCE / PairwiseWeightedBCE / BalancedBCE 损失函数
- 学习率调度：预热 + 退火，按优化步更新（见 TrainingConfig.LR_*）
- 训练时在验证集上评估，按验证集 loss 和逐日 IC 选择最佳模型
- 训练结束后用最佳模型在测试集上做最终评估
'''

import os, torch, numpy as np
import math
import copy
import random
import csv
from datetime import datetime
from config import (TrainingConfig,DataConfig,DeviceConfig,print_config_summary,LossConfig,RegularizationConfig)

from model import create_model
from visreg import VISRegLoss

from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,FeatureNormalizer,
    compute_label_distance_exclusions,
    apply_quality_exclusions,
    precompute_training_pool,
    sample_from_pool,
    sample_temporal_from_pool,
    TemporalSampler
)

from training_utils import (
    StepLRScheduler,
    evaluate_model,
    save_model_with_metadata,
    DynamicWeightedBCE,
    PairwiseWeightedBCE,
    BalancedBCE,
    EarlyStopping,
    print_dispersion_sparkline,
    create_optimizer_from_config_for_params,
    create_scheduler_from_config,
    training_step,
    create_eval_criterion,
    create_run_dir,
)

def train(model, train_stock_info, val_stock_info, test_stock_info,
          epochs=TrainingConfig.EPOCHS,
          learning_rate=None,
          device=None,
          batch_size=TrainingConfig.BATCH_SIZE,
          batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
          feature_normalizer=None,
          run_dir=None):
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
        run_dir: 本次训练的 run 目录（create_run_dir 创建）。所有训练产物
                 （model_loss/model_realistic 的 .pth、epoch_returns CSV）
                 都落在该目录内；None 时兜底到 DataConfig.OUTPUT_DIR。
    """
    # run 目录：None 时兜底到顶层输出目录（正常路径由 __main__ 用 create_run_dir 创建）
    run_dir = run_dir or DataConfig.OUTPUT_DIR

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
        eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_tradeable_mask = create_fixed_evaluation_dataset(
            val_stock_info, feature_normalizer,
            start_key='val_split_point', end_key='test_split_point'
        )
        print(f"  验证集样本数: {len(eval_inputs)}")
    else:
        print("⚠ 无验证集数据，使用测试集进行模型选择（不推荐）")
        # 必须显式给 start_date/end_date：否则预测日会溢出 TEST_END_DATE
        # （旧 end_key 口径下会多出约 CONTEXT_LENGTH 个交易日的 2026 年样本）
        eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_tradeable_mask = create_fixed_evaluation_dataset(
            test_stock_info, feature_normalizer,
            start_date=DataConfig.TEST_START_DATE, end_date=DataConfig.TEST_END_DATE
        )

    # 测试集评估数据集延后到训练结束后构建（整个训练期间不消费，
    # 提前构建只会让进入训练循环前多等一段只在收尾时用的预处理）
    test_eval_inputs, test_eval_targets, test_eval_cumulative_returns = None, None, None
    test_eval_day_indices, test_eval_daily_returns, test_eval_tradeable_mask = None, None, None

    # 单阶段端到端：Embedding 与主干一起训练。
    # Embedding 用独立参数组和更低学习率，避免 6.8 万参数的逐日编码器
    # 被稀疏的序列级标签一步带偏。
    embed_params = [p for p in list(model.embed_proj.parameters())
                    + list(model.embed_mlp.parameters()) if p.requires_grad]
    embed_param_ids = {id(p) for p in embed_params}
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in embed_param_ids]
    embed_lr = learning_rate * RegularizationConfig.EMBED_LR_RATIO
    param_groups = [
        {'params': other_params, 'lr': learning_rate},
        {'params': embed_params, 'lr': embed_lr},
    ]
    optimizer = create_optimizer_from_config_for_params(param_groups, lr=learning_rate)
    print(f"  参数组: 主干 {sum(p.numel() for p in other_params):,} (lr={learning_rate:g}), "
          f"Embedding {sum(p.numel() for p in embed_params):,} (lr={embed_lr:g})")

    # VISReg：主训练中的表示正则。辅助 batch 独立采样单日 K 线，
    # 不展平 45 天序列窗口（展平会破坏 VISReg 假设的 i.i.d. 单日边际分布）。
    visreg_loss_fn = VISRegLoss(
        num_slices=RegularizationConfig.VISREG_NUM_SLICES,
        w_scale=RegularizationConfig.VISREG_W_SCALE,
        w_shape=RegularizationConfig.VISREG_W_SHAPE,
        w_center=RegularizationConfig.VISREG_W_CENTER,
    )
    print(f"  VISReg: λ={RegularizationConfig.VISREG_MAIN_WEIGHT}, "
          f"单日辅助batch={RegularizationConfig.VISREG_AUX_BATCH_SIZE}, "
          f"target_std={RegularizationConfig.TARGET_STD}")

    # 创建学习率调度器（两段式：预热 → 退火，按步更新）
    lr_scheduler = create_scheduler_from_config(optimizer)
    # 预热区结束对应的周期数，只用于选模门槛（预热期内的测评不参与选模）
    warmup_epochs = math.ceil(TrainingConfig.LR_WARMUP_STEPS / batches_per_epoch)

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
    elif LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        print("损失函数: DynamicWeightedBCE (正样本权重4.0，负样本动态调整)")
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
    elif LossConfig.LOSS_TYPE.lower() == 'balanced_bce':
        print("损失函数: BalancedBCE (正负各占一半损失，量纲与标准BCE一致)")
        criterion = BalancedBCE(reduction='mean')
    else:
        raise ValueError(f"未知 LOSS_TYPE: {LossConfig.LOSS_TYPE} (支持: dynamic_bce / pairwise_bce / balanced_bce)")

    # 评估损失走统一入口 create_eval_criterion：pairwise_bce 路径下评估仍用纯 BCE，
    # 与 train.py 旧版本行为一致；dynamic_bce / balanced_bce 路径下按正负比例填权重。
    eval_criterion = create_eval_criterion(eval_targets)

    eval_set_name = "验证集" if has_val else "测试集"
    if isinstance(eval_criterion, DynamicWeightedBCE):
        val_targets = np.array(eval_targets)
        val_pos_count = int(np.sum(val_targets >= 0.5))
        val_neg_count = int(np.sum(val_targets < 0.5))
        val_neg_weight = eval_criterion.weight_0_0.item()
        print(f"{eval_set_name}权重: 正样本={LossConfig.POS_WEIGHT}, 负样本={val_neg_weight:.4f} (正负比例={val_pos_count}:{val_neg_count})")
    elif isinstance(eval_criterion, BalancedBCE):
        eval_targets_arr = np.array(eval_targets)
        pos_count = int(np.sum(eval_targets_arr >= 0.5))
        neg_count = int(np.sum(eval_targets_arr < 0.5))
        print(f"{eval_set_name}BalancedBCE权重: 正样本={eval_criterion.pos_weight.item():.4f}, 负样本={eval_criterion.neg_weight.item():.4f} (正{pos_count}:负{neg_count})")

    # 按loss保存的最佳模型（预热结束后即生效）
    #
    # 为什么去掉硬门槛：这里原来是 `top_return >= 0.006 and auc > 0.69`。
    # 低于门槛的 epoch 从不落盘，于是「模型怎么都过不了 70%」其实是**左截断假象**——
    # 真正存在的低分段你根本没看见。选模标准可以严，记录必须完整。
    best_loss = float('inf')
    best_loss_epoch = 0
    best_model_by_loss = None
    best_return_at_best_loss = 0.0
    best_auc_at_best_loss = 0.0
    best_threshold_at_best_loss = 0.0
    best_realistic_return_at_best_loss = 0.0
    best_daily_auc_at_best_loss = float('nan')
    best_ic_at_best_loss = float('nan')

    # 按逐日判别力保存的最佳模型（预热结束后）
    #
    # 为什么换成 IC：这里原来是 `realistic_return > best_realistic_return`——
    # 在约 243 天的验证集上对**收益**无条件取 argmax。这个量的标准误约 0.5%，
    # 也就是说在噪声主导的数上挑最大值，是典型的 winner's curse：
    # 挑出来的 epoch 到测试集必然回落。IC 的日间方差比收益小一到两个量级，
    # 且不吃「哪几天整体好」这种无法交易的部分。
    best_ic = -float('inf')
    best_ic_epoch = 0
    best_model_by_ic = None
    best_return_at_best_ic = 0.0
    best_auc_at_best_ic = 0.0
    best_daily_auc_at_best_ic = float('nan')
    best_threshold_at_best_ic = 0.0
    best_return_value_at_best_ic = 0.0

    # 逐日指标是否可用（依赖 eval_day_indices）。退化时只警告一次，不刷屏。
    daily_metrics_available = None

    # 早停机制（patience = EPOCHS * 0.25）
    patience = int(epochs * 0.25)
    early_stopping = EarlyStopping(patience=patience)

    # 预计算所有训练样本（验证+归一化+标签+收益率只做一次，后续epoch只做数组索引）
    train_rng = random.Random(DataConfig.RANDOM_SEED)
    pool_inputs, pool_targets, pool_returns, pos_indices, neg_indices, sample_key_map = precompute_training_pool(
        train_stock_info, feature_normalizer
    )

    # VISReg 辅助池：把 [N, 45, 18] 展平成 [N*45, 18] 的视图，
    # 每步随机取若干单日向量。这里取的是“某一天长什么样”的边际样本，
    # 与主 batch 的 45 天序列窗口是两条独立的数据流。
    if pool_inputs.ndim != 3:
        raise ValueError(f"VISReg 需要 [N, context, input_dim] 的训练池，实际 {pool_inputs.shape}")
    aux_pool_flat = pool_inputs.reshape(-1, pool_inputs.shape[-1])
    aux_rng = np.random.default_rng(DataConfig.RANDOM_SEED + 1)
    print(f"  VISReg 辅助池: {len(aux_pool_flat):,} 条单日 K 线")

    # 根据采样策略初始化采样器
    use_temporal = DataConfig.SAMPLING_STRATEGY == 'temporal'
    if use_temporal:
        temporal_sampler = TemporalSampler(train_stock_info)
        print("  使用时间顺序采样策略（预计算池化版）")

    # 记录每轮收益率
    epoch_returns = []

    global_step = 0   # 全训练累计的优化步数，学习率按它取值

    # 逐步指标日志：训练损失每一步都落盘（成本≈0），供监控页画稠密曲线。
    # 验证损失仍只在测评周期末算一次（全量验证集一次约 16 秒，不能每步做）。
    step_csv_path = os.path.join(run_dir, 'step_metrics.csv')
    step_csv = open(step_csv_path, 'w', newline='', encoding='utf-8')
    step_writer = csv.writer(step_csv)
    step_writer.writerow(['step', 'epoch', 'lr', 'train_loss', 'visreg', 'embed_z_std'])

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
        epoch_visreg_raw = 0.0
        epoch_z_std = 0.0
        epoch_embed_grad = 0.0
        visreg_steps = 0

        # 学习率按步更新：epoch 头只打印本周期起点的学习率
        start_step = global_step
        current_lr = lr_scheduler.lr_at(start_step)
        lr_status = f'{lr_scheduler.phase_name(start_step)}中'

        lr_str = f"{current_lr:.6g}"
        if len(optimizer.param_groups) > 1:
            lr_str += f" (embedding {current_lr * lr_scheduler.group_scales[1]:.6g})"
        print(f'Epoch {epoch + 1}/{epochs} · step {start_step + 1}~'
              f'{start_step + batches_per_epoch} · LR: {lr_str} ({lr_status})')

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
        # 标签目前为二分类（generate_label 返回 0/1），早期版本曾设计软标签边界，
        # 但实际从未启用——保留 0.5 阈值的二分类统计更贴合现状，避免误读
        count_positive = np.sum(epoch_targets >= 0.5)
        count_negative = np.sum(epoch_targets < 0.5)
        total_count = len(epoch_targets)
        print(f'  标签分布: 上涨={count_positive}({count_positive/total_count:.1%}), 不涨={count_negative}({count_negative/total_count:.1%})')

        # 转换为tensor
        epoch_inputs_tensor = torch.tensor(epoch_inputs, dtype=torch.float32).to(device)
        epoch_targets_tensor = torch.tensor(epoch_targets, dtype=torch.float32).to(device)

        # 计算实际可用的batch数量
        actual_batches = len(epoch_inputs_tensor) // batch_size
        if actual_batches < batches_per_epoch:
            print(f'  ⚠ 警告：实际batch数({actual_batches}) < 期望batch数({batches_per_epoch})，将使用实际数量')

        for step in range(actual_batches):
            # 每个优化步先更新学习率，再前向/反向
            lr_now = lr_scheduler.step(global_step)
            global_step += 1

            start_idx = step * batch_size
            end_idx = (step + 1) * batch_size

            batch_inputs = epoch_inputs_tensor[start_idx:end_idx]
            batch_targets = epoch_targets_tensor[start_idx:end_idx]

            # 独立采样单日辅助 batch（与主序列 batch 无重叠约束，采样有放回）
            aux_idx = aux_rng.integers(0, len(aux_pool_flat),
                                       size=RegularizationConfig.VISREG_AUX_BATCH_SIZE)
            aux_x = torch.from_numpy(aux_pool_flat[aux_idx]).float().to(device)

            aux_stats = {'visreg': 0.0, 'z_std': 0.0}

            def _loss_fn():
                output = model(batch_inputs)
                if hasattr(criterion, 'update_weights'):
                    criterion.update_weights(batch_targets)
                loss = criterion(output.squeeze(-1), batch_targets)

                # 辅助分支只过 Embedding；fp32 计算保证 VISReg 的分位数稳定
                z_aux = model.encode(aux_x).float()
                loss_visreg = visreg_loss_fn(z_aux / RegularizationConfig.TARGET_STD)
                loss = loss + RegularizationConfig.VISREG_MAIN_WEIGHT * loss_visreg
                aux_stats['visreg'] = loss_visreg.item()
                aux_stats['z_std'] = z_aux.std(dim=0).mean().item()

                return loss, output

            loss_val, _ = training_step(model, optimizer, _loss_fn)
            total_loss += loss_val * (end_idx - start_idx)
            embed_grad_norm = torch.sqrt(sum(
                p.grad.detach().pow(2).sum()
                for p in embed_params if p.grad is not None))
            epoch_embed_grad += embed_grad_norm.item()
            epoch_visreg_raw += aux_stats['visreg']
            epoch_z_std += aux_stats['z_std']
            visreg_steps += 1

            # 逐步指标落盘（每 50 步 flush 一次，监控页不必等训练结束）
            step_writer.writerow([
                global_step - 1, epoch + 1, f'{lr_now:.6e}',
                f'{loss_val:.6f}', f"{aux_stats['visreg']:.6f}", f"{aux_stats['z_std']:.6f}",
            ])
            if global_step % 50 == 0:
                step_csv.flush()

            # 进度显示
            progress = (step + 1) / actual_batches * 100
            processed_samples = (step + 1) * batch_size
            avg_loss = total_loss / processed_samples
            print(f'\r  训练进度: {progress:.1f}%, Loss: {avg_loss:.4f}, LR: {lr_now:.2e}', end='', flush=True)

        print()
        print()

        avg_visreg = epoch_visreg_raw / max(visreg_steps, 1)
        avg_z_std = epoch_z_std / max(visreg_steps, 1)
        avg_embed_grad = epoch_embed_grad / max(actual_batches, 1)

        # 清理内存
        del epoch_inputs_tensor, epoch_targets_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 学习率已在每个 batch 内按步更新，这里不再按轮推进

        # 评估模型（在验证集上）
        stats = evaluate_model(
            model, eval_inputs, eval_targets, eval_cumulative_returns,
            device,
            eval_day_indices=eval_day_indices,
            eval_daily_returns=eval_daily_returns,
            criterion=eval_criterion,
            tradeable_mask=eval_tradeable_mask
        )

        # 计算训练集平均损失
        total_samples = len(epoch_inputs)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0

        val_loss = stats['test_loss']
        eval_label = "验证集" if has_val else "测试集"

        # 两个 AUC 口径必须并列打印：pooled 会吸收「哪几天整体好」这种无法交易的成分，
        # 逐日 AUC 才是「同一天里买的和没买的有无区别」。二者差多少，就是虚高多少。
        print(f'  [模型] 训练损失: {avg_loss:.4f}, {eval_label}损失: {val_loss:.4f}, '
              f'AUC(pooled): {stats["auc"]:.4f}, AUC(逐日): {_fmt_or_na(stats.get("daily_auc"))}, '
              f'Prec@10%: {stats["precision_top10"]:.3f}, Prec@3%: {stats["precision_top3"]:.3f}, '
              f'Prec@1%: {stats["precision_top1"]:.3f} (基线: {stats["base_positive_rate"]:.3f})')
        print(f'         逐日判别力: IC {_fmt_or_na(stats.get("ic_mean"), "{:+.4f}")}, '
              f'ICIR {_fmt_or_na(stats.get("icir"), "{:+.3f}")}, '
              f'IC的t值 {_fmt_or_na(stats.get("ic_t"), "{:+.2f}")}, '
              f'日Top{DataConfig.TOP_K}%命中率 {_fmt_or_na(stats.get("daily_precision"), "{:.3f}")} '
              f'(统计天数 {stats.get("eval_days", 0)})')
        print(f'         表示正则: VISReg {avg_visreg:.4f} '
              f'(λ={RegularizationConfig.VISREG_MAIN_WEIGHT}), '
              f'Embedding逐维std均值 {avg_z_std:.3f} '
              f'(目标 {RegularizationConfig.TARGET_STD})')
        print(f'         Embedding 梯度范数: {avg_embed_grad:.4e}')
        if daily_metrics_available is None:
            daily_metrics_available = bool(stats.get('eval_days', 0) >= 2 and
                                           np.isfinite(stats.get('ic_mean', np.nan)))
            if not daily_metrics_available:
                print(f'         ⚠ 本轮拿不到逐日指标（eval_day_indices 缺失或每日样本过少），'
                      f'选模已退回 pooled AUC；该口径会把跨日行情算成判别力，不要拿它下结论。')
            elif stats.get('eval_days', 0) < DataConfig.MIN_EVAL_DAYS:
                print(f'         ⚠ {eval_label}只有 {stats.get("eval_days", 0)} 个预测日'
                      f'（建议 ≥ {DataConfig.MIN_EVAL_DAYS}）：在这个天数上 IC 的日间波动很大，'
                      f'选 epoch 看 ICIR，别看单期收益。')
        print(f'         预测均值: {stats["pred_mean"]:.3f}, 高置信(>0.7): {stats["high_conf_count"]}, 低置信(<0.2): {stats["low_conf_count"]}')
        daily_str = f', 日Top{DataConfig.TOP_K}%收益: {stats["daily_top_return"]*100:+.2f}%' if stats.get("daily_top_return") is not None else ''
        print(f'         Top{DataConfig.TOP_K}%收益: {stats["top_return"]*100:+.2f}%{daily_str}')

        if stats['realistic_stats'] is not None:
            rs = stats['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY},最大持仓{DataConfig.MAX_HOLDINGS}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
            print(f'         【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'         【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')

        epoch_return = {
            'turn': epoch + 1,
            'return': stats['top_return'] * 100,
            'daily_return': stats.get('daily_top_return'),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            'auc': stats['auc'],
            'daily_auc': stats.get('daily_auc'),
            'ic_mean': stats.get('ic_mean'),
            'icir': stats.get('icir'),
            'ic_t': stats.get('ic_t'),
            'daily_precision': stats.get('daily_precision'),
            'eval_days': stats.get('eval_days', 0),
            'precision_top10': stats['precision_top10'],
            'precision_top3': stats['precision_top3'],
            'precision_top1': stats['precision_top1'],
            'avg_realistic_return': stats['realistic_stats']['avg_realistic_return'] if stats.get('realistic_stats') else None,
            'dispersion_std': stats.get('dispersion_std', 0),
            'dispersion_range': stats.get('dispersion_range', 0),
            'dispersion_iqr': stats.get('dispersion_iqr', 0),
            'pos_ratio': stats.get('pred_mean', 0),
            'high_conf_ratio': stats.get('high_conf_count', 0) / len(eval_targets) if eval_targets is not None else 0,
            'visreg': avg_visreg,
            'embed_z_std': avg_z_std,
            'embed_grad_norm': avg_embed_grad,
        }
        epoch_returns.append(epoch_return)

        print_dispersion_sparkline(stats.get('all_preds', []), epoch_returns, all_targets=stats.get('all_targets'))

        # 早停检测（基于验证集指标）
        improved, improve_reason = early_stopping.check_improve(
            avg_loss=val_loss,
            top_return=stats['top_return'],
            auc=stats['auc'],
            threshold=stats['top_threshold'],
            ic=stats.get('ic_mean'),          # 首选判据；拿不到时内部退化到收益
        )

        if improved:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'         ✓ {improve_reason} (进度: {no_improve_count}/{patience_limit})')
        else:
            no_improve_count, patience_limit = early_stopping.get_progress()
            print(f'         ⚠ 无改善 ({no_improve_count}/{patience_limit})')

        # 按loss保存最佳模型（基于验证集指标）
        realistic_return = stats['realistic_stats']['avg_realistic_return'] if stats.get('realistic_stats') else 0.0

        # 选模主键：优先逐日 IC，其次逐日 AUC，最后才退回 pooled AUC。
        # 三者都不依赖全局阈值，也不把「哪几天整体好」算成选股能力。
        sel_key = stats.get('ic_mean')
        if sel_key is None or not np.isfinite(sel_key):
            sel_key = stats.get('daily_auc')
        if sel_key is None or not np.isfinite(sel_key):
            sel_key = stats.get('auc', 0.5)
        sel_key = float(sel_key)

        # 硬门槛已删除。原条件是 top_return >= 0.006 且 auc > 0.69，
        # 它会让低于门槛的 epoch 永远不落盘 —— 「过不了 70%」是这么来的，不是模型上限。
        if (epoch + 1) > warmup_epochs:
            if val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch + 1
                best_model_by_loss = copy.deepcopy(model.state_dict())
                best_return_at_best_loss = stats['top_return']
                best_auc_at_best_loss = stats['auc']
                best_threshold_at_best_loss = stats['top_threshold']
                best_realistic_return_at_best_loss = realistic_return
                best_daily_auc_at_best_loss = stats.get('daily_auc', float('nan'))
                best_ic_at_best_loss = stats.get('ic_mean', float('nan'))
                print(f'         ✓ 新最佳模型（{eval_label} loss）！Loss: {best_loss:.4f}, '
                      f'逐日AUC: {_fmt_or_na(best_daily_auc_at_best_loss)}, '
                      f'IC: {_fmt_or_na(best_ic_at_best_loss, "{:+.4f}")} (第{best_loss_epoch}轮)')

        # 按逐日判别力保存最佳模型（基于验证集指标）
        if (epoch + 1) > warmup_epochs:
            if sel_key > best_ic:
                best_ic = sel_key
                best_ic_epoch = epoch + 1
                best_model_by_ic = copy.deepcopy(model.state_dict())
                best_return_at_best_ic = stats['top_return']
                best_auc_at_best_ic = stats['auc']
                best_daily_auc_at_best_ic = stats.get('daily_auc', float('nan'))
                best_threshold_at_best_ic = stats['top_threshold']
                best_return_value_at_best_ic = realistic_return
                print(f'         ✓ 新最佳模型（{eval_label}判别力）！'
                      f'IC: {_fmt_or_na(stats.get("ic_mean"), "{:+.4f}")}, '
                      f'逐日AUC: {_fmt_or_na(best_daily_auc_at_best_ic)}, '
                      f'实战: {realistic_return*100:+.1f}% (第{best_ic_epoch}轮)')

        print("-" * 60)

        # 早停检查
        if early_stopping.should_stop() and TrainingConfig.OPEN_EARLY_STOPPING:
            print(f"\n⚠ 早停触发：连续{patience}轮无改善，停止训练")
            break

    step_csv.close()
    print(f"✓ 逐步指标已保存: {os.path.relpath(step_csv_path, DataConfig.OUTPUT_DIR)}")

    # ========== 最终测试集评估 → 保存模型 ==========
    print("\n" + "=" * 60)
    print(f"训练完成！")
    eval_label = "验证集" if has_val else "测试集"
    print(f"最佳模型（按{eval_label} loss）: 第{best_loss_epoch}轮, Loss: {best_loss:.4f}, "
          f"逐日AUC: {_fmt_or_na(best_daily_auc_at_best_loss)}, IC: {_fmt_or_na(best_ic_at_best_loss, '{:+.4f}')}")
    print(f"最佳模型（按{eval_label}判别力）: 第{best_ic_epoch}轮, "
          f"IC: {_fmt_or_na(best_ic, '{:+.4f}')}, 逐日AUC: {_fmt_or_na(best_daily_auc_at_best_ic)}, "
          f"实战收益率: {best_return_value_at_best_ic*100:+.1f}%, Top1%: {best_return_at_best_ic*100:+.2f}%")

    # 此刻才构建测试集评估数据集（训练期间不消费，延后构建缩短训练前等待）
    print("创建测试集评估数据集（训练结束后仅评估一次）...")
    # start_date/end_date 必须显式给：否则首个预测日会推迟约 CONTEXT_LENGTH 个交易日
    # （丢掉 2023 年头两个月），末个预测日又会溢出 TEST_END_DATE 进 2026 年。
    test_eval_inputs, test_eval_targets, test_eval_cumulative_returns, test_eval_day_indices, test_eval_daily_returns, test_eval_tradeable_mask = create_fixed_evaluation_dataset(
        test_stock_info, feature_normalizer,
        start_date=DataConfig.TEST_START_DATE, end_date=DataConfig.TEST_END_DATE
    )

    # 测试集评估（用测试集指标保存到模型文件中）
    test_return = 0.0
    test_realistic_return = 0.0

    # 用于保存的测试集指标（默认用验证集指标，有测试集时覆盖）
    save_loss_return = best_return_at_best_loss
    save_loss_threshold = best_threshold_at_best_loss
    save_loss_auc = best_auc_at_best_loss
    save_ic_return = best_return_at_best_ic
    save_ic_threshold = best_threshold_at_best_ic
    save_ic_auc = best_auc_at_best_ic

    if len(test_eval_inputs) > 0:
        print("\n" + "=" * 60)
        print("测试集最终评估（即将保存的模型）")
        print("=" * 60)

        # 创建测试集评估损失（统一走 create_eval_criterion，消除与 run.py / signal_flow 的复制粘贴）
        test_eval_criterion = create_eval_criterion(test_eval_targets)

        # 评估按loss选出的最佳模型
        if best_model_by_loss is not None:
            model.load_state_dict(best_model_by_loss)
            test_stats_loss = evaluate_model(
                model, test_eval_inputs, test_eval_targets, test_eval_cumulative_returns,
                device,
                eval_day_indices=test_eval_day_indices,
                eval_daily_returns=test_eval_daily_returns,
                criterion=test_eval_criterion,
                tradeable_mask=test_eval_tradeable_mask
            )
            _print_eval_summary("测试集·按loss选出的模型", test_stats_loss)
            # 用测试集指标覆盖保存指标
            save_loss_return = test_stats_loss['top_return']
            save_loss_threshold = test_stats_loss['top_threshold']
            save_loss_auc = test_stats_loss['auc']
            test_return = test_stats_loss['top_return']

        # 评估按逐日判别力选出的最佳模型
        if best_model_by_ic is not None:
            model.load_state_dict(best_model_by_ic)
            test_stats_ic = evaluate_model(
                model, test_eval_inputs, test_eval_targets, test_eval_cumulative_returns,
                device,
                eval_day_indices=test_eval_day_indices,
                eval_daily_returns=test_eval_daily_returns,
                criterion=test_eval_criterion,
                tradeable_mask=test_eval_tradeable_mask
            )
            _print_eval_summary("测试集·按判别力选出的模型", test_stats_ic)
            # 用测试集指标覆盖保存指标
            save_ic_return = test_stats_ic['top_return']
            save_ic_threshold = test_stats_ic['top_threshold']
            save_ic_auc = test_stats_ic['auc']
            test_realistic_return = test_stats_ic['realistic_stats']['avg_realistic_return'] if test_stats_ic.get('realistic_stats') else 0.0

        # 测试集窗口太短时结论不可信 —— 这一条必须在流程里，不能只写在文档里。
        # 依据：20 日反转这种固定因子，在 136 天窗口上的 95% 置信区间是 ±1.39%，
        # 比多数模型报告的日均超额本身还大。
        _test_days = 0
        if len(test_eval_day_indices) > 0:
            _test_days = int(len(np.unique(test_eval_day_indices)))
        if 0 < _test_days < DataConfig.MIN_EVAL_DAYS:
            print("\n" + "!" * 60)
            print(f"⚠ 测试集只有 {_test_days} 个预测日（建议 ≥ {DataConfig.MIN_EVAL_DAYS}）。")
            print("  在这个天数上，日均超额的标准误约 ±0.5%，")
            print("  与你要分辨的差异同量级 —— 上面这些数字区分不了「模型好」和「运气好」。")
            print("  不要拿它决定是否换架构、换标签。改用 IC / ICIR 判断，它们方差小得多。")
            print("!" * 60)

        print("=" * 60)

    # 保存模型（嵌入测试集评估指标）——统一落入本次 run 目录
    if best_model_by_loss is not None:
        save_path = save_model_with_metadata(
            best_model_by_loss,
            save_loss_return, save_loss_threshold, save_loss_auc,
            best_loss_epoch,
            model_prefix="model_loss",
            output_dir=run_dir,
        )
        print(f"✓ 模型(loss)已保存: {os.path.relpath(save_path, DataConfig.OUTPUT_DIR)}")
        print(f"  测试集 Top1%收益: {save_loss_return*100:+.2f}%, AUC: {save_loss_auc:.4f}")

    if best_model_by_ic is not None:
        save_path_ic = save_model_with_metadata(
            best_model_by_ic,
            save_ic_return, save_ic_threshold, save_ic_auc,
            best_ic_epoch,
            model_prefix="model_ic",
            output_dir=run_dir,
        )
        print(f"✓ 模型(判别力)已保存: {os.path.relpath(save_path_ic, DataConfig.OUTPUT_DIR)}")
        print(f"  测试集 Top1%收益: {save_ic_return*100:+.2f}%, AUC: {save_ic_auc:.4f}")
        print(f"  选模依据：验证集 IC {_fmt_or_na(best_ic, '{:+.4f}')}（第{best_ic_epoch}轮）"
              f"——不是收益 argmax，避免 winner's curse")

    # 保存每轮收益率到CSV（与模型同住本次 run 目录，产物集中追溯）
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(run_dir, f"epoch_returns_{timestamp}.csv")

    # 逐日口径字段与 pooled 并列落盘：事后要能看出「哪个 epoch 是 pooled 虚高」。
    # 选 epoch 看 daily_auc / ic_mean / icir，不要看 avg_realistic_return。
    fieldnames = ['turn', 'top_return', 'daily_return', 'train_loss', 'val_loss',
                  'auc', 'daily_auc', 'ic_mean', 'icir', 'ic_t', 'daily_precision',
                  'eval_days', 'prec_top10', 'prec_top3', 'prec_top1', 'avg_realistic_return',
                  'visreg', 'embed_z_std', 'embed_grad_norm']

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
                'daily_auc': _fmt_or_na(er.get('daily_auc'), '{:.4f}', ''),
                'ic_mean': _fmt_or_na(er.get('ic_mean'), '{:+.4f}', ''),
                'icir': _fmt_or_na(er.get('icir'), '{:+.4f}', ''),
                'ic_t': _fmt_or_na(er.get('ic_t'), '{:+.3f}', ''),
                'daily_precision': _fmt_or_na(er.get('daily_precision'), '{:.4f}', ''),
                'eval_days': er.get('eval_days', 0),
                'prec_top10': f"{er['precision_top10']:.4f}" if er.get('precision_top10') is not None else "",
                'prec_top3': f"{er['precision_top3']:.4f}" if er.get('precision_top3') is not None else "",
                'prec_top1': f"{er['precision_top1']:.4f}" if er.get('precision_top1') is not None else "",
                'avg_realistic_return': f"{er['avg_realistic_return']*100:.1f}" if er.get('avg_realistic_return') is not None else "",
                'visreg': _fmt_or_na(er.get('visreg'), '{:.4f}', ''),
                'embed_z_std': _fmt_or_na(er.get('embed_z_std'), '{:.4f}', ''),
                'embed_grad_norm': _fmt_or_na(er.get('embed_grad_norm'), '{:.4e}', ''),
            }
            writer.writerow(row)

    print(f"✓ 训练日志已保存: {os.path.relpath(returns_csv_path, DataConfig.OUTPUT_DIR)}")
    return test_return, test_realistic_return


def _fmt_or_na(x, fmt='{:.4f}', na='n/a'):
    """格式化数值；nan / None 一律显示成 n/a，绝不伪装成 0。

    选模指标里出现 nan 是有含义的（这一天/这个口径算不出来），
    打印成 0.0000 会让人误以为是「模型很烂」而不是「指标不可用」。
    """
    try:
        if x is None or not np.isfinite(float(x)):
            return na
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return na


def _print_eval_summary(tag, stats):
    """评估摘要：pooled 与逐日口径并列打印。

    只报 pooled AUC 会导致「AUC 很高但赚不到钱」——pooled 把跨日行情也算成了
    判别力。两个口径并列，差值就是虚高的部分。
    """
    print(f"\n  [{tag}] Loss: {_fmt_or_na(stats.get('test_loss'))}, "
          f"AUC(pooled): {_fmt_or_na(stats.get('auc'))}, "
          f"AUC(逐日): {_fmt_or_na(stats.get('daily_auc'))}")
    print(f"       IC: {_fmt_or_na(stats.get('ic_mean'), '{:+.4f}')}, "
          f"ICIR: {_fmt_or_na(stats.get('icir'), '{:+.3f}')}, "
          f"IC的t值: {_fmt_or_na(stats.get('ic_t'), '{:+.2f}')}, "
          f"统计天数: {stats.get('eval_days', 0)}")
    print(f"       Top{DataConfig.TOP_K}%收益: {_fmt_or_na(stats.get('top_return'), '{:+.4f}')}, "
          f"日Top{DataConfig.TOP_K}%命中率: {_fmt_or_na(stats.get('daily_precision'), '{:.3f}')}")
    if stats.get('realistic_stats'):
        rs = stats['realistic_stats']
        print(f"       实战收益率: {rs['avg_realistic_return']*100:+.1f}%")


def _run_selfcheck_before_training():
    """训练前的数据自检（不变量：关键步骤不可跳过）。

    校验的是「训练所依赖的数据前提还在不在」：
    审计范围是否等于训练用的池、派生状态是否比源数据旧、派生表是否只有一份表达。

    为什么要做成代码而不是文档叮嘱：审计必须在数据更新之后跑，这一步原来只写在
    数据维护文档里，而且那条流水线**还漏了复权物化**。
    漏跑不会报错，只会让训练拿到过期的排除区间，或让一部分股票带着未物化的
    复权列（特征基准静默异常）。
    """
    print("\n" + "=" * 60)
    print("训练前自检")
    print("=" * 60)
    try:
        from equinet_data_manager.selfcheck import prerequisites
        findings = prerequisites(DataConfig.DB_PATH)
    except Exception as e:
        print(f"  ⚠ 自检无法运行：{e}")
        print("  ⚠ 训练继续，但「数据前提是否成立」未经验证")
        return

    for f in findings:
        print(f"  {f}")
    if not findings:
        print("  ✓ 全部不变量成立")

    errors = [f for f in findings if f.level == 'error']
    if errors and getattr(DataConfig, 'STRICT_SELFCHECK', True):
        raise RuntimeError(
            f"训练前自检发现 {len(errors)} 个错误，已中止。"
            f"修完后重跑；确要强行训练就把 DataConfig.STRICT_SELFCHECK 设为 False。")
    if errors:
        print(f"  [!] {len(errors)} 个错误被忽略（STRICT_SELFCHECK=False）")
    print("=" * 60)


if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary()

    # 获取设备
    device = DeviceConfig.get_device()

    # 创建本次训练的 run 目录（目录即模型：本次 run 的所有产物都落在里面）
    run_dir = create_run_dir(DataConfig.OUTPUT_DIR)
    print(f"  本次训练 run 目录: {run_dir}")

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

    # 数据质量排除（真错价 / 停牌填充 / 量纲异常附近的样本）
    #
    # 在此之前先做自检：审计范围是否等于训练用的池、派生状态是否比源数据旧。
    # 「顺序很重要」原来是文档里的一句叮嘱（而且那条流水线还漏了复权物化），
    # 一旦漏跑，下游不会报错、只会拿到过期或残缺的结论 —— 所以把它变成代码。
    _run_selfcheck_before_training()

    if DataConfig.EXCLUDE_DATA_ISSUE_SAMPLES:
        apply_quality_exclusions(train_stock_info)
    else:
        print("  数据质量排除: 已由 EXCLUDE_DATA_ISSUE_SAMPLES=False 关闭")

    # 打印数据集统计
    print("\n" + "="*60)
    print("数据集统计")
    print(f" 训练集: {len(train_stock_info)} 只股票")
    print(f" 验证集: {len(val_stock_info)} 只股票")
    print(f" 测试集: {len(test_stock_info)} 只股票")

    # 创建模型（单阶段端到端训练）
    amp_str = "BF16混合精度" if TrainingConfig.USE_AMP else "FP32精度"
    print(f"\n正在创建模型 ({amp_str})...")
    model = create_model(seq_len=DataConfig.CONTEXT_LENGTH).to(device)

    # 开始训练
    print("\n开始模型训练...")
    best_return, best_realistic_return = train(
        model, train_stock_info, val_stock_info, test_stock_info,
        device=device,
        feature_normalizer=feature_normalizer,
        run_dir=run_dir,
    )

    print(f"\n最终结果:")
    print(f"  最佳Top1%收益={best_return*100:+.2f}%")
    print(f"  最佳实战收益率={best_realistic_return*100:.1f}%")
