'''
进化训练脚本

核心思想：
- 支持多教师模型（多个模型A）
- 每个教师对样本打分，取平均后排名
- 前1%的样本作为伪正标签
- 训练模型B，如果B的收益率超过所有教师的平均收益率，则保存
- 最终保存最佳模型B为N

这样就实现了模型的集成进化：M1, M2, ... → N
'''

import os, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F, numpy as np
import copy
import argparse
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
    generate_pseudo_labels,
    calculate_test_loss,
    DynamicWeightedBCE,
    save_model_with_metadata,
    print_dispersion_sparkline
)


def train_evolve_model(teacher_paths, student_path, train_stock_info, test_stock_info,
                       epochs=TrainingConfig.EPOCHS, 
                       learning_rate=TrainingConfig.LEARNING_RATE, 
                       device=None, 
                       batch_size=TrainingConfig.BATCH_SIZE, 
                       batches_per_epoch=TrainingConfig.BATCHES_PER_EPOCH,
                       pseudo_pos_ratio=0.01,
                       pseudo_neg_ratio=0.05,
                       use_return_weight=False,
                       return_weight_alpha=3.0,
                       return_weight_clip=0.20,
                       seed=DataConfig.RANDOM_SEED):
    """
    进化训练函数（支持多教师模型）
    
    训练策略：
    - 加载多个模型作为教师（模型A1, A2, ...，固定用于纠偏）
    - 加载指定模型作为学生B（训练）
    - 每个教师对样本打分，取平均后排名
    - 前pseudo_pos_ratio的样本作为伪正标签（强制=1）
    - 倒数pseudo_neg_ratio的样本作为伪负标签（强制=0）
    - 中间样本保持原始标签
    - 训练模型B
    - 如果B的收益率超过B自己之前的最佳收益率：
      - 保存当前B为最佳
      - 将当前B克隆一份加入教师集
    - 最终保存最佳模型B为N
    """
    # 确保teacher_paths是列表
    if isinstance(teacher_paths, str):
        teacher_paths = [teacher_paths]

    # 收益加权功能暂时禁用（TemporalSampler不提供收益率数据）
    if use_return_weight:
        print("  ⚠ 警告：use_return_weight与TemporalSampler不兼容，已自动禁用")
        use_return_weight = False

    num_teachers = len(teacher_paths)
    
    print("\n" + "="*60)
    print(f"进化训练（{num_teachers}个教师模型）")
    print("="*60)
    print(f"教师模型（固定纠偏）:")
    for i, path in enumerate(teacher_paths):
        print(f"  [{i+1}] {os.path.basename(path)}")
    print(f"学生模型（训练）:")
    print(f"  [B] {os.path.basename(student_path)}")
    print(f"训练策略：")
    print(f"  - 教师们的平均预测排名 → 前{pseudo_pos_ratio*100:.0f}%作为伪正标签")
    print(f"  - 教师们的平均预测排名 → 倒数{pseudo_neg_ratio*100:.0f}%作为伪负标签")
    print(f"  - B收益率 > B自己之前最佳 → 保存 + 克隆加入教师集")
    print(f"  - 最终保存最佳模型B为N")
    print("="*60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(DataConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(DataConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(DataConfig.RANDOM_SEED)
    
    # 创建评估数据集
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = create_fixed_evaluation_dataset(test_stock_info)
    
    # 加载所有教师模型
    teachers = []
    print(f"正在加载{num_teachers}个教师模型...")
    for i, model_path in enumerate(teacher_paths):
        teacher = create_model().to(device)
        teacher = teacher.to(dtype=torch.bfloat16)
        
        state_dict = torch.load(model_path, map_location=device)
        teacher.load_state_dict(state_dict)
        teacher.eval()  # 教师模型固定不训练
        teachers.append(teacher)
        
        # 评估教师模型
        stats = evaluate_model(teacher, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name=f"教师{i+1}", eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns)
        print(f"  教师{i+1}: AUC={stats['auc']:.4f}, Top1%收益={stats['top_return']*100:+.2f}%")
        if stats['realistic_stats'] is not None:
            rs = stats['realistic_stats']
            print(f"         【实战收益率】平均: {rs['avg_realistic_return']*100:.1f}%")
    
    # 加载学生模型B
    print(f"正在加载学生模型B: {student_path}")
    model_b = create_model().to(device)
    model_b = model_b.to(dtype=torch.bfloat16)
    state_dict = torch.load(student_path, map_location=device)
    model_b.load_state_dict(state_dict)
    
    # 评估初始学生B
    stats_b_init = evaluate_model(model_b, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="B(初始)", eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns)
    print(f"  学生B: AUC={stats_b_init['auc']:.4f}, Top1%收益={stats_b_init['top_return']*100:+.2f}%")
    if stats_b_init['realistic_stats'] is not None:
        rs = stats_b_init['realistic_stats']
        print(f"         【实战收益率】平均: {rs['avg_realistic_return']*100:.1f}%")
    
    # 进化训练使用更低的学习率（已训练模型需要更小的学习率避免破坏已学特征）
    evolve_lr = learning_rate * 0.2  # 使用原学习率的20%
    print(f"进化学习率: {evolve_lr:.6f} (原学习率的20%)")
    
    # 模型B的优化器（只训练B）
    if TrainingConfig.USE_MANO:
        from optimizers import create_optimizer
        optimizer_b = create_optimizer(
            model_b,
            optimizer_type='mano',
            lr=evolve_lr,
            momentum=TrainingConfig.MANO_MOMENTUM,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            betas=TrainingConfig.MANO_ADAMW_BETAS
        )
    elif TrainingConfig.USE_ADAMW:
        optimizer_b = optim.AdamW(model_b.parameters(), lr=evolve_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
    else:
        optimizer_b = optim.Adam(model_b.parameters(), lr=evolve_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
    
    # 进化训练不使用warmup，直接使用余弦退火
    total_main_epochs = epochs
    main_scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, 
        T_max=total_main_epochs,
        eta_min=evolve_lr * 0.01  # 最小学习率
    )
    
    # 损失函数：由全局配置控制
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

    def compute_loss(logits, targets, sample_weight=None):
        logits = logits.squeeze(-1)
        targets = targets.squeeze()

        if LossConfig.use_dynamic_bce():
            criterion.update_weights(targets)
            return criterion(logits, targets)

        loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction='none')
        if sample_weight is not None:
            weights = sample_weight.squeeze().to(dtype=loss.dtype)
            loss = loss * weights
        return loss.mean()

    # 记录最佳状态（以学生B的初始收益率为基准）
    best_return_b = stats_b_init['top_return']  # 初始基准为B自己的收益率
    best_auc_b = stats_b_init['auc']
    best_threshold_b = stats_b_init['top_threshold']
    best_model_state = copy.deepcopy(model_b.state_dict())
    best_epoch = 0
    evolution_count = 0  # 进化次数（B超越自己的次数）
    
    # 早停机制
    patience = TrainingConfig.EPOCHS*0.25
    no_improve_count = 0

    # 创建时间顺序采样器（使用train.py的统一采样机制）
    sampler = TemporalSampler(train_stock_info)
    train_rng = random.Random(seed)

    # 记录每轮收益率
    epoch_returns = []  # 格式: [{'turn': 1, 'return_b': 1.62}, ...]

    for epoch in range(epochs):
        # 所有教师固定不训练
        model_b.train()

        total_loss_b = 0
        total_pseudo_pos = 0
        total_unchanged = 0

        # 获取当前学习率（调度器在epoch结束后调用）
        current_lr = optimizer_b.param_groups[0]['lr']
        phase = "进化训练"

        # 使用时间顺序采样器生成训练数据（与train.py统一）
        train_inputs, train_targets = sample_with_pools(
            sampler, train_stock_info, batch_size, batches_per_epoch, train_rng
        )

        # 统计标签分布
        up_count = np.sum(train_targets == 1.0)
        boundary_count = np.sum((train_targets > 0) & (train_targets < 1.0))
        down_count = np.sum(train_targets == 0.0)

        print(f"Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f} ({phase})")
        print(f"  标签分布: 上涨={up_count}({up_count/len(train_targets)*100:.1f}%), "
              f"边界={boundary_count}({boundary_count/len(train_targets)*100:.1f}%), "
              f"不涨={down_count}({down_count/len(train_targets)*100:.1f}%)")

        # 用所有教师模型生成伪标签（取平均预测）
        all_teacher_preds = []
        for teacher in teachers:
            teacher.eval()
            original_dtype = next(teacher.parameters()).dtype
            use_fp32_eval = original_dtype == torch.bfloat16
            if use_fp32_eval:
                teacher = teacher.float()
            with torch.no_grad():
                teacher_preds = []
                for i in range(0, len(train_inputs), batch_size):
                    batch_inputs = torch.tensor(train_inputs[i:i+batch_size], 
                                               dtype=torch.float32).to(device)
                    preds = torch.sigmoid(teacher(batch_inputs))
                    teacher_preds.append(preds.float().cpu().numpy())
                teacher_preds = np.concatenate(teacher_preds).flatten()
                all_teacher_preds.append(teacher_preds)
            if use_fp32_eval:
                teacher = teacher.to(original_dtype)
        
        # 计算教师平均预测
        avg_preds = np.mean(all_teacher_preds, axis=0)

        # 使用统一的top-k方式生成伪标签
        pseudo_targets, pseudo_stats = generate_pseudo_labels(
            avg_preds, train_targets,
            pseudo_pos_ratio=pseudo_pos_ratio,
            pseudo_neg_ratio=pseudo_neg_ratio
        )

        total_pseudo_pos = pseudo_stats['pseudo_pos_count']
        total_pseudo_neg = pseudo_stats['pseudo_neg_count']
        total_unchanged = pseudo_stats['unchanged_count']
        
        # 训练模型B
        num_batches = len(train_inputs) // batch_size
        nan_detected = False
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = torch.tensor(train_inputs[start_idx:end_idx], 
                                        dtype=torch.bfloat16).to(device)
            batch_targets = torch.tensor(pseudo_targets[start_idx:end_idx], 
                                        dtype=torch.bfloat16).to(device)

            if use_return_weight:
                batch_returns = torch.tensor(train_returns[start_idx:end_idx], dtype=torch.float32).to(device)
                batch_returns = torch.clamp(batch_returns, -return_weight_clip, return_weight_clip)
                batch_weight = 1.0 + return_weight_alpha * torch.abs(batch_returns)
                batch_weight = batch_weight / (batch_weight.mean() + 1e-8)
                batch_weight = batch_weight.to(dtype=torch.bfloat16)
            
            optimizer_b.zero_grad()
            logits_b = model_b(batch_inputs)
            loss_b = compute_loss(logits_b, batch_targets, batch_weight if use_return_weight else None)
            
            # NaN检测
            if torch.isnan(loss_b) or torch.isinf(loss_b):
                nan_detected = True
                print(f"\n  ⚠ 检测到NaN/Inf，跳过本轮并重置模型B")
                break

            loss_b.backward()

            torch.nn.utils.clip_grad_norm_(model_b.parameters(), max_norm=1.0)
            optimizer_b.step()

            # 累加loss时乘以batch_size，得到该batch的总损失
            total_loss_b += loss_b.item() * (end_idx - start_idx)

            # 打印进度
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                # 使用已处理的样本数计算当前平均损失
                processed_samples = (batch_idx + 1) * batch_size
                print(f"\r  训练进度: {progress:.1f}%, Loss_B: {total_loss_b/processed_samples:.4f}", end="")
        
        # 如果检测到NaN，从教师1重新克隆B并重置优化器
        if nan_detected:
            model_b = copy.deepcopy(teachers[0])
            # 恢复时使用更低的学习率
            recover_lr = evolve_lr * 0.1  # 恢复时使用更低的学习率
            if TrainingConfig.USE_MANO:
                from optimizers import create_optimizer
                optimizer_b = create_optimizer(
                    model_b,
                    optimizer_type='mano',
                    lr=recover_lr,
                    momentum=TrainingConfig.MANO_MOMENTUM,
                    weight_decay=TrainingConfig.WEIGHT_DECAY,
                    betas=TrainingConfig.MANO_ADAMW_BETAS
                )
            elif TrainingConfig.USE_ADAMW:
                optimizer_b = optim.AdamW(model_b.parameters(), lr=recover_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
            else:
                optimizer_b = optim.Adam(model_b.parameters(), lr=recover_lr, weight_decay=TrainingConfig.WEIGHT_DECAY)
            print(f"  → B已从教师1重新克隆，学习率降至 {recover_lr:.6f}")
            print("-" * 60)
            continue  # 跳过本轮评估
        
        print()  # 换行
        
        # 评估模型B
        stats_b = evaluate_model(model_b, eval_inputs, eval_targets, eval_cumulative_returns, device, model_name="B", eval_day_indices=eval_day_indices, eval_daily_returns=eval_daily_returns)

        # 计算训练集平均损失（除以样本数，与测试损失保持一致）
        # total_loss_b已经是所有样本的总损失（累加时乘以了batch_size）
        total_samples = len(train_inputs)
        avg_loss_b = total_loss_b / total_samples if total_samples > 0 else 0

        # 计算测试集损失
        test_loss_b = calculate_test_loss(model_b, eval_inputs, eval_targets, eval_criterion, device)

        # 记录当前轮次收益率
        epoch_return = {
            'turn': epoch + 1,
            'return': stats_b['top_return'] * 100,
            'train_loss': avg_loss_b,
            'test_loss': test_loss_b,
            'dispersion_std': stats_b.get('dispersion_std', 0),
            'dispersion_range': stats_b.get('dispersion_range', 0),
            'dispersion_iqr': stats_b.get('dispersion_iqr', 0),
            'pos_ratio': stats_b.get('pred_mean', 0),
            'high_conf_ratio': stats_b.get('high_conf_count', 0) / len(eval_targets) if eval_targets is not None else 0,
        }
        epoch_returns.append(epoch_return)
        
        print_dispersion_sparkline(stats_b.get('all_preds', []), epoch_returns)
        
        print(f"  [教师数量] {len(teachers)}个")
        print(f"  [B最佳] Top1%收益: {best_return_b*100:+.2f}%")
        print(f"  [模型B] 损失: {avg_loss_b:.4f}, AUC: {stats_b['auc']:.4f}")
        print(f"          预测均值: {stats_b['pred_mean']:.3f}, 高置信(>0.7): {stats_b['high_conf_count']}, 低置信(<0.2): {stats_b['low_conf_count']}")
        print(f"          Top1%收益: {stats_b['top_return']*100:+.2f}%")
        
        if stats_b['realistic_stats'] is not None:
            rs = stats_b['realistic_stats']
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
            mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else "全局阈值,不限数量"
            print(f'          【实战收益率({mode_str})】每日统计: {{{daily_stats_str}}}')
            print(f'          【实战收益率({mode_str})】平均实战收益率: {rs["avg_realistic_return"]*100:.1f}%')
        
        if stats_b.get('smart_exit_stats') is not None:
            se = stats_b['smart_exit_stats']
            print(f'          【智能止损】收益率: {se["avg_realistic_return"]*100:.1f}%, Day1止损: {se["stop_loss_day1_count"]}次, 累计止损: {se["stop_loss_cum_count"]}次, 止盈: {se["take_profit_count"]}次')
        
        print(f"          伪标签统计: 伪正={total_pseudo_pos}, 伪负={total_pseudo_neg}, 不变={total_unchanged}")
        
        # 检查是否进化：B收益率 > B自己之前的最佳收益率
        if stats_b['top_return'] > best_return_b:
            evolution_count += 1
            print(f"          ★ 进化！B({stats_b['top_return']*100:+.2f}%) > 之前最佳({best_return_b*100:+.2f}%)")
            
            # 保存最佳状态
            best_return_b = stats_b['top_return']
            best_auc_b = stats_b['auc']
            best_threshold_b = stats_b['top_threshold']
            best_model_state = copy.deepcopy(model_b.state_dict())
            best_epoch = epoch + 1
            no_improve_count = 0
            
            # 将当前B克隆一份加入教师集
            new_teacher = copy.deepcopy(model_b)
            new_teacher.eval()
            teachers.append(new_teacher)
            print(f"            → B已克隆加入教师集（当前教师数: {len(teachers)}）")
        else:
            no_improve_count += 1
            print(f"          ⚠ 无改进 ({no_improve_count}/{patience})")
        
        # 学习率调度（在optimizer.step()之后调用）
        main_scheduler_b.step()
        
        # 早停检查
        if no_improve_count >= patience:
            print(f"\n早停触发！连续{patience}轮无进化")
            break
        
        print("-" * 60)
    
    # 训练完成
    print("\n" + "=" * 60)
    print(f"进化训练完成！")
    print(f"初始教师数: {num_teachers} → 最终教师数: {len(teachers)}")
    print(f"总改进次数: {evolution_count}")
    print(f"最佳模型: 第{best_epoch}轮, Top1%收益: {best_return_b*100:+.2f}%, AUC: {best_auc_b:.4f}")

    # 保存每轮收益率到CSV（使用时间戳避免多模型训练时覆盖）
    timestamp_csv = datetime.now().strftime("%m%d_%H%M%S")
    returns_csv_path = os.path.join(DataConfig.OUTPUT_DIR, f"evolve_epoch_returns_{timestamp_csv}.csv")
    with open(returns_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['turn', 'B', 'train_loss', 'test_loss'])
        writer.writeheader()

        for epoch_return in epoch_returns:
            row = {
                'turn': epoch_return['turn'],
                'B': f"{epoch_return['return_b']:.2f}",
                'train_loss': f"{epoch_return['train_loss']:.4f}",
                'test_loss': f"{epoch_return['test_loss']:.4f}"
            }
            writer.writerow(row)

    print(f"✓ 每轮收益率已保存: {os.path.basename(returns_csv_path)}")
    print(f"  共记录 {len(epoch_returns)} 轮训练数据")

    # 保存最佳模型N（使用与train_clone相同的命名风格）
    final_teacher_count = len(teachers)
    extra_info = f"t{final_teacher_count}"
    save_path_n = save_model_with_metadata(
        best_model_state,
        best_return_b,
        best_threshold_b,
        best_auc_b,
        best_epoch,
        model_prefix="evolved",
        extra_info=extra_info,
        output_dir=DataConfig.OUTPUT_DIR
    )
    filename_n = os.path.basename(save_path_n)
    print(f"✓ 进化模型N已保存: {filename_n}")
    print(f"  Top1%阈值: {best_threshold_b:.4f} (预测值≥此值即入选Top1%)")
    print("=" * 60)
    
    return best_return_b, best_auc_b, evolution_count


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='进化训练：多教师纠偏，学生自我进化')
    parser.add_argument('--teachers', '-t', type=str, nargs='+', required=True,
                        help='教师模型路径（支持多个，用空格分隔）')
    parser.add_argument('--student', '-s', type=str, required=True,
                        help='学生模型路径（将被训练）')
    parser.add_argument('--epochs', '-e', type=int, default=TrainingConfig.EPOCHS,
                        help=f'训练轮数（默认: {TrainingConfig.EPOCHS}）')
    parser.add_argument('--pseudo_pos', '-p', type=float, default=0.01,
                        help='伪正标签比例（默认: 0.01，即前1%）')
    parser.add_argument('--pseudo_neg', '-n', type=float, default=0.05,
                        help='伪负标签比例（默认: 0.05，即倒数5%）')
    parser.add_argument('--use_return_weight', action='store_true',
                        help='启用收益加权BCE（根据逐样本收益绝对值加权）')
    parser.add_argument('--return_weight_alpha', type=float, default=3.0,
                        help='收益加权强度alpha（默认: 3.0）')
    parser.add_argument('--return_weight_clip', type=float, default=0.20,
                        help='收益裁剪阈值clip（默认: 0.20，即±20%）')
    parser.add_argument('--seed', type=int, default=DataConfig.RANDOM_SEED,
                        help=f'随机种子（默认: {DataConfig.RANDOM_SEED}），用于训练采样可复现')
    args = parser.parse_args()
    
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查模型文件是否存在
    for model_path in args.teachers:
        if not os.path.exists(model_path):
            print(f"错误：教师模型文件不存在: {model_path}")
            exit(1)
    if not os.path.exists(args.student):
        print(f"错误：学生模型文件不存在: {args.student}")
        exit(1)
    
    # 打印配置摘要
    print_config_summary()
    
    # 获取设备
    device = DeviceConfig.print_device_info()
    
    # 创建输出目录
    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    print("正在加载和预处理数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()
    
    # 打印数据集统计
    print("\n" + "="*60)
    print("数据集统计")
    print("="*60)
    print(f"训练集: {len(train_stock_info)} 只股票")
    print(f"测试集: {len(test_stock_info)} 只股票")
    print("="*60)
    
    # 开始进化训练
    print(f"\n开始进化训练（{len(args.teachers)}个教师模型）...")
    best_return, best_auc, evolution_count = train_evolve_model(
        args.teachers, args.student, train_stock_info, test_stock_info,
        device=device,
        epochs=args.epochs,
        pseudo_pos_ratio=args.pseudo_pos,
        pseudo_neg_ratio=args.pseudo_neg,
        use_return_weight=args.use_return_weight,
        return_weight_alpha=args.return_weight_alpha,
        return_weight_clip=args.return_weight_clip,
        seed=args.seed
    )
    
    print(f"\n最终结果:")
    print(f"  最佳Top1%收益: {best_return*100:+.2f}%")
    print(f"  最佳AUC: {best_auc:.4f}")
    print(f"  总改进次数: {evolution_count}")
