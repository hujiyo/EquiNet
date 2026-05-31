'''
MC Dropout 推理与选股脚本

与 run.py 的评估流程完全一致，唯一区别：
推理阶段使用 MC Dropout（Monte Carlo Dropout）技术：
  1. 将模型中所有 Dropout 层的 p 强制设为 0.1
  2. 模型保持在 train 模式（使 Dropout 生效）
  3. 对每个样本执行 100 次前向传播，取 sigmoid 后平均
  4. 用平均后的预测值参与后续评估

用法: python run_with_mc_dropout.py
'''

import os, sys, torch, numpy as np

from config import (DataConfig, DeviceConfig, LossConfig)
from data import (load_and_preprocess_data, create_fixed_evaluation_dataset, FeatureNormalizer,
                  create_recent_days_dataset)
from training_utils import DynamicWeightedBCE, _get_amp_context
from run import (list_available_models, load_model,
                 visualize_classification, print_banner, print_section, print_section_end,
                 select_model, print_recent_days_chart)


# ==================== MC Dropout 核心逻辑 ====================
MC_DROPOUT_P = 0.1       # MC Dropout 概率
MC_SAMPLES = 100          # 每个样本的前向传播次数


def set_mc_dropout(model, p=MC_DROPOUT_P):
    """
    将模型中所有 Dropout 层的概率设为 p，
    用于 MC Dropout 推理。
    """
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = p
            count += 1
    return count


def mc_dropout_predict(model, eval_inputs, device, num_samples=MC_SAMPLES,
                       batch_size=DataConfig.EVAL_BATCH_SIZE):
    """
    MC Dropout 推理：对每个样本执行 num_samples 次前向传播，返回 sigmoid 后的平均预测。

    Args:
        model: 已加载的模型（会被设为 train 模式）
        eval_inputs: numpy 数组，形状 (N, seq_len, input_dim)
        device: torch.device
        num_samples: MC 采样次数
        batch_size: 每次前向传播的批大小

    Returns:
        all_preds: numpy 数组，形状 (N,)，每次预测经 sigmoid 后取平均
    """
    model.train()  # 保持 Dropout 激活
    amp_ctx = _get_amp_context(device)
    num_total = len(eval_inputs)
    num_batches = (num_total + batch_size - 1) // batch_size

    # 累加器：逐 batch、逐 sample 累加 sigmoid 预测值
    accum = np.zeros(num_total, dtype=np.float64)

    for t in range(num_samples):
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, num_total)
                batch = torch.tensor(eval_inputs[start:end], dtype=torch.float32, device=device)
                with amp_ctx:
                    logits = model(batch)
                preds = torch.sigmoid(logits.float()).cpu().numpy().flatten()
                accum[start:end] += preds
                del batch, logits

    # 取平均
    all_preds = (accum / num_samples).astype(np.float32)
    return all_preds


def mc_dropout_evaluate(model, eval_inputs, eval_targets, eval_cumulative_returns,
                        device, model_name="MC-Dropout模型",
                        eval_day_indices=None, eval_daily_returns=None,
                        criterion=None, enable_portfolio_simulation=False,
                        num_samples=MC_SAMPLES):
    """
    使用 MC Dropout 预测后，复用 evaluate_model 的指标计算逻辑。

    由于 evaluate_model 内部会调 model.eval() 并自行推理，
    我们不能直接调用它。因此，先手动执行 MC Dropout 推理，
    然后内联 evaluate_model 的统计计算部分。
    """
    num_samples_data = len(eval_inputs)
    if num_samples_data == 0:
        return {
            'auc': 0.5, 'top_return': 0.0, 'daily_top_return': None,
            'top_count': 0, 'top_threshold': 0.0,
            'high_conf_count': 0, 'low_conf_count': 0,
            'pred_mean': 0.0, 'pred_std': 0.0,
            'filtered_count': 0, 'realistic_stats': None,
            'test_loss': None
        }

    # ===== MC Dropout 推理 =====
    print(f"  MC Dropout 推理: {MC_SAMPLES} 次前向传播, dropout_p={MC_DROPOUT_P}")
    all_preds = mc_dropout_predict(model, eval_inputs, device, num_samples=num_samples)

    # ===== 计算测试损失（单次前向传播即可，用 eval 模式） =====
    from sklearn.metrics import roc_auc_score
    from training_utils import calculate_realistic_return, calculate_portfolio_simulation

    test_loss = None
    if criterion is not None:
        model.eval()
        batch_size = DataConfig.EVAL_BATCH_SIZE
        amp_ctx = _get_amp_context(device)
        total_loss = 0.0
        with torch.no_grad():
            for i in range(0, num_samples_data, batch_size):
                start = i
                end = min(i + batch_size, num_samples_data)
                batch_inputs = torch.tensor(eval_inputs[start:end], dtype=torch.float32, device=device)
                batch_targets = torch.tensor(eval_targets[start:end], dtype=torch.float32, device=device)
                with amp_ctx:
                    logits = model(batch_inputs)
                logits = logits.float()
                loss = criterion(logits.squeeze(-1), batch_targets)
                total_loss += loss.item() * (end - start)
                del batch_inputs, batch_targets, logits
        test_loss = total_loss / num_samples_data

    # ===== 统计计算（与 evaluate_model 完全一致） =====
    all_targets = np.array(eval_targets)
    all_returns = np.array(eval_cumulative_returns)

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    sorted_indices = np.argsort(all_preds)[::-1]

    base_positive_rate = float(np.mean(all_targets))
    precision_at = {'base_positive_rate': base_positive_rate}
    for top_pct in [10, 3, 1]:
        k = max(10, int(len(all_preds) * top_pct / 100))
        subset_targets = all_targets[sorted_indices[:k]]
        precision_at[f'precision_top{top_pct}'] = float(np.mean(subset_targets))

    percent = DataConfig.TOP_K
    top_k = max(1, int(len(all_preds) * percent / 100))
    top_indices = sorted_indices[:top_k]
    top_returns = all_returns[top_indices]

    top_return = np.mean(top_returns)
    top_threshold = all_preds[sorted_indices[top_k - 1]]

    high_conf = all_preds > 0.7
    low_conf = all_preds < 0.2

    daily_top_return = None
    if eval_day_indices is not None:
        all_day_idx = np.array(eval_day_indices)
        unique_days = np.unique(all_day_idx)
        daily_top_indices_list = []
        for day in unique_days:
            day_mask = all_day_idx == day
            day_preds = all_preds[day_mask]
            day_local_k = max(1, int(len(day_preds) * percent / 100))
            day_sorted = np.argsort(day_preds)[::-1]
            day_top_local = day_sorted[:day_local_k]
            day_global_indices = np.where(day_mask)[0][day_top_local]
            daily_top_indices_list.append(day_global_indices)
        if daily_top_indices_list:
            all_daily_top_indices = np.concatenate(daily_top_indices_list)
            daily_top_return = float(np.mean(all_returns[all_daily_top_indices]))

    stats = {
        'auc': auc,
        **precision_at,
        'top_return': top_return,
        'daily_top_return': daily_top_return,
        'top_count': top_k,
        'top_threshold': top_threshold,
        'high_conf_count': int(np.sum(high_conf)),
        'low_conf_count': int(np.sum(low_conf)),
        'pred_mean': float(np.mean(all_preds)),
        'pred_std': float(np.std(all_preds)),
        'filtered_count': 0,
        'dispersion_std': float(np.std(all_preds)),
        'dispersion_range': float(np.max(all_preds) - np.min(all_preds)),
        'dispersion_iqr': float(np.percentile(all_preds, 75) - np.percentile(all_preds, 25)),
        'test_loss': test_loss,
    }

    if eval_day_indices is not None:
        actual_top_n = DataConfig.TOP_N_PER_DAY
        if actual_top_n == 0:
            actual_top_n = None
        stats['realistic_stats'] = calculate_realistic_return(
            all_preds, all_returns, eval_day_indices, percent, actual_top_n)

        if enable_portfolio_simulation and eval_daily_returns is not None:
            stats['portfolio_stats'] = calculate_portfolio_simulation(
                all_preds, all_returns, eval_daily_returns, eval_day_indices, percent, actual_top_n)
        else:
            stats['portfolio_stats'] = None
    else:
        stats['realistic_stats'] = None
        stats['portfolio_stats'] = None

    stats['all_preds'] = all_preds
    stats['all_targets'] = all_targets
    return stats


# ==================== 评估入口 ====================

def run_mc_dropout_evaluation(model, test_stock_info, device, feature_normalizer=None):
    """
    使用 MC Dropout 执行模型评估（与 run.py 的 run_evaluation 流程一致）
    """
    print(f"正在创建评估数据集...")

    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = \
        create_fixed_evaluation_dataset(test_stock_info, feature_normalizer)

    print(f"- 评估样本数: {len(eval_inputs)}")

    # 设置 MC Dropout
    num_dropout = set_mc_dropout(model, p=MC_DROPOUT_P)
    print(f"- 已将 {num_dropout} 个 Dropout 层设为 p={MC_DROPOUT_P}")

    print(f"正在评估模型 (MC Dropout ×{MC_SAMPLES})...")

    # 创建评估损失函数
    if LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
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
    else:
        import torch.nn as nn
        eval_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    stats = mc_dropout_evaluate(
        model, eval_inputs, eval_targets, eval_cumulative_returns,
        device, model_name="MC-Dropout选中模型",
        eval_day_indices=eval_day_indices,
        eval_daily_returns=eval_daily_returns,
        criterion=eval_criterion,
        enable_portfolio_simulation=True
    )
    test_loss = stats['test_loss']

    # 打印评估结果（与 run.py 格式一致）
    print()
    print(f"┌── 评估结果 (MC Dropout: p={MC_DROPOUT_P}, samples={MC_SAMPLES}) ──┐")
    print(f"│  测试损失:          {test_loss:.4f}")
    print(f"│  AUC:              {stats['auc']:.4f}")
    base_rate = stats.get('base_positive_rate', 0)
    print(f"│  正样本基线:       {base_rate:.3f}")
    print(f"│  Prec@10%:         {stats.get('precision_top10', 0):.3f}  ({stats.get('precision_top10', 0)/base_rate:.1f}x)" if base_rate > 0 else f"│  Prec@10%:         {stats.get('precision_top10', 0):.3f}")
    print(f"│  Prec@3%:          {stats.get('precision_top3', 0):.3f}  ({stats.get('precision_top3', 0)/base_rate:.1f}x)" if base_rate > 0 else f"│  Prec@3%:          {stats.get('precision_top3', 0):.3f}")
    print(f"│  Prec@1%:          {stats.get('precision_top1', 0):.3f}  ({stats.get('precision_top1', 0)/base_rate:.1f}x)" if base_rate > 0 else f"│  Prec@1%:          {stats.get('precision_top1', 0):.3f}")
    print(f"│  预测均值:          {stats['pred_mean']:.3f}")
    print(f"│  预测标准差:        {stats['pred_std']:.4f}")
    print(f"│  高置信(>0.7):      {stats['high_conf_count']} 个")
    print(f"│  低置信(<0.2):      {stats['low_conf_count']} 个")
    print(f"│  Top{DataConfig.TOP_K}%样本数:        {stats['top_count']} 个")
    print(f"│  Top{DataConfig.TOP_K}%平均收益:      {stats['top_return']*100:+.2f}%")
    if stats.get('daily_top_return') is not None:
        print(f"│  日Top{DataConfig.TOP_K}%平均收益:    {stats['daily_top_return']*100:+.2f}%")
    print(f"│")
    print(f"│  ★ Top{DataConfig.TOP_K}%阈值:        {stats['top_threshold']:.10f}")
    print(f"│")

    # 实战收益率
    if stats['realistic_stats'] is not None:
        rs = stats['realistic_stats']
        daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
        mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else \
                   f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY},最大持仓{DataConfig.MAX_HOLDINGS}" if DataConfig.MAX_SELECT_PER_DAY > 0 else \
                   "全局阈值,不限数量"
        print(f"│  【实战收益率({mode_str})】")
        print(f"│  每日统计: {{{daily_stats_str}}}")
        print(f"│  平均实战收益率: {rs['avg_realistic_return']*100:.1f}%")
        print(f"│")

    if stats.get('portfolio_stats') is not None:
        ps = stats['portfolio_stats']
        if ps['total_days'] > 0:
            print(f"│  【实战资金模拟（串行逐日买卖）】")
            print(f"│  总交易日: {ps['total_days']}天")
            print(f"│  总交易笔数: {ps['trade_count']}笔（买入{ps['buy_count']}, 卖出{ps['sell_count']}）")
            print(f"│  最终资金: {ps['final_value']:.4f}")
            print(f"│  总收益率: {ps['total_return_pct']:+.2f}%")
            print(f"│  最大回撤: {ps['max_drawdown']*100:.2f}%")

            dt = ps['daily_trades']
            print(f"│")
            print(f"│  *** 逐日交易明细（共{len(dt)}天） ***")
            for d in dt:
                cash_pct = d['cash_ratio'] * 100
                pv = d['portfolio_value']
                print(f"│  Day{d['day']:>3}: 买{d['buys']} 卖{d['open_sells']+d['close_sells']} "
                      f"资金={pv:.4f} (现金{cash_pct:.0f}%)")
            print(f"│")

    print(f"└───────────────────────────────────────────────┘")

    stats['eval_targets'] = np.array(eval_targets)

    return stats


# ==================== 选股（MC Dropout 版本） ====================

def mc_dropout_score_all_stocks(model, stock_list, device, feature_normalizer=None):
    """
    使用 MC Dropout 为所有股票打分
    """
    from run import generate_latest_input

    results = []
    skipped = 0

    all_inputs = []
    all_codes = []
    all_dates = []
    all_closes = []
    all_changes = []

    for item in stock_list:
        fname, data, latest_date, times = item
        result = generate_latest_input(data, fname, feature_normalizer)
        if result is None:
            skipped += 1
            continue

        input_seq, stock_code = result
        all_inputs.append(input_seq)
        all_codes.append(stock_code)
        all_dates.append(latest_date)

        latest_close = data[-1, 3]
        if len(data) >= 2 and data[-2, 3] > 0:
            change_pct = (data[-1, 3] - data[-2, 3]) / data[-2, 3] * 100
        else:
            change_pct = 0.0
        all_closes.append(latest_close)
        all_changes.append(change_pct)

    if len(all_inputs) == 0:
        return [], skipped

    # MC Dropout 批量推理
    all_inputs_np = np.array(all_inputs)
    all_scores = mc_dropout_predict(model, all_inputs_np, device)

    for i in range(len(all_codes)):
        results.append((all_codes[i], float(all_scores[i]), all_dates[i], all_closes[i], all_changes[i]))

    return results, skipped


def run_mc_dropout_stock_selection(model, threshold, device, feature_normalizer=None):
    """使用 MC Dropout 执行选股"""
    from run import load_all_stock_data

    print_section("选股推理 (MC Dropout)")
    print(f"│  正在加载全部股票数据...")

    stock_list = load_all_stock_data()
    total_stocks = len(stock_list)
    print(f"│  共加载 {total_stocks} 只股票数据")

    dates = set(s[2] for s in stock_list)
    if len(dates) > 1:
        date_counts = {}
        for s in stock_list:
            d = s[2]
            date_counts[d] = date_counts.get(d, 0) + 1
        main_date = max(date_counts, key=date_counts.get)
        print(f"│  ⚠ 数据日期不完全一致，主要日期: {main_date} ({date_counts[main_date]}只)")
    else:
        main_date = list(dates)[0]

    print(f"│  数据截至日期: {main_date}")
    print(f"│  使用阈值: {threshold:.10f}")
    print(f"│  正在对所有股票打分 (MC Dropout ×{MC_SAMPLES})...")

    results, skipped = mc_dropout_score_all_stocks(model, stock_list, device, feature_normalizer)

    print(f"│  有效股票: {len(results)} 只，跳过（涨停/数据不足）: {skipped} 只")

    results.sort(key=lambda x: x[1], reverse=True)

    threshold_idx = -1
    for i, (code, score, date, close, change) in enumerate(results):
        if score < threshold:
            threshold_idx = i
            break

    if threshold_idx == -1:
        threshold_idx = len(results)

    print(f"│")
    print(f"│  超过阈值: {threshold_idx} 只")
    print(f"│  低于阈值: {len(results) - threshold_idx} 只")
    print_section_end()

    # 打印选股列表
    print()
    print("╔" + "═"*73 + "╗")
    print("║" + " "*22 + "选 股 结 果 列 表 (MC Dropout)" + " "*22 + "    ║")
    print("╠" + "═"*73 + "╣")
    print(f"║  {'排名':^4} {'代码':^7}{'模型分数':^10} {'收盘价':^7}{'涨跌幅':^8}{'日期':^10}{'':^6} ║")
    print("╠" + "═"*73 + "╣")

    show_below = max(10, 30 - threshold_idx)
    total_show = min(len(results), threshold_idx + show_below)

    if threshold_idx > 50:
        display_ranges = []
        display_ranges.append((0, min(20, threshold_idx)))
        if threshold_idx > 30:
            display_ranges.append(('ellipsis', threshold_idx - 20, threshold_idx))
            display_ranges.append((max(20, threshold_idx - 10), threshold_idx))
        display_ranges.append((threshold_idx, min(len(results), threshold_idx + 5)))
    else:
        display_ranges = [(0, total_show)]

    printed_indices = set()

    for item in display_ranges:
        if isinstance(item, tuple) and item[0] == 'ellipsis':
            if not any(i in printed_indices for i in range(item[1], item[2])):
                print(f"║  {'':^4}  {'...':^8}  {'':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}    ║")
            continue

        start_r, end_r = item if isinstance(item, tuple) else item
        for i in range(start_r, end_r):
            if i in printed_indices:
                continue
            printed_indices.add(i)

            code, score, date, close, change = results[i]
            rank = i + 1
            change_str = f"{change:+.2f}%"

            marker = ""
            if i == threshold_idx - 1 and threshold_idx > 0:
                marker = "-阈值-"
            elif i == threshold_idx:
                marker = "  ↓  "
            elif i < threshold_idx:
                marker = "  ★  "

            print(f"║  {rank:>4}   {code:>8}   {score:>10.8f}   {close:>8.2f}  {change_str:>8}   {date:>10} {marker:^6}  ║")

            if i == threshold_idx - 1 and threshold_idx < len(results):
                print("╠" + "─"*73 + "╣")
                print(f"║  {'':^4}  {'':^8}  {'↑ 超过阈值 ↑':^10}  {'│':^8}  {'↓ 低于阈值 ↓':^8}  {'':^10}     ║")
                print("╠" + "─"*73 + "╣")

    remaining = len(results) - len(printed_indices)
    if remaining > 0:
        print(f"║  {'':^4}  {'':^8}  {f'... 还有 {remaining} 只未显示':^10}  {'':^8}  {'':^8}     {'':^6}  ║")

    print("╚" + "═"*73 + "╝")

    print()
    print_section("选股汇总")
    above_scores = [r[1] for r in results[:threshold_idx]]
    if len(above_scores) > 0:
        print(f"│  超过阈值的股票: {threshold_idx} 只")
        print(f"│  最高分: {above_scores[0]:.8f}")
        print(f"│  最低分: {above_scores[-1]:.8f}")
        print(f"│  平均分: {np.mean(above_scores):.8f}")
        print(f"│")
        print(f"│  推荐关注（Top{DataConfig.TOP_K}%阈值以上）:")
        for i in range(min(threshold_idx, 10)):
            code, score, date, close, change = results[i]
            print(f"│    {i+1}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%")
    else:
        print(f"│  没有股票超过阈值 ({threshold:.10f})")
        print(f"│  当前最高分: {results[0][1]:.8f}" if results else "│  无有效股票")

    print_section_end()

    return results


# ==================== 主函数 ====================

def main():
    print_banner()
    print(f"  *** MC Dropout 测试模式 (p={MC_DROPOUT_P}, samples={MC_SAMPLES}) ***")

    # 获取设备
    device = DeviceConfig.get_device()
    if device.type == "cuda":
        print(f"  设备: GPU ({torch.cuda.get_device_name()})")
    else:
        print(f"  设备: CPU")

    # 列出可用模型
    models = list_available_models()
    if not models:
        print("\n  没有可用的模型，请先训练模型。")
        return

    # 选择模型
    model_idx = select_model(models)
    selected_file = models[model_idx]
    model_path = os.path.join(DataConfig.OUTPUT_DIR, selected_file)

    print(f"正在加载模型: {selected_file}")
    model, metadata = load_model(model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    if metadata is not None:
        arch = metadata.get('model_arch') or {}
        tp = metadata.get('train_params') or {}
        es = metadata.get('eval_stats') or {}
        print(f"  ┌── 模型内嵌元数据 ──────────────────────────────")
        print(f"  │  架构: "
              f"d_model={arch.get('d_model','?')}  "
              f"layers={arch.get('num_layers','?')}  "
              f"heads={arch.get('nhead','?')}  "
              f"ctx={arch.get('context_length','?')}")
        print(f"  │  训练: lr={tp.get('learning_rate','?')}  "
              f"bs={tp.get('batch_size','?')}  "
              f"loss={tp.get('loss_type','?')}")
        print(f"  │  评估: Top{es.get('top_k','?')}%收益={es.get('top_return',0)*100:+.2f}%  "
              f"AUC={es.get('auc',0):.4f}  "
              f"Ep={es.get('epoch','?')}")
        print(f"  └───────────────────────────────────────────────")
    else:
        print(f"  (旧格式模型，无内嵌元数据，使用当前 config.py 参数)")

    # 加载数据
    print(f"\n正在加载数据集...")

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        print(f"正在加载归一化器...")
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n  ⚠ 错误: 归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")
        print(f"  请先运行: python data.py")
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    # 运行 MC Dropout 评估
    stats = run_mc_dropout_evaluation(model, test_stock_info, device, feature_normalizer)
    threshold = stats['top_threshold']

    visualize_classification(
        preds=stats['all_preds'],
        targets=stats['eval_targets'],
        title=f"MC Dropout 分类能力可视化 (AUC={stats['auc']:.4f}, p={MC_DROPOUT_P}, ×{MC_SAMPLES})"
    )

    # 询问是否选股
    print()
    while True:
        choice = input("是否进入选股模式？(y/n): ").strip().lower()
        if choice in ('y', 'yes', ''):
            break
        elif choice in ('n', 'no', 'q'):
            print("  已退出。")
            return
        print("  请输入 y 或 n")

    # 确保选股时也使用 MC Dropout
    set_mc_dropout(model, p=MC_DROPOUT_P)

    # 执行选股
    results = run_mc_dropout_stock_selection(model, threshold, device, feature_normalizer)

    # 计算最近10天实战收益率（使用 MC Dropout）
    from run import create_recent_days_dataset
    recent_inputs, recent_returns, recent_day_indices, recent_available_days = \
        create_recent_days_dataset(test_stock_info, feature_normalizer, max_days=15)

    if recent_inputs is not None and len(recent_inputs) > 0:
        model.train()  # MC Dropout 需要 train 模式
        recent_preds = mc_dropout_predict(model, recent_inputs, device)

        all_recent_returns = np.array(recent_returns)
        all_recent_available = np.array(recent_available_days)
        unique_days = np.sort(np.unique(recent_day_indices))

        daily_stats = []
        max_select = DataConfig.MAX_SELECT_PER_DAY

        # 使用全局阈值模式
        above_threshold_mask = recent_preds > threshold
        for day in unique_days:
            day_mask = recent_day_indices == day
            day_above_threshold = above_threshold_mask & day_mask
            day_indices_arr = np.where(day_above_threshold)[0]

            count = len(day_indices_arr)
            if count > 0:
                if max_select > 0 and count > max_select:
                    day_preds = recent_preds[day_indices_arr]
                    top_local = np.argsort(day_preds)[::-1][:max_select]
                    selected_indices = day_indices_arr[top_local]
                    count = max_select
                else:
                    selected_indices = day_indices_arr

                day_return = np.mean(all_recent_returns[selected_indices])
                min_available = int(np.min(all_recent_available[selected_indices]))
                daily_stats.append((count, day_return, min_available))
            else:
                daily_stats.append((0, 0.0, 0))

        if daily_stats:
            print_recent_days_chart(daily_stats, last_n=10)

    # 自定义阈值
    while True:
        print()
        choice = input("  输入自定义阈值重新筛选（直接回车退出）: ").strip()
        if not choice:
            break
        try:
            custom_threshold = float(choice)
            if 0 <= custom_threshold <= 1:
                print(f"\n  使用自定义阈值: {custom_threshold:.10f}")

                threshold_idx = 0
                for i, (code, score, date, close, change) in enumerate(results):
                    if score < custom_threshold:
                        threshold_idx = i
                        break
                else:
                    threshold_idx = len(results)

                print(f"  超过阈值: {threshold_idx} 只")
                if threshold_idx > 0:
                    print(f"\n  推荐关注:")
                    for i in range(min(threshold_idx, 20)):
                        code, score, date, close, change = results[i]
                        marker = "-阈值-" if i == threshold_idx - 1 else "  ★  "
                        print(f"    {i+1:>3}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%  {marker}")
                    if threshold_idx > 20:
                        print(f"    ... 还有 {threshold_idx - 20} 只")
            else:
                print("  ✗ 阈值应在 0 到 1 之间")
        except ValueError:
            print("  ✗ 无效输入")


if __name__ == "__main__":
    main()
