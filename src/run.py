'''
EquiNet 模型推理与选股脚本

功能：
1. 读取 out/ 下可用的模型列表，用户选择模型
2. 加载模型后执行评估（与 train.py 对模型A的评估一致）
3. 打印评估结果 + Top1%阈值（高精度）
4. 用户决定是否进入选股模式
5. 选股：用 data/ 最新数据作为最后一天，模型打分，按分数排序输出
'''

import os, sys, torch, numpy as np, glob, re
from datetime import datetime

# 设置工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (ModelConfig, DataConfig, DeviceConfig, LossConfig)
from model import create_model
from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,
    create_recent_days_dataset,
    generate_sample_from_index
)
from training_utils import evaluate_model, calculate_test_loss, DynamicWeightedBCE


# ==================== 工具函数 ====================

def parse_model_filename(filename):
    """
    从模型文件名中解析元信息
    格式示例: modelA_top1_p1_05pct_thr0_512_auc0_6377_ep15_0227_2110.pth
    """
    info = {'filename': filename, 'prefix': '', 'return_pct': '', 'threshold': '', 'auc': '', 'epoch': '', 'time': ''}
    
    # 提取模型前缀 (modelA, modelB, modelB_dft)
    prefix_match = re.match(r'^(modelA|modelB_dft|modelB)', filename)
    if prefix_match:
        info['prefix'] = prefix_match.group(1)
    
    # 提取收益率
    ret_match = re.search(r'_([pn]\d+_\d+)pct_', filename)
    if ret_match:
        ret_str = ret_match.group(1)
        ret_str = ret_str.replace('p', '+').replace('n', '-').replace('_', '.')
        info['return_pct'] = ret_str + '%'
    
    # 提取阈值
    thr_match = re.search(r'_thr(\d+_\d+)_', filename)
    if thr_match:
        thr_str = thr_match.group(1).replace('_', '.', 1)
        info['threshold'] = thr_str
    
    # 提取AUC
    auc_match = re.search(r'_auc(\d+_\d+)_', filename)
    if auc_match:
        auc_str = auc_match.group(1).replace('_', '.', 1)
        info['auc'] = auc_str
    
    # 提取epoch
    ep_match = re.search(r'_ep(\d+)_', filename)
    if ep_match:
        info['epoch'] = ep_match.group(1)
    
    # 提取时间戳
    time_match = re.search(r'_(\d{4}_\d{4})\.pth$', filename)
    if time_match:
        info['time'] = time_match.group(1)
    
    return info


def list_available_models(output_dir=DataConfig.OUTPUT_DIR):
    """列出 out/ 目录下所有可用的 .pth 模型文件"""
    if not os.path.exists(output_dir):
        print(f"  ✗ 输出目录 {output_dir} 不存在")
        return []
    
    pth_files = sorted(glob.glob(os.path.join(output_dir, '*.pth')))
    if not pth_files:
        print(f"  ✗ {output_dir} 下没有找到 .pth 模型文件")
        return []
    
    return [os.path.basename(f) for f in pth_files]


def load_model(model_path, device):
    """加载模型"""
    model = create_model()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
      
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def generate_latest_input(stock_data, file_name):
    """
    为单只股票生成最新一天的模型输入（不需要未来数据）
    
    逻辑与 generate_sample_from_index 一致，但只生成输入序列，不生成标签。
    取数据最后 CONTEXT_LENGTH 天作为输入窗口。
    
    返回: (input_seq, stock_code) 或 None
    """
    context_length = DataConfig.CONTEXT_LENGTH
    data_length = len(stock_data)
    
    # 需要 context_length + 1 天数据（第一天需要前一天做参照）
    if data_length < context_length + 1:
        return None
    
    # 取最后 context_length 天作为输入窗口
    start_idx = data_length - context_length
    input_seq_raw = stock_data[start_idx:]  # [context_length, 6]
    prev_day_data = stock_data[start_idx - 1]
    
    prev_close = prev_day_data[3]
    prev_volume = prev_day_data[4]
    if prev_close == 0 or prev_volume == 0 or np.any(prev_day_data[:4] == 0):
        return None
    
    closes = input_seq_raw[:, 3]
    volumes = input_seq_raw[:, 4]
    if np.any(closes == 0) or np.any(volumes == 0):
        return None
    
    # 涨停过滤（窗口内任何一天涨跌超过11%则跳过）
    limit_threshold = 0.11
    all_data = stock_data[start_idx - 1:]  # 包含prev_day
    for day_idx in range(1, len(all_data)):
        today_close = all_data[day_idx, 3]
        yesterday_close = all_data[day_idx - 1, 3]
        if yesterday_close > 0:
            daily_return = (today_close - yesterday_close) / yesterday_close
            if abs(daily_return) > limit_threshold:
                return None
    
    # 最后一天涨停检查（最后一天涨停≥9.5%则跳过，因为次日大概率无法买入）
    prev_day_idx = data_length - 2
    last_day_idx = data_length - 1
    prev_day_close = stock_data[prev_day_idx, 3]
    last_day_close = stock_data[last_day_idx, 3]
    if prev_day_close > 0:
        last_day_return = (last_day_close - prev_day_close) / prev_day_close
        if last_day_return >= 0.095:
            return None
    
    # 构建输入特征（与 generate_sample_from_index 完全一致）
    input_seq = np.empty((context_length, 6), dtype=np.float32)
    
    # OHLC 涨跌幅
    input_seq[0, :4] = (input_seq_raw[0, :4] - prev_close) / prev_close
    if context_length > 1:
        input_seq[1:, :4] = (input_seq_raw[1:, :4] - closes[:-1, np.newaxis]) / closes[:-1, np.newaxis]
    
    # 量比
    input_seq[0, 4] = (volumes[0] - prev_volume) / prev_volume
    if context_length > 1:
        input_seq[1:, 4] = (volumes[1:] - volumes[:-1]) / volumes[:-1]
    
    # 换手率
    input_seq[:, 5] = input_seq_raw[:, 5] / 100.0
    
    # 裁剪
    np.clip(input_seq[:, :4], -0.1, 0.1, out=input_seq[:, :4])
    np.clip(input_seq[:, 4], -5.0, 5.0, out=input_seq[:, 4])
    input_seq[:, 4] = input_seq[:, 4] / 10.0 + 0.5
    np.clip(input_seq[:, 4:6], 0.0, 1.0, out=input_seq[:, 4:6])
    
    if np.any(~np.isfinite(input_seq)):
        return None
    
    # 提取股票代码（去掉.csv后缀）
    stock_code = file_name.replace('.csv', '')
    
    return input_seq, stock_code


def load_all_stock_data(data_dir=DataConfig.DATA_DIR):
    """
    加载所有股票原始数据（用于选股推理）
    返回: [(file_name, data_array, latest_date), ...]
    """
    import pandas as pd
    
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    stock_list = []
    
    for fname in all_files:
        fpath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(fpath)
            # 原始数据按时间倒序，翻转为正序（早→晚）
            df = df.iloc[::-1].reset_index(drop=True)
            data = df[['start', 'max', 'min', 'end', 'volume', 'exchange']].values
            latest_date = str(df['time'].iloc[-1])  # 最新交易日期
            stock_list.append((fname, data, latest_date))
        except Exception as e:
            pass  # 静默跳过异常文件
    
    return stock_list


def score_all_stocks(model, stock_list, device):
    """
    为所有股票打分
    
    返回: [(stock_code, score, latest_date, latest_close, latest_change_pct), ...]
    """
    results = []
    skipped = 0
    
    all_inputs = []
    all_codes = []
    all_dates = []
    all_closes = []
    all_changes = []
    
    for fname, data, latest_date in stock_list:
        result = generate_latest_input(data, fname)
        if result is None:
            skipped += 1
            continue
        
        input_seq, stock_code = result
        all_inputs.append(input_seq)
        all_codes.append(stock_code)
        all_dates.append(latest_date)
        
        # 最新收盘价和涨跌幅
        latest_close = data[-1, 3]
        if len(data) >= 2 and data[-2, 3] > 0:
            change_pct = (data[-1, 3] - data[-2, 3]) / data[-2, 3] * 100
        else:
            change_pct = 0.0
        all_closes.append(latest_close)
        all_changes.append(change_pct)
    
    if len(all_inputs) == 0:
        return [], skipped
    
    # 批量推理
    batch_size = DataConfig.EVAL_BATCH_SIZE
    all_inputs_np = np.array(all_inputs)
    all_scores = []
    
    num_batches = (len(all_inputs_np) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(all_inputs_np))
            batch = torch.tensor(all_inputs_np[start:end], dtype=torch.float32, device=device)
            preds = torch.sigmoid(model(batch)).cpu().numpy().flatten()
            all_scores.extend(preds)
            del batch
    
    # 组合结果
    for i in range(len(all_codes)):
        results.append((all_codes[i], float(all_scores[i]), all_dates[i], all_closes[i], all_changes[i]))
    
    return results, skipped


# ==================== 界面函数 ====================

def print_banner():
    """打印欢迎界面"""
    print()
    print("╔" + "═"*62 + "╗")
    print("║" + " "*14 + "EquiNet v2 · 模型推理与选股" + " "*14 + "    ║")
    print("╚" + "═"*62 + "╝")
    print()


def print_section(title):
    """打印分节标题"""
    print()
    print(f"┌─── {title} " + "─" * max(1, 55 - len(title)*2) + "┐")


def print_section_end():
    """打印分节结束"""
    print(f"└" + "─"*62 + "┘")


def select_model(models):
    """模型选择界面"""
    print_section("可用模型列表")
    print(f"│")
    
    for i, fname in enumerate(models):
        info = parse_model_filename(fname)
        
        # 格式化显示
        prefix_display = {
            'modelA': '模型A(原始)',
            'modelB': '模型B(克隆)',
            'modelB_dft': '模型B(DFT)',
        }.get(info['prefix'], info['prefix'])
        
        detail_parts = []
        if info['return_pct']:
            detail_parts.append(f"收益{info['return_pct']}")
        if info['auc']:
            detail_parts.append(f"AUC={info['auc']}")
        if info['threshold']:
            detail_parts.append(f"阈值={info['threshold']}")
        if info['epoch']:
            detail_parts.append(f"Ep{info['epoch']}")
        
        detail_str = ', '.join(detail_parts)
        
        print(f"│  [{i+1}] {prefix_display}")
        print(f"│      {detail_str}")
        print(f"│      文件: {fname}")
        if i < len(models) - 1:
            print(f"│")
    
    print(f"│")
    print_section_end()
    
    while True:
        try:
            choice = input(f"\n  请选择模型 [1-{len(models)}]（输入 q 退出）: ").strip()
            if choice.lower() == 'q':
                print("  已退出。")
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return idx
            print(f"  ✗ 请输入 1 到 {len(models)} 之间的数字")
        except ValueError:
            print(f"  ✗ 无效输入，请输入数字")


def run_evaluation(model, test_stock_info, device):
    """
    执行模型评估（与 train.py 中对模型A的评估完全一致）
    返回评估统计字典
    """
    print_section("模型评估")
    print(f"│  正在创建评估数据集...")
    
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns = \
        create_fixed_evaluation_dataset(test_stock_info)
    
    print(f"│  评估样本数: {len(eval_inputs)}")
    print(f"│  正在评估模型...")
    
    stats = evaluate_model(
        model, eval_inputs, eval_targets, eval_cumulative_returns,
        device, model_name="选中模型",
        eval_day_indices=eval_day_indices,
        eval_daily_returns=eval_daily_returns
    )
    
    # 创建评估损失函数（与 train.py 一致）
    if LossConfig.use_dynamic_bce():
        eval_criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        
        # 测试集权重
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
    
    # 计算测试集损失
    test_loss = calculate_test_loss(model, eval_inputs, eval_targets, eval_criterion, device)
    
    # 打印评估结果（与 train.py 格式一致）
    print(f"│")
    print(f"│  ┌── 评估结果 ──────────────────────────────────┐")
    print(f"│  │  测试损失:          {test_loss:.4f}")
    print(f"│  │  AUC:              {stats['auc']:.4f}")
    print(f"│  │  预测均值:          {stats['pred_mean']:.3f}")
    print(f"│  │  预测标准差:        {stats['pred_std']:.4f}")
    print(f"│  │  高置信(>0.7):      {stats['high_conf_count']} 个")
    print(f"│  │  低置信(<0.2):      {stats['low_conf_count']} 个")
    print(f"│  │  Top{DataConfig.TOP_PERCENT}%样本数:        {stats['top_count']} 个")
    print(f"│  │  Top{DataConfig.TOP_PERCENT}%平均收益:      {stats['top_return']*100:+.2f}%")
    print(f"│  │")
    print(f"│  │  ★ Top{DataConfig.TOP_PERCENT}%阈值:        {stats['top_threshold']:.10f}")
    print(f"│  │")
    
    # 实战收益率
    if stats['realistic_stats'] is not None:
        rs = stats['realistic_stats']
        daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in rs['daily_stats']])
        mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else \
                   f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY}" if DataConfig.MAX_SELECT_PER_DAY > 0 else \
                   "全局阈值,不限数量"
        print(f"│  │  【实战收益率({mode_str})】")
        print(f"│  │  每日统计: {{{daily_stats_str}}}")
        print(f"│  │  平均实战收益率: {rs['avg_realistic_return']*100:.1f}%")
        print(f"│  │")
    
    if stats.get('smart_exit_stats') is not None:
        se = stats['smart_exit_stats']
        print(f"│  │  【智能止损】")
        print(f"│  │  收益率: {se['avg_realistic_return']*100:.1f}%")
        print(f"│  │  Day1止损: {se['stop_loss_day1_count']}次, 累计止损: {se['stop_loss_cum_count']}次, 止盈: {se['take_profit_count']}次")
        print(f"│  │")
    
    print(f"│  └───────────────────────────────────────────────┘")
    print_section_end()
    
    return stats


def run_stock_selection(model, threshold, device):
    """
    执行选股
    """
    print_section("选股推理")
    print(f"│  正在加载全部股票数据...")
    
    stock_list = load_all_stock_data()
    total_stocks = len(stock_list)
    print(f"│  共加载 {total_stocks} 只股票数据")
    
    # 检查数据日期一致性
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
    print(f"│  正在对所有股票打分...")
    
    results, skipped = score_all_stocks(model, stock_list, device)
    
    print(f"│  有效股票: {len(results)} 只，跳过（涨停/数据不足）: {skipped} 只")
    
    # 按分数降序排列
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 找到阈值分界位置
    threshold_idx = -1
    for i, (code, score, date, close, change) in enumerate(results):
        if score < threshold:
            threshold_idx = i
            break
    
    if threshold_idx == -1:
        threshold_idx = len(results)  # 全部都在阈值之上
    
    # 打印结果
    print(f"│")
    print(f"│  超过阈值: {threshold_idx} 只")
    print(f"│  低于阈值: {len(results) - threshold_idx} 只")
    print_section_end()
    
    # 打印选股列表
    print()
    print("╔" + "═"*78 + "╗")
    print("║" + " "*26 + "选 股 结 果 列 表" + " "*26 + "    ║")
    print("╠" + "═"*78 + "╣")
    print(f"║  {'排名':^4}  {'代码':^8}  {'模型分数':^10}  {'收盘价':^8}  {'涨跌幅':^8}  {'日期':^10}  {'':^6}  ║")
    print("╠" + "═"*78 + "╣")
    
    # 决定显示多少条
    # 阈值线上方全部显示 + 阈值线下方显示到前30名或阈值线后10条（取较大者）
    show_below = max(10, 30 - threshold_idx)
    total_show = min(len(results), threshold_idx + show_below)
    
    # 但如果阈值线上方已经超过50条，只显示上方前50条 + 下方5条
    if threshold_idx > 50:
        # 显示前20条 + 省略 + 阈值线附近10条 + 阈值线下方5条
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
                print(f"║  {'':^4}  {'...':^8}  {'':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}  ║")
            continue
        
        start_r, end_r = item if isinstance(item, tuple) else item
        for i in range(start_r, end_r):
            if i in printed_indices:
                continue
            printed_indices.add(i)
            
            code, score, date, close, change = results[i]
            rank = i + 1
            
            # 涨跌幅颜色标记
            change_str = f"{change:+.2f}%"
            
            # 阈值标记
            marker = ""
            if i == threshold_idx - 1 and threshold_idx > 0:
                marker = "┈阈值┈"
            elif i == threshold_idx:
                marker = "  ↓  "
            elif i < threshold_idx:
                marker = "  ★  "
            
            print(f"║  {rank:>4}   {code:>8}   {score:>10.8f}   {close:>8.2f}  {change_str:>8}   {date:>10}  {marker:^6}  ║")
            
            # 在阈值分界处画线
            if i == threshold_idx - 1 and threshold_idx < len(results):
                print("╠" + "─"*78 + "╣")
                print(f"║  {'':^4}  {'':^8}  {'↑ 超过阈值 ↑':^10}  {'│':^8}  {'↓ 低于阈值 ↓':^8}  {'':^10}  {'':^6}  ║")
                print("╠" + "─"*78 + "╣")
    
    # 如果还有更多未显示的
    remaining = len(results) - len(printed_indices)
    if remaining > 0:
        print(f"║  {'':^4}  {'':^8}  {f'... 还有 {remaining} 只未显示':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}  ║")
    
    print("╚" + "═"*78 + "╝")
    
    # 打印汇总统计
    print()
    print_section("选股汇总")
    above_scores = [r[1] for r in results[:threshold_idx]]
    if len(above_scores) > 0:
        print(f"│  超过阈值的股票: {threshold_idx} 只")
        print(f"│  最高分: {above_scores[0]:.8f}")
        print(f"│  最低分: {above_scores[-1]:.8f}")
        print(f"│  平均分: {np.mean(above_scores):.8f}")
        print(f"│")
        print(f"│  推荐关注（Top{DataConfig.TOP_PERCENT}%阈值以上）:")
        for i in range(min(threshold_idx, 10)):
            code, score, date, close, change = results[i]
            print(f"│    {i+1}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%")
    else:
        print(f"│  没有股票超过阈值 ({threshold:.10f})")
        print(f"│  当前最高分: {results[0][1]:.8f}" if results else "│  无有效股票")
    
    print_section_end()
    
    return results


def print_recent_days_chart(daily_stats, last_n=10):
    """
    打印最近N天的实战收益率表格
    
    参数:
        daily_stats: 每日统计列表 [(count, return, available_days), ...]
        last_n: 显示最近多少天
    """
    if not daily_stats or len(daily_stats) == 0:
        return
    
    total_days = len(daily_stats)
    start_idx = max(0, total_days - last_n)
    recent_stats = daily_stats[start_idx:]
    
    print()
    print("╔" + "═"*52 + "╗")
    title = f"最近{last_n}天实战收益率"
    padding = (52 - 2 - len(title)) // 2
    print("║" + " "*padding + title + " "*(52 - 2 - padding - len(title)) + "║")
    print("╠" + "═"*52 + "╣")
    print("║  Day  │ Count │ Return   │ 相对日期   │ 数据      ║")
    print("╠" + "─"*52 + "╣")
    
    for i, (count, ret, available_days) in enumerate(recent_stats):
        day_num = start_idx + i + 1
        
        days_from_end = total_days - day_num + 1
        if days_from_end == 1:
            relative_date = "昨天"
        elif days_from_end == 2:
            relative_date = "前天"
        elif days_from_end == 3:
            relative_date = "大前天"
        else:
            relative_date = f"T-{days_from_end}"
        
        if available_days == 3:
            data_status = "完整"
        elif available_days == 2:
            data_status = "临时(2天)"
        elif available_days == 1:
            data_status = "临时(1天)"
        else:
            data_status = "-"
        
        ret_str = f"{ret*100:+.1f}%"
        
        print(f"║  {day_num:>3}  │  {count:>3}  │ {ret_str:>8} │ {relative_date:<8} │ {data_status:<9} ║")
    
    print("╚" + "═"*52 + "╝")


def calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=4):
    """
    计算最近几天的实战收益率（包含临时数据）
    
    返回: daily_stats [(count, return, available_days), ...]
    """
    recent_inputs, recent_returns, recent_day_indices, recent_available_days = \
        create_recent_days_dataset(test_stock_info)
    
    if recent_inputs is None or len(recent_inputs) == 0:
        return []
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        batch_size = DataConfig.EVAL_BATCH_SIZE
        for i in range(0, len(recent_inputs), batch_size):
            batch = torch.tensor(recent_inputs[i:i+batch_size], dtype=torch.float32, device=device)
            preds = torch.sigmoid(model(batch)).cpu().numpy().flatten()
            all_preds.extend(preds)
    
    all_preds = np.array(all_preds)
    
    unique_days = np.unique(recent_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    
    for day in unique_days:
        day_mask = recent_day_indices == day
        day_indices = np.where(day_mask)[0]
        
        if len(day_indices) == 0:
            daily_stats.append((0, 0.0, 0))
            continue
        
        day_preds = all_preds[day_indices]
        day_returns = recent_returns[day_indices]
        day_available = recent_available_days[day_indices]
        
        sorted_local_indices = np.argsort(day_preds)[::-1]
        select_count = min(top_n_per_day, len(day_indices))
        top_local_indices = sorted_local_indices[:select_count]
        
        day_return = np.mean(day_returns[top_local_indices])
        min_available = int(np.min(day_available[top_local_indices]))
        
        daily_stats.append((select_count, day_return, min_available))
    
    return daily_stats


# ==================== 主函数 ====================

def main():
    print_banner()
    
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
    
    print(f"\n  正在加载模型: {selected_file}")
    model = load_model(model_path, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  ✓ 模型加载成功")
    
    # 加载数据并评估
    print(f"\n  正在加载数据集...")
    train_stock_info, test_stock_info = load_and_preprocess_data()
    
    # 运行评估
    stats = run_evaluation(model, test_stock_info, device)
    threshold = stats['top_threshold']
    
    # 询问是否选股
    print()
    while True:
        choice = input("  是否进入选股模式？(y/n): ").strip().lower()
        if choice in ('y', 'yes', ''):
            break
        elif choice in ('n', 'no', 'q'):
            print("  已退出。")
            return
        print("  请输入 y 或 n")
    
    # 执行选股
    results = run_stock_selection(model, threshold, device)
    
    # 计算并打印最近10天实战收益率表格（包含临时数据）
    recent_stats = calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=DataConfig.TOP_N_PER_DAY)
    if recent_stats:
        print_recent_days_chart(recent_stats, last_n=10)
    
    # 询问是否使用自定义阈值重新选股
    while True:
        print()
        choice = input("  输入自定义阈值重新筛选（直接回车退出）: ").strip()
        if not choice:
            break
        try:
            custom_threshold = float(choice)
            if 0 <= custom_threshold <= 1:
                print(f"\n  使用自定义阈值: {custom_threshold:.10f}")
                
                # 重新标记
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
                        marker = "┈阈值┈" if i == threshold_idx - 1 else "  ★  "
                        print(f"    {i+1:>3}. {code}  分数={score:.8f}  价格={close:.2f}  涨跌={change:+.2f}%  {marker}")
                    if threshold_idx > 20:
                        print(f"    ... 还有 {threshold_idx - 20} 只")
            else:
                print("  ✗ 阈值应在 0 到 1 之间")
        except ValueError:
            print("  ✗ 无效输入")
    
    print("\n  感谢使用 EquiNet v2！")


if __name__ == "__main__":
    main()
