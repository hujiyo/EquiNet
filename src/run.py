'''
EquiNet 模型推理与选股脚本

核心流程：
1. 模型选择：读取 out/ 下可用的模型列表，用户选择模型
2. 模型评估：加载模型后执行评估（与 train.py 对模型A的评估完全一致）
3. 选股模式：用 data/ 最新数据作为最后一天，模型打分，按分数排序输出

数据一致性保证：
- 评估数据集：只包含完整样本（available_days == 3），与 train.py 完全一致
- 最近几天展示：包含临时样本（available_days < 3），仅用于展示，不参与阈值计算
'''

import os, sys, torch, numpy as np, glob, re, json, argparse
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

from config import (ModelConfig, DataConfig, DeviceConfig, LossConfig)
from model import create_model
from data import (load_and_preprocess_data, create_fixed_evaluation_dataset,FeatureNormalizer,
                  create_recent_days_dataset, normalize_and_validate_context_window)
from training_utils import evaluate_model, create_eval_criterion, _get_amp_context


# 每日统计 JSON 与可视化输出目录（项目根 /out_run，已被 .gitignore 的 out*/ 覆盖）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_RUN_DIR = os.path.join(_PROJECT_ROOT, 'out_run')


# ==================== MC Dropout 推理（可选，--mc-dropout 启用） ====================

MC_DROPOUT_P = 0.1       # MC Dropout 概率（默认值，可用 --mc-p 覆盖）
MC_SAMPLES = 100          # 每个样本的前向传播次数（默认值，可用 --mc-samples 覆盖）


def set_mc_dropout(model, p=MC_DROPOUT_P):
    """
    将模型中所有 Dropout 层的概率设为 p，用于 MC Dropout 推理。
    返回被修改的 Dropout 层数量。
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
    MC Dropout 推理：对每个样本执行 num_samples 次前向传播（保持 Dropout 激活），
    返回 sigmoid 后的平均预测。

    Args:
        model: 已加载的模型（会被设为 train 模式以使 Dropout 生效）
        eval_inputs: numpy 数组，形状 (N, seq_len, input_dim)
        device: torch.device
        num_samples: MC 采样次数
        batch_size: 每次前向传播的批大小

    Returns:
        numpy 数组，形状 (N,)
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

    return (accum / num_samples).astype(np.float32)


# ==================== 工具函数 ====================

def parse_model_filename(filename):
    """
    从模型文件名中解析元信息
    格式示例: modelA_top1_p1_05pct_thr0_512_auc0_6377_ep15_0227_2110.pth
    """
    info = {'filename': filename, 'prefix': '', 'return_pct': '', 'threshold': '', 'auc': '', 'epoch': '', 'time': ''}
    
    # 提取模型前缀 (model_loss, model_realistic 等)
    prefix_match = re.match(r'^(model_[a-z]+)', filename)
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
    """
    加载模型，支持两种 .pth 格式：
    - 新格式（checkpoint 字典）：含 model_arch / train_params / eval_stats / state_dict
    - 旧格式（裸 state_dict）：直接是权重字典，按 config.py 当前参数创建模型

    返回: (model, metadata)
        metadata: 新格式时为包含元数据的字典，旧格式时为 None
    """
    raw = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and 'state_dict' in raw:
        # 新格式：从内嵌的 model_arch 重建与训练时完全一致的模型
        model_arch   = raw.get('model_arch')
        train_params = raw.get('train_params')
        eval_stats   = raw.get('eval_stats')
        state_dict   = raw['state_dict']
        metadata     = {'model_arch': model_arch, 'train_params': train_params, 'eval_stats': eval_stats}
        model = create_model(model_arch=model_arch)
    else:
        # 旧格式：裸 state_dict，使用当前 config.py 参数
        state_dict = raw
        metadata   = None
        model = create_model()

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, metadata


def generate_latest_input(stock_data, file_name, feature_normalizer=None):
    """
    为单只股票生成最新一天的模型输入（不需要未来数据）用于预测
    
    使用 data.py 的统一归一化函数，确保与训练时的数据处理逻辑完全一致。
    取数据最后 CONTEXT_LENGTH 天作为输入窗口。
    
    Args:
        stock_data: 股票原始数据
        file_name: 文件名
        feature_normalizer: 可选的特征归一化器实例
    返回: (input_seq, stock_code) 或 None
    """
    context_length = DataConfig.CONTEXT_LENGTH
    data_length = len(stock_data)
    
    if data_length < context_length + 1:
        return None
    
    start_idx = data_length - context_length
    
    input_seq = normalize_and_validate_context_window(
        stock_data,
        start_idx,
        context_length,
        check_limit_up=True,
        required_length=context_length,
        feature_normalizer=feature_normalizer
    )
    
    if input_seq is None:
        return None
    
    # stock_code 现在直接就是股票代码（不再需要从文件名提取）
    stock_code = file_name
    
    return input_seq, stock_code


def load_all_stock_data(db_path=DataConfig.DB_PATH):
    """
    加载所有股票原始数据（用于选股推理）
    从 SQLite 数据库读取训练池(selected)中的股票。

    返回: [(stock_code, data_array, latest_date, times_array), ...]
    """
    import pandas as pd
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        query = """SELECT sd.stock_code, sd.date, sd.open, sd.high, sd.low, sd.close,
                          sd.vwap, sd.volume, sd.exchange, sd.m5, sd.m10, sd.m20,
                          sd.dif, sd.dea, sd.macd_hist, sd.macd_hist_diff, sd.bb_upper, sd.bb_lower
                   FROM stock_daily sd
                   JOIN stock_pool sp ON sd.stock_code = sp.stock_code
                   WHERE sp.pool_type='selected' AND sp.is_active=1
                   ORDER BY sd.stock_code, sd.date ASC"""
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    stock_list = []
    if len(df) == 0:
        return stock_list
    cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'exchange', 'm5', 'm10', 'm20', 'dif', 'dea', 'macd_hist', 'macd_hist_diff', 'bb_upper', 'bb_lower']
    for stock_code, group in df.groupby('stock_code', sort=False):
        data = group[cols].values
        latest_date = str(group['date'].iloc[-1])
        times = group['date'].values
        stock_list.append((stock_code, data, latest_date, times))
    return stock_list


def score_all_stocks(model, stock_list, device, feature_normalizer=None,
                     mc_dropout=False, mc_samples=MC_SAMPLES):
    """
    为所有股票打分

    Args:
        model: 模型实例
        stock_list: 股票数据列表
        device: 设备
        feature_normalizer: 可选的特征归一化器实例
        mc_dropout: 启用 MC Dropout 推理（多次前向取平均）
        mc_samples: MC Dropout 前向传播次数

    返回: [(stock_code, score, latest_date, latest_close, latest_change_pct), ...]
    """
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
    
    # 批量推理（MC Dropout 模式下多次前向取平均）
    batch_size = DataConfig.EVAL_BATCH_SIZE
    all_inputs_np = np.array(all_inputs)
    if mc_dropout:
        all_scores = mc_dropout_predict(model, all_inputs_np, device, num_samples=mc_samples)
    else:
        all_scores = []
        amp_ctx = _get_amp_context(device)

        num_batches = (len(all_inputs_np) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, len(all_inputs_np))
                batch = torch.tensor(all_inputs_np[start:end], dtype=torch.float32, device=device)
                with amp_ctx:
                    logits = model(batch)
                preds = torch.sigmoid(logits.float()).cpu().numpy().flatten()
                all_scores.extend(preds)
                del batch, logits
    
    # 组合结果
    for i in range(len(all_codes)):
        results.append((all_codes[i], float(all_scores[i]), all_dates[i], all_closes[i], all_changes[i]))
    
    return results, skipped


# ==================== 每日统计：日期映射 / 导出 ====================

def build_day_to_date_map(test_stock_info, start_date=None):
    """
    构建 day_index -> 交易日历日期(int YYYYMMDD) 的映射。

    关键：split_point 必须与 create_fixed_evaluation_dataset 完全一致，否则日期会错位。
        - start_date 给定时(--begin 口径)：split_point = max(1, begin_idx - CONTEXT_LENGTH + 1)，
          使首个预测日落在 begin 当天
        - 否则(默认测试集口径)：split_point = test_split_point

    推导：
        eval_day_index = start_idx + CONTEXT_LENGTH - split_point
        上下文最后一天（预测日）的绝对索引 = start_idx + CONTEXT_LENGTH - 1
            = day_index + split_point - 1
        该日日期 = stock_info['times'][day_index + split_point - 1]

    所有股票共享同一交易日历，理论上同一 day_index 对应同一日期；
    但停牌/数据缺失会让个别股票错位，故对每个 day_index 采用"多数投票"
    取出现次数最多的日期，保证鲁棒。
    """
    context_length = DataConfig.CONTEXT_LENGTH
    votes = defaultdict(lambda: defaultdict(int))  # day_index -> {date: 票数}

    for stock_info in test_stock_info:
        times = stock_info.get('times')
        if times is None:
            continue
        times = np.asarray(times)
        if len(times) == 0:
            continue

        if start_date is not None:
            # 与 create_fixed_evaluation_dataset 完全一致：首个预测日落在 begin 当天
            mask = np.where(times >= start_date)[0]
            if len(mask) == 0:
                continue
            begin_idx = int(mask[0])
            sp = max(1, begin_idx - context_length + 1)
        else:
            sp = int(stock_info.get('test_split_point', 0))

        first_abs = sp + context_length - 1          # 最早的预测日绝对索引
        if first_abs < 0 or first_abs >= len(times):
            continue
        abs_indices = np.arange(first_abs, len(times))
        day_indices = (abs_indices - sp + 1).astype(np.int64)
        dates = times[abs_indices].astype(np.int64)
        for di, dt in zip(day_indices.tolist(), dates.tolist()):
            votes[di][dt] += 1

    # 多数投票；票数并列时取日期较小者（更早的交易日），保证结果确定、不依赖股票遍历顺序
    return {di: max(counter.items(), key=lambda x: (x[1], -x[0]))[0]
            for di, counter in votes.items()}


def _format_date(yyyymmdd):
    """20260602 -> '2026-06-02'；无法解析时原样返回字符串。"""
    try:
        s = str(int(yyyymmdd))
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    except (ValueError, TypeError):
        return str(yyyymmdd)


def _json_default(o):
    """json.dump 的 default 钩子：把 numpy 标量/数组转成原生 Python 类型，
    避免 'Object of type float32 is not JSON serializable'（meta 里的
    avg_realistic_return / cumulative_return 等可能是 np.float32/float64）。"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def build_daily_kept(realistic_stats, day_to_date):
    """
    将 realistic_stats 的每日统计对齐到日历日期，产出展示/导出用列表。

    - 区间已在 create_fixed_evaluation_dataset(start_date=...) 处确定（--begin 时为
      [begin, 最新]，默认为测试集区间），此处不再做日期过滤；
    - 丢弃 count==0（无股票过阈值）的日子——这些日子没有真实交易，
      不应计入"赚钱效应"收益序列；
    - 返回 [(date_int, count, return), ...]，按 day_index 升序（即日期升序）。
    """
    daily_stats = realistic_stats['daily_stats']
    day_indices = realistic_stats.get('day_indices')
    if day_indices is None:
        day_indices = list(range(len(daily_stats)))

    kept = []
    for (count, ret), di in zip(daily_stats, day_indices):
        if count <= 0:
            continue
        date = day_to_date.get(int(di))
        if date is None:
            continue
        kept.append((int(date), int(count), float(ret)))
    return kept


def write_daily_json(kept, meta, out_dir=OUT_RUN_DIR):
    """
    将过滤后的每日统计写入 out_run/ 下的 JSON。

    JSON 结构（自设计）：
        {
          "meta": { 模型名、起止日期、阈值、平均/累计收益、生成时间, ... },
          "daily": [ {"date": "2026-01-06", "yyyymmdd": 20260106,
                      "return_pct": 0.9, "count": 4}, ... ]
        }
    每条 daily 记录即"每日统计"中的第二位数（当日收益率%），并附带年月日。
    """
    os.makedirs(out_dir, exist_ok=True)
    daily = [
        {
            "date": _format_date(d),
            "yyyymmdd": d,
            "return_pct": round(float(r) * 100, 4),
            "count": int(c),
        }
        for (d, c, r) in kept
    ]
    payload = {"meta": meta, "daily": daily}

    fname = f"daily_stats_{meta.get('generated_tag', 'run')}.json"
    path = os.path.join(out_dir, fname)
    with open(path, 'w', encoding='utf-8') as f:
        # default=_json_default 兜底 meta 中可能残留的 numpy 标量
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return path


# ==================== 界面函数 ====================

def print_banner():
    """打印欢迎界面"""
    print("╔" + "═"*56 + "╗")
    print("║" + " "*14 + "EquiNet · 模型推理与选股" + " "*14 + "    ║")
    print("╚" + "═"*56 + "╝")


def print_section(title):
    """打印分节标题"""
    print()
    print(f"┌─── {title} " + "─" * max(1, 55 - len(title)*2))


def print_section_end():
    """打印分节结束"""
    print(f"└" + "─"*62)


def visualize_classification(preds, targets, title="模型分类能力可视化"):
    """
    可视化模型的分类能力

    Args:
        preds: 模型预测值数组 (0-1)
        targets: 真实标签数组 (0或1)
        title: 图表标题
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    preds = np.array(preds)
    targets = np.array(targets)

    np.random.seed(42)
    y_coords = np.random.uniform(0, 1, size=len(preds))

    mask_0 = targets < 0.5
    mask_1 = targets >= 0.5

    plt.figure(figsize=(10, 8))

    plt.scatter(preds[mask_0], y_coords[mask_0],
                c='black', alpha=0.6, s=20, label=f'标签=0 ({np.sum(mask_0)}个)', marker='o')

    plt.scatter(preds[mask_1], y_coords[mask_1],
                c='blue', alpha=0.6, s=20, label=f'标签=1 ({np.sum(mask_1)}个)', marker='o')

    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='决策边界(0.5)')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('模型预测值', fontsize=12)
    plt.ylabel('随机Y坐标', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    correct_0 = np.sum((preds[mask_0] < 0.5))
    correct_1 = np.sum((preds[mask_1] >= 0.5))
    total_correct = correct_0 + correct_1
    accuracy = total_correct / len(preds) * 100

    info_text = f'准确率: {accuracy:.1f}%\n标签0预测<0.5: {correct_0}/{np.sum(mask_0)}\n标签1预测>=0.5: {correct_1}/{np.sum(mask_1)}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    os.makedirs(DataConfig.OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(DataConfig.OUTPUT_DIR, 'classification_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'分类可视化已保存: {save_path}')
    plt.close()


def select_model(models):
    """模型选择界面"""
    print_section("可用模型列表")
    print(f"│")
    
    for i, fname in enumerate(models):
        info = parse_model_filename(fname)
        
        # 格式化显示
        prefix_display = {
            'model_loss': '模型(按loss)',
            'model_realistic': '模型(按实战收益率)',
            'modelA': '模型(按loss)',
            'modelA_realistic': '模型(按实战收益率)',
            'modelA_loss': '模型(按loss)',
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
            choice = input(f"\n请选择模型 [1-{len(models)}]（输入 q 退出）: ").strip()
            if choice.lower() == 'q':
                print("  已退出。")
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return idx
            print(f"  ✗ 请输入 1 到 {len(models)} 之间的数字")
        except ValueError:
            print(f"  ✗ 无效输入，请输入数字")


def run_evaluation(model, test_stock_info, device, feature_normalizer=None,
                   begin_date=None, mc_dropout=False, mc_p=MC_DROPOUT_P, mc_samples=MC_SAMPLES):
    """
    执行模型评估（与 train.py 中对模型A的评估完全一致）
    返回评估统计字典

    Args:
        feature_normalizer: 可选的特征归一化器实例
        begin_date: 可选，每日统计的起始测评日期(int YYYYMMDD)。
                    仅展示/导出 预测日 >= begin_date 的交易日，不影响阈值计算。
        mc_dropout: 启用 MC Dropout 推理（多次前向取平均）
        mc_p: MC Dropout 概率
        mc_samples: MC Dropout 前向传播次数
    """
    print(f"正在创建评估数据集...")
    if begin_date is not None:
        print(f"- 评估区间: --begin {_format_date(begin_date)} ~ 最新（忽略训练/验证/测试集划分）")

    # --begin 给定时，评估区间变为 [begin, 最新]；否则为默认测试集区间
    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_tradeable_mask = \
        create_fixed_evaluation_dataset(test_stock_info, feature_normalizer, start_date=begin_date)

    print(f"- 评估样本数: {len(eval_inputs)}")

    # MC Dropout：强制所有 Dropout 层生效
    predict_fn = None
    if mc_dropout:
        num_dropout = set_mc_dropout(model, p=mc_p)
        print(f"- 已将 {num_dropout} 个 Dropout 层设为 p={mc_p}")
        print(f"正在评估模型 (MC Dropout ×{mc_samples})...")
        predict_fn = lambda m, x, d: mc_dropout_predict(m, x, d, num_samples=mc_samples)
    else:
        print(f"正在评估模型...")

    # 创建评估损失函数（与 train.py 一致；走统一入口 create_eval_criterion）
    eval_criterion = create_eval_criterion(eval_targets)

    # 评估模型（同时计算测试损失，避免冗余前向传播；MC Dropout 时预测走 predict_fn）
    stats = evaluate_model(
        model, eval_inputs, eval_targets, eval_cumulative_returns,
        device, model_name="选中模型",
        eval_day_indices=eval_day_indices,
        eval_daily_returns=eval_daily_returns,
        criterion=eval_criterion,
        enable_portfolio_simulation=True,
        tradeable_mask=eval_tradeable_mask,
        predict_fn=predict_fn
    )
    test_loss = stats['test_loss']

    # ========== 每日统计：对齐日历日期（供展示与 out_run/ JSON 导出）==========
    # split_point 口径必须与上面 create_fixed_evaluation_dataset 一致（start_date=begin_date）
    day_to_date = build_day_to_date_map(test_stock_info, start_date=begin_date)
    daily_kept = []  # [(date, count, return), ...] 有效交易日(count>0)
    if stats.get('realistic_stats') is not None:
        daily_kept = build_daily_kept(stats['realistic_stats'], day_to_date)
    stats['daily_kept'] = daily_kept  # 供 main() 写入 out_run/ JSON

    # 打印评估结果（与 train.py 格式一致）
    print()
    if mc_dropout:
        print(f"┌── 评估结果 (MC Dropout: p={mc_p}, samples={mc_samples}) ──┐")
    else:
        print(f"┌── 评估结果 ──────────────────────────────────┐")
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
        ds = rs['daily_stats']
        mode_str = f"每日Top{DataConfig.TOP_N_PER_DAY}" if rs.get('mode') == 'top_n_per_day' else \
                   f"全局阈值,每日上限{DataConfig.MAX_SELECT_PER_DAY},最大持仓{DataConfig.MAX_HOLDINGS}" if DataConfig.MAX_SELECT_PER_DAY > 0 else \
                   "全局阈值,不限数量"
        print(f"│  【实战收益率({mode_str})】")
        if daily_kept:
            print(f"│  交易区间: {_format_date(daily_kept[0][0])} ~ {_format_date(daily_kept[-1][0])}"
                  f"（{len(daily_kept)}个有效交易日）")
        if begin_date is not None:
            print(f"│  --begin 起始测评日期: {_format_date(begin_date)}（全区间回测，不分训练/验证/测试集）")
        # 内联展示：区间短则全量；区间长(--begin 多年)则只显示首尾，完整数据见 out_run/ JSON
        if len(ds) <= 60:
            daily_stats_str = ', '.join([f'({c},{r*100:.1f}%)' for c, r in ds])
            print(f"│  每日统计: {{{daily_stats_str}}}")
        else:
            head = ', '.join([f'({c},{r*100:.1f}%)' for c, r in ds[:8]])
            tail = ', '.join([f'({c},{r*100:.1f}%)' for c, r in ds[-4:]])
            print(f"│  每日统计: {{{head}, ... 共{len(ds)}天 ..., {tail}}}")
            print(f"│            （区间较长，完整逐日数据见 out_run/ JSON）")
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
            # 区间短则全量；区间长(--begin 多年)则只显示首10+末5，避免刷屏
            if len(dt) <= 40:
                show_idx = list(range(len(dt)))
            else:
                show_idx = list(range(10)) + list(range(len(dt) - 5, len(dt)))
            last_shown = -1
            for i in show_idx:
                if last_shown >= 0 and i > last_shown + 1:
                    print(f"│  ... （省略 {i - last_shown - 1} 天） ...")
                d = dt[i]
                cash_pct = d['cash_ratio'] * 100
                pv = d['portfolio_value']
                print(f"│  Day{d['day']:>4}: 买{d['buys']} 卖{d['open_sells']+d['close_sells']} "
                      f"资金={pv:.4f} (现金{cash_pct:.0f}%)")
                last_shown = i
            print(f"│")
    
    print(f"└───────────────────────────────────────────────┘")

    stats['eval_targets'] = stats['all_targets']

    return stats


def run_stock_selection(model, threshold, device, feature_normalizer=None,
                        mc_dropout=False, mc_samples=MC_SAMPLES):
    """
    执行选股

    Args:
        model: 模型实例
        threshold: 选股阈值
        device: 设备
        feature_normalizer: 可选的特征归一化器实例
        mc_dropout: 启用 MC Dropout 推理（多次前向取平均）
        mc_samples: MC Dropout 前向传播次数
    """
    print_section("选股推理 (MC Dropout)" if mc_dropout else "选股推理")
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
    if mc_dropout:
        print(f"│  正在对所有股票打分 (MC Dropout ×{mc_samples})...")
    else:
        print(f"│  正在对所有股票打分...")

    results, skipped = score_all_stocks(model, stock_list, device, feature_normalizer,
                                        mc_dropout=mc_dropout, mc_samples=mc_samples)
    
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
    print("╔" + "═"*73 + "╗")
    print("║" + " "*26 + "选 股 结 果 列 表" + " "*26 + "    ║")
    print("╠" + "═"*73 + "╣")
    print(f"║  {'排名':^4} {'代码':^7}{'模型分数':^10} {'收盘价':^7}{'涨跌幅':^8}{'日期':^10}{'':^6} ║")
    print("╠" + "═"*73 + "╣")
    
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
                print(f"║  {'':^4}  {'...':^8}  {'':^10}  {'':^8}  {'':^8}  {'':^10}  {'':^6}    ║")
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
                marker = "-阈值-"
            elif i == threshold_idx:
                marker = "  ↓  "
            elif i < threshold_idx:
                marker = "  ★  "
            
            print(f"║  {rank:>4}   {code:>8}   {score:>10.8f}   {close:>8.2f}  {change_str:>8}   {date:>10} {marker:^6}  ║")
            
            # 在阈值分界处画线
            if i == threshold_idx - 1 and threshold_idx < len(results):
                print("╠" + "─"*73 + "╣")
                print(f"║  {'':^4}  {'':^8}  {'↑ 超过阈值 ↑':^10}  {'│':^8}  {'↓ 低于阈值 ↓':^8}  {'':^10}     ║")
                print("╠" + "─"*73 + "╣")
    
    # 如果还有更多未显示的
    remaining = len(results) - len(printed_indices)
    if remaining > 0:
        print(f"║  {'':^4}  {'':^8}  {f'... 还有 {remaining} 只未显示':^10}  {'':^8}  {'':^8}     {'':^6}  ║")
    
    print("╚" + "═"*73 + "╝")
    
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
        print(f"│  推荐关注（Top{DataConfig.TOP_K}%阈值以上）:")
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
    title = f"Situation in the last {last_n} days"
    padding = (52 - len(title)) // 2
    print("║" + " "*padding + title + " "*(52 - padding - len(title)) + "║")
    print("╠" + "═"*52 + "╣")
    print("║  Day  │ Count │ Return   │ 相对日期   │ 数据       ║")
    print("╠" + "─"*52 + "╣")
    
    for i, (count, ret, available_days) in enumerate(recent_stats):
        # day_num 表示"倒数第几天"
        day_num = last_n - i
        
        # 相对日期
        relative_date = f"T-{day_num}"
        
        if available_days == DataConfig.FUTURE_DAYS:
            data_status = "完整"
        elif available_days == DataConfig.FUTURE_DAYS - 1:
            data_status = f"临时({available_days}天)"
        elif available_days == 1:
            data_status = "临时(1天)"
        else:
            data_status = "-"
        
        ret_str = f"{ret*100:+.1f}%"
        
        print(f"║  {day_num:>3}  │  {count:>3}  │ {ret_str:>8} │ {relative_date:<8}   │ {data_status:<9}║")
    
    print("╚" + "═"*52 + "╝")


def calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=4, threshold=None, feature_normalizer=None,
                                mc_dropout=False, mc_samples=MC_SAMPLES):
    """
    计算最近几天的实战收益率（用于展示，包含临时数据）

    关键设计：
    - 阈值来源：直接使用传入的阈值（由 run_evaluation 计算，基于固定评估集）
    - 选股范围：所有样本（包括临时样本），用于展示最近几天的选股情况
    - 临时样本：仅用于展示，方便用户决策，不参与任何阈值计算

    Args:
        model: 模型实例
        test_stock_info: 测试集股票信息列表
        device: 设备
        top_n_per_day: 每日选股数量
        threshold: 选股阈值
        feature_normalizer: 可选的特征归一化器实例
        mc_dropout: 启用 MC Dropout 推理（多次前向取平均）
        mc_samples: MC Dropout 前向传播次数

    返回: daily_stats [(count, return, available_days), ...]
    """
    recent_inputs, recent_returns, recent_day_indices, recent_available_days = \
        create_recent_days_dataset(test_stock_info, feature_normalizer, max_days=15)

    if recent_inputs is None or len(recent_inputs) == 0:
        return []

    if mc_dropout:
        all_preds = mc_dropout_predict(model, np.array(recent_inputs), device, num_samples=mc_samples)
    else:
        model.eval()
        all_preds = []
        amp_ctx = _get_amp_context(device)

        with torch.no_grad():
            batch_size = DataConfig.EVAL_BATCH_SIZE
            for i in range(0, len(recent_inputs), batch_size):
                batch = torch.tensor(recent_inputs[i:i+batch_size], dtype=torch.float32, device=device)
                with amp_ctx:
                    logits = model(batch)
                preds = torch.sigmoid(logits.float()).cpu().numpy().flatten()
                all_preds.extend(preds)
                del batch, logits

        all_preds = np.array(all_preds)
    all_returns = np.array(recent_returns)
    all_available_days = np.array(recent_available_days)
    
    unique_days = np.unique(recent_day_indices)
    unique_days = np.sort(unique_days)
    
    daily_stats = []
    
    use_threshold_mode = (top_n_per_day == 0 and threshold is not None)
    
    if use_threshold_mode:
        max_select = DataConfig.MAX_SELECT_PER_DAY
        
        above_threshold_mask = all_preds > threshold
        
        for day in unique_days:
            day_mask = recent_day_indices == day
            day_above_threshold = above_threshold_mask & day_mask
            day_indices = np.where(day_above_threshold)[0]
            
            count = len(day_indices)
            if count > 0:
                if max_select > 0 and count > max_select:
                    day_preds = all_preds[day_indices]
                    top_local = np.argsort(day_preds)[::-1][:max_select]
                    selected_indices = day_indices[top_local]
                    count = max_select
                else:
                    selected_indices = day_indices
                
                day_return = np.mean(all_returns[selected_indices])
                min_available = int(np.min(all_available_days[selected_indices]))
                daily_stats.append((count, day_return, min_available))
            else:
                daily_stats.append((0, 0.0, 0))
    else:
        for day in unique_days:
            day_mask = recent_day_indices == day
            day_indices = np.where(day_mask)[0]
            
            if len(day_indices) == 0:
                daily_stats.append((0, 0.0, 0))
                continue
            
            day_preds = all_preds[day_indices]
            day_returns = all_returns[day_indices]
            day_available = all_available_days[day_indices]
            
            sorted_local_indices = np.argsort(day_preds)[::-1]
            select_count = min(top_n_per_day, len(day_indices))
            top_local_indices = sorted_local_indices[:select_count]
            
            if select_count == 0 or len(top_local_indices) == 0:
                daily_stats.append((0, 0.0, 0))
                continue
            
            day_return = np.mean(day_returns[top_local_indices])
            min_available = int(np.min(day_available[top_local_indices]))
            
            daily_stats.append((select_count, day_return, min_available))
    
    return daily_stats


# ==================== 主函数 ====================

def main():
    # ===== 命令行参数 =====
    parser = argparse.ArgumentParser(description='EquiNet 模型推理与选股')
    parser.add_argument('--begin', type=str, default=None, metavar='YYYYMMDD',
                        help='每日统计的起始测评日期(如 20260301)：仅展示并导出 该日及之后的'
                             '交易日，不影响选股阈值计算。')
    parser.add_argument('--mc-dropout', action='store_true',
                        help='启用 MC Dropout 推理：评估、选股、最近几天统计均使用多次前向传播取平均，'
                             'Dropout 层在推理时保持激活以估计预测不确定性。')
    parser.add_argument('--mc-p', type=float, default=MC_DROPOUT_P, metavar='P',
                        help=f'MC Dropout 概率 (默认 {MC_DROPOUT_P})')
    parser.add_argument('--mc-samples', type=int, default=MC_SAMPLES, metavar='N',
                        help=f'MC Dropout 每个样本的前向传播次数 (默认 {MC_SAMPLES})')
    args = parser.parse_args()

    begin_date = None
    if args.begin:
        raw = re.sub(r'\D', '', args.begin)  # 容错：容忍 2026-03-01 / 2026/03/01 等写法
        if len(raw) != 8:
            print(f"  ✗ --begin 需要 8 位日期 YYYYMMDD，收到: {args.begin}")
            sys.exit(1)
        begin_date = int(raw)

    print_banner()
    if args.mc_dropout:
        print(f"  *** MC Dropout 模式 (p={args.mc_p}, samples={args.mc_samples}) ***")

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
        tp   = metadata.get('train_params') or {}
        es   = metadata.get('eval_stats') or {}
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
    
    # 加载数据并评估
    print(f"\n正在加载数据集...")

    # ========== 特征归一化器配置 ==========
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        print(f"正在加载归一化器...")
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print(f"\n  ⚠ 错误: 归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")
        print(f"  请先运行: python data.py")
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    # 运行评估
    stats = run_evaluation(model, test_stock_info, device, feature_normalizer,
                           begin_date=begin_date,
                           mc_dropout=args.mc_dropout, mc_p=args.mc_p, mc_samples=args.mc_samples)
    threshold = stats['top_threshold']

    # 导出每日统计 JSON 到 out_run/（功能2）
    daily_kept = stats.get('daily_kept') or []
    if daily_kept and stats.get('realistic_stats') is not None:
        rs = stats['realistic_stats']
        now = datetime.now()
        meta = {
            'model': selected_file,
            'mode': rs.get('mode'),
            'threshold': float(threshold),
            'max_select_per_day': DataConfig.MAX_SELECT_PER_DAY,
            'top_n_per_day': DataConfig.TOP_N_PER_DAY,
            'begin_date': _format_date(begin_date) if begin_date else None,
            'start_date': _format_date(daily_kept[0][0]),
            'end_date': _format_date(daily_kept[-1][0]),
            'total_days': len(daily_kept),
            'avg_return_pct': round(rs['avg_realistic_return'] * 100, 4),
            'cumulative_return_pct': round(rs['cumulative_return'] * 100, 4),
            'generated_at': now.strftime('%Y-%m-%d %H:%M:%S'),
            'generated_tag': now.strftime('%Y%m%d_%H%M%S'),
        }
        json_path = write_daily_json(daily_kept, meta)
        print(f"\n  📄 每日统计已导出: {json_path}")
        print(f"     可视化: python src/visualize_daily.py \"{json_path}\"")
    else:
        print(f"\n  （无可导出的每日统计数据）")

    viz_title = f"模型分类能力可视化 (AUC={stats['auc']:.4f})"
    if args.mc_dropout:
        viz_title += f", MC Dropout (p={args.mc_p}, ×{args.mc_samples})"
    visualize_classification(
        preds=stats['all_preds'],
        targets=stats['eval_targets'],
        title=viz_title
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
    
    # 执行选股（MC Dropout 模式下选股前确保 Dropout 已激活）
    if args.mc_dropout:
        set_mc_dropout(model, p=args.mc_p)
    results = run_stock_selection(model, threshold, device, feature_normalizer,
                                  mc_dropout=args.mc_dropout, mc_samples=args.mc_samples)

    # 计算并打印最近10天实战收益率表格（包含临时数据，仅用于展示）
    recent_stats = calculate_recent_days_stats(model, test_stock_info, device, top_n_per_day=DataConfig.TOP_N_PER_DAY, threshold=threshold, feature_normalizer=feature_normalizer,
                                               mc_dropout=args.mc_dropout, mc_samples=args.mc_samples)
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
