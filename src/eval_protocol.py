"""
Phase 0 —— 可信评估协议

## 为什么先写这个而不是先改模型

在拿到「模型有没有 alpha」这个问题的答案之前，必须先确认**问问题的尺子是准的**。
一把没校准过的尺子会给出正数，而正数会让人停止怀疑 —— 这是本项目此前最危险的处境：
报告里写着「Top1% 收益 +1.11%」，但没人知道这 1.11% 里多少来自市场本身、
多少来自止损规则、多少来自模型。

所以本模块的第一用途不是评估模型，而是**评估评估方法本身**：

    python src/eval_protocol.py --selftest

它会用三个已知的打分器跑同一套协议：
  1. 随机打分   → 协议必须说「没有 alpha」（若说有，协议坏了）
  2. 注入信号   → 协议必须说「有 alpha」（若说没有，协议坏了）
  3. 20 日反转  → 协议必须说「有 alpha」，且给出 Transformer 必须超越的门槛

三条都通过，尺子才算校准完成，之后的数字才有资格被相信。

## 本模块做了什么（相对现行 training_utils.calculate_realistic_return 的差异）

| 项 | 现行做法 | 本协议 |
|---|---|---|
| 收益口径 | 止损策略下的实现收益 | T+1 开盘买、T+3 收盘卖，无止损（止损贡献单独量化） |
| 基准 | 无 | 同一预测日的全市场等权收益 |
| 阈值 | 用整个评估集定（回看） | 每日 Top-N，不需要全局阈值 |
| 空仓日 | 剔除，不进均值 | 每天必选 N 只，不存在空仓 |
| 显著性 | 无 | Newey-West t 值（3 天窗口重叠 → lag=2） |
| 截面信号 | 无 | 逐日 Spearman IC、ICIR |
| 噪声对照 | 无 | 每日内随机选股 Monte Carlo，给出百分位 |

## 成本

按 hzy 2026-09-26 的决定：EquiNet 不建模交易成本（与资金体量相关）。
本协议因此**只回答「有没有超额」，不回答「能不能赚钱」**。
把成本加回来只需在 `evaluate()` 里对 daily_excess 减一个常数。

## 判读标准（团队共识，写死在常量里，避免每次重新讨论）

    IC      > 0.02   且
    ICIR    > 0.30   且
    |t|     > 2.0    且
    随机对照百分位 > 95

四条同时成立才算「检出可交易信号」。这是量化行业对日频短周期信号的常规门槛，
不是本项目自定的宽松标准。
"""

import os
import sys
import json
import argparse
import sqlite3
from collections import defaultdict

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

# 池口径从 config 取（config 是 EquiNet 与 equinet-data-manager 的唯一接口处），
# 本文件不再自己 import 数据包 —— 否则就出现了第二个知道外部路径的地方。
from config import (  # noqa: E402
    DataConfig, pool_join_sql, CURRENT_POOL, pool_scope_label,
)


# ==================== 判读门槛 ====================

IC_OK = 0.02        # 日均截面 IC
ICIR_OK = 0.30      # IC 信息比率
T_OK = 2.0          # Newey-West t 绝对值
MC_PCTL_OK = 95.0   # 随机对照百分位

# 评估期默认：测试集起点到数据最新日
DEFAULT_BEGIN = DataConfig.TEST_START_DATE

# T+1 开盘涨停视为买不进（与 data.py tradeable_mask 同一口径）
OPEN_LIMIT = DataConfig.LIMIT_THRESHOLD   # 0.095

# 过去收益窗口（朴素因子用）
PAST_WINDOWS = (5, 10, 20)

# 自检时「已知异象应被检出」所需的最短预测日数。
# 短于此长度时，连 20 日反转这种稳健异象也会被噪声淹没 ——
# 实测 2026H1（136 天）反转 IC 仅 +0.0017，而 2023–2025（724 天）为 +0.0517。
# 窗口不够就让自检跳过这条，而不是把它误判成「协议坏了」。
MIN_DAYS_FOR_ANOMALY = 250

# 每天至少这么多只可交易股票才统计该日（否则样本太少，日收益噪声主导）
MIN_STOCKS_PER_DAY = 50


# ==================== 面板数据 ====================

class Panel:
    """逐预测日的截面面板。

    day[date] 里每个数组长度相同，第 j 项是同一只股票。
    ret3       : 无止损口径，T+1 开盘买入、T+3 收盘卖出的收益
    ret3_stop  : 现行止损口径（config.calculate_returns 的三条规则）
    tradable   : T+1 开盘未涨停（买得进）
    """

    def __init__(self, dates, day, pool_label):
        self.dates = list(dates)
        self.day = day
        self.pool_label = pool_label

    def __len__(self):
        return len(self.dates)

    @property
    def n_days(self):
        return len([d for d in self.dates if self.day[d]['n'] >= MIN_STOCKS_PER_DAY])

    def summary(self):
        total = sum(self.day[d]['n'] for d in self.dates)
        return (f"预测日 {len(self.dates)} 天，样本 {total:,} 条，"
                f"股票池 {self.pool_label}")


def _connect_ro(db_path):
    return sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)


def _latest_date(con):
    row = con.execute("SELECT MAX(date) FROM stock_daily").fetchone()
    return int(row[0]) if row and row[0] else None


def load_panel(begin=DEFAULT_BEGIN, end=None, db_path=None, pool=None,
               lookback_days=70, verbose=True):
    """构建评估面板。

    lookback_days: 为了算 20 日反转因子，需要预测日之前约 20 个交易日的数据，
                   70 个自然日足够覆盖（含春节等长假）。
    """
    db_path = db_path or DataConfig.DB_PATH
    pool = pool or CURRENT_POOL

    con = _connect_ro(db_path)
    try:
        if end is None:
            end = _latest_date(con)
        # 自然日回退：int 日期不能直接减天数，这里用 SQL 排序后取前 N 个 distinct 日期
        begin_int = int(begin)
        sql_dates = f"""
            SELECT DISTINCT date FROM stock_daily
            WHERE date < ? ORDER BY date DESC LIMIT ?
        """
        # 往回数 lookback_days 个**交易日**（而非自然日），保证 20 日因子足够
        pre = [r[0] for r in con.execute(sql_dates, (begin_int, lookback_days))]
        fetch_from = min(pre) if pre else begin_int

        sql = f"""
            SELECT sd.stock_code, sd.date, sd.open_adj, sd.close_adj
            FROM stock_daily sd
            {pool_join_sql('sd', pool)}
            WHERE sd.date >= ? AND sd.date <= ?
            ORDER BY sd.stock_code, sd.date
        """
        rows = con.execute(sql, (fetch_from, int(end))).fetchall()
    finally:
        con.close()

    if not rows:
        raise RuntimeError(f"面板为空：{fetch_from}~{end} 在池 {pool} 内没有任何行情")

    by_stock = defaultdict(list)
    for code, date, o, c in rows:
        by_stock[code].append((date, o, c))

    day = defaultdict(lambda: defaultdict(list))
    n_skipped_short = 0

    for code, series in by_stock.items():
        n = len(series)
        if n < 25:
            n_skipped_short += 1
            continue
        dates = np.array([r[0] for r in series], dtype=np.int64)
        op = np.array([r[1] for r in series], dtype=np.float64)
        cl = np.array([r[2] for r in series], dtype=np.float64)

        max_past = max(PAST_WINDOWS)
        i = np.arange(max_past, n - 3)
        if len(i) == 0:
            n_skipped_short += 1
            continue

        # 只保留评估期内的预测日
        keep = dates[i] >= begin_int
        i = i[keep]
        if len(i) == 0:
            continue

        c0 = cl[i]
        o1 = op[i + 1]
        c1 = cl[i + 1]
        o2 = op[i + 2]
        c2 = cl[i + 2]
        c3 = cl[i + 3]

        safe_o1 = np.where(o1 != 0, o1, np.nan)
        ret3 = (c3 - o1) / safe_o1

        # 现行止损口径，逐条对齐 config.calculate_returns
        d1 = (c1 - o1) / safe_o1                       # LABEL_DAY1_USE_OPEN=True
        d2 = (c2 - c1) / np.where(c1 != 0, c1, np.nan)
        ret_stop = ret3.copy()
        m1 = d1 <= -0.03                                        # 规则1：次日开盘止损
        ret_stop = np.where(m1, (o2 - o1) / safe_o1, ret_stop)
        m23 = (~m1) & (
            ((d1 + d2) < -0.02) | ((d1 < 0.01) & (d2 < 0.01))
        )                                                       # 规则2/3：次日收盘止损
        ret_stop = np.where(m23, (c2 - o1) / safe_o1, ret_stop)

        open_gap = (o1 - c0) / np.where(c0 != 0, c0, np.nan)
        tradable = open_gap < OPEN_LIMIT

        # 标签：config.generate_label 的五条硬规则（逐位等价的向量化版本）。
        # 加进来是为了把「模型在标签上的 AUC」换算成「收益尺度上的期望值」——
        # 这是本项目此前完全缺失的一层校准。
        day1 = (c1 - c0) / np.where(c0 != 0, c0, np.nan)
        day2 = (c2 - c1) / np.where(c1 != 0, c1, np.nan)
        day3 = (c3 - c2) / np.where(c2 != 0, c2, np.nan)
        cum2 = day1 + day2
        cum3 = day1 + day2 + day3
        mx = np.maximum(np.maximum(day1, day2), day3)
        label = (
            ((day1 >= 0.05) & (cum3 >= 0.03))
            | ((cum2 >= 0.06) & (day1 > 0.01) & (day2 > 0.01) & (cum3 >= 0.03))
            | ((day1 >= 0.01) & (day2 >= 0.01) & (day3 >= 0.01) & (cum3 >= 0.06))
            | ((mx >= 0.08) & (cum3 >= 0.06) & (day1 >= -0.02))
            | ((cum3 >= 0.08) & (day1 >= -0.02))
        ).astype(np.float64)

        finite = np.isfinite(ret3) & np.isfinite(ret_stop) & np.isfinite(open_gap)
        finite = finite & np.isfinite(cum3)
        tradable = tradable & finite
        if not tradable.any():
            continue

        past = {}
        for w in PAST_WINDOWS:
            base = cl[i - w]
            past[w] = (c0 - base) / np.where(base != 0, base, np.nan)

        for j, idx in enumerate(i):
            if not tradable[j]:
                continue
            d = int(dates[idx])
            day[d]['code'].append(code)
            day[d]['ret3'].append(ret3[j])
            day[d]['ret3_stop'].append(ret_stop[j])
            day[d]['label'].append(label[j])
            day[d]['day1'].append(day1[j])
            for w in PAST_WINDOWS:
                day[d][f'past{w}'].append(past[w][j])

    out = {}
    for d in sorted(day.keys()):
        rec = {k: np.asarray(v, dtype=np.float64) for k, v in day[d].items() if k != 'code'}
        rec['code'] = np.array(day[d]['code'])
        rec['n'] = len(rec['code'])
        out[d] = rec

    if not out:
        raise RuntimeError("面板为空：过滤后没有任何可交易样本")

    panel = Panel(sorted(out.keys()), out, pool_scope_label(pool))
    if verbose:
        print(f"  面板: {panel.summary()}")
        if n_skipped_short:
            print(f"  （{n_skipped_short} 只股票因历史太短被跳过）")
    return panel


# ==================== 打分器 ====================
# 统一接口：dict[date] -> np.ndarray[float]，长度与该日可交易股票数一致。

def scorer_random(panel, seed=42):
    """随机打分。用于校准：协议在它身上必须输出「无 alpha」。"""
    rng = np.random.default_rng(seed)
    return {d: rng.random(panel.day[d]['n']) for d in panel.dates}


def scorer_reversal(panel, window=20):
    """反转因子：过去 window 日跌得多的股票给高分。

    A 股最稳健的短周期异象之一，是 Transformer 必须跨过的第一道门槛。
    """
    key = f'past{window}'
    return {d: -panel.day[d][key] for d in panel.dates}


def scorer_momentum(panel, window=5):
    """动量因子。作为反转的对照组。"""
    key = f'past{window}'
    return {d: panel.day[d][key] for d in panel.dates}


def scorer_on_label(panel, target_auc=0.70, seed=42):
    """构造在**本项目五条规则标签**上 AUC = target_auc 的打分器。

    用途：把「模型在标签上有多少判别力」翻译成「在收益尺度上应该值多少」。
    项目优化的是 config.generate_label 的五条硬规则，验收的却是止损策略下的
    实现收益 —— 这条链路从来没有被校准过，所以没人知道 AUC 0.70 到底该对应
    百分之几。这个函数给出参照值。

    构造：score|y=1 ~ N(+delta/2, 1)，score|y=0 ~ N(-delta/2, 1)
          AUC = Phi(delta / sqrt(2))  →  delta = sqrt(2) * Phi^-1(AUC)

    **这是保守下界**：构造出的打分器除标签关联外全是逐日独立的噪声，不具备
    真实模型那种跨期一致的结构化偏好，Top-N 会被噪声稀释。真实模型若有同等
    判别力，收益应当**不低于**此值。若实测远低于此值，说明判别力没有转化。
    """
    from statistics import NormalDist
    inv = NormalDist().inv_cdf
    target_auc = min(max(target_auc, 1e-6), 1 - 1e-6)
    delta = np.sqrt(2.0) * inv(target_auc)

    rng = np.random.default_rng(seed)
    scores = {}
    for d in panel.dates:
        y = panel.day[d]['label']
        scores[d] = rng.standard_normal(len(y)) + delta * (y - 0.5)
    return scores


def panel_auc_label(panel, scores, ret_key='ret3'):
    """在项目标签上的 AUC。返回 (日均值, 日均值的真实标准误)。

    **标准误必须按交易日算，不能按样本条数算。**
    每天的几百只股票并不独立 —— 它们共享同一天的市场状态，而且相邻样本
    的 45 天上下文重叠 44/45 天。用总样本数 n 去除标准差会把置信区间缩窄
    一到两个数量级，让一个其实是 ±0.02 的 AUC 看起来像 ±0.002。

    正确做法：先算逐日 AUC，得到 243 个（或 n_days 个）观测，
    再用 std(日AUC)/sqrt(n_days) 作为均值的标准误。
    """
    aucs = []
    for d in panel.dates:
        rec = panel.day[d]
        y = rec['label']
        s = np.asarray(scores[d], dtype=np.float64)
        ok = np.isfinite(s) & (np.isfinite(y))
        if ok.sum() < MIN_STOCKS_PER_DAY:
            continue
        yy, ss = y[ok], s[ok]
        if (yy > 0).sum() < 3 or (yy <= 0).sum() < 3:
            continue
        aucs.append(auc_binary(yy, ss))
    if not aucs:
        return float('nan'), float('nan')
    a = np.array(aucs)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) / np.sqrt(len(a)))


def label_audit(panel):
    """标签本身的基本面：正样本率、.ylabel 分布是否可分离。"""
    rates = []
    pos_ret, neg_ret = [], []
    for d in panel.dates:
        rec = panel.day[d]
        y = rec['label']
        ok = np.isfinite(y)
        if ok.sum() < MIN_STOCKS_PER_DAY:
            continue
        yy, rr = y[ok], rec['ret3'][ok]
        rates.append(float((yy > 0).mean()))
        m = yy > 0
        if m.sum() >= 3:
            pos_ret.append(rr[m].mean())
        if (~m).sum() >= 3:
            neg_ret.append(rr[~m].mean())
    return {
        'pos_rate': float(np.mean(rates)) if rates else float('nan'),
        'pos_ret': float(np.mean(pos_ret)) if pos_ret else float('nan'),
        'neg_ret': float(np.mean(neg_ret)) if neg_ret else float('nan'),
    }


def scorer_injected(panel, target_ic=0.08, seed=42):
    """注入已知信号的打分器 —— **仅用于校准尺子**。

    把未来收益按指定相关系数掺进打分里，构造一个「确实有 alpha」的阳性对照。

    target_ic 取 0.08 而不是接近 1：阳性对照的强度必须和真实异象同量级才有校准
    意义——用 IC=0.98 的完美信号去验证协议「检得出」，等于什么都没验证。
    实测 A 股 20 日反转在这个池里 IC ≈ 0.05，0.08 是同量级、略强一点的水平。

    它必然前视，**绝对不能用于真实策略评估**，只用来证明协议检得出信号。
    """
    rng = np.random.default_rng(seed)
    scores = {}
    for d in panel.dates:
        r = panel.day[d]['ret3']
        sd = r.std()
        if not np.isfinite(sd) or sd <= 0:
            scores[d] = rng.standard_normal(len(r))
            continue
        signal = (r - r.mean()) / sd
        noise = rng.standard_normal(len(r))
        # corr(score, r) ≈ target_ic
        scores[d] = target_ic * signal + np.sqrt(1.0 - target_ic ** 2) * noise
    return scores


# ==================== 统计 ====================

def newey_west_t(x, lag=2):
    """重叠窗口下的均值 t 值。

    标签窗口 3 天 → 相邻观测重叠 2 天，OLS 标准误会低估约 √3 倍。
    lag = future_days - 1 = 2 是对应的 Bartlett 核截断阶数。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float('nan')
    m = x.mean()
    s = np.var(x, ddof=1)
    for l in range(1, lag + 1):
        if l >= n:
            break
        cov = np.cov(x[:-l], x[l:])[0, 1]
        s += 2.0 * (1.0 - l / (lag + 1.0)) * cov
    if s <= 0:
        return float('nan')
    return m / np.sqrt(s / n)


def moving_block_bootstrap(x, block=20, n_boot=2000, seed=42):
    """移动块自助法（moving block bootstrap）。

    日超额序列不是独立同分布的 —— 市场状态会持续若干周，相邻观测强相关。
    naive bootstrap 会把这种自相关当成独立信息，从而**低估真实方差**。
    block 取 20 个交易日（约一个月）足以吸收状态持续性，是金融序列的常规选择。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        out[b] = np.concatenate([x[s:s + block] for s in starts])[:n].mean()
    return out


def auc_binary(y, s):
    """二分类 AUC（rank-based Mann-Whitney，无需 sklearn）。"""
    y = np.asarray(y).astype(float)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos = int((y > 0).sum())
    n_neg = int((y <= 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # 并列值取平均秩
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y > 0].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_ic(a, b):
    """Spearman 秩相关。样本数 < 5 时返回 NaN。"""
    if len(a) < 5:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom == 0:
        return float('nan')
    return float((ra * rb).sum() / denom)


# ==================== 评估主流程 ====================

def evaluate(panel, scores, top_n=4, n_mc=200, seed=42, label="打分器",
             use_stop_returns=False):
    """跑一遍完整评估协议。

    Args:
        top_n: 每天选多少只。用每日 Top-N 而不是全局阈值，
               是为了避免「阈值用整个评估集反推」这个回看问题。
        n_mc:  随机对照的 Monte Carlo 次数。
        use_stop_returns: True 时用现行止损口径的收益（用于量化止损贡献）。

    Returns:
        dict，字段见 report() 的打印内容。
    """
    ret_key = 'ret3_stop' if use_stop_returns else 'ret3'
    rng = np.random.default_rng(seed)

    dates, sel_ret, bench_ret, excess, ics = [], [], [], [], []
    counts = []

    for d in panel.dates:
        rec = panel.day[d]
        if rec['n'] < MIN_STOCKS_PER_DAY:
            continue
        s = np.asarray(scores[d], dtype=np.float64)
        r = rec[ret_key]
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < MIN_STOCKS_PER_DAY:
            continue
        s, r = s[ok], r[ok]

        k = min(top_n, len(s))
        top = np.argsort(s)[::-1][:k]

        dates.append(d)
        sel_ret.append(r[top].mean())
        bench_ret.append(r.mean())          # 全市场等权 = 不做任何选择的收益
        excess.append(r[top].mean() - r.mean())
        ics.append(spearman_ic(s, r))
        counts.append(k)

    dates = np.array(dates)
    sel_ret = np.array(sel_ret)
    bench_ret = np.array(bench_ret)
    excess = np.array(excess)
    ics = np.array(ics)

    # 随机对照：每日内随机选同样数量，重复 n_mc 次
    # n_mc=0 跳过 —— 批量实验（稳定性诊断跑几十次）时百分位不是必需的，
    # 而 MC 是本协议里唯一 O(n_mc × n_days) 的部分，跳过能省几十倍时间。
    mc_means = np.empty(max(n_mc, 1))
    if n_mc > 0:
        for b in range(n_mc):
            acc = []
            for d in dates:
                rec = panel.day[d]
                r = rec[ret_key]
                ok = np.isfinite(r)
                r = r[ok]
                if len(r) < MIN_STOCKS_PER_DAY:
                    continue
                k = min(top_n, len(r))
                pick = rng.choice(len(r), size=k, replace=False)
                acc.append(r[pick].mean() - r.mean())
            mc_means[b] = np.mean(acc) if acc else np.nan

    obs = excess.mean()
    if n_mc > 0 and np.isfinite(mc_means).all():
        pctl = float((mc_means < obs).mean() * 100.0)
    else:
        mc_means = np.array([np.nan])
        pctl = float('nan')

    ic = ics[np.isfinite(ics)]
    ic_mean = float(ic.mean()) if len(ic) else float('nan')
    icir = float(ic_mean / ic.std(ddof=1)) if len(ic) > 1 and ic.std(ddof=1) > 0 else float('nan')
    t_stat = newey_west_t(excess, lag=DataConfig.FUTURE_DAYS - 1)

    # 月度 / 年度分解（YYYYMM 为键，显示时再格式化，避免跨年月份撞在一起）
    monthly = defaultdict(list)
    for d, e in zip(dates, excess):
        monthly[d // 100].append(e)
    months = {m: float(np.mean(v)) for m, v in sorted(monthly.items())}

    yearly = defaultdict(list)
    for m, v in months.items():
        yearly[m // 100].append(v)
    years = {y: float(np.mean(v)) for y, v in sorted(yearly.items())}

    return {
        'label': label,
        'n_days': int(len(dates)),
        'top_n': top_n,
        'ret_key': ret_key,
        'sel_mean': float(sel_ret.mean()),
        'bench_mean': float(bench_ret.mean()),
        'excess_mean': obs,
        'excess_std': float(excess.std(ddof=1)) if len(excess) > 1 else float('nan'),
        't_stat': t_stat,
        'win_rate': float((excess > 0).mean()),
        'ic_mean': ic_mean,
        'icir': icir,
        'ic_pos_rate': float((ic > 0).mean()) if len(ic) else float('nan'),
        'mc_mean': float(np.nanmean(mc_means)),
        'mc_std': float(np.nanstd(mc_means, ddof=1)),
        'mc_pctl': pctl,
        'months': months,
        'years': years,
        'dates': dates,
        'excess': excess,
        'ics': ics,
    }


def _fmt_ym(m):
    return f"{m // 100}-{m % 100:02d}"


def report_dual(panel, scores, label, top_n, n_mc=200):
    """同一打分器跑两种收益口径，把「止损规则贡献」单独量化出来。

    存在感很低但很重要：现行 calculate_returns 的三条止损规则与正样本标签
    的条件高度重合（规则 2/3 也要求 day1、day2 > 1%），等于止损在替模型做
    择时筛选。不把它分离出来，那部分收益会被误记在模型头上。
    """
    r_no = evaluate(panel, scores, top_n=top_n, n_mc=n_mc, label=label,
                    use_stop_returns=False)
    r_st = evaluate(panel, scores, top_n=top_n, n_mc=n_mc, label=label,
                    use_stop_returns=True)
    report(r_no, show_months=False)
    delta = r_st['excess_mean'] - r_no['excess_mean']
    p = print
    p(f"    止损口径超额:      {r_st['excess_mean']*100:+7.3f}%")
    p(f"    → 止损规则净贡献:  {delta*100:+7.3f}%   "
      f"({'止损在帮模型' if delta > 0 else '止损在拖后腿'})")
    return r_no, r_st, delta


def _jsonable(v):
    """转 JSON 可序列化：np 标量 → float，字典键 → str。

    months/years 的键来自 `date // 100`，date 是 np.int64 ——
    json 不接受 numpy 整数做键，会在写到一半时抛 TypeError，
    留下一个截断的、不可解析的 JSON 文件。
    """
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save_report(results, path):
    """把评估结果存成 JSON，供后续模型对比（基线档案）。"""
    out = {}
    for name, res in results.items():
        out[name] = {k: _jsonable(v) for k, v in res.items()
                     if k not in ('dates', 'excess', 'ics')}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  基线档案已保存: {path}")
    return path


def load_baseline(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ==================== 稳定性诊断 ====================
# 回答 hzy 2026-09-26 的三个观察：
#   1. 不同 epoch 日均收益在 0.3% ~ 1.4% 之间跳，AUC 却稳在 70%
#   2. AUC 卡在 70% 上不去（回归卡 63%） —— 像遇到了天花板
#   3. 同样 AUC 70% 的模型，最终收益率大相径庭
# 这三个现象有一个共同的候选解释：**全局排序指标与尾部收益指标之间的信息落差**。
# 下面三个实验用已知答案的对照组把它量化出来，不需要训练任何模型。

def _sub_panel(panel, dates):
    """取原面板的一段日期子集。evaluate 只按 dates 遍历 day dict，可直接复用。"""
    return Panel(list(dates), panel.day, panel.pool_label)


def panel_auc(panel, scores, ret_key='ret3'):
    """整个面板上的平均逐日 AUC（标签 = 当日截面收益中位数以上）。

    中位数二分而不是 0 二分：后者在普涨/普跌日会出现极端不平衡的类比，
    AUC 会失真。中位数二分衡量的是纯粹的「截面排序质量」。
    """
    aucs = []
    for d in panel.dates:
        rec = panel.day[d]
        r = rec[ret_key]
        s = np.asarray(scores[d], dtype=np.float64)
        ok = np.isfinite(r) & np.isfinite(s)
        if ok.sum() < MIN_STOCKS_PER_DAY:
            continue
        rr = r[ok]
        thr = np.nanmedian(rr)
        aucs.append(auc_binary((rr > thr).astype(float), s[ok]))
    return float(np.nanmean(aucs)) if aucs else float('nan')


def scorer_synthetic_epoch(panel, target_ic=0.05, rho=0.5, seed=42, ret_key='ret3'):
    """模拟「训练到不同 epoch 的模型」：IC 期望相同，但具体选股系统性不同。

    要判断 epoch 间的收益波动是「模型真的变了」还是「测不准」，需要一个
    **答案已知的对照组**：构造一批打分器，其截面 IC 的期望值由构造固定在同一个
    数，唯一区别是随机种子 —— 等价于「每个 epoch 的能力都一样，只是把这套能力
    落成了不同的具体偏好」。

    构造：打分 = 跨 epoch 共享的成分 u + epoch 特有成分 e_t

        score_t = sqrt(rho) * u + sqrt(1 - rho) * e_t
        corr(score_t, r) = sqrt(rho) * corr(u, r)

    令 corr(u, r) = target_ic / sqrt(rho)，每个 epoch 的期望 IC 就都是 target_ic。
    rho 控制「epoch 之间有多像」：0.9 = 高度稳定，0.3 = 几乎每次重学一遍。

    [第一版踩过的坑 —— 保留此教训]
    最初用「标签上的 AUC」而不是「连续收益上的 IC」做锚点：把 target_auc=0.70
    换算成 u 与标签的距离 delta，结果隐含 IC 高达 **0.30**（中位数二分的标签与
    收益的秩相关性远强于直觉）。信号强到 Top-N 选股几乎确定，跨 epoch 收益极差
    只有 0.26%，**完全复现不出**「同样 AUC 收益差很多」。
    锚点必须是 IC 这种直接对应「对连续收益的排序强度」的量，不能用标签 AUC。
    """
    rng = np.random.default_rng(seed)
    c = target_ic / np.sqrt(rho)          # corr(u, r)，经 sqrt(rho) 缩放后 = target_ic
    c = float(np.clip(c, -0.999, 0.999))

    shared = {}
    for d in panel.dates:
        r = panel.day[d][ret_key]
        z = np.zeros_like(r)
        ok = np.isfinite(r)
        if ok.sum() > 5:
            rr = r[ok]
            sd = rr.std()
            if sd > 0:
                z[ok] = (rr - rr.mean()) / sd
        shared[d] = c * z + np.sqrt(1.0 - c * c) * rng.standard_normal(len(r))

    scores = {}
    for d in panel.dates:
        e = rng.standard_normal(len(shared[d]))
        scores[d] = np.sqrt(rho) * shared[d] + np.sqrt(1.0 - rho) * e
    return scores


def diagnose_stability(panel, top_n=4, window=136, step=20,
                       target_ics=(0.03, 0.05, 0.10),
                       target_aucs=(0.60, 0.65, 0.70, 0.75),
                       n_epochs=20, n_epochs_label=6,
                       rhos=(0.9, 0.5), n_boot=2000, seed=42):
    """稳定性诊断。四个实验共用同一份 panel，全跑一次约 4-5 分钟。"""
    p = print
    ds = panel.dates

    # ---------- 实验 1：什么都没变，数字也在跳 ----------
    p("\n" + "=" * 70)
    p("实验 1 —— 打分器完全不变，只换测量窗口，日均超额能跳多少")
    p("=" * 70)
    p(f"  做法：20 日反转是个零自由度的固定因子（连随机数都没有），")
    p(f"        在 {len(ds)} 个预测日上滑动 {window} 天窗口（步长 {step} 天），每个窗口测一次。")
    p(f"        窗口之间**因子没有任何变化**，变的只是样本期。")
    p("")

    rev_scores = scorer_reversal(panel, 20)
    rows = []
    for start in range(0, max(1, len(ds) - window + 1), step):
        sub = _sub_panel(panel, ds[start:start + window])
        r = evaluate(sub, rev_scores, top_n=top_n, n_mc=0, seed=seed)
        rows.append((sub.dates[0], sub.dates[-1], r['excess_mean'], r['ic_mean'],
                     r['t_stat']))
    ex = np.array([r[2] for r in rows])
    if len(rows) >= 2:
        p(f"  共 {len(rows)} 个窗口，日均超额分布：")
        p(f"    最小 {ex.min()*100:+.3f}%   25分位 {np.percentile(ex,25)*100:+.3f}%   "
          f"中位 {np.median(ex)*100:+.3f}%")
        p(f"    75分位 {np.percentile(ex,75)*100:+.3f}%   最大 {ex.max()*100:+.3f}%   "
          f"极差 {(ex.max()-ex.min())*100:.3f}%")
        p(f"    标准差 {ex.std(ddof=1)*100:.3f}%")
        p("")
        p("    [窗口起止]               日均超额      IC")
        for a, b, e_, ic_, t_ in rows[:14]:
            p(f"    {_fmt_date(a)}~{_fmt_date(b)}   {e_*100:+7.3f}%   {ic_:+7.4f}")
        if len(rows) > 14:
            p(f"    ...（省略 {len(rows)-14} 个窗口）")
        p("")
        p("  >>> 读法：如果连一个固定的、连随机数都没有的因子，")
        p(f"      换套样本都能在 {ex.min()*100:+.2f}% ~ {ex.max()*100:+.2f}% 之间跳，")
        p("      那么 epoch 之间的那点差异，未必是模型变了。")

    # ---------- 实验 2：定量刻画「测不准」有多宽 ----------
    p("\n" + "=" * 70)
    p("实验 2 —— 同一份预测，95% 置信区间有多宽")
    p("=" * 70)
    full = evaluate(panel, rev_scores, top_n=top_n, n_mc=0, seed=seed)
    ex_full = full['excess']
    se_full = ex_full.std(ddof=1) / np.sqrt(len(ex_full))
    t_full = full['t_stat']
    se_nw = (full['excess_mean'] / t_full) if np.isfinite(t_full) and t_full != 0 else np.nan
    boot = moving_block_bootstrap(ex_full, block=20, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    p(f"  全窗口 {len(ex_full)} 个预测日，日均超额 {full['excess_mean']*100:+.3f}%")
    p(f"    朴素标准误        {se_full*100:.3f}%（假设日间独立，会低估）")
    p(f"    Newey-West 标准误 {se_nw*100:.3f}%（修正 3 天标签重叠）")
    p(f"    块自助法 95% 区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  宽度 {(hi-lo)*100:.3f}%")
    p("")
    se_w = float('nan')
    if len(ds) > window:
        # 方差 ∝ 1/n，天数缩到 window 时标准误放大 sqrt(N/window)
        scale = np.sqrt(len(ex_full) / window)
        se_w = se_nw * scale
        p(f"  但你们实际验收用的是 ~{window} 个预测日。天数从 {len(ex_full)} 缩到 {window}，")
        p(f"    标准误放大 ×{scale:.2f} → {se_w*100:.3f}%")
        p(f"    ⇒ 单次测量的 95% 区间约 ±{1.96*se_w*100:.3f}%")
        p("")
        p(f"  >>> 读法：在 {window} 天窗口上，同一模型两次测量的差只要小于 "
          f"{2*1.96*se_w*100:.2f}%，")
        p("      统计上就**无法区分**「模型真变了」和「什么都没变」。")

    # ---------- 实验 3：同样能力水平的两个模型，收益能差多少 ----------
    p("\n" + "=" * 70)
    p("实验 3 —— 两个 IC 完全相同的模型，收益能差到什么程度")
    p("=" * 70)
    p("  做法：构造一批打分器，对连续收益的截面 IC **由构造固定**在同一档位，")
    p("        它们之间只有随机种子不同 —— 等价于「每个 epoch 能力一样，")
    p("        但把这套能力落成了不同的具体偏好（选出来的股票不一样）」。")
    p("        rho = epoch 之间共享的信息比例，0.9 表示两个 epoch 很像，")
    p("        0.3 表示几乎各学各的。")
    p("")
    p(f"  每日 Top{top_n}，{n_epochs} 个合成 epoch，{len(ds)} 个预测日：")
    p("")
    p("   IC档位  rho     实测IC          AUC       日均超额均值   跨epoch标准差  相对波动")
    p("  " + "-" * 84)

    summary = []
    for ic_lv in target_ics:
        for rho in rhos:
            aucs, excs, ics = [], [], []
            for t in range(n_epochs):
                s = scorer_synthetic_epoch(panel, target_ic=ic_lv, rho=rho,
                                           seed=seed + 977 * t)
                r = evaluate(panel, s, top_n=top_n, n_mc=0, seed=seed)
                aucs.append(panel_auc(panel, s))
                excs.append(r['excess_mean'])
                ics.append(r['ic_mean'])
            aucs, excs, ics = map(np.array, (aucs, excs, ics))
            sd = excs.std(ddof=1)
            rel = sd / abs(excs.mean()) if abs(excs.mean()) > 1e-9 else float('nan')
            summary.append((ic_lv, rho, excs.mean(), sd, rel))
            p(f"   {ic_lv:.3f}   {rho:.1f}   {ics.mean():+.4f}±{ics.std(ddof=1):.4f}   "
              f"{aucs.mean():.4f}     {excs.mean()*100:+7.3f}%      "
              f"{sd*100:6.3f}%      {rel*100:6.1f}%")
    p("")

    p("  >>> 读法三条：")
    p("      (a) AUC 那一列是同一批打分器在「收益中位数以上」二分标签上的表现。")
    p("          **把这张表当成 IC ↔ AUC 的换算尺**：你们的标签是另外五条硬规则，")
    p("          所以不能直接对号入座，但量级不会差到另一个世界。")
    p("      (b) 相对波动那一列是关键：信号越弱（IC 越小），同样的能力差异")
    p("          换来的收益差异占比越大 —— 弱信号模型天然更容易「看着忽好忽坏」。")
    p("      (c) 但注意这张表的标准差是在**全窗口**上测的。叠加实验 2 的窗口噪声后，")
    p("          实际观察到的跨 epoch 波动会远大于这里。")

    # 把用户的观察换算成 IC，给出可直接对照的数字
    rev_ic = full['ic_mean']
    rev_ex = full['excess_mean']
    if np.isfinite(rev_ic) and abs(rev_ic) > 1e-6:
        per01 = rev_ex / rev_ic * 0.01
        p("")
        p(f"  【换算尺】本池标定：20 日反转 IC {rev_ic:.4f} ↔ 日均超额 {rev_ex*100:+.3f}%")
        p(f"      即在本池，IC 每高 0.01，日均超额约高 {per01*100:.3f}%。")
        p(f"      你看到的 0.30% ~ 1.40% 区间，换算成 IC 约为 "
          f"{0.003/per01/100:.3f} ~ {0.014/per01/100:.3f}。")

    # ---------- 实验 4：标签上的 AUC 到底值多少钱 ----------
    p("\n" + "=" * 70)
    p("实验 4 —— 在你们的五条规则标签上 AUC=0.70，收益应该有多少")
    p("=" * 70)
    la = label_audit(panel)
    p(f"  标签基本面（config.generate_label 五条硬规则）：")
    p(f"    正样本率        {la['pos_rate']*100:5.1f}%")
    p(f"    正样本日均收益  {la['pos_ret']*100:+7.3f}%")
    p(f"    负样本日均收益  {la['neg_ret']*100:+7.3f}%")
    p(f"    → 标签的理论收益空间 {(la['pos_ret']-la['neg_ret'])*100:+.3f}%  "
      f"（完美分类器能拿到的上限）")
    p("")
    p("  构造一批**只**在这个标签上有判别力、其余成分全是独立噪声的打分器，")
    p("  用它标定「标签上的判别力 → 收益」的换算关系（{n} 次平均）："
      .format(n=n_epochs_label))
    p("")
    p(f"   目标AUC   实测AUC(±真实标准误)   日均超额        95%区间跨度      超额t值")
    p("  " + "-" * 80)
    calib = []
    for t_auc in target_aucs:
        aucs, auses, excs, ts = [], [], [], []
        for k in range(n_epochs_label):
            s = scorer_on_label(panel, target_auc=t_auc, seed=seed + 31 * k)
            r = evaluate(panel, s, top_n=top_n, n_mc=0, seed=seed)
            m, se = panel_auc_label(panel, s)
            aucs.append(m)
            auses.append(se)
            excs.append(r['excess_mean'])
            ts.append(r['t_stat'])
        aucs, auses, excs, ts = map(np.array, (aucs, auses, excs, ts))
        calib.append((t_auc, float(aucs.mean()), float(auses.mean()),
                      float(excs.mean()), float(ts.mean())))
        p(f"    {t_auc:.2f}     {aucs.mean():.4f} ± {auses.mean():.4f}      "
          f"{excs.mean()*100:+7.3f}%   [{excs.min()*100:+.3f}%,{excs.max()*100:+.3f}%]   "
          f"{ts.mean():+7.2f}")
    p("")
    p("  >>> 读法三条：")
    p("      (a) **置信区间那一 ± 号务必看**。它是按交易日算的标准误，不是按样本条数。")
    p("          AUC 0.70 若配 ±0.015，那么 0.69 和 0.71 的差异完全不可区分 ——")
    p("          「卡在 70% 上不去」很可能不是天花板，是分辨力不够。")
    p("      (b) 日均超额这一列是**保守下界**：构造的打分器除标签关联外全是噪声，")
    p("          没有真实模型那种跨期一致的结构化偏好，Top-N 会被稀释。")
    p("          真实模型若有同等判别力，收益应**不低于**此值。")
    p(f"      (c) 这是 {len(ds)} 天全窗口的结果。窗口缩到 {window} 天要再叠加")
    p(f"          实验 2 的 ±{1.96*se_w*100:.2f}% 抖动。")

    # ---------- 结论与替代指标 ----------
    p("\n" + "=" * 70)
    p("由此得到的四条结论")
    p("=" * 70)
    p("  1. epoch 间的收益波动，绝大部分不是模型变了，是**测不准**。")
    p(f"     证据：一个零自由度的固定因子，只换样本期就能在 "
      f"{ex.min()*100:+.2f}% ~ {ex.max()*100:+.2f}% 之间跳 —— 比你们模型跳得更凶。")
    p("  2. 别再用「某次 run 的日均收益」选 epoch。改用 IC / ICIR：")
    p("     它们衡量全截面排序，方差比尾部收益小一到两个量级。")
    p("  3. 全局 AUC 也不是好目标。它平均了整个分布，而收益只吃最右尾部。")
    p("     改看**头部区分度**：截断 AUC（只在高分前 5% 内部算）、Precision@Top-N、NDCG@N。")
    p("  4. 唯一能压缩这些噪声的办法是**增加有效样本**：")
    p("     拉长预测期 > 增加每日选股只数 > 调参。Shortening 验收窗口会反向放大噪声。")

    return {'window_rows': rows, 'full': full, 'boot_ci': (lo, hi),
            'epoch_summary': summary, 'auc_calibration': calib,
            'label_audit': la}


# ==================== 头部指标 + epoch 对比 ====================
# 动机：全局 AUC 平均了整个分布，而收益只吃最右尾部 —— 两者信息落差极大
# （实验 3 已量化）。选 epoch、选阈值、决定是否上线，都应该看头部指标。

def head_auc(y, s, frac=0.05):
    """截断 AUC：只在打分最高的前 frac 比例内部计算。

    全局 AUC 里绝大部分判别力来自分布中段（样本多），而中段排序对 Top-N 组合
    毫无影响。截断后看的是「头部区还分得清好坏吗」，这才是决定收益的部分。
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < 20:
        return float('nan')
    thr = np.nanquantile(s, 1.0 - frac)
    m = s >= thr
    if m.sum() < 10:
        return float('nan')
    return auc_binary(y[m], s[m])


def precision_at_n(y, s, n):
    """Top-N 命中率。n 取实际每天选股数。"""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < n + 1:
        return float('nan')
    top = np.argsort(s)[::-1][:n]
    return float((y[top] > 0).mean())


def ndcg_at_n(y, s, n):
    """Top-N 的 NDCG。相对 precision 多考虑了排序位置：把好股排第一比排第十值钱。"""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) < n + 1:
        return float('nan')
    top = np.argsort(s)[::-1][:n]
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = float((y[top] * disc).sum())
    ideal = np.sort(y)[::-1][:n]
    idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else float('nan')


def align_scores(panel, pred):
    """把 pred[date][code] = score 对齐到 panel 的行顺序，缺失填 NaN。

    panel 内部的行顺序是按数据库 SELECT 顺序排的，外部预测必须按 code 对齐，
    不能靠位置对齐 —— 那是本项目已经踩过的坑（污染区间错位 74.5% 漏排）。
    """
    out = {}
    for d in panel.dates:
        rec = panel.day[d]
        if d not in pred:
            continue
        idx = {c: i for i, c in enumerate(rec['code'])}
        s = np.full(len(rec['code']), np.nan)
        for c, v in pred[d].items():
            i = idx.get(c)
            if i is not None:
                s[i] = v
        out[d] = s
    return out


def epoch_metrics(panel, aligned, top_n=4, ret_key='ret3', head_frac=0.05,
                  n_boot=500, seed=42):
    """单个 epoch 的全套指标。aligned 是 align_scores 的输出。"""
    ics, h_aucs, precs, ndcgs, excess, dates = [], [], [], [], [], []
    for d, s in sorted(aligned.items()):
        rec = panel.day[d]
        if rec['n'] < MIN_STOCKS_PER_DAY:
            continue
        r = rec[ret_key]
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < MIN_STOCKS_PER_DAY:
            continue
        ss, rr = s[ok], r[ok]
        y = (rr > np.nanmedian(rr)).astype(float)   # 截面中位数二分
        dates.append(d)
        ics.append(spearman_ic(ss, rr))
        h_aucs.append(head_auc(y, ss, head_frac))
        precs.append(precision_at_n(y, ss, top_n))
        ndcgs.append(ndcg_at_n(y, ss, top_n))
        k = min(top_n, len(ss))
        excess.append(rr[np.argsort(ss)[::-1][:k]].mean() - rr.mean())

    dates = np.array(dates)
    excess = np.array(excess)
    ics = np.array(ics)
    h_aucs = np.array(h_aucs)
    boot = moving_block_bootstrap(excess, block=20, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'dates': dates,
        'excess': excess,
        'ics': ics,
        'excess_mean': float(excess.mean()),
        'excess_ci': (float(lo), float(hi)),
        't_stat': newey_west_t(excess, lag=DataConfig.FUTURE_DAYS - 1),
        'ic_mean': float(np.nanmean(ics)),
        'icir': float(np.nanmean(ics) / np.nanstd(ics, ddof=1)),
        'head_auc': float(np.nanmean(h_aucs)),
        'precision': float(np.nanmean(precs)),
        'ndcg': float(np.nanmean(ndcgs)),
    }


def compare_epochs(panel, preds, top_n=4, ret_key='ret3', head_frac=0.05,
                   n_boot=500, seed=42):
    """对比多个 epoch 的预测，判断它们是不是真的不同。

    preds: dict[name] -> dict[date][code] = score

    输出两张表：
      1. 逐 epoch 指标（含超额收益的 bootstrap 置信区间）
      2. 两两「日超额序列」相关系数 —— 这张才是重点

    第二张表回答一个具体问题：**两个 epoch 每天选出的是同一批股吗**。
    corr 高（>0.5）= 模型输出稳定，收益差异主要是常数平移；
    corr 低（<0.2）= 模型在不同 epoch 之间几乎重选了一批股，
                     此时「挑收益最高的 epoch」等价于在噪声里挑最大值，必然失望。
    """
    p = print
    names = list(preds.keys())
    mets = {}
    for name in names:
        mets[name] = epoch_metrics(panel, align_scores(panel, preds[name]),
                                   top_n=top_n, ret_key=ret_key,
                                   head_frac=head_frac, n_boot=n_boot, seed=seed)

    p("\n" + "=" * 92)
    p(f"逐 epoch 指标对比（每日 Top{top_n}，头部区间前 {head_frac*100:.0f}%）")
    p("=" * 92)
    p(f"  {'epoch':<16}{'超额均值':>10}{'95%CI':>22}{'t值':>7}"
      f"{'IC':>9}{'ICIR':>7}{'头部AUC':>9}{'P@N':>8}{'NDCG':>8}")
    p("  " + "-" * 92)
    for n in names:
        m = mets[n]
        lo, hi = m['excess_ci']
        p(f"  {n:<16}{m['excess_mean']*100:>+9.3f}%"
          f"[{lo*100:>+7.3f}%,{hi*100:>+7.3f}%]{m['t_stat']:>7.2f}"
          f"{m['ic_mean']:>+9.4f}{m['icir']:>7.2f}{m['head_auc']:>9.4f}"
          f"{m['precision']:>8.3f}{m['ndcg']:>8.3f}")
    p("")
    p("  置信区间大面积重叠 = 这些 epoch 的收益差异**统计上不可区分**。")

    if len(names) > 1:
        p("\n" + "=" * 92)
        p("两两日超额序列相关系数 —— 模型在不同 epoch 间是否选了同一批股")
        p("=" * 92)
        p(f"  {'':<16}" + "".join(f"{n[:9]:>11}" for n in names))
        for a in names:
            row = f"  {a[:15]:<16}"
            for b in names:
                if a == b:
                    row += f"{'—':>11}"
                    continue
                xa, xb = mets[a]['excess'], mets[b]['excess']
                na, nb = mets[a]['dates'], mets[b]['dates']
                common = np.intersect1d(na, nb)
                ia = np.searchsorted(na, common)
                ib = np.searchsorted(nb, common)
                va, vb = xa[ia], xb[ib]
                if len(common) < 10 or va.std() == 0 or vb.std() == 0:
                    row += f"{'n/a':>11}"
                    continue
                row += f"{np.corrcoef(va, vb)[0,1]:>11.3f}"
            p(row)
        p("")
        p("  >>> 读法：corr < 0.2 意味着两个 epoch 每天选的股票几乎完全不同，")
        p("      收益高点很可能是某次抽样碰上的 —— 挑它会系统性高估。")
        p("      corr > 0.5 说明模型输出稳定，此时按 IC / 头部AUC 挑 epoch 才安全。")

    return mets


def calibrate_label_auc(panel, top_n=4, target_aucs=(0.55, 0.60, 0.65, 0.70, 0.75),
                        n_rep=8, seed=42, observed=None):
    """标定「项目标签上的 AUC」到「收益尺度」的换算表（独立入口，跑得快）。

    为什么需要这张表：本项目优化的目标是 config.generate_label 的五条硬规则，
    但验收指标是止损策略下的实现收益。中间这条链路从来没有人标定过 ——
    所以「AUC 0.70」到底该对应百分之几，此前没有任何依据可以判断。

    做法是构造一批**只**在标签上有给定判别力、其余成分全是独立噪声的打分器，
    然后按正常口径跑评估。得到的是**保守下界**：真实模型若有同等判别力，
    通常还多一层跨期一致的结构化偏好，收益应不低于此值。

    observed: 你实测的日均超额（小数），给出它落在这张表的哪个 AUC 档位。
    """
    p = print
    la = label_audit(panel)
    p("\n" + "=" * 74)
    p(f"标签 AUC → 收益 换算表（每日 Top{top_n}，{len(panel.dates)} 个预测日）")
    p("=" * 74)
    p(f"  标签正样本率      {la['pos_rate']*100:5.1f}%")
    p(f"  正样本日均收益    {la['pos_ret']*100:+7.3f}%")
    p(f"  负样本日均收益    {la['neg_ret']*100:+7.3f}%")
    p(f"  理论上限（完美分类器） {(la['pos_ret']-la['neg_ret'])*100:+.3f}%")
    p("")
    p(f"   {'目标AUC':>8}{'实测AUC':>18}{'日均超额':>11}{'t值':>8}{'对收益IC':>10}")
    p("  " + "-" * 60)

    rows = []
    for t_auc in target_aucs:
        aucs, auses, excs, ics, ts = [], [], [], [], []
        for k in range(n_rep):
            s = scorer_on_label(panel, target_auc=t_auc, seed=seed + 31 * k)
            r = evaluate(panel, s, top_n=top_n, n_mc=0, seed=seed)
            m, se = panel_auc_label(panel, s)
            aucs.append(m); auses.append(se)
            excs.append(r['excess_mean']); ics.append(r['ic_mean']); ts.append(r['t_stat'])
        rows.append((t_auc, float(np.mean(aucs)), float(np.mean(auses)),
                     float(np.mean(excs)), float(np.mean(ics)), float(np.mean(ts))))
        a, ase, e, ic, t = rows[-1][1:]
        p(f"   {t_auc:>8.2f}{a:>10.4f} ± {ase:.4f}{e*100:>+9.3f}%{t:>8.2f}{ic:>+10.4f}")

    p("")
    p("  AUC 那一列的 ± 是按**交易日**算的真实标准误，不是按样本条数。")

    if observed is not None:
        xs = [r[3] for r in rows]
        aus = [r[0] for r in rows]
        p("")
        if observed <= xs[0]:
            p(f"  你实测的日均超额 {observed*100:+.3f}% —— 低于表中最低档 "
              f"（AUC {aus[0]:.2f} 对应 {xs[0]*100:+.3f}%），")
            p(f"  换算过来标签上的有效判别力大约只有 AUC {aus[0]-0.03:.2f} 以下。")
        elif observed >= xs[-1]:
            p(f"  你实测的日均超额 {observed*100:+.3f}% —— 高于表中最高档 "
              f"（AUC {aus[-1]:.2f} 对应 {xs[-1]*100:+.3f}%）。")
        else:
            eq = float(np.interp(observed, xs, aus))
            p(f"  你实测的日均超额 {observed*100:+.3f}% 落在这张表里，")
            p(f"  按本表换算，相当于标签上的有效判别力 **AUC ≈ {eq:.3f}**。")
            p(f"  若你的模型在同一批评测上报告的是 AUC 0.70，中间这 "
              f"{0.70-eq:.3f} 的落差就是判别力没有转化成收益的部分。")
    return rows


def pooled_auc(panel, scores):
    """把整个评估期所有天的样本混在一起算一次 AUC（**pooled 口径**）。

    这是最容易骗人的算法：不同日期之间的系统性差异（比如某些天普遍上涨）
    会被计入判别力。模型只要学会分辨「今天是好日子还是坏日子」，pooled AUC
    就会很高 —— 但这对**每天该买哪几只**毫无帮助，策略用不上。

    正确口径是逐日算再平均（daily AUC）。两者的差就是「被浪费掉的判别力」。
    """
    ys, ss = [], []
    for d in panel.dates:
        rec = panel.day[d]
        y = rec['label']
        s = np.asarray(scores[d], dtype=np.float64)
        ok = np.isfinite(y) & np.isfinite(s)
        if ok.sum() < 5:
            continue
        ys.append(y[ok])
        ss.append(s[ok])
    if not ys:
        return float('nan')
    return auc_binary(np.concatenate(ys), np.concatenate(ss))


def scorer_pooled_trap(panel, day_strength=1.0, stock_strength=0.0, seed=42):
    """构造一个「pooled AUC 很高但完全不能选股」的打分器。

        score = day_strength * (当日标签均值) + stock_strength * (个股真信号) + 噪声

    第一项只对「今天整体怎样」有判别力，对「今天买哪只」毫无贡献。
    现实里模型很容易学到这一项：只要网络里有任何跨样本共享的成分，
    同一天的样本就会带上共同的偏移。
    """
    from statistics import NormalDist
    inv = NormalDist().inv_cdf
    rng = np.random.default_rng(seed)
    delta = np.sqrt(2.0) * inv(0.70)      # 个股信号的强度，使自身 AUC ≈ 0.70

    by_label_mean = {}
    for d in panel.dates:
        y = panel.day[d]['label']
        by_label_mean[d] = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
    g = np.array([by_label_mean[d] for d in panel.dates])
    g = (g - g.mean()) / (g.std() + 1e-12)

    scores = {}
    for i, d in enumerate(panel.dates):
        y = panel.day[d]['label']
        n = len(y)
        scores[d] = (day_strength * g[i]
                     + stock_strength * (delta * (y - 0.5) + rng.standard_normal(n))
                     + rng.standard_normal(n))
    return scores


def demo_pooled_trap(panel, top_n=4, seed=42):
    """演示 pooled AUC 与 daily AUC 的落差，以及它对收益的影响。

    读法：看最后两列。同样一个「pooled AUC 有 0.65 甚至 0.70」的打分器，
    只要判别力来自跨日而非截面，日均超额就塌到接近 0。
    """
    p = print
    p("\n" + "=" * 78)
    p("陷阱演示 —— pooled AUC 高，并不代表能选股")
    p("=" * 78)
    p("  构造一批打分器：")
    p("    score = 日强度 × (当日标签均值) + 个股强度 × (个股真信号) + 噪声")
    p("  第一项只对「今天整体好不好」有判别力（每天都一样），第二项才是对「买哪只」有用的。")
    p("")
    p(f"   {'日强度':>6}{'个股强度':>10}{'pooled AUC':>13}{'daily AUC':>12}"
      f"{'日均超额':>11}{'超额t值':>9}")
    p("  " + "-" * 62)
    rows = []
    for ds_, ss_ in ((1.0, 0.0), (1.0, 0.3), (1.0, 1.0), (0.0, 1.0), (0.0, 0.5)):
        sc = scorer_pooled_trap(panel, day_strength=ds_, stock_strength=ss_, seed=seed)
        r = evaluate(panel, sc, top_n=top_n, n_mc=0, seed=seed)
        pa = pooled_auc(panel, sc)
        da = panel_auc_label(panel, sc)[0]
        rows.append((ds_, ss_, pa, da, r['excess_mean'], r['t_stat']))
        p(f"   {ds_:>6.1f}{ss_:>10.1f}{pa:>13.4f}{da:>12.4f}"
          f"{r['excess_mean']*100:>+9.3f}%{r['t_stat']:>9.2f}")
    p("")
    p("  >>> 读法：第 1 行是最纯粹的陷阱 —— pooled AUC 明显大于 0.5，")
    p("      daily AUC 却是 0.5（同一天内完全没判别力），超额自然接近 0。")
    p("      第 4 行是干净的真信号：pooled 与 daily 一致，收益实打实。")
    p("")
    p("  **行动**：去查你们的 AUC 是 pooled 还是 daily。若是 pooled 且远高于 daily，")
    p("      那么「AUC 卡在 70%」里的很大一部分本来就不属于可用的选股能力 ——")
    p("      调模型结构不会让它变成收益，只能先把这两个口径分开再看。")
    return rows


def compare_to_baseline(res, baseline, baseline_name='20日反转'):
    """模型结果 vs 朴素因子基线。返回逐项对比。"""
    b = baseline.get(baseline_name)
    if b is None:
        return None
    return {
        'baseline_name': baseline_name,
        'excess': (res['excess_mean'], b['excess_mean']),
        'ic': (res['ic_mean'], b['ic_mean']),
        'icir': (res['icir'], b['icir']),
        't': (res['t_stat'], b['t_stat']),
        'beats': res['icir'] > b['icir'] and res['excess_mean'] > b['excess_mean'],
    }


def verdict(res):
    """四条件判读。返回 (是否检出信号, 未通过的理由列表)。"""
    reasons = []
    if not (res['ic_mean'] > IC_OK):
        reasons.append(f"IC {res['ic_mean']:+.4f} 未过 {IC_OK}")
    if not (res['icir'] > ICIR_OK):
        reasons.append(f"ICIR {res['icir']:+.3f} 未过 {ICIR_OK}")
    if not (abs(res['t_stat']) > T_OK):
        reasons.append(f"t {res['t_stat']:+.2f} 未过 {T_OK}")
    if not (res['mc_pctl'] > MC_PCTL_OK):
        reasons.append(f"随机对照百分位 {res['mc_pctl']:.1f} 未过 {MC_PCTL_OK}")
    return (len(reasons) == 0), reasons


def report(res, show_months=True):
    p = print
    p(f"\n  [{res['label']}]  {res['n_days']} 个预测日，每日 Top{res['top_n']}，"
      f"收益口径 {'止损' if res['ret_key'] == 'ret3_stop' else '无止损'}")
    p("  " + "-" * 62)
    p(f"    选股日均收益:      {res['sel_mean']*100:+7.3f}%")
    p(f"    全市场等权基准:    {res['bench_mean']*100:+7.3f}%   <- 什么都不做随便买的收益")
    p(f"    日均超额:          {res['excess_mean']*100:+7.3f}%   <- 模型的真实贡献")
    p(f"    超额标准差:        {res['excess_std']*100:7.3f}%")
    p(f"    超额 t 值 (NW):    {res['t_stat']:+7.2f}")
    p(f"    日胜率:            {res['win_rate']*100:7.1f}%")
    p(f"    日均 IC:           {res['ic_mean']:+7.4f}")
    p(f"    ICIR:              {res['icir']:+7.3f}")
    p(f"    IC>0 天数占比:     {res['ic_pos_rate']*100:7.1f}%")
    p(f"    随机对照:          {res['mc_mean']*100:+7.3f}% ± {res['mc_std']*100:.3f}%")
    p(f"    模型超额百分位:    {res['mc_pctl']:7.1f}%   <- 随机分布中的位置")
    if show_months and res['months']:
        if len(res['months']) > 24:
            # 长窗口按月铺开会刷屏，按年汇总
            p("    分年超额:          " + "  ".join(
                f"{y}年{v*100:+.2f}%" for y, v in res['years'].items()))
        else:
            p("    分月超额:          " + "  ".join(
                f"{_fmt_ym(m)}{v*100:+.2f}%" for m, v in res['months'].items()))
    ok, reasons = verdict(res)
    if ok:
        p("    >>> 判定：检出可交易信号（四项门槛全部通过）")
    else:
        p("    >>> 判定：未检出可交易信号 —— " + "；".join(reasons))


# ==================== 自检：校准尺子 ====================

def selftest(panel, top_n=4, n_mc=200):
    """用已知答案的三个打分器校准评估协议。

    这是「先造尺子、再校准尺子」的校准步骤。三个期望：
      随机   → 必须判「无信号」，且超额、IC 都贴近 0
      注入   → 必须判「有信号」（协议得检得出东西）
      反转   → 必须判「有信号」，并给出 Transformer 的及格线
    """
    print("\n" + "=" * 66)
    print("评估协议自检 —— 用已知答案校准尺子")
    print("=" * 66)

    cases = []

    rnd = evaluate(panel, scorer_random(panel), top_n=top_n, n_mc=n_mc,
                   label="随机打分（期望：无信号）")
    report(rnd)
    # 判据用 IC 与 t，不用裸超额：裸超额的量级随窗口长度和横截面大小漂移，
    # 而「IC≈0 且 t 不显著」才是「没有截面信息」这件事本身。
    ok_rnd = (not verdict(rnd)[0]) and abs(rnd['ic_mean']) < 0.01 and abs(rnd['t_stat']) < T_OK
    cases.append(("随机打分应判无信号", ok_rnd,
                  f"超额 {rnd['excess_mean']*100:+.3f}%, IC {rnd['ic_mean']:+.4f}, "
                  f"t {rnd['t_stat']:+.2f}"))

    inj = evaluate(panel, scorer_injected(panel), top_n=top_n, n_mc=n_mc,
                   label="注入信号（期望：有信号，用于证明协议检得出）")
    report(inj)
    cases.append(("注入信号应判有信号", verdict(inj)[0],
                  f"超额 {inj['excess_mean']*100:+.3f}%, IC {inj['ic_mean']:+.4f}"))

    rev = evaluate(panel, scorer_reversal(panel, 20), top_n=top_n, n_mc=n_mc,
                   label="20日反转（Transformer 的及格线）")
    report(rev)
    if len(panel.dates) >= MIN_DAYS_FOR_ANOMALY:
        cases.append(("20日反转应判有信号", verdict(rev)[0],
                      f"超额 {rev['excess_mean']*100:+.3f}%, IC {rev['ic_mean']:+.4f}"))
    else:
        # 窗口太短时已知异象也会被噪声淹没，此时不检出**不是协议的错**。
        # 实测：2026H1（136 天）反转 IC +0.0017；2023–2025（724 天）IC +0.0517。
        print(f"    [注意] 仅 {len(panel.dates)} 个预测日 < {MIN_DAYS_FOR_ANOMALY}，"
              f"已知异象可能被噪声淹没，本条不作通过条件。")
        print(f"           要校准这条，请用更长的窗口重跑，例如 --begin 20230101 --end 20251231")

    mom = evaluate(panel, scorer_momentum(panel, 5), top_n=top_n, n_mc=n_mc,
                   label="5日动量（对照组）")
    report(mom)
    cases.append(("5日动量仅作参考，不作为通过条件", True,
                  f"超额 {mom['excess_mean']*100:+.3f}%, IC {mom['ic_mean']:+.4f}"))

    print("\n" + "=" * 66)
    print("自检结论")
    print("=" * 66)
    all_ok = True
    for name, ok, detail in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  ({detail})")
        all_ok = all_ok and ok

    if all_ok:
        print("\n  尺子校准通过：随机无信号、注入有信号、已知异象可检出。")
        print("  此后用同一协议跑出的数字才有资格被相信。")
    else:
        print("\n  尺子未通过校准 —— 先修协议，不要用它的输出做任何判断。")
    return all_ok, {'random': rnd, 'injected': inj, 'reversal': rev, 'momentum': mom}


# ==================== 入口 ====================

def _fmt_date(d):
    s = str(int(d))
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def main():
    ap = argparse.ArgumentParser(
        description="Phase 0 可信评估协议",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法：
  python src/eval_protocol.py --selftest      先校准尺子（不需要模型）
  python src/eval_protocol.py --scan          跑朴素因子，给出模型必须超越的门槛
        """)
    ap.add_argument('--begin', type=int, default=DEFAULT_BEGIN,
                    help=f"评估起始日（默认 {DEFAULT_BEGIN} = 测试集起点）")
    ap.add_argument('--end', type=int, default=None, help="评估截止日（默认数据最新日）")
    ap.add_argument('--top-n', type=int, default=DataConfig.MAX_SELECT_PER_DAY or 4,
                    help="每日选股数量（默认 MAX_SELECT_PER_DAY）")
    ap.add_argument('--mc', type=int, default=200, help="随机对照重复次数")
    ap.add_argument('--selftest', action='store_true', help="校准评估协议本身")
    ap.add_argument('--scan', action='store_true', help="扫描朴素因子基线")
    ap.add_argument('--stability', action='store_true',
                    help="稳定性诊断：epoch 间波动是模型变了还是测量噪声")
    ap.add_argument('--window', type=int, default=136,
                    help="稳定性诊断的滑动窗口长度（预测日数，默认 136≈7个月）")
    ap.add_argument('--calib', action='store_true',
                    help="标定「标签 AUC → 收益」换算表（快，便于换 top_n 反复对照）")
    ap.add_argument('--trap', action='store_true',
                    help="演示 pooled AUC 与 daily AUC 的落差陷阱")
    ap.add_argument('--observed', type=float, default=None,
                    help="你实测的日均超额（如 0.008 表示 0.8%%），与换算表对照")
    ap.add_argument('--save', action='store_true', help="把结果存成 JSON 基线档案")
    args = ap.parse_args()

    print("\n" + "=" * 66)
    print("Phase 0 可信评估协议")
    print("=" * 66)
    print(f"  区间: {_fmt_date(args.begin)} ~ {'最新' if args.end is None else _fmt_date(args.end)}")
    print(f"  池: {CURRENT_POOL}")
    print(f"  每日选股: Top{args.top_n}（每日 Top-N，不使用全局阈值 → 无回看）")
    print(f"  成本: 不建模（按 hzy 2026-09-26 决定，本协议只回答有无超额）")

    print("\n加载面板...")
    panel = load_panel(begin=args.begin, end=args.end, verbose=True)

    if args.selftest:
        ok, _ = selftest(panel, top_n=args.top_n, n_mc=args.mc)
        return 0 if ok else 1

    if args.calib:
        print("\n" + "=" * 66)
        print("标签 AUC → 收益 换算标定")
        print("=" * 66)
        calibrate_label_auc(panel, top_n=args.top_n, observed=args.observed)
        return 0

    if args.trap:
        print("\n" + "=" * 66)
        print("AUC 口径陷阱")
        print("=" * 66)
        demo_pooled_trap(panel, top_n=args.top_n)
        return 0

    if args.stability:
        print("\n" + "=" * 66)
        print("稳定性诊断 —— epoch 间的收益波动，是模型变了还是测不准")
        print("=" * 66)
        diagnose_stability(panel, top_n=args.top_n, window=args.window,
                           n_epochs=20, n_boot=1000)
        return 0

    if args.scan:
        print("\n" + "=" * 66)
        print("朴素因子基线扫描 —— 这些是 Transformer 必须超越的门槛")
        print("=" * 66)
        results = {}
        for w in (20, 10, 5):
            r_no, r_st, _ = report_dual(panel, scorer_reversal(panel, w),
                                        f"{w}日反转", args.top_n, n_mc=args.mc)
            results[f'reversal{w}'] = r_no
        for w in (5, 20):
            r_no, r_st, _ = report_dual(panel, scorer_momentum(panel, w),
                                        f"{w}日动量", args.top_n, n_mc=args.mc)
            results[f'momentum{w}'] = r_no
        results['random'] = evaluate(panel, scorer_random(panel), top_n=args.top_n,
                                     n_mc=args.mc, label="随机打分")
        report(results['random'], show_months=False)

        if args.save:
            tag = f"{args.begin}_{args.end or 'latest'}"
            save_report(results, os.path.join(
                DataConfig.OUTPUT_DIR, f"eval_baseline_{tag}.json"))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
