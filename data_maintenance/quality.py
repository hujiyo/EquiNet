"""
行情数据质量规则集（纯函数，不依赖数据库）

设计原则：
1. 每条规则只做一件事，可单独单测
2. 规则只"报告"，不"修复" —— 修复策略由调用方决定
3. 所有阈值显式暴露，便于按板块/日期调整

术语对齐下游（src/data.py）：
- 上下文 CONTEXT_LENGTH = 45：样本 t 的输入是 [t-44, t]
- 前瞻 FUTURE_DAYS = 3 + BUFFER_DAY = 1：样本 t 的标签落在 [t+1, t+4]
- 故「日期 g 的数据有问题」会污染的样本位置是 t ∈ [g-4, g+44]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# ==================== 常量 ====================

# 样本污染半径（与 src/config.py 保持一致，此处冗余以免 quality 依赖 config）
CONTEXT_LENGTH = 45
FUTURE_DAYS = 3
BUFFER_DAY = 1
LABEL_BACKSPAN = FUTURE_DAYS + BUFFER_DAY      # 4：标签侧回溯
CONTAMINATE_LEFT = LABEL_BACKSPAN              # g-4
CONTAMINATE_RIGHT = CONTEXT_LENGTH - 1         # g+44

# 涨跌幅限制（按板块 / 日期）
LIMIT_MAIN = 0.10        # 主板 ±10%
LIMIT_GEM = 0.20         # 创业板 ±20%（2020-08-24 起）
LIMIT_STAR = 0.20        # 科创板 ±20%（设立起）
GEM_20PCT_FROM = 20200824
LIMIT_TOLERANCE = 1.15   # 容差：允许超过理论上限 15%（应对四舍五入/数据源误差）

# 新股上市初期无涨跌幅限制，这段时间的巨大波动是正常的
IPO_FREE_DAYS = 5

# 僵尸价格判定
ZOMBIE_MIN_RUN = 5       # 连续多少天收盘价完全不变即判为僵尸段

# 占位垃圾行判定
PLACEHOLDER_MAX_AMOUNT = 1.0     # 成交额 <= 1 元
PLACEHOLDER_MIN_VOLUME = 1.0     # 成交量 <= 1 股


# ==================== 数据结构 ====================

@dataclass
class Issue:
    """一条数据质量问题"""
    stock_code: str
    date: int
    issue_type: str
    detail: str

    def as_row(self) -> tuple:
        return (self.stock_code, self.date, self.issue_type, self.detail)


# ==================== 板块 / 涨跌幅限制 ====================

def price_limit(stock_code: str, date: int) -> float:
    """返回该股票在该日期的涨跌幅限制（小数形式）

    依据：
    - 科创板 688xxx：设立起 ±20%
    - 创业板 300xxx：2020-08-24 起 ±20%，此前 ±10%
    - 北交所 4xxxxx / 8xxxxx：±30%（本项目暂不涉及）
    - 其余（主板）：±10%
    """
    if stock_code.startswith('688'):
        return LIMIT_STAR
    if stock_code.startswith('300'):
        return LIMIT_GEM if date >= GEM_20PCT_FROM else LIMIT_MAIN
    if stock_code.startswith(('4', '8')):
        return 0.30
    return LIMIT_MAIN


# ==================== 单条规则 ====================

def check_ohlc_consistency(df: pd.DataFrame) -> np.ndarray:
    """OHLC 自洽：low <= min(open, close)、high >= max(open, close)、low <= high"""
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    bad = (h < l) | (c > h) | (c < l) | (o > h) | (o < l)
    return np.asarray(bad, dtype=bool)


def check_positive_prices(df: pd.DataFrame) -> np.ndarray:
    """价格必须为正"""
    for col in ('open', 'high', 'low', 'close'):
        v = df[col].values
        if np.any(v <= 0):
            pass
    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    return (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)


def check_volume(df: pd.DataFrame) -> np.ndarray:
    """成交量 / 成交额必须为正；零成交是停牌快照，不应作为 K 线入库"""
    return (df['volume'].values <= 0) | (df['amount'].values <= 0)


def check_vwap_range(df: pd.DataFrame) -> np.ndarray:
    """vwap 必须落在 [low, high] 内。越界说明 amount/volume 量纲不一致（数据源混用指纹）"""
    vwap = df['vwap'].values
    lo = np.minimum(df['low'].values, df['high'].values)
    hi = np.maximum(df['low'].values, df['high'].values)
    tol = np.abs(hi) * 1e-3 + 1e-9
    return (vwap < lo - tol) | (vwap > hi + tol)


def check_placeholder(df: pd.DataFrame) -> np.ndarray:
    """占位垃圾行：成交额/成交量小到不可能真实（如 OHLC 全 99.0、amount=1.0）"""
    return (df['amount'].values <= PLACEHOLDER_MAX_AMOUNT) | \
           (df['volume'].values <= PLACEHOLDER_MIN_VOLUME)


def check_price_anomaly(df: pd.DataFrame, stock_code: str) -> np.ndarray:
    """价格异常跳变：不复权数据下的除权日 / 错价

    判定：|日涨跌幅| > 板块涨跌幅上限 × 容差
    豁免：新股上市前 IPO_FREE_DAYS 个交易日（注册制下无涨跌幅限制）
    """
    close = df['close'].values.astype(np.float64)
    dates = df['date'].values
    n = len(close)
    bad = np.zeros(n, dtype=bool)
    if n < 2:
        return bad

    prev = close[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ret = np.where(prev > 0, (close[1:] - prev) / prev, 0.0)

    limits = np.array([price_limit(stock_code, int(d)) for d in dates[1:]])
    thresh = limits * LIMIT_TOLERANCE

    hit = np.abs(ret) > thresh
    # 豁免新股上市初期
    hit[:IPO_FREE_DAYS] = False

    bad[1:] = hit
    return bad


def find_zombie_runs(df: pd.DataFrame, stock_code: str) -> List[Issue]:
    """僵尸价格：连续多日 OHLC 完全不变（停牌期用最后一刻 K 线填充）

    真正的停牌应该是「没有记录」，而不是「复制一条记录」。
    """
    issues: List[Issue] = []
    if len(df) < ZOMBIE_MIN_RUN:
        return issues

    o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
    flat = (o == h) & (h == l) & (l == c)          # 一字（含填充）
    dates = df['date'].values

    # 找出连续 flat 段，且段内收盘价完全不变
    i = 0
    n = len(flat)
    while i < n:
        if not flat[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flat[j + 1] and c[j + 1] == c[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= ZOMBIE_MIN_RUN:
            issues.append(Issue(
                stock_code, int(dates[i]), 'zombie_price',
                f'收盘价 {c[i]:.2f} 连续 {run_len} 日 OHLC 零变化 '
                f'({int(dates[i])}~{int(dates[j])})，疑似停牌期填充'
            ))
            # 段内每日各记一条，便于下游逐日排除
            for k in range(i + 1, j + 1):
                issues.append(Issue(
                    stock_code, int(dates[k]), 'zombie_price',
                    f'属于自 {int(dates[i])} 起连续 {run_len} 日零变化段'
                ))
        i = j + 1
    return issues


# ==================== 组合扫描 ====================

def scan_stock(df: pd.DataFrame, stock_code: str) -> List[Issue]:
    """对单只股票的完整历史执行全部规则

    Args:
        df: 至少包含 date/open/high/low/close/volume/amount/vwap，按 date 升序
        stock_code: 股票代码（用于板块判定）

    Returns:
        Issue 列表
    """
    if df is None or len(df) == 0:
        return []

    issues: List[Issue] = []
    dates = df['date'].values

    def _emit(mask: np.ndarray, kind: str, fmt) -> None:
        idx = np.flatnonzero(mask)
        for i in idx:
            issues.append(Issue(stock_code, int(dates[i]), kind, fmt(i)))

    _emit(check_ohlc_consistency(df), 'ohlc_inconsistent',
          lambda i: f"O={df['open'].values[i]} H={df['high'].values[i]} "
                    f"L={df['low'].values[i]} C={df['close'].values[i]}")

    _emit(check_positive_prices(df), 'nonpositive_price',
          lambda i: f"close={df['close'].values[i]}")

    _emit(check_volume(df), 'zero_volume',
          lambda i: f"volume={df['volume'].values[i]} amount={df['amount'].values[i]}")

    _emit(check_placeholder(df), 'placeholder_row',
          lambda i: f"amount={df['amount'].values[i]} volume={df['volume'].values[i]} "
                    f"close={df['close'].values[i]}")

    _emit(check_vwap_range(df), 'vwap_out_of_range',
          lambda i: f"vwap={df['vwap'].values[i]} low={df['low'].values[i]} "
                    f"high={df['high'].values[i]}")

    _emit(check_price_anomaly(df, stock_code), 'price_anomaly',
          lambda i: f"较前日跳变，close={df['close'].values[i]}（不复权下疑似除权或错价）")

    issues.extend(find_zombie_runs(df, stock_code))
    return issues


# ==================== 污染区间 ====================

def contaminate_span(g_date: int) -> tuple:
    """返回问题日 g 会污染的样本位置区间 [left, right]（闭区间，以样本末日 t 表示）

    样本 t 的输入 = [t-44, t]，标签 = [t+1, t+4]
    - 输入被污染 <=> t-44 <= g <= t  <=> t ∈ [g, g+44]
    - 标签被污染 <=> t+1 <= g <= t+4 <=> t ∈ [g-4, g-1]
    合并得 t ∈ [g-4, g+44]
    """
    return (g_date - CONTAMINATE_LEFT, g_date + CONTAMINATE_RIGHT)


def build_exclusions(issues: List[Issue]) -> Dict[str, List[tuple]]:
    """把问题日展开成「样本位置排除区间」，按股票分组

    Returns:
        {stock_code: [(left, right, reason), ...]}
    """
    out: Dict[str, List[tuple]] = {}
    for iss in issues:
        left, right = contaminate_span(iss.date)
        out.setdefault(iss.stock_code, []).append((left, right, iss.issue_type))
    for code in out:
        out[code].sort()
    return out


def gate_dataframe(df: pd.DataFrame, stock_code: str) -> tuple:
    """写入门禁：在入库前剔除「结构性不可能」的行

    只做上下文无关的硬校验（不需要历史数据）：
    - OHLC 不自洽 / 价格非正
    - 零成交量（停牌快照，不是 K 线）
    - 占位垃圾行（amount<=1 或 volume<=1）

    vwap 越界与价格跳变属于「软问题」（可能是数据源量纲差异或真实除权），
    不在门禁里删除，交由 audit.py 记录为 data_issues 供人工决策。

    Returns:
        (df_clean, dropped)  dropped 为被剔除行的 Issue 列表
    """
    if df is None or len(df) == 0:
        return df, []

    n = len(df)
    dates = df['date'].values

    hard = check_ohlc_consistency(df) | check_positive_prices(df) | \
        check_volume(df) | check_placeholder(df)

    if not hard.any():
        return df, []

    dropped = []
    for i in np.flatnonzero(hard):
        kinds = []
        if check_ohlc_consistency(df)[i]:
            kinds.append('ohlc_inconsistent')
        if check_positive_prices(df)[i]:
            kinds.append('nonpositive_price')
        if check_volume(df)[i]:
            kinds.append('zero_volume')
        if check_placeholder(df)[i]:
            kinds.append('placeholder_row')
        dropped.append(Issue(
            stock_code, int(dates[i]), '|'.join(kinds),
            f"O={df['open'].values[i]} H={df['high'].values[i]} L={df['low'].values[i]} "
            f"C={df['close'].values[i]} vol={df['volume'].values[i]} amt={df['amount'].values[i]}"
        ))

    return df.loc[~hard].reset_index(drop=True), dropped


def merge_spans(spans: List[tuple]) -> List[tuple]:
    """合并重叠区间，减少下游排除判定的开销。reason 合并为去重后的字符串"""
    if not spans:
        return []
    spans = sorted(spans)
    merged = []
    cur_l, cur_r, reasons = spans[0][0], spans[0][1], {spans[0][2]}
    for l, r, reason in spans[1:]:
        if l <= cur_r + 1:
            cur_r = max(cur_r, r)
            reasons.add(reason)
        else:
            merged.append((cur_l, cur_r, '|'.join(sorted(reasons))))
            cur_l, cur_r, reasons = l, r, {reason}
    merged.append((cur_l, cur_r, '|'.join(sorted(reasons))))
    return merged
