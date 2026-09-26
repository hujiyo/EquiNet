"""
行情数据质量规则集（纯函数，不依赖数据库）

设计原则：
1. 每条规则只做一件事，可单独单测
2. 规则只"报告"，不"修复" —— 修复策略由调用方决定
3. 所有阈值显式暴露，便于按板块/日期调整
4. **本模块只产出「哪天有问题」（Issue），不产出「哪些样本不能用」**
   —— 后者是索引空间的事，由消费方用 `contract.SAMPLING` 换算。
   历史缺陷：本模块曾把序号空间的 ±4/±44 偏移直接加在日期整数上
   （`g + 44` 得到的 20150955 不是合法日期），下游按日期查回来时
   右界永远落在「g 所在月月末」，导致 74.5% 的污染样本漏排。
   单位只在一个地方定义，才能避免这类错误。
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# ==================== 常量 ====================
# 注意：采样几何（上下文长度/前瞻天数/污染半径）**不在这里定义**。
# 唯一定义处是 contract.SAMPLING；本模块只判定「哪天有问题」，不碰索引空间。

# 涨跌幅限制（按板块 / 日期）
LIMIT_MAIN = 0.10        # 主板 ±10%
LIMIT_GEM = 0.20         # 创业板 ±20%（2020-08-24 起）
LIMIT_STAR = 0.20        # 科创板 ±20%（设立起）
GEM_20PCT_FROM = 20200824
# 注意：不要用「百分比容差」判涨跌停（曾用 LIMIT_TOLERANCE=1.15）——
# 低价股一个最小变动价位就是百分之几，正确做法是比较舍入后的涨跌停价，
# 见 check_price_anomaly。

# 新股上市初期无涨跌幅限制，这段时间的巨大波动是正常的
IPO_FREE_DAYS = 5

# 僵尸价格判定
ZOMBIE_MIN_RUN = 5       # 连续多少天收盘价完全不变即判为僵尸段

# 占位垃圾行判定
PLACEHOLDER_MAX_AMOUNT = 1.0     # 成交额 <= 1 元
PLACEHOLDER_MIN_VOLUME = 1.0     # 成交量 <= 1 股

# 复牌判定（**退化口径**，仅在 audit 未提供 `is_resume` 列时使用）：
# 正常最长假期（春节/国庆连休）约 8 个自然日，超过它基本只能是停牌。
# 但长假本身会被误判 → 只作为兜底，正式口径是真实交易日历（见 find_resume_gaps）。
RESUME_GAP_DAYS = 8


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
    """OHLC 必须为正"""
    o, h, l, c = (df['open'].values, df['high'].values,
                  df['low'].values, df['close'].values)
    return (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)


# 参与模型输入、因此不允许缺失的列。任何一列非有限数即视为该行不可用。
REQUIRED_FINITE_COLS = ('open', 'high', 'low', 'close', 'vwap', 'volume', 'amount')


def check_nonfinite(df: pd.DataFrame) -> np.ndarray:
    """必需列存在 NULL / NaN / inf。

    为什么必须单独列一条规则：其余规则全是 `<=` / `>=` 这类比较，而 **NaN 参与
    比较恒为 False** —— 一行 OHLC 全是 NULL 的记录会被所有比较型规则静默放过，
    顺利写入数据库（`upsert` 还会把它当成正常行覆盖已有值）。
    守门员不能依赖上游恰好没给 NULL。
    """
    v = np.asarray(df[list(REQUIRED_FINITE_COLS)], dtype=np.float64)
    return ~np.isfinite(v).all(axis=1)


def check_volume(df: pd.DataFrame) -> np.ndarray:
    """成交量 / 成交额必须为正；零成交是停牌快照，不应作为 K 线入库"""
    return (df['volume'].values <= 0) | (df['amount'].values <= 0)


# vwap 越界容差
#   有振幅日：vwap 由 amount/volume 反推，必须落在 [low, high] 内，留千分之一相对容差
#   一字板日（low == high）：整日只有一个成交价，区间退化成一点，
#       此时 vwap 与价格的偏差只反映 amount/volume 的精度。
#       实测 908/919 条越界都发生在一字板日，中位偏差 0.234%、871 条在 1% 以内，
#       而 amount/volume 的整数精度只能解释 0.0002%~0.05% —— 说明这是**源的精度伪报**，
#       不是量纲问题。1% 以上的才可能是真问题（实测 37 条）。
VWAP_TOL_RANGE = 1e-3
VWAP_TOL_FLAT = 0.01


def check_vwap_range(df: pd.DataFrame) -> np.ndarray:
    """vwap 必须与当日价格区间自洽

    本规则原本想抓的是「amount/volume 量纲不一致」（那会产生数量级偏差），
    但原实现对**一字板日**用了和有振幅日一样的千分之一容差，
    于是把 908 条纯精度伪报当成问题报了出来 —— 占当时全部问题的 77%，
    既把真问题淹没，又连带排除掉 2015-2017 一片本来干净的样本。
    一字板日改用 1% 容差（见 VWAP_TOL_FLAT 的依据）。
    """
    vwap = df['vwap'].values
    lo = np.minimum(df['low'].values, df['high'].values)
    hi = np.maximum(df['low'].values, df['high'].values)

    flat = hi <= lo + np.abs(hi) * 1e-9          # 一字：区间退化成一点
    tol = np.where(flat, np.abs(hi) * VWAP_TOL_FLAT,
                   np.abs(hi) * VWAP_TOL_RANGE + 1e-9)

    return (vwap < lo - tol) | (vwap > hi + tol)


def check_placeholder(df: pd.DataFrame) -> np.ndarray:
    """占位垃圾行：成交额/成交量小到不可能真实（如 OHLC 全 99.0、amount=1.0）"""
    return (df['amount'].values <= PLACEHOLDER_MAX_AMOUNT) | \
           (df['volume'].values <= PLACEHOLDER_MIN_VOLUME)


def find_resume_gaps(df: pd.DataFrame) -> np.ndarray:
    """停牌后复牌的首个交易日。

    复牌日的涨跌幅**不受常规涨跌幅限制**（重组类甚至无限制），
    所以它既不是数据错误，也不能用「日涨跌幅 > 板块上限」去判。
    实测：131 条 price_anomaly 里 **113 条（86%）属于这一类**
    （前一记录在 346~538 天前，跳幅 +87%/-72% 配正常量能）。

    判定必须用**真实交易日历**：某行的前一条记录若不等于「市场在该日之前的
    最后一个交易日」，说明该股在中间有停牌。

    不要用「自然日间隔」启发式 —— 实测阈值 3 天会把每个周末都算成停牌
    （86,849 条 vs 真实的 113 条），阈值 8 天又会把春节/国庆长假误判成停牌。
    日历由 audit.py 通过 `df['is_resume']` 列提供（外部事实随数据一起传，
    与 `is_dividend` / `series_truncated` 同一模式）。

    缺列时退化为「自然日间隔 >= RESUME_GAP_DAYS（=8）」，
    该退化口径**偏保守**（宁可少豁免，多报几个复牌跳空），并在自检里可被察觉。
    """
    n = len(df)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out
    if 'is_resume' in df.columns:
        return df['is_resume'].values.astype(bool)
    d = pd.to_datetime(df['date'].astype(str), format='%Y%m%d').values
    gaps = np.zeros(n, dtype=np.int64)
    gaps[1:] = (d[1:] - d[:-1]).astype('timedelta64[D]').astype(np.int64)
    out[1:] = gaps[1:] >= RESUME_GAP_DAYS
    return out


def check_price_anomaly(df: pd.DataFrame, stock_code: str) -> np.ndarray:
    """价格异常跳变：真错价（除权日与复牌日已豁免）

    判定：**连续交易日之间** |日涨跌幅| > 板块涨跌幅上限 × 容差。
    规则必须显式列出「合法例外」，否则会把正常事件报成问题、并把真问题淹没。
    本规则有两类豁免，都是实测逼出来的：

    1. **除权日**（`df['is_dividend'] == 1`）。后复权体系（见 adjust_factor.py）
       落地后，除权造成的跳变在模型输入里已经消失，`stock_daily` 的不复权价
       只是事实层。实测原 3,491 条里 **3,360 条（96.2%）当天恰有除权事件**。

    2. **复牌日**（`find_resume_gaps`）。前一记录相隔多日的跳空是停牌复牌，
       不受常规涨跌幅限制。实测剩下 131 条里 **113 条（86%）属于这一类**。

    两类加起来，原来 3,491 条「问题」里只有个位数是真正需要人看的错价。

    3. **上市初期**（前 IPO_FREE_DAYS 个交易日，注册制下无涨跌幅限制）。
       仅在该股数据未被库起点截断时适用：数据首日恰好是库内最早日期，
       说明这是「数据起始」而不是「上市」（训练池里 960/2307 只如此）。

    豁免上下文通过 DataFrame 上的可选列传入（`is_dividend` / `series_truncated`，
    由 audit.py 提供）。**缺列时不豁免** —— 宁可多报，不可漏报。
    """
    close = df['close'].values.astype(np.float64)
    dates = df['date'].values
    n = len(close)
    bad = np.zeros(n, dtype=bool)
    if n < 2:
        return bad

    prev = close[:-1]

    limits = np.array([price_limit(stock_code, int(d)) for d in dates[1:]])
    # 涨跌停价按交易所规则取「前收 × (1±涨跌幅) 后四舍五入到最小变动价位（0.01 元）」，
    # 再与当日收盘比较 —— **不能用百分比容差判**。
    #
    # 原因：低价股一个 tick 就是百分之几。面值退市股在 0.1~0.3 元区间，
    # 0.26→0.23 恰好是 3 个 tick、也正是 round(0.26*0.9,2)=0.23 的跌停价，
    # 用百分比（±11.5%）判会把正常涨跌停报成跳变 —— 实测 23 条「真错价候选」
    # 里 20 条是这一类，占当前驱动排除量的 70%。
    #
    # 比较时放 1 个 tick + 极小 epsilon：
    #   - 1 tick：交易所的四舍五入无法用二进制浮点精确复现。例：2.55×1.1 的精确值
    #     是 2.805，交易所给 2.81；而 2.55 的浮点表示略小，浮点积 ≈2.80499999...，
    #     round() 给 2.80 —— 恰差一个 tick。实测 627 条「+10.01%~+10.20%」全是这种
    #     合法涨停（中位 +10.07%）。
    #   - epsilon：浮点加法误差会吃掉恰好 1 tick 的容差（2.80+0.01 在 IEEE754 下
    #     略小于字面量 2.81），不加它这 1 tick 等于没加。
    # 真异常（跌 22%、跌 36%、+38%）远超 1 tick，不受影响。
    TICK = 0.01
    EPS = 1e-9
    up = np.round(prev * (1 + limits), 2) + TICK + EPS
    dn = np.round(prev * (1 - limits), 2) - TICK - EPS
    with np.errstate(invalid='ignore'):
        hit = (close[1:] > up) | (close[1:] < dn)

    if 'is_dividend' in df.columns:
        # hit 对应 dates[1:]，除权标记要按同一偏移对齐
        hit &= df['is_dividend'].values[1:] == 0

    # 复牌日豁免（同样对齐到 dates[1:]）
    hit &= ~find_resume_gaps(df)[1:]

    truncated = True
    if 'series_truncated' in df.columns:
        truncated = bool(df['series_truncated'].iloc[0])
    if not truncated:
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


# ==================== 规则注册表 ====================
#
# 门禁（写库前丢弃）与审计（全库记录）**共用这一份规则清单**。
# 历史上两者各列一份，是口径分叉的直接来源：同一种行在一处被拒、在另一处被保留，
# 于是「库里有什么」与「门禁允许什么」永远对不上。
#
# 判定函数签名统一为 detect(df, stock_code) -> bool mask
# 详情函数签名统一为 detail(df, i) -> str

@dataclass(frozen=True)
class Rule:
    kind: str
    detect: object
    detail: object


def _fmt_ohlc(df, i) -> str:
    return (f"O={df['open'].values[i]} H={df['high'].values[i]} "
            f"L={df['low'].values[i]} C={df['close'].values[i]}")


def _fmt_nonfinite(df, i) -> str:
    bad = [c for c in REQUIRED_FINITE_COLS
           if not np.isfinite(np.float64(df[c].values[i]))]
    return '必需列非有限数: ' + ', '.join(bad)


# 硬规则：这些行在任何数据源下都不是一根 K 线，写库前直接丢弃。
#
# 关于 zero_volume 的口径（曾自相矛盾）：门禁拒收 `volume/amount <= 0`，
# 但 8.8 节曾决定「保留停牌记录」。两者取一，本处统一为**拒收**，
# 依据是本文件 `find_zombie_runs` 已经写下的原则：
# 「真正的停牌应该是「没有记录」，而不是「复制一条记录」」。
# 存量的 4,763 行停牌记录属于历史遗留（写入时还没有门禁），
# 由审计报出、由 selfcheck 断言其数量，不再作为「应当保留」的口径。
HARD_RULES = (
    Rule('nonfinite_value', lambda df, code: check_nonfinite(df), _fmt_nonfinite),
    Rule('ohlc_inconsistent', lambda df, code: check_ohlc_consistency(df), _fmt_ohlc),
    Rule('nonpositive_price', lambda df, code: check_positive_prices(df), _fmt_ohlc),
    Rule('zero_volume', lambda df, code: check_volume(df),
         lambda df, i: f"volume={df['volume'].values[i]} amount={df['amount'].values[i]}"),
    Rule('placeholder_row', lambda df, code: check_placeholder(df),
         lambda df, i: f"amount={df['amount'].values[i]} volume={df['volume'].values[i]} "
                       f"close={df['close'].values[i]}"),
)

# 软规则：可能是真实数据（除权、复牌、量纲差异、数据源特性），只记录、不下发删除。
SOFT_RULES = (
    Rule('resume_gap', lambda df, code: find_resume_gaps(df),
         lambda df, i: f"停牌后复牌首日（前一记录 "
                       f"{int(df['date'].values[i-1]) if i else '?'}），"
                       f"跳空属正常事件 —— 与「真错价」分开列，否则真错价会被淹没"),
    Rule('vwap_out_of_range', lambda df, code: check_vwap_range(df),
         lambda df, i: f"vwap={df['vwap'].values[i]} low={df['low'].values[i]} "
                       f"high={df['high'].values[i]}"),
    Rule('price_anomaly', lambda df, code: check_price_anomaly(df, code),
         lambda df, i: f"连续交易日间跳变，close={df['close'].values[i]}"
                       f"（除权日与复牌日已豁免，剩下的才是真错价）"),
)

ALL_RULES = HARD_RULES + SOFT_RULES


# ==================== 组合扫描 ====================

def scan_stock(df: pd.DataFrame, stock_code: str) -> List[Issue]:
    """对单只股票的完整历史执行全部规则（硬 + 软 + 僵尸段）

    Args:
        df: 至少包含 date/open/high/low/close/volume/amount/vwap，按 date 升序。
            ⚠️ **必须是原始（不复权）OHLC**。价格跳变与「一字」判定都以不复权价为
            事实依据：后复权会把除权日抹平（跳变检不出来），同时把复权因子差
            混进 OHLC 恒等关系（一字板判别失真）。
        stock_code: 股票代码（用于板块判定）

    Returns:
        Issue 列表（只到「哪天有问题」，不展开污染区间）
    """
    if df is None or len(df) == 0:
        return []

    issues: List[Issue] = []
    dates = df['date'].values
    for rule in ALL_RULES:
        for i in np.flatnonzero(rule.detect(df, stock_code)):
            issues.append(Issue(stock_code, int(dates[i]), rule.kind,
                                rule.detail(df, int(i))))
    issues.extend(find_zombie_runs(df, stock_code))
    return issues


def gate_dataframe(df: pd.DataFrame, stock_code: str) -> tuple:
    """写入门禁：在入库前剔除「结构性不可能」的行

    应用 `HARD_RULES`（与审计同一份清单，口径不可能分叉）。
    `SOFT_RULES` 不在此处删除 —— 它们可能是真实数据，交给审计记录后由人工决策。

    Returns:
        (df_clean, dropped)  dropped 为被剔除行的 Issue 列表
    """
    if df is None or len(df) == 0:
        return df, []

    dates = df['date'].values
    masks = [(rule, rule.detect(df, stock_code)) for rule in HARD_RULES]
    hard = np.zeros(len(df), dtype=bool)
    for _, m in masks:
        hard |= m

    if not hard.any():
        return df, []

    dropped = []
    for i in np.flatnonzero(hard):
        kinds = [rule.kind for rule, m in masks if m[i]]
        dropped.append(Issue(
            stock_code, int(dates[i]), '|'.join(kinds),
            f"O={df['open'].values[i]} H={df['high'].values[i]} L={df['low'].values[i]} "
            f"C={df['close'].values[i]} vol={df['volume'].values[i]} amt={df['amount'].values[i]}"
        ))

    return df.loc[~hard].reset_index(drop=True), dropped

