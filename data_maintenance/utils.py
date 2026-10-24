"""
数据标准化工具模块

统一处理从不同数据源（Baostock / AKShare）获取的原始 DataFrame，
输出标准化的行情数据格式，消除 update.py / check.py 中的重复逻辑。
"""

import datetime

import pandas as pd
import numpy as np
from typing import Optional

# 标准输出列
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close',
                     'amount', 'volume', 'exchange', 'vwap']


# ==================== 取数窗口收口 ====================

def latest_closed_trading_date() -> int:
    """当前时刻最近一个「已收盘的交易日」（YYYYMMDD 整数）。

    规则：
    1. A 股 15:00 收盘，数据源通常 18:00 后才有当日数据；此前视为未更新 → 回退一天。
    2. 周六 / 周日回退到周五。
    3. 法定节假日不穷举，留给服务端判断。

    **为什么所有取数路径都必须走这里**：数据源在盘中会返回当天**尚未收盘的半根 K 线**。
    一旦入库，之后所有增量更新都从 last_date+1 起拉，这行错误数据**永远不会再被覆盖** ——
    2026-05-22 的一次更新就是这样把 626 只股票的盘中快照永久写进库的。
    `update.py` 的抓取路径已收口；`check.py` 的**修复路径**曾直接用
    `datetime.datetime.now()`，而修复恰恰是唯一会写这种永久脏行的地方。
    """
    now = datetime.datetime.now()
    if now.hour < 18:
        now -= datetime.timedelta(days=1)
    weekday = now.weekday()          # 0=Mon ... 6=Sun
    if weekday == 5:                 # 周六 → 周五
        now -= datetime.timedelta(days=1)
    elif weekday == 6:               # 周日 → 周五
        now -= datetime.timedelta(days=2)
    return int(now.strftime('%Y%m%d'))


def bs_date(compact: int) -> str:
    """20260731 → '2026-07-31'（数据源日期参数的格式）"""
    return f'{compact // 10000}-{compact // 100 % 100:02d}-{compact % 100:02d}'


def normalize_stock_df(
    df: pd.DataFrame,
    source: str = 'baostock',
    volume_scale_factor: float = 1.0
) -> Optional[pd.DataFrame]:
    """标准化股票行情 DataFrame。

    统一处理：
    1. AKShare 中文列名 → 英文
    2. 日期 YYYY-MM-DD → int(YYYYMMDD)
    3. 数值类型转换 + fillna
    4. turn → exchange（换手率）
    5. VWAP 计算
    6. 不做行过滤 —— 非法行统一由 DatabaseManager 的门禁拦截并报告

    Args:
        df: 原始 DataFrame
        source: 'baostock' 或 'akshare'
        volume_scale_factor: AKShare 成交量需 ×100（手→股）

    Returns:
        标准 9 列 DataFrame，或 None（输入为空）
    """
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    # --- 列重命名（AKShare 中文→英文）---
    if source == 'akshare':
        df = df.rename(columns={
            '日期': 'date', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close',
            '成交量': 'volume', '成交额': 'amount', '换手率': 'turn'
        })

    # --- 日期处理 ---
    # 判据用「是不是整数列」而不是「是不是 object」：pandas 3.x 起字符串列默认推断为
    # `str` dtype 而非 `object`，按 dtype == object 判断会让日期串原样进入 astype(int)，
    # 抛 `invalid literal for int() with base 10: '2026-06-03'`。
    if not pd.api.types.is_integer_dtype(df['date']):
        df['date'] = df['date'].astype(str).str.replace('-', '', regex=False)
        df = df[df['date'].str.strip() != '']
        if len(df) == 0:
            return None
    df['date'] = df['date'].astype(int)

    # --- 数值转换 ---
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
    if volume_scale_factor != 1.0:
        df['volume'] = df['volume'] * volume_scale_factor

    # --- exchange（换手率）---
    if 'turn' in df.columns:
        df['turn'] = pd.to_numeric(df['turn'], errors='coerce')
        df['exchange'] = df['turn'].fillna(0.0).astype(float)
    elif 'exchange' in df.columns:
        df['exchange'] = pd.to_numeric(df['exchange'], errors='coerce').fillna(0.0)
    else:
        df['exchange'] = 0.0

    # --- VWAP ---
    df['vwap'] = df['amount'] / df['volume'].replace(0, float('nan'))
    df['vwap'] = df['vwap'].fillna(df['close'])

    # --- 清洗：这里**不做任何行过滤** ---
    #
    # 历史上这里有两行静默过滤（dropna(OHLC) + volume>0），与门禁拦的是同一批行，
    # 但不打印、不计数 —— 门禁汇总因此会谎报「无非法行被剔除」，而数据已经丢了。
    # 「哪些行非法」只有一处定义（quality.HARD_RULES），过滤只发生在
    # DatabaseManager 的门禁里（带报告）。本函数只做重塑：列名 / 类型 / 量纲。

    return df[STANDARD_COLUMNS]
