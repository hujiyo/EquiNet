"""
数据标准化工具模块

统一处理从不同数据源（Baostock / AKShare）获取的原始 DataFrame，
输出标准化的行情数据格式，消除 update.py / check.py 中的重复逻辑。
"""

import pandas as pd
import numpy as np
from typing import Optional

# 标准输出列
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close',
                     'amount', 'volume', 'exchange', 'vwap']


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
    6. 丢弃 OHLC 含 NaN 的行
    7. 过滤零成交幽灵记录（volume <= 0）

    Args:
        df: 原始 DataFrame
        source: 'baostock' 或 'akshare'
        volume_scale_factor: AKShare 成交量需 ×100（手→股）

    Returns:
        标准 9 列 DataFrame，或 None（输入为空 / 结果为空）
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
    if df['date'].dtype == object:
        df['date'] = df['date'].astype(str).str.replace('-', '')
        df = df[df['date'] != '']
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

    # --- 清洗 ---
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[df['volume'] > 0]  # 过滤零成交幽灵记录（停牌快照等）

    if len(df) == 0:
        return None

    return df[STANDARD_COLUMNS]
