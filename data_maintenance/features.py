"""
衍生特征计算模块

计算并填充以下列：
- m5:  (close - MA5)  / MA5
- m10: (close - MA10) / MA10
- m20: (close - MA20) / MA20
- dif:       (EMA12 - EMA26) / close       （MACD快线偏离度）
- dea:       EMA9(DIF) / close             （MACD信号线偏离度）
- macd_hist: 2 * (DIF - DEA) / close       （MACD柱状图偏离度）
- bb_upper:  (close - UPPER) / close       （收盘价相对布林上轨偏离）
- bb_lower:  (LOWER - close) / close       （布林下轨相对收盘价偏离）
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List

from .database import DatabaseManager


MA_WINDOWS = [5, 10, 20]
MA_COL_NAMES = ['m5', 'm10', 'm20']

# MACD 参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 布林带参数
BB_WINDOW = 20
BB_STD_MULT = 2


def compute_ma_features(closes: np.ndarray, window: int) -> np.ndarray:
    """
    计算均线偏离度

    Args:
        closes: 收盘价数组（时间正序）
        window: 均线窗口大小

    Returns:
        result: (close - MA) / MA 数组
    """
    n = len(closes)
    ma = np.empty(n, dtype=np.float64)

    cumsum = np.empty(n + 1, dtype=np.float64)
    cumsum[0] = 0
    np.cumsum(closes, out=cumsum[1:])

    for i in range(n):
        if i >= window:
            ma[i] = (cumsum[i] - cumsum[i - window]) / window
        else:
            left_part = closes[0:i]
            deficit = window - i
            right_end = min(i + 1 + deficit, n)
            right_part = closes[i + 1:right_end]
            combined = np.concatenate([left_part, right_part])
            if len(combined) > 0:
                ma[i] = np.mean(combined)
            else:
                ma[i] = closes[i]

    return np.where(ma > 0, (closes - ma) / ma, 0.0)


def compute_ema(prices: np.ndarray, window: int) -> np.ndarray:
    """
    计算指数移动平均（EMA）

    递推公式: EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
    alpha = 2 / (window + 1)

    前期数据不足时使用左侧已有数据 + 右侧借数据补齐窗口（与 MA 特征一致）。

    Args:
        prices: 价格数组（时间正序）
        window: EMA 窗口大小

    Returns:
        EMA 数组
    """
    n = len(prices)
    alpha = 2.0 / (window + 1)
    ema = np.empty(n, dtype=np.float64)

    # 初始值：用窗口内可用数据做 SMA 作为 EMA 起点
    if n >= window:
        ema[0] = np.mean(prices[:window])
    else:
        ema[0] = np.mean(prices)

    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema


def compute_macd_features(closes: np.ndarray) -> tuple:
    """
    计算 MACD 衍生特征（无量纲化，除以 close）

    Args:
        closes: 收盘价数组（时间正序）

    Returns:
        (dif, dea, macd_hist) 三个数组，均为 np.float32
        - dif:       (EMA12 - EMA26) / close
        - dea:       EMA9(DIF) / close
        - macd_hist: 2 * (DIF - DEA) / close
    """
    ema_fast = compute_ema(closes, MACD_FAST)
    ema_slow = compute_ema(closes, MACD_SLOW)

    dif_raw = ema_fast - ema_slow
    dea_raw = compute_ema(dif_raw, MACD_SIGNAL)
    macd_hist_raw = 2.0 * (dif_raw - dea_raw)

    safe_closes = np.where(closes > 0, closes, 1.0)

    dif = (dif_raw / safe_closes).astype(np.float32)
    dea = (dea_raw / safe_closes).astype(np.float32)
    macd_hist = (macd_hist_raw / safe_closes).astype(np.float32)

    return dif, dea, macd_hist


def compute_bb_features(closes: np.ndarray) -> tuple:
    """
    计算布林带偏离特征（无量纲化）

    布林带 = MA20 ± k × STD20（k=2）
    - bb_upper: (close - UPPER) / close  > 0 表示突破上轨
    - bb_lower: (LOWER - close) / close  > 0 表示跌破下轨

    前期数据不足时采用与 compute_ma_features 一致的借数据策略。

    Args:
        closes: 收盘价数组（时间正序）

    Returns:
        (bb_upper, bb_lower) 两个数组，均为 np.float32
    """
    n = len(closes)
    window = BB_WINDOW
    k = BB_STD_MULT

    ma = np.empty(n, dtype=np.float64)
    std = np.empty(n, dtype=np.float64)

    for i in range(n):
        if i >= window:
            segment = closes[i - window:i]
        else:
            left_part = closes[0:i]
            deficit = window - i
            right_end = min(i + 1 + deficit, n)
            right_part = closes[i + 1:right_end]
            segment = np.concatenate([left_part, right_part]) if (len(left_part) > 0 or len(right_part) > 0) else closes[i:i+1]

        ma[i] = np.mean(segment)
        std[i] = np.std(segment, ddof=0)

    upper = ma + k * std
    lower = ma - k * std

    safe_closes = np.where(closes > 0, closes, 1.0)
    bb_upper = ((closes - upper) / safe_closes).astype(np.float32)
    bb_lower = ((lower - closes) / safe_closes).astype(np.float32)

    return bb_upper, bb_lower


def _process_single_stock(args):
    """处理单只股票的特征计算（多进程 worker）"""
    db_path, stock_code = args
    try:
        db = DatabaseManager(db_path)
        df = db.get_stock_data(stock_code, chronological=True)

        if len(df) == 0:
            db.close()
            return (stock_code, 'empty')

        closes = df['close'].values.astype(np.float64)
        dates = df['date'].values

        m5 = compute_ma_features(closes, 5).astype(np.float32)
        m10 = compute_ma_features(closes, 10).astype(np.float32)
        m20 = compute_ma_features(closes, 20).astype(np.float32)

        dif, dea, macd_hist = compute_macd_features(closes)

        bb_upper, bb_lower = compute_bb_features(closes)

        feature_records = list(zip(
            dates.tolist(),
            m5.tolist(),
            m10.tolist(),
            m20.tolist(),
            dif.tolist(),
            dea.tolist(),
            macd_hist.tolist(),
            bb_upper.tolist(),
            bb_lower.tolist()
        ))

        db.update_features(stock_code, feature_records)
        db.close()

        return (stock_code, 'ok')
    except Exception as e:
        return (stock_code, f'error: {e}')


def compute_features(db: DatabaseManager, pool_type: str = 'selected',
                     stock_codes: List[str] = None, force: bool = False):
    """
    为指定池的股票计算 MA + MACD 特征

    Args:
        db: 数据库管理器
        pool_type: 股票池类型 ('all' 或 'selected')
        stock_codes: 指定股票列表，None 表示处理整个池
        force: 是否强制重新计算（即使已有特征）
    """
    if stock_codes is None:
        if force:
            stock_codes = db.get_pool_stocks(pool_type)
        else:
            stock_codes = db.get_stocks_missing_features(pool_type)

    if not stock_codes:
        print("✓ 所有股票的特征已计算，无需处理")
        return

    print(f"\n计算 MA + MACD + BB 特征: {len(stock_codes)} 只股票 (pool={pool_type})")

    db_path = db.db_path
    file_args = [(db_path, code) for code in stock_codes]
    num_workers = min(cpu_count(), 8)

    with Pool(num_workers) as pool:
        results = pool.map(_process_single_stock, file_args)

    ok_count = sum(1 for _, status in results if status == 'ok')
    error_count = sum(1 for _, status in results if status.startswith('error'))
    skip_count = sum(1 for _, status in results if status == 'empty')

    print(f"✓ 特征计算完成: {ok_count} 成功, {skip_count} 跳过(空数据), {error_count} 出错")

    if error_count > 0:
        print("\n错误详情:")
        for code, status in results:
            if status.startswith('error'):
                print(f"  {code}: {status}")
