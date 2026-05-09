"""
MA 均线偏离度特征计算模块

从 data_field.py 重构而来，使用 SQLite 替代 CSV 文件。

计算并填充 m5/m10/m20 列：
- m5:  (close - MA5)  / MA5
- m10: (close - MA10) / MA10
- m20: (close - MA20) / MA20
"""

import numpy as np
from multiprocessing import Pool, cpu_count
from typing import List

from .database import DatabaseManager


MA_WINDOWS = [5, 10, 20]
MA_COL_NAMES = ['m5', 'm10', 'm20']


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

        feature_records = list(zip(
            dates.tolist(),
            m5.tolist(),
            m10.tolist(),
            m20.tolist()
        ))

        db.update_features(stock_code, feature_records)
        db.close()

        return (stock_code, 'ok')
    except Exception as e:
        return (stock_code, f'error: {e}')


def compute_features(db: DatabaseManager, pool_type: str = 'selected',
                     stock_codes: List[str] = None, force: bool = False):
    """
    为指定池的股票计算 MA 特征

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

    print(f"\n计算 MA 特征: {len(stock_codes)} 只股票 (pool={pool_type})")

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
