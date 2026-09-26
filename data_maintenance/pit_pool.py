"""
Point-in-Time 股票池（修正幸存者偏差）

问题背景：
原 select.py 用「最新一天」的市值与 ST 状态筛一次，把结果套用到 2016-2026 全历史，
导致池里只剩「活到今天且今天市值仍在区间内」的股票 —— 幸存者偏差。

本模块的做法：
- 逐日重算流通市值： market_cap(t) = amount(t) * 100 / exchange(t)
- 逐日判定是否满足筛选条件（前缀 / 市值区间 / 当日可交易）
- 只输出**满足条件的连续区间**（interval），而不是逐日打标
  → 表体量小，下游 JOIN 一天到位
- 退市股不会被剔除：它只在自己的存续期内属于池，之后自动退出

产出表 pool_membership：
    (stock_code, start_date, end_date, market_cap_median, reason)
    end_date = 99999999 表示「存续至今」

用法：
    python -m data_maintenance.pit_pool --write
    python -m data_maintenance.pit_pool --min-run-days 60 --write
"""

import argparse
import os
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DataConfig  # noqa: E402  只读常量，不引入训练链路

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equinet.db')
OPEN_ENDED = 99999999

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pool_membership (
    stock_code        TEXT    NOT NULL,
    start_date        INTEGER NOT NULL,
    end_date          INTEGER NOT NULL,
    market_cap_median REAL,
    reason            TEXT,
    updated_at        TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (stock_code, start_date)
);
CREATE INDEX IF NOT EXISTS idx_membership_span ON pool_membership (start_date, end_date);
"""

# 市值估算的滑动窗口（交易日）
MC_ROLL_WINDOW = 21
MC_FFILL_LIMIT = 250          # 停牌最多回填 250 个交易日
MC_SANITY_MAX = 5e12          # 5 万亿，超过视为估算失败


def _pit_one(args):
    """单只股票的 PIT 判定，返回 [ (start, end, mc_median, reason), ... ]"""
    db_path, stock_code, min_cap, max_cap, prefixes, min_run = args
    try:
        con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
        df = pd.read_sql_query(
            "SELECT date, amount, exchange, volume FROM stock_daily "
            "WHERE stock_code = ? ORDER BY date ASC",
            con, params=(stock_code,)
        )
        con.close()
        if df.empty:
            return stock_code, []

        # 前缀过滤（不满足则该股永不入池）
        if prefixes and not any(stock_code.startswith(p) for p in prefixes):
            return stock_code, []

        # --- 逐日流通市值 ---
        amount = df['amount'].values.astype(np.float64)
        exch = df['exchange'].values.astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            mc = np.where((exch > 0) & (amount > 0), amount * 100.0 / exch, np.nan)
        mc = np.where(mc > MC_SANITY_MAX, np.nan, mc)

        # 停牌/换手率为 0 的日期用前值回填，再做滚动中位数平滑
        mc_s = pd.Series(mc).ffill(limit=MC_FFILL_LIMIT)
        mc_smooth = mc_s.rolling(MC_ROLL_WINDOW, min_periods=3).median()

        # --- 逐日判定 ---
        tradable = (df['volume'].values > 0) & (amount > 0)
        in_range = (mc_smooth.values >= min_cap) & (mc_smooth.values <= max_cap)
        eligible = tradable & in_range

        # --- 连续区间 ---
        dates = df['date'].values
        runs = []
        i = 0
        n = len(eligible)
        while i < n:
            if not eligible[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and eligible[j + 1]:
                j += 1
            if (j - i + 1) >= min_run:
                med = float(np.nanmedian(mc_smooth.values[i:j + 1]))
                runs.append((int(dates[i]), int(dates[j]), med))
            i = j + 1

        # 最后一段若延伸到数据末尾，视为「存续至今」
        if runs and runs[-1][1] == int(dates[-1]):
            s, e, med = runs[-1]
            runs[-1] = (s, OPEN_ENDED, med)

        out = [(s, e, med, f'cap∈[{min_cap/1e8:.0f}亿,{max_cap/1e8:.0f}亿]')
               for s, e, med in runs]
        return stock_code, out
    except Exception as e:
        return stock_code, [(-1, -1, 0.0, f'error: {e}')]


def main():
    ap = argparse.ArgumentParser(description='生成 Point-in-Time 股票池')
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--min-run-days', type=int, default=60,
                    help='连续满足条件不足该天数的片段丢弃（默认 60）')
    ap.add_argument('--workers', type=int, default=min(cpu_count(), 8))
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    min_cap = DataConfig.MARKET_CAP_MIN
    max_cap = DataConfig.MARKET_CAP_MAX
    prefixes = DataConfig.VALID_STOCK_PREFIXES

    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    codes = [r[0] for r in con.execute('SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    con.close()

    print(f'股票数={len(codes):,}  市值区间=[{min_cap/1e8:.0f}亿, {max_cap/1e8:.0f}亿]')
    print(f'前缀={prefixes}  最短区间={args.min_run_days} 天  并行度={args.workers}')

    t0 = time.time()
    results = {}
    with Pool(args.workers) as pool:
        tasks = [(db_path, c, min_cap, max_cap, prefixes, args.min_run_days) for c in codes]
        for i, (code, runs) in enumerate(pool.imap_unordered(_pit_one, tasks, chunksize=16), 1):
            results[code] = runs
            if i % 1000 == 0:
                print(f'  进度 {i}/{len(codes)} ({i/len(codes)*100:.0f}%)')

    errs = [(c, r) for c, rs in results.items() for r in rs if r[0] == -1]
    rows = [(c, s, e, m, r) for c, rs in results.items() for (s, e, m, r) in rs if s != -1]

    elapsed = time.time() - t0
    print(f'\n耗时 {elapsed:.1f}s')
    print(f'入池股票数: {sum(1 for c, rs in results.items() if rs):,} / {len(codes):,}')
    print(f'区间总数:   {len(rows):,}')
    if errs:
        print(f'[!] 出错 {len(errs)} 只，示例: {errs[:3]}')

    # --- 与旧池对比：这是幸存者偏差的量化 ---
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    old = {r[0] for r in con.execute(
        "SELECT stock_code FROM stock_pool WHERE pool_type='selected' AND is_active=1")}
    con.close()
    new = {c for c, rs in results.items() if rs}
    print(f'\n旧 selected 池: {len(old):,} 只')
    print(f'PIT 池覆盖:     {len(new):,} 只')
    print(f'  仅新池有(历史曾符合条件、现已退出): {len(new - old):,} 只')
    print(f'  仅旧池有(PIT 下从未符合条件):       {len(old - new):,} 只')

    if args.write:
        con = sqlite3.connect(db_path)
        con.executescript(SCHEMA_SQL)
        con.execute('DELETE FROM pool_membership')
        con.executemany(
            'INSERT OR REPLACE INTO pool_membership '
            '(stock_code, start_date, end_date, market_cap_median, reason) VALUES (?,?,?,?,?)',
            rows)
        con.commit()
        con.close()
        print(f'\n已写入 pool_membership: {len(rows):,} 行')
    else:
        print('\n(未写入。加 --write 持久化 pool_membership)')


if __name__ == '__main__':
    main()
