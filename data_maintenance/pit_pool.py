"""
Point-in-Time 股票池（修正幸存者偏差与前视偏差）

问题背景：
原 `select.py` 用「最新一天」的市值与 ST 状态筛一次，把结果套用到 2010-2026 全历史，
导致池里只剩「活到今天、今天市值仍在区间内、今天不是 ST」的股票。两种偏差混在一起：

- **幸存者偏差**：退市/长停的股票整只消失，模型从未见过它们（实测约 190 只，8.2%）
- **前视偏差**：用今天的市值/ST 状态去决定十年前该不该看某只股票
  （实测约 646 只市值口径 + 169 只 ST 口径）

> ⚠️ 报告里「仅 PIT 有 N 只 = 44.7%」这种写法把两种偏差混成了一个数，会造成误判
> （例如以为该优先补 ST 历史，实际上市值前视是大头）。main() 里已经把两者拆开打印。

本模块的做法：
- 逐日重算流通市值： market_cap(t) = amount(t) * 100 / exchange(t)
- 逐日判定是否满足筛选条件（前缀 / 市值区间 / 当日可交易 / **当日非 ST**）
- 只输出**满足条件的连续区间**（interval），而不是逐日打标
  → 表体量小，下游 JOIN 一天到位
- 退市股不会被剔除：它只在自己的存续期内属于池，之后自动退出

ST 判定来自 `stock_status.is_st`（逐日外部事实，见 fetch_status.py）。
该表不存在时**直接拒绝运行**，不做「没有 ST 信息也照跑」的降级 ——
降级的结果就是把「当时是 ST」的股票当成正常股票放进池里。

产出表 pool_membership：
    (stock_code, start_date, end_date, market_cap_median, reason)
    end_date = 99999999 表示「其数据末段仍符合条件」（注：退市股若在退市前仍符合条件，
    也会被记为 99999999。这一列只用于 downstream 的日期区间 JOIN，不要拿它判断
    「今天该选谁」）

用法：
    python -m data_maintenance.pit_pool --write
    python -m data_maintenance.pit_pool --min-run-days 60 --write
"""

import argparse
import os
import sqlite3
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DataConfig  # noqa: E402  只读常量，不引入训练链路
from .contract import DEFAULT_DB, MAX_WORKERS  # noqa: E402
from . import provenance  # noqa: E402

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
    """单只股票的 PIT 判定，返回 [ (start, end, mc_median, reason), ... ]

    入池条件（全部必须**在该日**成立）：
      1. 代码前缀在有效范围内
      2. 当日可交易（volume>0 且 amount>0）
      3. 当日**不是 ST**（来自 stock_status.is_st，逐日外部事实）
      4. 21 日滚动中位数估算的流通市值落在 [min_cap, max_cap]

    条件 3 曾经缺失：那时 PIT 池只判市值与可交易性，会把「当时是 ST」的股票
    当成正常股票放进历史池 —— 那不是修偏差，是把一种偏差换成另一种。
    """
    db_path, stock_code, min_cap, max_cap, prefixes, min_run = args
    try:
        con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
        df = pd.read_sql_query(
            "SELECT date, amount, exchange, volume FROM stock_daily "
            "WHERE stock_code = ? ORDER BY date ASC",
            con, params=(stock_code,)
        )
        st = pd.read_sql_query(
            "SELECT date, tradestatus, is_st FROM stock_status "
            "WHERE stock_code = ? ORDER BY date ASC",
            con, params=(stock_code,)
        )
        con.close()
        if df.empty:
            return stock_code, []

        # 前缀过滤（不满足则该股永不入池）
        if prefixes and not any(stock_code.startswith(p) for p in prefixes):
            return stock_code, []

        if st.empty:
            # 没有逐日状态 → 无法执行 ST 过滤。**保守丢弃**：
            # 宁可这只股票不进池，也不要把「可能是 ST」的股票混进来
            # （这正是原实现缺少 ST 判定时犯的错）。
            return stock_code, [(-3, -3, 0.0, 'no_status')]

        # 按日期对齐外部状态（外部表可能含库外日期，stock_daily 也可能有库外缺口）
        st = st.set_index('date')
        dates_arr = df['date'].values
        is_st = st['is_st'].reindex(dates_arr).fillna(0).values.astype(bool)
        st_tradable = st['tradestatus'].reindex(dates_arr).fillna(0).values.astype(bool)

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
        eligible = tradable & st_tradable & ~is_st & in_range

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

        out = [(s, e, med, f'cap∈[{min_cap/1e8:.0f}亿,{max_cap/1e8:.0f}亿] 且非ST')
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
    ap.add_argument('--workers', type=int, default=MAX_WORKERS,
                    help=f'并行度（默认 {MAX_WORKERS}）')
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    min_cap = DataConfig.MARKET_CAP_MIN
    max_cap = DataConfig.MARKET_CAP_MAX
    prefixes = DataConfig.VALID_STOCK_PREFIXES

    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    codes = [r[0] for r in con.execute('SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    try:
        n_status = con.execute('SELECT COUNT(DISTINCT stock_code) FROM stock_status').fetchone()[0]
    except sqlite3.OperationalError:
        con.close()
        print('✗ stock_status 表不存在 —— PIT 池需要逐日 isST 才能排除「当时是 ST」的股票。')
        print('  先跑：python -m data_maintenance.fetch_status --fetch')
        sys.exit(1)
    con.close()
    if n_status < len(codes) * 0.95:
        print(f'✗ stock_status 只覆盖 {n_status:,}/{len(codes):,} 只，不足以做 ST 过滤。')
        print('  先补齐：python -m data_maintenance.fetch_status --fetch --only-missing')
        sys.exit(1)

    print(f'股票数={len(codes):,}  市值区间=[{min_cap/1e8:.0f}亿, {max_cap/1e8:.0f}亿]')
    print(f'前缀={prefixes}  最短区间={args.min_run_days} 天  并行度={args.workers}')
    print(f'ST 状态覆盖={n_status:,} 只（逐日 isST 已接入入池条件）')

    t0 = time.time()
    results = {}
    with Pool(args.workers) as pool:
        tasks = [(db_path, c, min_cap, max_cap, prefixes, args.min_run_days) for c in codes]
        for i, (code, runs) in enumerate(pool.imap_unordered(_pit_one, tasks, chunksize=16), 1):
            results[code] = runs
            if i % 1000 == 0:
                print(f'  进度 {i}/{len(codes)} ({i/len(codes)*100:.0f}%)')

    errs = [(c, r) for c, rs in results.items() for r in rs if r[0] == -1]
    no_status = [c for c, rs in results.items() if any(r[0] == -3 for r in rs)]
    rows = [(c, s, e, m, r) for c, rs in results.items() for (s, e, m, r) in rs if s > 0]

    elapsed = time.time() - t0
    print(f'\n耗时 {elapsed:.1f}s')
    print(f'入池股票数: {sum(1 for c, rs in results.items() if any(r[0] > 0 for r in rs)):,} / {len(codes):,}')
    print(f'区间总数:   {len(rows):,}')
    if no_status:
        print(f'[!] {len(no_status):,} 只因无逐日状态被保守丢弃（示例 {no_status[:5]}）')
    if errs:
        print(f'[!] 出错 {len(errs)} 只，示例: {errs[:3]}')

    # --- 与旧池对比：这是偏差的量化（注意口径差异，见下方说明） ---
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    old = {r[0] for r in con.execute(
        "SELECT stock_code FROM stock_pool WHERE pool_type='selected' AND is_active=1")}
    late = dict(con.execute('SELECT stock_code, MAX(date) FROM stock_daily GROUP BY stock_code'))
    con.close()
    new = {c for c, rs in results.items() if any(r[0] > 0 for r in rs)}
    only_new = new - old
    # 把「仅 PIT 有」拆成两类，因为它们性质完全不同：
    #   已退市/长停 = 真·幸存者偏差；仍在交易 = 旧池「用今天的市值/ST 套历史」的前视偏差
    delisted = [c for c in only_new if late.get(c, 0) < 20250701]
    print(f'\n旧 selected 池: {len(old):,} 只')
    print(f'PIT 池覆盖:     {len(new):,} 只')
    print(f'  仅 PIT 有: {len(only_new):,} 只')
    print(f'    ├─ 数据已停止更新（退市/长停）→ **真·幸存者偏差**: {len(delisted):,} 只 '
          f'({len(delisted)/max(len(old),1)*100:.1f}% of 旧池)')
    print(f'    └─ 仍在交易 → 「用今天的市值/ST 套历史」的前视偏差: '
          f'{len(only_new)-len(delisted):,} 只')
    print(f'  仅旧池有(PIT 下从未符合条件): {len(old - new):,} 只')

    if args.write:
        con = sqlite3.connect(db_path)
        try:
            con.executescript(SCHEMA_SQL)
            con.execute('DELETE FROM pool_membership')
            con.executemany(
                'INSERT OR REPLACE INTO pool_membership '
                '(stock_code, start_date, end_date, market_cap_median, reason) VALUES (?,?,?,?,?)',
                rows)
            provenance.record(key=provenance.KEY_POOL_PIT, scope=f'scope=all, {len(codes)} 只候选',
                              rows=len(rows),
                              detail=f'入池 {len(new)} 只; 无状态丢弃 {len(no_status)}; 出错 {len(errs)}',
                              con=con)
            con.commit()
        finally:
            con.close()
        print(f'\n已写入 pool_membership: {len(rows):,} 行')
        print(f'provenance 已登记: {provenance.KEY_POOL_PIT}')
    else:
        print('\n(未写入。加 --write 持久化 pool_membership)')


if __name__ == '__main__':
    main()
