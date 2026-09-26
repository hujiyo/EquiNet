"""
复权因子（后复权）采集与物化

## 为什么只拉因子、不重拉 K 线

实测（2026-09-25，见 tmp_tests/probe_hfq.py）：

    后复权下 close 被调整，而 volume / amount / turn **完全不变**（比值恒 = 1.0）

所以「后复权价 = 不复权价 × backAdjustFactor」，可以在**现有数据上就地派生**，
不需要重拉 5456 只股票的历史 K 线 —— 零重拉风险。

## 因子推导规则（实测确认）

    对日期 d：k(d) = 「dividOperateDate <= d 的最近一条 backAdjustFactor」
    若 d 早于所有除权日，则 k(d) = 1.0

验证：对 600519 / 601003 全历史逐日比对 `close_hfq / close_unadj` 与推导值，
6,086 / 4,763 个交易日**零不符**（最大相对误差 2e-16）。

⚠️ **必须从 1990-01-01 开始查询**。实测 000001：
- start=1990-01-01 → 42 条，首条 1991-04-03 / k=1.000000 ✓
- start=2000-01-01 → 25 条，首条 2000-11-06 / k=28.953 ✗（历史被截断，
  导致 2000 年之前的日期推不出正确因子）

## 产出

- `adjust_factor` 表：除权事件级因子（每只股票约 6~46 条）
- `stock_daily` 新增列：`adj_factor` + `open_adj/high_adj/low_adj/close_adj/vwap_adj`
  - vwap 必须一并换算：`vwap = amount/volume` 用的是**不复权**的 amount/volume，
    而 close 是后复权价 → 不换算会让 `(vwap - close)/close` 特征彻底失准

用法：
    python -m data_maintenance.adjust_factor --fetch --materialize   # 完整流程
    python -m data_maintenance.adjust_factor --fetch                 # 只拉因子表
    python -m data_maintenance.adjust_factor --materialize           # 只按因子填充复权列
    python -m data_maintenance.adjust_factor --verify                # 抽样验证
"""

import argparse
import os
import sqlite3
import sys
import time
from multiprocessing import Pool, cpu_count

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equinet.db')
FACTOR_START = '1990-01-01'      # 必须早于所有股票的首次除权
ADJ_COLS = ('open_adj', 'high_adj', 'low_adj', 'close_adj', 'vwap_adj')
RAW_FOR_ADJ = {'open_adj': 'open', 'high_adj': 'high', 'low_adj': 'low',
               'close_adj': 'close', 'vwap_adj': 'vwap'}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS adjust_factor (
    stock_code         TEXT    NOT NULL,
    divid_operate_date INTEGER NOT NULL,
    fore_adjust_factor REAL,
    back_adjust_factor REAL,
    updated_at         TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (stock_code, divid_operate_date)
);
CREATE INDEX IF NOT EXISTS idx_adjfactor_code ON adjust_factor (stock_code, divid_operate_date);
"""


# ==================== 采集 ====================

def _to_bs_code(code: str) -> str:
    if code.startswith(('6', '9')):
        return f'sh.{code}'
    if code.startswith(('0', '3')):
        return f'sz.{code}'
    if code.startswith(('4', '8')):
        return f'bj.{code}'
    return code


def _fetch_worker(args):
    """子进程：登录 baostock，拉一批股票的复权因子

    实测 baostock 会话会在长时间批量查询中途失效（返回「用户未登录」），
    因此每条查询失败时自动重登并重试。
    """
    db_path, codes, end_date = args
    import baostock as bs
    out = []

    def do_login():
        for attempt in range(3):
            try:
                lg = bs.login()
                if lg.error_code == '0':
                    return True
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        return False

    def do_reconnect():
        try:
            bs.logout()
        except Exception:
            pass
        time.sleep(1.0)
        return do_login()

    if not do_login():
        return [(-1, 'login failed')]

    try:
        for code in codes:
            if not code.isdigit():          # 跳过 __TEST__ 之类的脏代码
                continue
            bs_code = _to_bs_code(code)

            rows_local, done = [], False
            for attempt in range(4):
                try:
                    rs = bs.query_adjust_factor(code=bs_code,
                                                start_date=FACTOR_START,
                                                end_date=end_date)
                except Exception as e:
                    rs = None
                    err = str(e)
                else:
                    err = rs.error_msg if rs.error_code != '0' else ''

                if rs is not None and rs.error_code == '0':
                    while rs.next():
                        r = rs.get_row_data()
                        d = int(str(r[1]).replace('-', ''))
                        rows_local.append((code, d, float(r[2]), float(r[3])))
                    done = True
                    break

                if '登录' in err or 'login' in err.lower():
                    if not do_reconnect():
                        break
                else:
                    break

            if done:
                if rows_local:
                    out.extend(rows_local)
                else:
                    out.append((code, -1, 1.0, 1.0))     # 无除权 → 恒定 1.0
            else:
                out.append((-2, f'{code} failed: {err}'))
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    return out


def fetch_factors(db_path, workers=4, end_date=None, only_missing=False):
    """拉取全市场复权因子，写入 adjust_factor 表

    Args:
        only_missing: True 时只补拉 adjust_factor 表里尚无记录的股票（断点续拉），
                      已拉到的数据保留不删；False 时先清空整表再全量重拉。
    """
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    all_codes = [r[0] for r in con.execute(
        'SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    if only_missing:
        try:
            have = {r[0] for r in con.execute('SELECT DISTINCT stock_code FROM adjust_factor')}
        except sqlite3.OperationalError:
            have = set()
        codes = [c for c in all_codes if c not in have and c.isdigit()]
        print(f'断点续拉: 全库 {len(all_codes):,} 只, 已有 {len(have):,} 只, 待补 {len(codes):,} 只')
    else:
        codes = [c for c in all_codes if c.isdigit()]
    con.close()

    if not codes:
        print('没有需要拉取的股票，跳过')
        return 0

    print(f'拉取 {len(codes):,} 只股票的复权因子（start={FACTOR_START}, end={end_date}）')
    print(f'并行进程数={workers}')

    # 分片
    chunks = [codes[i::workers] for i in range(workers)]
    tasks = [(db_path, ch, end_date) for ch in chunks if ch]

    t0 = time.time()
    rows, errs = [], []
    with Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_fetch_worker, tasks), 1):
            for item in res:
                if len(item) == 4:
                    rows.append(item)
                else:
                    errs.append(item)
            print(f'  完成分片 {i}/{len(tasks)}  累计事件 {len(rows):,}  错误 {len(errs):,}')

    print(f'\n耗时 {time.time()-t0:.1f}s  抓到事件 {len(rows):,} 条  错误 {len(errs):,}')
    if errs:
        print('  错误样本:', errs[:5])

    if not rows:
        print('未抓到任何数据，跳过写入')
        return 0

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_SQL)
    if not only_missing:
        con.execute('DELETE FROM adjust_factor')
    con.executemany(
        'INSERT OR REPLACE INTO adjust_factor '
        '(stock_code, divid_operate_date, fore_adjust_factor, back_adjust_factor) '
        'VALUES (?,?,?,?)', rows)
    con.commit()

    n_stock = con.execute('SELECT COUNT(DISTINCT stock_code) FROM adjust_factor').fetchone()[0]
    n_noevent = con.execute(
        'SELECT COUNT(*) FROM adjust_factor WHERE divid_operate_date = -1').fetchone()[0]
    con.close()
    print(f'已写入 adjust_factor: {len(rows):,} 行, 覆盖 {n_stock:,} 只股票 '
          f'(其中 {n_noevent:,} 只为「无除权」占位)')
    return len(rows)


# ==================== 物化 ====================

def ensure_adj_columns(db_path):
    con = sqlite3.connect(db_path)
    existing = {r[1] for r in con.execute('PRAGMA table_info(stock_daily)')}
    added = []
    for col in ('adj_factor',) + ADJ_COLS:
        if col not in existing:
            con.execute(f'ALTER TABLE stock_daily ADD COLUMN {col} REAL')
            added.append(col)
    con.commit()
    con.close()
    if added:
        print(f'  新增列: {", ".join(added)}')
    return added


def load_spans(con, stock_code):
    """把事件级因子展开成 (start_date, end_date, k) 区间（闭区间，end=None 表示开放）"""
    ev = con.execute(
        'SELECT divid_operate_date, back_adjust_factor FROM adjust_factor '
        'WHERE stock_code = ? ORDER BY divid_operate_date', (stock_code,)).fetchall()
    # 过滤掉「无除权」占位
    ev = [(d, k) for d, k in ev if d > 0]
    spans = []
    if not ev:
        return [(None, None, 1.0)]
    if ev[0][0] > 0:
        spans.append((None, ev[0][0] - 1, 1.0))       # 首次除权之前
    for i, (d, k) in enumerate(ev):
        end = ev[i + 1][0] - 1 if i + 1 < len(ev) else None
        spans.append((d, end, k))
    return spans


def materialize(db_path, only_codes=None):
    """按因子区间填充 adj_factor 与 5 个复权价列"""
    ensure_adj_columns(db_path)
    con = sqlite3.connect(db_path)
    # 只在本连接内放宽持久化保证以加速批量 UPDATE；
    # 不动 journal_mode（它是持久化到 DB header 的，改了会影响其他连接）
    con.execute('PRAGMA synchronous = OFF')

    codes = only_codes or [r[0] for r in con.execute(
        'SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    print(f'物化复权列: {len(codes):,} 只股票')

    upd = ('UPDATE stock_daily SET adj_factor=:k, '
           + ', '.join(f'{a} = {RAW_FOR_ADJ[a]} * :k' for a in ADJ_COLS)
           + ' WHERE stock_code=:c AND date BETWEEN :s AND :e')

    t0 = time.time()
    total_stmt = 0
    for i, code in enumerate(codes, 1):
        spans = load_spans(con, code)
        dmin, dmax = con.execute(
            'SELECT MIN(date), MAX(date) FROM stock_daily WHERE stock_code=?',
            (code,)).fetchone()
        if dmin is None:
            continue
        for s, e, k in spans:
            s2 = max(s, dmin) if s else dmin
            e2 = min(e, dmax) if e else dmax
            if e2 < s2:
                continue
            con.execute(upd, {'k': k, 'c': code, 's': s2, 'e': e2})
            total_stmt += 1
        if i % 500 == 0:
            con.commit()
            print(f'  进度 {i}/{len(codes)} ({i/len(codes)*100:.0f}%)  '
                  f'语句 {total_stmt:,}  耗时 {time.time()-t0:.0f}s')
    con.commit()

    n_null = con.execute('SELECT COUNT(*) FROM stock_daily WHERE adj_factor IS NULL').fetchone()[0]
    con.close()
    print(f'\n完成: {total_stmt:,} 条 UPDATE, 耗时 {time.time()-t0:.0f}s')
    print(f'adj_factor 仍为 NULL 的行: {n_null:,}')
    return total_stmt


# ==================== 验证 ====================

def verify(db_path, n_stocks=5):
    """抽样：用因子派生的 close_adj 与 baostock 直接拉的后复权 close 比对"""
    import baostock as bs
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    codes = [r[0] for r in con.execute(
        "SELECT stock_code FROM stock_daily WHERE close_adj IS NOT NULL "
        "GROUP BY stock_code ORDER BY stock_code LIMIT ?", (n_stocks,))]

    lg = bs.login()
    print(f'登录 {lg.error_code}')
    ok = bad = 0
    for code in codes:
        rs = bs.query_history_k_data_plus(
            _to_bs_code(code), "date,close", start_date='2000-01-01',
            end_date='2026-09-25', frequency='d', adjustflag='1')
        rows = []
        while rs.error_code == '0' and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            continue
        truth = {int(r[0].replace('-', '')): float(r[1]) for r in rows if r[1]}
        mine = dict(con.execute(
            'SELECT date, close_adj FROM stock_daily WHERE stock_code=?', (code,)))
        n_ok = n_bad = 0
        for d, v in mine.items():
            if d in truth and v and truth[d]:
                if abs(v - truth[d]) / truth[d] < 1e-6:
                    n_ok += 1
                else:
                    n_bad += 1
        print(f'  {code}: 一致 {n_ok:,}  不一致 {n_bad:,}')
        ok += n_ok; bad += n_bad
    bs.logout()
    con.close()
    print(f'\n合计: 一致 {ok:,}, 不一致 {bad:,}  →  {"通过" if bad == 0 else "未通过"}')
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description='后复权因子采集与物化')
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--fetch', action='store_true', help='拉取全市场因子')
    ap.add_argument('--only-missing', action='store_true',
                    help='断点续拉：只补 adjust_factor 表里尚无记录的股票')
    ap.add_argument('--materialize', action='store_true', help='按因子填充复权列')
    ap.add_argument('--verify', action='store_true', help='抽样验证')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not (args.fetch or args.materialize or args.verify):
        ap.print_help()
        return

    if args.fetch:
        print('=' * 60)
        print('步骤 1/3  拉取复权因子')
        print('=' * 60)
        fetch_factors(db_path, workers=args.workers, only_missing=args.only_missing)

    if args.materialize:
        print('\n' + '=' * 60)
        print('步骤 2/3  物化复权列')
        print('=' * 60)
        materialize(db_path)

    if args.verify:
        print('\n' + '=' * 60)
        print('步骤 3/3  抽样验证')
        print('=' * 60)
        verify(db_path)


if __name__ == '__main__':
    main()
