"""
离线数据质量审计（不联网、不修改任何行情数据）

用法：
    python -m data_maintenance.audit                    # 只报告，不写库
    python -m data_maintenance.audit --write            # 额外写入 data_issues / sample_exclusion
    python -m data_maintenance.audit --pool all         # 扫全量池
    python -m data_maintenance.audit --workers 8        # 指定并行度
    python -m data_maintenance.audit --report-only      # 从已有 data_issues 读，重新出报告

产出两张表（--write 时）：
- data_issues      : 逐条问题（股票, 日期, 类型, 详情）
- sample_exclusion : 展开后的「样本位置排除区间」，下游 src/data.py 直接 JOIN 使用

设计要点：
- 与 src/ 完全隔离，只读 stock_daily / stock_pool
- 不删除、不修改任何行情行。所有"修复"动作留给人工决策
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

from .quality import scan_stock, build_exclusions, merge_spans, Issue

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equinet.db')

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS data_issues (
    stock_code  TEXT    NOT NULL,
    date        INTEGER NOT NULL,
    issue_type  TEXT    NOT NULL,
    detail      TEXT,
    detected_at TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (stock_code, date, issue_type)
);

CREATE INDEX IF NOT EXISTS idx_issues_type ON data_issues (issue_type);

CREATE TABLE IF NOT EXISTS sample_exclusion (
    stock_code TEXT    NOT NULL,
    start_date INTEGER NOT NULL,
    end_date   INTEGER NOT NULL,
    reason     TEXT,
    updated_at TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (stock_code, start_date, end_date)
);
"""

ISSUE_LABELS = {
    'ohlc_inconsistent': 'OHLC 不自洽',
    'nonpositive_price': '价格非正',
    'zero_volume': '零成交量',
    'placeholder_row': '占位垃圾行',
    'vwap_out_of_range': 'vwap 越界（量纲不一致）',
    'price_anomaly': '价格异常跳变（疑似除权/错价）',
    'zombie_price': '僵尸价格（停牌填充）',
}


# ==================== worker ====================

def _scan_one(args):
    """子进程任务：扫单只股票（只读连接）"""
    db_path, stock_code = args
    try:
        con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
        con.execute('PRAGMA cache_size = -20000')
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume, amount, vwap "
            "FROM stock_daily WHERE stock_code = ? ORDER BY date ASC",
            con, params=(stock_code,)
        )
        con.close()
        if df.empty:
            return stock_code, [], 0
        issues = scan_stock(df, stock_code)
        return stock_code, issues, len(df)
    except Exception as e:
        return stock_code, [Issue(stock_code, 0, 'scan_error', str(e))], 0


# ==================== 主流程 ====================

def load_stock_codes(db_path: str, pool: str):
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    if pool == 'all':
        codes = [r[0] for r in con.execute('SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    else:
        codes = [r[0] for r in con.execute(
            "SELECT stock_code FROM stock_pool WHERE pool_type='selected' AND is_active=1 ORDER BY stock_code")]
    con.close()
    return codes


def ensure_schema(db_path: str):
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA_SQL)
    con.commit()
    con.close()


def persist(db_path: str, all_issues, exclusions, verbose=True):
    """写入 data_issues 与 sample_exclusion（先清后写，保证与本次扫描一致）"""
    con = sqlite3.connect(db_path)
    con.execute('PRAGMA synchronous = NORMAL')
    con.execute('DELETE FROM data_issues')
    con.execute('DELETE FROM sample_exclusion')

    n_iss = 0
    for code, issues in all_issues.items():
        rows = [i.as_row() for i in issues if i.issue_type != 'scan_error']
        if rows:
            con.executemany(
                'INSERT OR REPLACE INTO data_issues (stock_code, date, issue_type, detail) VALUES (?,?,?,?)',
                rows)
            n_iss += len(rows)

    n_span = 0
    span_rows = []
    for code, spans in exclusions.items():
        merged = merge_spans(spans)
        for l, r, reason in merged:
            span_rows.append((code, l, r, reason))
        n_span += len(merged)
    con.executemany(
        'INSERT OR REPLACE INTO sample_exclusion (stock_code, start_date, end_date, reason) VALUES (?,?,?,?)',
        span_rows)
    con.commit()

    cov = con.execute('SELECT COUNT(*) FROM stock_daily').fetchone()[0]
    exc_cov = con.execute("SELECT COUNT(*) FROM stock_daily sd WHERE EXISTS ("
                          "SELECT 1 FROM sample_exclusion e WHERE e.stock_code=sd.stock_code "
                          "AND sd.date BETWEEN e.start_date AND e.end_date)").fetchone()[0]
    con.close()

    if verbose:
        print(f'\n已写入: data_issues {n_iss:,} 条, sample_exclusion {n_span:,} 个区间')
        print(f'受排除区间覆盖的行情行: {exc_cov:,} / {cov:,} ({exc_cov/cov*100:.2f}%)')
    return n_iss, n_span


def print_report(all_issues, n_rows_scanned, exclusions, elapsed):
    counter = Counter()
    by_stock = defaultdict(int)
    for code, issues in all_issues.items():
        for iss in issues:
            counter[iss.issue_type] += 1
            if iss.issue_type != 'scan_error':
                by_stock[code] += 1

    print()
    print('=' * 68)
    print('数据质量审计报告')
    print('=' * 68)
    print(f'扫描股票数: {len(all_issues):,}   扫描行情行数: {n_rows_scanned:,}   耗时: {elapsed:.1f}s')

    print('\n--- 问题分布 ---')
    if not counter:
        print('  未发现任何问题')
    total = sum(counter.values())
    for kind, n in counter.most_common():
        label = ISSUE_LABELS.get(kind, kind)
        share = n / n_rows_scanned * 100 if n_rows_scanned else 0
        print(f'  {label:32s} {n:>9,}  ({share:6.3f}% of rows)')

    errs = [i for code, lst in all_issues.items() for i in lst if i.issue_type == 'scan_error']
    if errs:
        print(f'\n  [!] 扫描出错 {len(errs)} 只:')
        for i in errs[:10]:
            print(f'      {i.stock_code}: {i.detail}')

    print('\n--- 问题最多的股票（前 12）---')
    for code, n in sorted(by_stock.items(), key=lambda x: -x[1])[:12]:
        kinds = Counter(i.issue_type for i in all_issues[code])
        brief = ', '.join(f'{ISSUE_LABELS.get(k, k)}×{v}' for k, v in kinds.most_common(3))
        print(f'  {code}: {n:>6,} 条   {brief}')

    print('\n--- 污染样本区间 ---')
    n_spans = sum(len(merge_spans(v)) for v in exclusions.values())
    total_spans_raw = sum(len(v) for v in exclusions.values())
    print(f'  原始问题日展开区间: {total_spans_raw:,} 个')
    print(f'  合并后区间:         {n_spans:,} 个')
    print(f'  受影响股票数:       {len(exclusions):,} / {len(all_issues):,}')
    print()
    print('  (区间含义：样本以 [start_date, end_date] 内任一交易日为「上下文末日」时，')
    print('   其 45 天输入或 3+1 天标签会包含被污染的数据，下游应跳过该样本)')
    print('=' * 68)


def main():
    ap = argparse.ArgumentParser(description='EquiNet 离线数据质量审计')
    ap.add_argument('--db', default=DEFAULT_DB, help='数据库路径')
    ap.add_argument('--pool', default='selected', choices=['selected', 'all'])
    ap.add_argument('--write', action='store_true', help='把结果写入 data_issues / sample_exclusion')
    ap.add_argument('--workers', type=int, default=min(cpu_count(), 8))
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f'✗ 数据库不存在: {db_path}')
        sys.exit(1)

    t0 = time.time()
    codes = load_stock_codes(db_path, args.pool)
    print(f'池={args.pool}  股票数={len(codes):,}')
    print(f'并行度={args.workers}  数据库={db_path}')

    all_issues = {}
    n_rows = 0
    with Pool(args.workers) as pool:
        for i, (code, issues, n) in enumerate(
                pool.imap_unordered(_scan_one, [(db_path, c) for c in codes], chunksize=8), 1):
            all_issues[code] = issues
            n_rows += n
            if i % 500 == 0:
                print(f'  进度 {i}/{len(codes)} ({i/len(codes)*100:.0f}%)  行数 {n_rows:,}')

    exclusions = build_exclusions([i for lst in all_issues.values() for i in lst])
    elapsed = time.time() - t0

    print_report(all_issues, n_rows, exclusions, elapsed)

    if args.write:
        ensure_schema(db_path)
        persist(db_path, all_issues, exclusions, verbose=True)
    else:
        print('\n(未写入数据库。加 --write 可持久化 data_issues / sample_exclusion)')


if __name__ == '__main__':
    main()
