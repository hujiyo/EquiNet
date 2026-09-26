"""
离线数据质量审计（不联网、不修改任何行情数据）

用法：
    python -m data_maintenance.audit                    # 只报告，不写库
    python -m data_maintenance.audit --write            # 写入 data_issues + 登记 provenance
    python -m data_maintenance.audit --pool pit         # 扫 PIT 池（默认取 contract.CURRENT_POOL）
    python -m data_maintenance.audit --workers 4

产出**只有一张表** `data_issues`（逐条问题：股票 / 日期 / 类型 / 详情）。

## 为什么不再有 sample_exclusion

原来还产出一张 `sample_exclusion`：把问题日展开成「哪些采样位置不能用」的
日期区间。这是**同一个语义的第二份表达**，而且展开本身出过错 ——
序号空间的 ±4/±44 被直接加在日期整数上（`20150911 + 44 = 20150955` 不是合法
日期，`searchsorted` 于是把右界定位到「问题日所在月的月末」），
实测 74.5% 的污染采样位置被漏排，3,961/3,961 个区间日期非法。

现在展开只发生在消费侧一处：`src/data.py:apply_quality_exclusions` 用
`contract.SAMPLING.excluded_starts()` 把 issue 日期映射成采样起始索引。
单位只在一个地方定义，且换算函数只有一个。

## 扫描范围纪律

`--pool` 默认取 `contract.CURRENT_POOL`，**即训练所用的同一个池**。
两者不一致时训练会用到一个没有被审计覆盖的股票池 —— 这种静默失效由
`selfcheck` 断言拦截（见 data_maintenance/selfcheck.py）。
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from .contract import (
    DEFAULT_DB, MAX_WORKERS, CURRENT_POOL, SAMPLING, is_exclusion_type,
    load_stock_codes, pool_scope_label,
)
from .quality import scan_stock, Issue
from . import provenance

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

-- 扫描台账：记录「本次审计到底扫过哪些股票」。
-- 没有它，「干净的股票」和「没被扫过的股票」在库里长得一模一样
-- （都没有 data_issues 行）—— 而这正是「审计范围没覆盖训练池」这类
-- 静默失效的症状。有了台账，覆盖率可以精确断言，不必靠百分比猜。
CREATE TABLE IF NOT EXISTS audit_scan (
    stock_code TEXT PRIMARY KEY,
    n_rows     INTEGER,
    n_issues   INTEGER,
    scanned_at TEXT DEFAULT (datetime('now'))
);
"""

# 单一事实来源：展开已移到消费侧，这张派生表没有存在理由了。
# 留着它只会让「库里有什么」和「代码按什么算」再次分叉。
DROP_LEGACY_SQL = "DROP TABLE IF EXISTS sample_exclusion;"

ISSUE_LABELS = {
    'ohlc_inconsistent': 'OHLC 不自洽',
    'nonfinite_value': '必需列非有限数（NULL/NaN）',
    'nonpositive_price': '价格非正',
    'zero_volume': '零成交量（停牌快照）',
    'placeholder_row': '占位垃圾行',
    'vwap_out_of_range': 'vwap 越界（一字板精度伪报已剔除）',
    'price_anomaly': '价格异常跳变（真错价；除权日与复牌日已豁免）',
    'resume_gap': '停牌后复牌跳空（正常事件，单独列出）',
    'zombie_price': '僵尸价格（停牌填充）',
    'scan_error': '扫描失败',
}


def connect_ro(db_path: str) -> sqlite3.Connection:
    """只读连接。路径统一转正斜杠 —— 反斜杠在 SQLite URI 里是未定义行为，
    历史上只有部分模块做了这个转换（同一个坑两种写法）。"""
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    con.execute('PRAGMA cache_size = -20000')
    return con


# ==================== worker ====================

def _scan_one(args):
    """子进程任务：扫单只股票（只读连接）

    注意：`pandas` 必须在这里 import。Windows 上 Pool 用 spawn 启动子进程，
    父进程里的模块级赋值不会继承（顶层 import 因模块重新导入而生效，
    但 `globals()['pd'] = ...` 这种运行时注入不会）。

    除 Issue 列表外还回传两个数，供报告与自检使用：
    - n_rows：扫描到的行情行数
    - n_excl_pos：按 `contract.SAMPLING` 展开后、**会污染的采样起始位置数**
      （用与消费侧同一个函数计算，因此报告与实际行为一致）
    """
    db_path, stock_code, global_min_date, market_dates = args
    import pandas as pd
    try:
        con = connect_ro(db_path)
        df = pd.read_sql_query(
            "SELECT sd.date, sd.open, sd.high, sd.low, sd.close, sd.volume, "
            "       sd.amount, sd.vwap, "
            "       CASE WHEN af.divid_operate_date IS NOT NULL THEN 1 ELSE 0 END "
            "            AS is_dividend "
            "FROM stock_daily sd "
            "LEFT JOIN adjust_factor af "
            "       ON af.stock_code = sd.stock_code "
            "      AND af.divid_operate_date = sd.date "
            "WHERE sd.stock_code = ? ORDER BY sd.date ASC",
            con, params=(stock_code,))
        con.close()
        if df.empty:
            return stock_code, [], 0, 0, {}

        # 该股数据首日就是库内最早日期 → 数据被库起点截断，不是上市
        df['series_truncated'] = int(df['date'].values[0]) <= int(global_min_date)

        # 复牌日（外部事实，随数据一起传）：前一条记录若不等于「市场在该日之前的
        # 最后一个交易日」，说明该股中间停牌过。用真实交易日历而不是自然日间隔 ——
        # 后者会把每个周末当成停牌（实测 86,849 vs 真实的 113）。
        d = df['date'].values
        pos = np.searchsorted(market_dates, d, side='left')
        prev_market = market_dates[np.maximum(pos - 1, 0)]
        is_resume = np.zeros(len(d), dtype=bool)
        is_resume[1:] = d[:-1] != prev_market[1:]
        df['is_resume'] = is_resume

        issues = scan_stock(df, stock_code)

        # 与消费侧同一个函数展开，保证「报告 == 实际行为」
        dates = df['date'].values
        n = len(dates)
        mask_ex = np.zeros(n, dtype=bool)      # 驱动排除的（真正的问题）
        by_type = {}
        for iss in issues:
            p = int(np.searchsorted(dates, iss.date, side='left'))
            if p < n and dates[p] == iss.date:
                lo, hi = SAMPLING.excluded_starts(p, n)
                if hi >= lo:
                    # 按问题类型分别统计。排除是有成本的（直接减少样本量），
                    # 不按类型归属的话这个成本是隐形的 —— 看不出「为了修 13 条真错价，
                    # 顺手排掉了 10% 的样本」这种事。
                    t = by_type.get(iss.issue_type)
                    if t is None:
                        t = by_type[iss.issue_type] = np.zeros(n, dtype=bool)
                    t[lo:hi + 1] = True
                    if is_exclusion_type(iss.issue_type):
                        mask_ex[lo:hi + 1] = True
        return (stock_code, issues, n, int(mask_ex.sum()),
                {k: int(v.sum()) for k, v in by_type.items()})
    except Exception as e:
        return stock_code, [Issue(stock_code, 0, 'scan_error', str(e))], 0, 0, {}


# ==================== 落库 ====================

def ensure_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        con.executescript(DROP_LEGACY_SQL)
        provenance.ensure_schema(con)
        con.commit()
    finally:
        con.close()


def persist(db_path: str, all_issues, stats, scope: str) -> int:
    """写入 data_issues + audit_scan 台账，并登记 provenance（同一事务）"""
    con = sqlite3.connect(db_path)
    try:
        con.execute('PRAGMA synchronous = NORMAL')
        con.execute('DELETE FROM data_issues')
        con.execute('DELETE FROM audit_scan')

        n_iss = 0
        scan_rows = []
        for code, issues in all_issues.items():
            rows = [i.as_row() for i in issues if i.issue_type != 'scan_error']
            if rows:
                con.executemany(
                    'INSERT OR REPLACE INTO data_issues '
                    '(stock_code, date, issue_type, detail) VALUES (?,?,?,?)', rows)
                n_iss += len(rows)
            # 台账记「扫过」，与「有问题」无关 —— 干净股票也要留下被扫过的证据
            if issues and issues[0].issue_type == 'scan_error':
                continue
            scan_rows.append((code, stats['rows_by_stock'].get(code, 0), len(rows)))
        con.executemany(
            'INSERT OR REPLACE INTO audit_scan (stock_code, n_rows, n_issues) VALUES (?,?,?)',
            scan_rows)

        counts = Counter(i.issue_type
                         for lst in all_issues.values() for i in lst
                         if i.issue_type != 'scan_error')
        detail = '; '.join(f'{k}×{v}' for k, v in sorted(counts.items()))

        # 与上面同一事务登记：不可能出现「表写了但没登记」的中间态
        provenance.record(key=provenance.KEY_ISSUES, scope=scope, rows=n_iss,
                          detail=detail, con=con)
        con.commit()
    finally:
        con.close()

    n_rows = stats['n_rows']
    print(f'\n已写入: data_issues {n_iss:,} 条（范围={scope}）')
    print(f'扫描台账 audit_scan: {len(scan_rows):,} 只（含无问题的股票）')
    print(f'会排除的采样起始位置: {stats["n_excluded_positions"]:,} 个'
          f'（占已扫行数 {n_rows:,} 的 '
          f'{stats["n_excluded_positions"]/max(n_rows,1)*100:.3f}%）')
    print(f'provenance 已登记: {provenance.KEY_ISSUES} scope={scope}')
    return n_iss


def print_report(all_issues, stats, elapsed):
    counter = Counter()
    by_stock = defaultdict(int)
    for code, issues in all_issues.items():
        for iss in issues:
            counter[iss.issue_type] += 1
            if iss.issue_type != 'scan_error':
                by_stock[code] += 1

    n_rows = stats['n_rows']
    print()
    print('=' * 68)
    print('数据质量审计报告')
    print('=' * 68)
    print(f'扫描股票数: {len(all_issues):,}   扫描行情行数: {n_rows:,}   耗时: {elapsed:.1f}s')

    print('\n--- 问题分布 ---')
    if not counter:
        print('  未发现任何问题')
    for kind, n in counter.most_common():
        share = n / n_rows * 100 if n_rows else 0
        print(f'  {ISSUE_LABELS.get(kind, kind):34s} {n:>9,}  ({share:6.3f}% of rows)')

    errs = [i for lst in all_issues.values() for i in lst if i.issue_type == 'scan_error']
    if errs:
        print(f'\n  [!] 扫描出错 {len(errs)} 只，前 10:')
        for i in errs[:10]:
            print(f'      {i.stock_code}: {i.detail}')

    print('\n--- 问题最多的股票（前 12）---')
    for code, n in sorted(by_stock.items(), key=lambda x: -x[1])[:12]:
        kinds = Counter(i.issue_type for i in all_issues[code])
        brief = ', '.join(f'{ISSUE_LABELS.get(k, k)}×{v}' for k, v in kinds.most_common(3))
        print(f'  {code}: {n:>6,} 条   {brief}')

    print('\n--- 影响面 ---')
    print(f'  受影响股票数: {len(by_stock):,} / {len(all_issues):,}')
    print(f'  会排除的采样起始位置: {stats["n_excluded_positions"]:,} 个'
          f'（占已扫行数 {n_rows:,} 的 '
          f'{stats["n_excluded_positions"]/max(n_rows,1)*100:.3f}%）')
    by_type = stats.get('excl_by_type') or {}
    if by_type:
        excl_kinds = [(k, v) for k, v in by_type.items() if is_exclusion_type(k)]
        info_kinds = [(k, v) for k, v in by_type.items() if not is_exclusion_type(k)]
        print('  驱动排除的类型（各自单独展开，合计数 > 总数是因为区间重叠）：')
        for kind, n in sorted(excl_kinds, key=lambda x: -x[1]):
            print(f'    {ISSUE_LABELS.get(kind, kind):38s} {n:>9,}')
        if info_kinds:
            print('  仅记录、不驱动排除的类型（正常市场事件）：')
            for kind, n in sorted(info_kinds, key=lambda x: -x[1]):
                print(f'    {ISSUE_LABELS.get(kind, kind):38s} {n:>9,}')
            print(f'    └ 这些**不计入排除**：排除策略由 '
                  f'contract.INFORMATIONAL_ISSUE_TYPES 显式声明。'
                  f'把它们计进去会让「数据质量」机制变成样本过滤器。')
    print(f'  展开规则: 采样起始索引 s ∈ [p-{SAMPLING.label_backspan + SAMPLING.context_length - 1}, p]'
          f'（p = 问题日位次，每个问题日 {SAMPLING.contaminate_t_span} 个位置）')
    print('  该数字在扫描侧用 contract.SAMPLING 算出，与 src/data.py 消费侧同一函数，')
    print('  因此是「实际会排除多少」，不再是只在纸上成立的数字。')
    print('=' * 68)


def main():
    ap = argparse.ArgumentParser(description='EquiNet 离线数据质量审计')
    ap.add_argument('--db', default=DEFAULT_DB, help='数据库路径')
    ap.add_argument('--pool', default=CURRENT_POOL, choices=['selected', 'all', 'pit'],
                    help=f'扫描范围（默认 {CURRENT_POOL}，须与训练所用池一致）')
    ap.add_argument('--write', action='store_true',
                    help='写入 data_issues 并登记 provenance')
    ap.add_argument('--workers', type=int, default=MAX_WORKERS,
                    help=f'并行度（默认 {MAX_WORKERS}；本机内存受限，见 contract.MAX_WORKERS）')
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f'✗ 数据库不存在: {db_path}')
        sys.exit(1)

    t0 = time.time()
    if args.pool == 'all':
        con = connect_ro(db_path)
        codes = [r[0] for r in con.execute(
            'SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
        con.close()
    else:
        codes = load_stock_codes(db_path, args.pool)
    if not codes:
        print(f'✗ 池 {args.pool!r} 为空（PIT 池需先跑 pit_pool --write）')
        sys.exit(1)

    con = connect_ro(db_path)
    global_min_date = con.execute('SELECT MIN(date) FROM stock_daily').fetchone()[0] or 0
    market_dates = np.array(
        [r[0] for r in con.execute('SELECT DISTINCT date FROM stock_daily ORDER BY date')],
        dtype=np.int64)
    con.close()

    print(f'池={args.pool}  股票数={len(codes):,}')
    print(f'并行度={args.workers}  数据库={db_path}')
    print(f'库内最早日期={global_min_date}（该日仍有数据的股票视为「被库起点截断」）')
    print(f'交易日历={len(market_dates):,} 天（用于判定停牌复牌，不用自然日间隔）')

    all_issues = {}
    stats = {'n_rows': 0, 'n_excluded_positions': 0, 'rows_by_stock': {},
             'excl_by_type': Counter()}
    with Pool(args.workers) as pool:
        for i, (code, issues, n, n_ex, by_type) in enumerate(
                pool.imap_unordered(_scan_one,
                                    [(db_path, c, global_min_date, market_dates)
                                     for c in codes],
                                    chunksize=8), 1):
            all_issues[code] = issues
            stats['n_rows'] += n
            stats['n_excluded_positions'] += n_ex
            stats['rows_by_stock'][code] = n
            stats['excl_by_type'].update(by_type)
            if i % 500 == 0:
                print(f'  进度 {i}/{len(codes)} ({i/len(codes)*100:.0f}%)  '
                      f'行数 {stats["n_rows"]:,}')

    elapsed = time.time() - t0
    print_report(all_issues, stats, elapsed)

    if args.write:
        ensure_schema(db_path)
        persist(db_path, all_issues, stats, pool_scope_label(args.pool))
    else:
        print('\n(未写入数据库。加 --write 可持久化 data_issues 并登记 provenance)')


if __name__ == '__main__':
    main()
