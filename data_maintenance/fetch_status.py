"""
逐日交易状态与 ST 标记采集（tradestatus / isST）

## 为什么需要这张表

两个硬缺口，都堵在「不知道历史状态」上：

1. **PIT 池缺 ST 标记**。`pool_membership` 只判了市值与可交易性，
   于是它会把「当时是 ST」的股票当成正常股票放进历史池 —— 这不是修偏差，
   是把一种偏差换成另一种（见 pit_pool.py 的说明）。
   实测：`stock_metadata` 是**空表**，历史 ST 状态在库里根本不存在。

2. **4,763 行「停牌记录」无从定论**。库里 `amount=0/volume=0` 的行，
   到底是数据源噪声还是真实停牌日？门禁口径已统一为「拒收」（停牌不是 K 线），
   但存量该不该清，需要 `tradestatus` 这个外部事实来判定，而不是靠猜。

## 产出

`stock_status(stock_code, date, tradestatus, is_st)`

| 列 | 含义 |
|---|---|
| `tradestatus` | 1 = 正常交易，0 = 停牌 |
| `is_st` | 1 = ST，0 = 非 ST |

**存全量，不存「例外」**。用「缺少记录 = 默认可交易且非 ST」这种隐式约定
虽然省空间，但会让「这个日期根本没有状态数据」和「这个日期状态正常」变得无法区分 ——
正是这类隐式默认在过去制造了静默错误。

顺带得到 `MIN(date)` = 该股真实上市日（首日交易日），
可以用来把 IPO 豁免从「数据首日」这个近似换成事实判据。

## 用法

    python -m data_maintenance.fetch_status --fetch                # 全量（约 15 分钟）
    python -m data_maintenance.fetch_status --fetch --only-missing # 断点续拉
    python -m data_maintenance.fetch_status --report               # 只读统计
    python -m data_maintenance.fetch_status --verify               # 与库内停牌行对照

## 内存与并发纪律

5,455 只 × 平均约 3,500 天的历史 ≈ 2,000 万行。两条硬约束：

1. **并行查、单写。** worker 只负责查询并把结果回传，父进程是唯一的写者。
   多进程同时写 SQLite 会撞 `database is locked` —— 实测**即使显式设了
   `busy_timeout = 30000` 也照样失败**（隐式 BEGIN 升级写锁时不受 busy 重试保护）。
   单写者天然无竞争，也不需要猜 SQLite 的锁语义。
2. **不在内存里攒。** 每 10 只股票一个任务（约 4 MB），worker 只驻一个分块。
   （对比 adjust_factor.py：每只股票只有几十行因子，攒在内存里无所谓；
   这里照抄那种写法会在 4 个 worker 里各堆 1~2 GB。）
"""

import argparse
import os
import sqlite3
import sys
import time
from multiprocessing import Pool

from .contract import DEFAULT_DB, MAX_WORKERS
from . import provenance

# 起始日期。取库内最早日期（20100104）之前一年即可。
#
# ⚠️ 这里**可以**截断，与 adjust_factor.FACTOR_START = '1990-01-01' 的处理**故意不同**：
# 复权因子是**累计推导量**（k(d) 取决于「d 之前最近一次除权」），截断历史会让早期
# 日期推出错误的因子 —— 那边必须全历史。而 tradestatus / isST 是**逐日独立事实**，
# 今天的状态不依赖 1990 年的状态，截断不产生任何推导误差。
# 截断的代价只是失去 2009 年之前上市股票的真实上市日，而我们判断「是否 IPO 豁免」
# 只需要「库内序列首日是否等于真实上市日」——那些老股票的序列首日本来就是库起点。
STATUS_START = '2009-01-01'
FETCH_FIELDS = 'date,code,tradestatus,isST'
# 每个并行任务负责多少只股票。20 只 ≈ 10 万行 ≈ 8 MB（pickle 后），
# 保证 worker 内存只驻一个分块、父进程也只缓冲少量在途结果。
CHUNK = 20

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS stock_status (
    stock_code  TEXT    NOT NULL,
    date        INTEGER NOT NULL,
    tradestatus INTEGER,
    is_st       INTEGER,
    PRIMARY KEY (stock_code, date)
);
CREATE INDEX IF NOT EXISTS idx_status_code ON stock_status (stock_code);
"""

INSERT_SQL = ('INSERT OR REPLACE INTO stock_status '
              '(stock_code, date, tradestatus, is_st) VALUES (?,?,?,?)')


def ensure_schema(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()


def _to_bs_code(code: str) -> str:
    if code.startswith(('6', '9')):
        return f'sh.{code}'
    if code.startswith(('0', '3')):
        return f'sz.{code}'
    if code.startswith(('4', '8')):
        return f'bj.{code}'
    return code


def _fetch_chunk(args):
    """子进程：查一小批股票的状态，**只查不写**，把行回传给父进程。

    为什么不在 worker 里写库：多个进程同时写 SQLite 会撞
    `database is locked` —— 实测即使显式设了 `busy_timeout = 30000`
    也照样失败（隐式 BEGIN 升级写锁时不受 busy 重试保护）。
    改成「并行查、单写」：写入侧永远只有一个写者，无竞争；
    worker 只驻一个分块（<10 MB），父进程只缓冲在途结果。

    baostock 会话在长时间批量查询中途会失效（返回「用户未登录」），
    这里和 adjust_factor.py 一样内置自动重登 + 重试。
    """
    codes, start_date, end_date = args
    import baostock as bs

    def do_login():
        for attempt in range(3):
            try:
                if bs.login().error_code == '0':
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
        return [], list(codes)

    out, failed = [], []
    try:
        for code in codes:
            if not code.isdigit():          # 跳过 __TEST__ 之类脏代码
                continue
            rows, done = [], False
            for _ in range(4):
                try:
                    rs = bs.query_history_k_data_plus(
                        _to_bs_code(code), FETCH_FIELDS,
                        start_date=start_date, end_date=end_date,
                        frequency='d', adjustflag='3')
                except Exception as e:
                    rs, err = None, str(e)
                else:
                    err = rs.error_msg if rs.error_code != '0' else ''

                if rs is not None and rs.error_code == '0':
                    while rs.next():
                        r = rs.get_row_data()
                        rows.append((code, int(r[0].replace('-', '')),
                                     int(r[2] or 0), int(r[3] or 0)))
                    done = True
                    break
                if '登录' in err or 'login' in err.lower():
                    if not do_reconnect():
                        break
                else:
                    break

            if done:
                out.extend(rows)            # 空 rows 表示该代码确无历史，不算失败
            else:
                failed.append(code)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
    return out, failed


def fetch_status(db_path: str = DEFAULT_DB, workers: int = MAX_WORKERS,
                 end_date: str = None, only_missing: bool = False) -> int:
    import datetime
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')

    ensure_schema(db_path)
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    all_codes = [r[0] for r in con.execute(
        'SELECT DISTINCT stock_code FROM stock_daily ORDER BY stock_code')]
    if only_missing:
        have = {r[0] for r in con.execute('SELECT DISTINCT stock_code FROM stock_status')}
        codes = [c for c in all_codes if c not in have and c.isdigit()]
        print(f'断点续拉: 全库 {len(all_codes):,} 只, 已有 {len(have):,} 只, 待补 {len(codes):,} 只')
    else:
        codes = [c for c in all_codes if c.isdigit()]
    con.close()

    if not codes:
        print('没有需要拉取的股票，跳过')
        return 0

    if not only_missing:
        con = sqlite3.connect(db_path)
        con.execute('DELETE FROM stock_status')
        con.commit()
        con.close()

    chunks = [codes[i:i + CHUNK] for i in range(0, len(codes), CHUNK)]
    print(f'拉取 {len(codes):,} 只股票的逐日状态（{STATUS_START} ~ {end_date}）')
    print(f'并行度={workers}，每任务 {CHUNK} 只，共 {len(chunks):,} 个任务（并行查 / 单写）')

    t0 = time.time()
    total_rows, all_failed = 0, []
    con = sqlite3.connect(db_path, timeout=60)
    con.execute('PRAGMA synchronous = NORMAL')
    try:
        tasks = [(ch, STATUS_START, end_date) for ch in chunks]
        with Pool(workers) as pool:
            for i, (rows, failed) in enumerate(pool.imap_unordered(_fetch_chunk, tasks), 1):
                if rows:
                    con.executemany(INSERT_SQL, rows)
                    total_rows += len(rows)
                all_failed.extend(failed)
                if i % 25 == 0:
                    con.commit()
                    done = min(i * CHUNK, len(codes))
                    print(f'  进度 {done:,}/{len(codes):,}  行 {total_rows:,}  '
                          f'失败 {len(all_failed):,}  耗时 {time.time()-t0:.0f}s')
        con.commit()
    finally:
        con.close()

    elapsed = time.time() - t0
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    n_stock = con.execute('SELECT COUNT(DISTINCT stock_code) FROM stock_status').fetchone()[0]
    n_susp = con.execute('SELECT COUNT(*) FROM stock_status WHERE tradestatus=0').fetchone()[0]
    n_st = con.execute('SELECT COUNT(*) FROM stock_status WHERE is_st=1').fetchone()[0]
    con.close()

    print(f'\n耗时 {elapsed/60:.1f} 分钟  写入 {total_rows:,} 行  覆盖 {n_stock:,} 只')
    print(f'  停牌日 {n_susp:,} 行    ST 日 {n_st:,} 行')
    if all_failed:
        print(f'  [!] 失败 {len(all_failed):,} 只，示例: {all_failed[:8]}')
        print('      （用 --only-missing 续拉）')

    provenance.record(db_path, key=provenance.KEY_STATUS,
                      scope=f'{n_stock} 只 / {STATUS_START}~{end_date}',
                      rows=total_rows,
                      detail=f'停牌日 {n_susp}; ST 日 {n_st}; 失败 {len(all_failed)}')
    print(f'provenance 已登记: {provenance.KEY_STATUS}')
    return total_rows


def report(db_path: str = DEFAULT_DB) -> None:
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    try:
        n = con.execute('SELECT COUNT(*) FROM stock_status').fetchone()[0]
    except sqlite3.OperationalError:
        print('stock_status 表不存在，先跑 --fetch')
        return
    print(f'stock_status 共 {n:,} 行，覆盖 '
          f'{con.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_status").fetchone()[0]:,} 只')
    print(f'日期范围: {con.execute("SELECT MIN(date), MAX(date) FROM stock_status").fetchone()}')
    print(f'停牌日 {con.execute("SELECT COUNT(*) FROM stock_status WHERE tradestatus=0").fetchone()[0]:,}')
    st_stocks = con.execute(
        'SELECT COUNT(DISTINCT stock_code) FROM stock_status WHERE is_st=1').fetchone()[0]
    print(f'曾被 ST 的股票: {st_stocks:,} 只')
    print('\nST 天数最多的股票（前 10）:')
    for r in con.execute(
            'SELECT stock_code, COUNT(*) c FROM stock_status WHERE is_st=1 '
            'GROUP BY stock_code ORDER BY c DESC LIMIT 10'):
        print(f'  {r[0]}: {r[1]:,} 天')
    con.close()


def verify(db_path: str = DEFAULT_DB) -> None:
    """用外部状态判定库里 4,763 行「停牌记录」的真实性质。

    结论会直接决定这批存量行是清理还是保留 —— 不靠猜。
    """
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    try:
        rows = con.execute(
            "SELECT sd.stock_code, sd.date, ss.tradestatus "
            "FROM stock_daily sd LEFT JOIN stock_status ss "
            "  ON ss.stock_code=sd.stock_code AND ss.date=sd.date "
            "WHERE sd.amount<=0 OR sd.volume<=0").fetchall()
    except sqlite3.OperationalError:
        print('stock_status 表不存在，先跑 --fetch')
        return
    con.close()

    total = len(rows)
    susp = sum(1 for r in rows if r[2] == 0)
    tradable = sum(1 for r in rows if r[2] == 1)
    unknown = total - susp - tradable
    print(f'库内 amount<=0 或 volume<=0 的行: {total:,}')
    print(f'  外部状态为【停牌 tradestatus=0】: {susp:,}  ({susp/max(total,1)*100:.1f}%)')
    print(f'  外部状态为【正常交易 tradestatus=1】: {tradable:,}')
    print(f'  外部无对应记录: {unknown:,}')
    print('\n判定：若绝大多数落在「停牌」，则这批行确实是停牌快照而非 K 线，')
    print('      与门禁「停牌不入库」的口径一致，应清理；否则说明存在另一种数据源噪声。')


def main():
    ap = argparse.ArgumentParser(description='逐日 tradestatus / isST 采集')
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--fetch', action='store_true', help='拉取（全量；--only-missing 时续拉）')
    ap.add_argument('--only-missing', action='store_true', help='只补尚无记录的股票')
    ap.add_argument('--report', action='store_true', help='只读统计')
    ap.add_argument('--verify', action='store_true', help='与库内停牌行对照')
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    ap.add_argument('--end-date')
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f'✗ 数据库不存在: {db_path}')
        sys.exit(1)

    if args.fetch:
        fetch_status(db_path, workers=args.workers, end_date=args.end_date,
                     only_missing=args.only_missing)
    if args.report:
        report(db_path)
    if args.verify:
        verify(db_path)
    if not (args.fetch or args.report or args.verify):
        ap.print_help()


if __name__ == '__main__':
    main()
