"""
派生状态的血缘登记（provenance）

## 解决什么问题

「这份派生数据，是拿哪个股票池、哪份源数据、什么时候算出来的？」
—— 这个问题原来无处可答，于是出现了一整类静默失效：

- `data_issues` / `sample_exclusion` 只覆盖 `selected` 池（2,307 只），
  而 PIT 池有 3,338 只 —— 下游不知道，切换池就等于 47% 的股票失去质量筛查。
- 库内数据更新了但没重跑审计：过期的结论照样喂给训练，没有提示。
- 特征是按 `close_adj` 还是 `close` 算的，只能靠读源码猜。

## 设计

一张表 `data_meta`，按 `key` 记录每份派生状态的四个事实：

| 字段 | 含义 |
|---|---|
| `scope` | 覆盖范围（`selected` / `all` / `pit` / 具体股票数） |
| `source_max_date` | 生成时源数据（`stock_daily`）的最新日期 —— **新鲜度的时钟** |
| `rows` | 覆盖行数 |
| `generated_at` | 生成时间 |

新鲜度判据很直接：**源数据的最新日期 > 派生状态生成时看到的源数据最新日期，
这份派生状态就过期了。** 不需要时间戳比较（时间戳会被时区/机器时间污染），
只用数据本身的事实。

## 为什么不放文件

派生状态本身就在数据库里，放同一处才能在同一事务里保持一致。
进程外的记忆文件无法保证与数据同步。
"""

import os
import sqlite3
from typing import Dict, List, Optional

from .contract import DEFAULT_DB

__all__ = [
    'SCHEMA_SQL', 'KEY_INGEST', 'KEY_ADJUST', 'KEY_FEATURES', 'KEY_ISSUES',
    'KEY_POOL_PIT', 'DEPENDENCIES',
    'ensure_schema', 'record', 'read_all', 'source_max_date',
    'staleness_report', 'assert_fresh',
]

# ---- 规范化的 key。不要在各处手写字符串，否则又是「多处各自表达」。----
KEY_INGEST = 'stock_daily.ingest'        # 行情增量更新
KEY_ADJUST = 'stock_daily.adjust'        # 后复权因子物化
KEY_FEATURES = 'stock_daily.features'    # 9 个衍生特征计算
KEY_STATUS = 'stock_status'              # 逐日 tradestatus / isST
KEY_ISSUES = 'data_issues'               # 离线质量审计
KEY_POOL_PIT = 'pool_membership'         # PIT 股票池

# 谁依赖谁：value 里的每个 key 都必须「不早于」源数据（source_max_date）。
# 顺序（增量更新 → 复权物化 → 特征 → 审计 → PIT）曾是文档里一段
# 「顺序很重要」的文字叮嘱，现在是可断言的依赖表。
DEPENDENCIES: Dict[str, List[str]] = {
    KEY_INGEST: [],
    KEY_STATUS: [KEY_INGEST],
    KEY_ADJUST: [KEY_INGEST],
    KEY_FEATURES: [KEY_INGEST, KEY_ADJUST],
    KEY_ISSUES: [KEY_INGEST],
    KEY_POOL_PIT: [KEY_INGEST, KEY_STATUS],
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS data_meta (
    key              TEXT PRIMARY KEY,
    scope            TEXT,
    source_max_date  INTEGER,
    rows             INTEGER,
    detail           TEXT,
    generated_at     TEXT DEFAULT (datetime('now'))
);
"""


def _connect(db_path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    else:
        con = sqlite3.connect(db_path)
    return con


def _ensure_schema_con(con: sqlite3.Connection) -> None:
    """建表（幂等）。**已存在时不调用 executescript** ——

    sqlite3 的 `executescript` 会先隐式 COMMIT 尚未提交的事务。
    在调用方的事务中间触发它，会把「写派生表 + 登记 provenance」拆成两个事务，
    原子性就没了（进程死在中间会得到「表更新了但没有血缘」的状态）。
    表已存在时完全不碰事务。
    """
    try:
        con.execute('SELECT 1 FROM data_meta LIMIT 1')
        return
    except sqlite3.OperationalError:
        pass
    con.executescript(SCHEMA_SQL)


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)


def source_max_date(con: sqlite3.Connection) -> int:
    """源数据（stock_daily）的最新日期 —— 全系统的新鲜度时钟。"""
    row = con.execute('SELECT MAX(date) FROM stock_daily').fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def record(db_path: str = DEFAULT_DB,
           key: str = '',
           scope: str = '',
           rows: int = 0,
           detail: Optional[str] = None,
           con: Optional[sqlite3.Connection] = None) -> dict:
    """登记/更新一份派生状态。传 con 时复用已有连接（便于放进调用方事务）。

    `source_max_date` 取登记时刻的 `stock_daily` 最新日期 —— 不需要调用方计算，
    避免「忘了传」导致登记了一个假的新鲜度。
    """
    own = con is None
    con = con or _connect(db_path)
    try:
        _ensure_schema_con(con)
        smd = source_max_date(con)
        con.execute(
            'INSERT OR REPLACE INTO data_meta '
            '(key, scope, source_max_date, rows, detail, generated_at) '
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (key, scope, smd, int(rows), detail))
        if own:
            con.commit()
        return {'key': key, 'scope': scope, 'source_max_date': smd, 'rows': int(rows)}
    finally:
        if own:
            con.close()


def read_all(db_path: str = DEFAULT_DB) -> Dict[str, dict]:
    """返回 {key: 登记信息}。表不存在时返回空 dict（向后兼容）。"""
    if not os.path.exists(db_path):
        return {}
    con = _connect(db_path, readonly=True)
    try:
        rows = con.execute(
            'SELECT key, scope, source_max_date, rows, detail, generated_at '
            'FROM data_meta').fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()
    return {r[0]: {'scope': r[1], 'source_max_date': r[2], 'rows': r[3],
                   'detail': r[4], 'generated_at': r[5]} for r in rows}


def staleness_report(db_path: str = DEFAULT_DB) -> Dict[str, dict]:
    """逐 key 判定新鲜度。

    state:
        'fresh'   已登记且不早于源数据
        'stale'   已登记但源数据之后又更新过 —— 这份结论过期了
        'missing' 从未登记
    """
    con = _connect(db_path, readonly=True)
    try:
        smd = source_max_date(con)
    finally:
        con.close()

    out = {}
    for key, info in read_all(db_path).items():
        seen = info['source_max_date'] or 0
        out[key] = {'state': 'fresh' if seen >= smd else 'stale',
                    'scope': info['scope'],
                    'generated_at': info['generated_at'],
                    'seen_source_max_date': seen,
                    'current_source_max_date': smd,
                    'rows': info['rows']}
    for key in DEPENDENCIES:
        out.setdefault(key, {'state': 'missing', 'scope': None,
                             'generated_at': None, 'seen_source_max_date': None,
                             'current_source_max_date': smd, 'rows': None})
    return out


def assert_fresh(db_path: str = DEFAULT_DB, keys: Optional[List[str]] = None,
                 allow_missing: bool = True) -> List[str]:
    """校验派生状态不早于源数据。

    Returns:
        问题描述列表（空 = 全部新鲜）。不抛异常 —— 由调用方决定是警告还是中止
        （训练会选择警告，selfcheck 会选择非零退出）。
    """
    report = staleness_report(db_path)
    problems = []
    for key in (keys or list(DEPENDENCIES)):
        st = report.get(key, {'state': 'missing'})
        if st['state'] == 'stale':
            problems.append(
                f'{key} 已过期：生成时源数据到 {st["seen_source_max_date"]}，'
                f'现在到 {st["current_source_max_date"]}（生成于 {st["generated_at"]}）')
        elif st['state'] == 'missing' and not allow_missing:
            problems.append(f'{key} 从未登记过 provenance')
    return problems
