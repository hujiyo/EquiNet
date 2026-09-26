"""
自检：把「系统内部是否自洽」编码成可执行断言

## 为什么需要它

「永久自洽」如果只是口号，最终的落点一定是某个人的记忆或某段文档叮嘱。
本项目已经付过这个学费：运维流水线（docs/data-quality-system.md 第 7 节）
漏了「复权物化」这一步，而审计又默认只扫 `selected` 池 —— 两者都不会报错，
只会让结论悄悄过期或让一部分股票失去质量筛查。

所以不变量必须是**机器可验证**的，违反时给出明确失败，而不是让人去记得。

## 两档

| 档位 | 用途 | 成本 |
|---|---|---|
| `run_fast()` | 训练/更新前自动跑（provenance、池覆盖、契约） | 亚秒级 |
| `run_all()` | 独立命令 `python -m data_maintenance.selfcheck` | 全库扫描，1~2 分钟 |

## 覆盖的不变量

1. 单一事实来源 —— 契约与 config 的派生量一致
2. 机器可验证 —— 物化一致性、NULL 完整性
3. 唯一写入收口 —— 派生表只有一份表达（`sample_exclusion` 必须不存在）
4. 派生状态带 provenance —— 新鲜度与扫描范围可判定
5. 关键步骤不可跳过 —— 训练前 `prerequisites()` 强制检查
"""

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import List, Optional

from .contract import (
    DEFAULT_DB, SAMPLING, SAMPLE_COLUMNS, CLOSE_IDX, CLOSE_RAW_IDX, CURRENT_POOL,
    load_stock_codes,
)
from . import provenance

__all__ = ['Finding', 'run_fast', 'run_all', 'prerequisites', 'main']


@dataclass
class Finding:
    key: str        # 违反的是哪条不变量
    level: str      # 'error' = 必须修；'warn' = 需要知道
    message: str

    def __str__(self) -> str:
        return f'[{self.level.upper():5s}] {self.key}: {self.message}'


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    con.execute('PRAGMA cache_size = -40000')
    return con


def _q(con, sql, params=()):
    try:
        return con.execute(sql, params).fetchone()[0]
    except sqlite3.OperationalError:
        return None


# ==================== 快速档（训练/更新前） ====================

def run_fast(db_path: str = DEFAULT_DB) -> List[Finding]:
    """亚秒级检查：provenance 新鲜度、扫描范围、契约、派生表唯一性。"""
    out: List[Finding] = []

    # --- 不变量 1：契约与 config 的派生量必须一致 ---
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'src'))
        import config  # 导入即触发 config._assert_contract()
        from config import DataConfig
        if DataConfig.CONTEXT_LENGTH != SAMPLING.context_length:
            out.append(Finding('单一事实来源', 'error',
                               f'config.CONTEXT_LENGTH={DataConfig.CONTEXT_LENGTH} '
                               f'≠ contract={SAMPLING.context_length}'))
    except AssertionError as e:
        out.append(Finding('单一事实来源', 'error', f'契约断言失败: {e}'))
    except Exception as e:
        out.append(Finding('单一事实来源', 'warn', f'无法导入 src/config.py 做交叉校验: {e}'))

    # --- 不变量 3：派生表只能有一份表达 ---
    con = _connect(db_path)
    legacy = _q(con, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
                     "AND name='sample_exclusion'")
    if legacy:
        out.append(Finding('唯一表达', 'error',
                           'sample_exclusion 表仍然存在 —— 它是 data_issues 的预展开副本，'
                           '曾因单位错位导致 74.5% 漏排。跑 audit --write 迁移。'))

    # --- 不变量 4：审计范围必须等于当前生效池 ---
    meta = provenance.read_all(db_path)
    scope = (meta.get(provenance.KEY_ISSUES) or {}).get('scope')
    if scope is None:
        out.append(Finding('provenance', 'warn',
                           'data_issues 没有登记 provenance，无法判断它扫的是哪个池'))
    elif scope != CURRENT_POOL and scope != f'{CURRENT_POOL}':
        out.append(Finding('provenance', 'error',
                           f'data_issues 的范围是 {scope!r}，而当前生效池是 '
                           f'{CURRENT_POOL!r} —— 训练会用到一个没被审计覆盖的池'))

    # 池内的股票是否都被审计扫过。依据 `audit_scan` 台账而不是 data_issues：
    # 干净的股票本来就没有 data_issues 行，用后者会把「扫过且干净」误判成「没扫过」。
    if scope in ('selected', 'pit'):
        universe = set(load_stock_codes(db_path, CURRENT_POOL))
        try:
            scanned = {r[0] for r in con.execute('SELECT stock_code FROM audit_scan')}
        except sqlite3.OperationalError:
            out.append(Finding('池覆盖', 'warn',
                               'audit_scan 台账不存在（旧版审计产出），无法判定扫描覆盖'))
            scanned = None
        if scanned is not None:
            missing = universe - scanned
            if missing:
                out.append(Finding('池覆盖', 'error',
                                   f'{len(missing):,}/{len(universe):,} 只池内股票未被本次审计扫描 '
                                   f'（示例 {sorted(missing)[:5]}）—— 它们没有质量筛查却会进训练集'))

    # --- 不变量 4：新鲜度 ---
    for p in provenance.assert_fresh(db_path, allow_missing=True):
        out.append(Finding('新鲜度', 'warn', p))

    con.close()
    return out


# ==================== 完整档（独立命令） ====================

def run_all(db_path: str = DEFAULT_DB) -> List[Finding]:
    out = run_fast(db_path)
    con = _connect(db_path)

    total = _q(con, 'SELECT COUNT(*) FROM stock_daily') or 0
    if not total:
        out.append(Finding('数据存在性', 'error', 'stock_daily 为空'))
        con.close()
        return out

    # --- 不变量 2：后复权物化一致 ---
    n_null = _q(con, 'SELECT COUNT(*) FROM stock_daily WHERE adj_factor IS NULL') or 0
    if n_null:
        out.append(Finding('复权物化', 'error',
                           f'{n_null:,} 行 adj_factor 为 NULL —— 增量更新后的新行必然如此。'
                           f'运行 adjust_factor --fetch --only-missing --materialize'))
    n_mismatch = _q(con, 'SELECT COUNT(*) FROM stock_daily '
                         'WHERE close>0 AND abs(close_adj - close*adj_factor) > 1e-6') or 0
    if n_mismatch:
        out.append(Finding('复权物化', 'error',
                           f'{n_mismatch:,} 行 close_adj ≠ close × adj_factor'))

    # --- 不变量 2：必需列不得为空（门禁的兜底断言） ---
    for col in ('open', 'high', 'low', 'close', 'vwap', 'volume', 'amount'):
        n = _q(con, f'SELECT COUNT(*) FROM stock_daily WHERE "{col}" IS NULL') or 0
        if n:
            out.append(Finding('NULL 完整性', 'error', f'{col} 有 {n:,} 行为 NULL'))

    # --- 不变量 2：池内特征必须齐备 ---
    pool_sql = {
        'selected': "stock_code IN (SELECT stock_code FROM stock_pool "
                    "WHERE pool_type='selected' AND is_active=1)",
        'pit': "stock_code IN (SELECT DISTINCT stock_code FROM pool_membership)",
    }.get(CURRENT_POOL)
    if pool_sql:
        for col in ('m5', 'm10', 'm20', 'dif', 'dea', 'macd_hist',
                    'macd_hist_diff', 'bb_upper', 'bb_lower'):
            n = _q(con, f'SELECT COUNT(*) FROM stock_daily WHERE {pool_sql} '
                        f'AND "{col}" IS NULL') or 0
            if n:
                out.append(Finding('特征完整性', 'error',
                                   f'池内 {col} 有 {n:,} 行为 NULL'))

    # --- 停牌行存量（口径为「不应存在」，见 quality.HARD_RULES 的说明） ---
    n_kline = _q(con, 'SELECT COUNT(*) FROM stock_daily WHERE amount<=0 OR volume<=0') or 0
    if n_kline:
        verified = _q(con, 'SELECT COUNT(*) FROM stock_daily sd JOIN stock_status ss '
                           'ON ss.stock_code=sd.stock_code AND ss.date=sd.date '
                           'WHERE (sd.amount<=0 OR sd.volume<=0) AND ss.tradestatus=0')
        if verified is None:
            out.append(Finding('停牌口径', 'warn',
                               f'{n_kline:,} 行 amount/volume<=0（应清理）；'
                               f'stock_status 尚未采集，无法判定其性质'))
        elif verified >= n_kline * 0.9:
            out.append(Finding('停牌口径', 'warn',
                               f'{n_kline:,} 行 amount/volume<=0，其中 {verified:,} 行被外部状态'
                               f'确认为停牌（tradestatus=0）→ 与门禁「停牌不入库」一致，应清理'))
        else:
            out.append(Finding('停牌口径', 'error',
                               f'{n_kline:,} 行 amount/volume<=0，但只有 {verified:,} 行是停牌 —— '
                               f'其余是另一种数据源噪声，需人工判断'))

    # --- 不变量 5：污染区间换算不得越界 ---
    # 注意：`lo > hi` 是契约定义的「无需排除」信号（例如 p 超出序列长度时），
    # 不是越界。要断言的只有「给出的区间不得落到 [0, T-1] 之外」。
    try:
        bad = []
        for p, T in ((0, 10), (5, 10), (9, 10), (100, 50), (3, 3)):
            lo, hi = SAMPLING.excluded_starts(p, T)
            if lo < 0 or hi > T - 1:
                bad.append(f'p={p},T={T} → ({lo},{hi})')
            elif lo <= hi and not (0 <= lo <= hi <= T - 1):
                bad.append(f'p={p},T={T} → ({lo},{hi})')
        if bad:
            out.append(Finding('污染区间', 'error',
                               f'excluded_starts 在 {len(bad)} 个边界样例上越界: {bad}'))
    except Exception as e:
        out.append(Finding('污染区间', 'error', f'excluded_starts 抛异常: {e}'))

    # --- PIT 池自洽（存在则检查） ---
    n_pit = _q(con, 'SELECT COUNT(*) FROM pool_membership')
    if n_pit is None:
        out.append(Finding('PIT 池', 'warn', 'pool_membership 表不存在（尚未生成）'))
    elif n_pit == 0:
        out.append(Finding('PIT 池', 'warn', 'pool_membership 是空表'))
    else:
        bad_span = _q(con, 'SELECT COUNT(*) FROM pool_membership WHERE start_date > end_date') or 0
        if bad_span:
            out.append(Finding('PIT 池', 'error', f'{bad_span:,} 个区间的 start_date > end_date'))
        cap = _q(con, 'SELECT COUNT(DISTINCT stock_code) FROM pool_membership') or 0
        if CURRENT_POOL != 'pit':
            out.append(Finding('PIT 池', 'warn',
                               f'PIT 池 {cap:,} 只已生成，但当前生效池是 {CURRENT_POOL!r} —— '
                               f'训练还没用上它（幸存者/前视偏差未修）'))
        else:
            # 只在「池真的用 PIT」时才有意义。检查时必须把日期限制在**数据范围内**：
            # end_date=99999999 表示「其数据末段仍符合条件」，会一直延伸到表尾，
            # 而 stock_status 采集到了「今天」—— 股票在数据停更之后才变成 ST 的日子
            # 会被这个区间罩进去，造成假阳性（实测 002743/600439/600530 三只全是）。
            st_leak = _q(con, 'SELECT COUNT(DISTINCT pm.stock_code) FROM pool_membership pm '
                              'JOIN stock_status ss ON ss.stock_code=pm.stock_code '
                              'AND ss.date BETWEEN pm.start_date AND pm.end_date '
                              'AND ss.date <= (SELECT MAX(date) FROM stock_daily) '
                              'WHERE ss.is_st=1')
            if st_leak is None:
                out.append(Finding('PIT 池', 'warn',
                                   f'PIT 池 {cap:,} 只，但缺 stock_status 无法排除 ST 期'))
            elif st_leak:
                out.append(Finding('PIT 池', 'error',
                                   f'{st_leak:,} 只股票的池区间在数据范围内覆盖了 ST 交易日 —— '
                                   f'pit_pool 的 isST 判定有洞'))
            else:
                print(f'  PIT 池 {cap:,} 只：数据范围内无 ST 泄漏 ✓')

    con.close()
    return out


# ==================== 训练前强制检查 ====================

def prerequisites(db_path: Optional[str] = None) -> List[Finding]:
    """训练开始前必须成立的条件。由 src/train.py 调用。

    只做「快速档」+ 池覆盖，不做全库扫描（不能为了自检让训练多等两分钟）。
    """
    from .contract import DEFAULT_DB as _D
    return run_fast(db_path or _D)


def main():
    ap = argparse.ArgumentParser(description='EquiNet 数据系统自检')
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--fast', action='store_true', help='只跑亚秒级检查')
    args = ap.parse_args()

    db_path = os.path.abspath(args.db)
    if not os.path.exists(db_path):
        print(f'✗ 数据库不存在: {db_path}')
        sys.exit(1)

    import time
    t0 = time.time()
    findings = run_fast(db_path) if args.fast else run_all(db_path)
    elapsed = time.time() - t0

    errors = [f for f in findings if f.level == 'error']
    warns = [f for f in findings if f.level == 'warn']

    print('=' * 70)
    print(f'自检结果（{elapsed:.1f}s）  池={CURRENT_POOL}')
    print('=' * 70)
    if not findings:
        print('✓ 全部不变量成立')
    for f in errors + warns:
        print(' ', f)
    print()
    print(f'错误 {len(errors)}  警告 {len(warns)}')
    print('=' * 70)
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
