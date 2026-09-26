"""
EquiNet 数据契约 —— 系统内部「同一份数据以什么口径被理解」的**唯一定义处**。

## 为什么必须存在这个文件

数据正确性缺陷几乎全是同一个病的变体：
**同一个语义在多处各自表达，且没有任何机制校验它们是否仍然一致。**

最直接的一例：采样几何 `45 / 3 / 1` 曾经同时在 `src/config.py` 和
`data_maintenance/quality.py` 里各写了一遍（后者的注释还写着「此处冗余以免依赖」）。
结果是改动 `CONTEXT_LENGTH`（config 里明确标注的「核心参数」）会让污染区间
**静默错位**，而错位方向恰好是「排漏」—— 脏数据照常进训练集，且不报错。

因此本文件是这些语义的唯一定义处。其他任何模块需要它们，只能 import，不得复述。
`src/config.py` 的 `DataConfig` 已经把相关项指向本文件，并带有启动断言。

## 本文件认领的口径

| 语义 | 位置 |
|---|---|
| 采样几何（几天算一个样本、标签落在哪） | `SAMPLING` |
| 样本矩阵的列语义（第几列是什么） | `SAMPLE_COLUMNS` / `CLOSE_IDX` / `CLOSE_RAW_IDX` |
| 污染半径（一个脏数据日污染哪些样本） | `SAMPLING.excluded_starts()` |
| 股票池口径（训练与审计必须是同一个池） | `CURRENT_POOL` / `pool_join_sql()` / `load_stock_codes()` |
| 特征计算基准 | `FEATURE_PRICE_COL` |
| 并行度上限（本机 31.7 GB，与用户共用） | `MAX_WORKERS` |

## 依赖约束

本文件**只依赖标准库**。上游需要一个能在没有 torch / numpy 的环境里被导入的
契约（离线审计、自检命令都会 import 它），所以不要把 numpy/pandas/torch 引进来。
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

__all__ = [
    'SAMPLING', 'Sampling',
    'SAMPLE_COLUMNS', 'SAMPLE_SELECT_SQL', 'CLOSE_IDX', 'CLOSE_RAW_IDX', 'MODEL_INPUT_DIM',
    'FEATURE_PRICE_COL', 'MAX_WORKERS', 'DEFAULT_DB',
    'INFORMATIONAL_ISSUE_TYPES', 'is_exclusion_type',
    'CURRENT_POOL', 'pool_join_sql', 'load_stock_codes', 'pool_scope_label',
]

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equinet.db')

# 本机 31.7 GB 物理内存且与用户共用（会被拿去打游戏）。
# 历史上 `min(cpu_count(), 8)` 这个「按核数定并行度」的默认值把机器压死过一次，
# 20 个逻辑核不等于有 20 份内存。所有多进程入口都引用这里的上限。
MAX_WORKERS = 4


# ==================== 采样几何 ====================

@dataclass(frozen=True)
class Sampling:
    """一个训练样本的定义。

    样本以「上下文末日」t 描述：
        输入 = [t - (context_length - 1), t]      共 context_length 天
        标签 = [t + 1, t + future_days]           共 future_days 天
    实现上习惯用**样本起始索引** s 表达（`src/data.py` 的 `all_starts`）：
        s = t - (context_length - 1)

    buffer_days 不参与标签计算，但会进入涨停/有效性过滤的读取窗口，
    因此也计入污染范围（保守方向：宁可多排一个位置，不可漏排）。
    """

    context_length: int = 45
    future_days: int = 3
    buffer_days: int = 1

    @property
    def required_length(self) -> int:
        """一个样本需要读多少行（用于序列尾部守卫）"""
        return self.context_length + self.future_days + self.buffer_days

    @property
    def label_backspan(self) -> int:
        """从上下文末日 t 往前的污染天数（标签侧 + buffer）"""
        return self.future_days + self.buffer_days

    @property
    def contaminate_right(self) -> int:
        """从 t 往后的污染天数（输入侧）"""
        return self.context_length - 1

    @property
    def contaminate_t_span(self) -> int:
        """单个脏数据日会污染的「上下文末日」位置总数"""
        return self.label_backspan + self.contaminate_right + 1

    def excluded_starts(self, p: int, series_len: int) -> tuple:
        """脏数据日在该股序列中的位次为 p 时，被污染的**样本起始索引**闭区间。

        推导（全部在序号空间，与日期无关）：
            输入被污染  <=>  s <= p <= s + C - 1     -> s ∈ [p-(C-1), p]
            标签被污染  <=>  t+1 <= p <= t+L, t=s+C-1 -> s ∈ [p-L-(C-1), p-1-(C-1)]
            合并        -> s ∈ [p-L-(C-1), p]

        返回 (lo, hi)，已对 [0, series_len-1] 做裁剪；hi < lo 表示无需排除。
        """
        lo = p - self.label_backspan - (self.context_length - 1)
        hi = p
        return max(lo, 0), min(hi, series_len - 1)


SAMPLING = Sampling()


# ==================== 样本矩阵的列语义 ====================

# `src/data.py:load_and_preprocess_data` 消费的列顺序，去掉 stock_code / date。
# 下游用魔法下标索引的只有两处（复权收盘价与不复权收盘价），
# 这里给出唯一定义，避免「第 16 列是什么」靠注释和约定维护。
SAMPLE_COLUMNS = (
    'open', 'high', 'low', 'close', 'vwap', 'volume', 'exchange',
    'm5', 'm10', 'm20', 'dif', 'dea', 'macd_hist', 'macd_hist_diff',
    'bb_upper', 'bb_lower',
    'close_raw',          # 第 16 列：**不复权**收盘价，仅供高价股过滤
)
CLOSE_IDX = SAMPLE_COLUMNS.index('close')          # 3：后复权收盘价（模型输入基准）
CLOSE_RAW_IDX = SAMPLE_COLUMNS.index('close_raw')  # 16：不复权收盘价（过滤基准）

# 契约列名 → `stock_daily` 物理列名。
# 价格列取**后复权**（`*_adj`），`close_raw` 取不复权原始收盘价。
# 这份映射是 SELECT 语句与下游下标契约的共同来源：由它生成 SQL，
# 就不存在「SELECT 改了但 cols 没改」这种静默错位。
_SAMPLE_SOURCE = {
    'open': 'open_adj', 'high': 'high_adj', 'low': 'low_adj', 'close': 'close_adj',
    'vwap': 'vwap_adj', 'volume': 'volume', 'exchange': 'exchange',
    'm5': 'm5', 'm10': 'm10', 'm20': 'm20', 'dif': 'dif', 'dea': 'dea',
    'macd_hist': 'macd_hist', 'macd_hist_diff': 'macd_hist_diff',
    'bb_upper': 'bb_upper', 'bb_lower': 'bb_lower',
    'close_raw': 'close',
}
assert set(_SAMPLE_SOURCE) == set(SAMPLE_COLUMNS), \
    'SAMPLE_COLUMNS 与 _SAMPLE_SOURCE 的列名集合不一致'

SAMPLE_SELECT_SQL = ', '.join(
    f'sd.{_SAMPLE_SOURCE[c]} AS {c}' for c in SAMPLE_COLUMNS
)

# `src/data.py` 归一到 ModelConfig.INPUT_DIM 维输入后，前 16 列是上面这些，
# 再追加 wick_up / wick_dn / body_ratio 三列 K 线形态占比。
MODEL_INPUT_DIM = 19


# ==================== 排除策略：哪些问题驱动「样本排除」 ====================

# `data_issues` 记录的是**事实**（哪天有什么问题）；
# 「哪些事实该导致样本被排除」是**另一个决策**，必须显式分开。
#
# 为什么必须分开：`resume_gap`（停牌复牌跳空）是正常市场事件 ——
# 记录它有价值（可审计、可解释），但它不是「坏数据」。
# 一旦把它计进排除，这个「数据质量」机制就悄悄变成了「停牌附近样本过滤器」：
# 实测 687,279 个被排除的采样位置里 **685,868 个（99.8%）来自 resume_gap**，
# 真正的问题（真错价/停牌残留/占位行/vwap/僵尸）合计只有约 2,900 个。
# 10% 的训练样本因为一个未声明的策略变化被丢掉，不该是默认行为。
#
# 若想利用「复牌日标签是虚假强势信号」这一点去排除它们，那是**另一个实验**
# （训练集规模直接变 10%），应当显式打开并重跑基线。
INFORMATIONAL_ISSUE_TYPES = ('resume_gap',)


def is_exclusion_type(kind: str) -> bool:
    """该问题类型是否驱动样本排除（False = 只记录，不影响训练集）"""
    return kind not in INFORMATIONAL_ISSUE_TYPES


# ==================== 特征基准 ====================

# MA/MACD/BB 建立在哪个价格上。必须是后复权，否则窗口跨除权日时特征断裂。
# 见 adjust_factor.py 与 docs/data-quality-system.md 第 8 节。
FEATURE_PRICE_COL = 'close_adj'
# 该列尚未物化时的行为：**报错**而不是静默退回不复权价。
# 历史缺陷：`features.py` 曾用 `not np.all(np.isfinite(closes))` 判定，
# 只要该股有任意一行 adj 列为空（增量更新后新行必然为空），
# 就会把**整只股票的历史特征**静默按不复权价重算 —— 一次增量更新即可
# 悄悄回退掉后复权的全部收益，且没有任何提示。
FEATURE_FALLBACK_IS_ERROR = True


# ==================== 股票池口径 ====================

# 训练与审计**必须**使用同一个池。历史上审计默认扫 'selected' 而下游训练的池
# 由另一段写死的 SQL 决定，两者靠「现在恰好一致」维持；一旦切换池，
# 审计覆盖不到的部分会静默失去质量筛查。
CURRENT_POOL = 'pit'   # 'selected'（今日口径） | 'pit'（逐日口径，见 pit_pool.py）

# 切到 'pit' 的前置（均已满足，2026-09-26）：
#   1. stock_status 已采集逐日 isST   → fetch_status --fetch
#   2. pool_membership 已带 isST 判定 → pit_pool --write
#   3. data_issues 覆盖 PIT 池        → audit --write（切池后必须重扫，见下）
#
# ⚠️ 内存提示：PIT 池 3,334 只 / 827 万行，比 selected（2,307 只 / 678 万行）多 22%。
#    全量加载父进程峰值本来就是数 GB 级，切池后线性放大；
#    并行度受 MAX_WORKERS（=4）限制，机器内存紧张时训练用 max_stocks 冒烟或再压 workers。
#
# ⚠️ 切池会改变训练数据，与所有历史实验不可比 —— 首次训练需重跑基线。

_POOL_JOIN = {
    'selected': (
        "JOIN stock_pool sp ON {alias}.stock_code = sp.stock_code\n"
        "               WHERE sp.pool_type='selected' AND sp.is_active=1"
    ),
    'pit': (
        "JOIN pool_membership pm ON pm.stock_code = {alias}.stock_code\n"
        "                       AND {alias}.date BETWEEN pm.start_date AND pm.end_date"
    ),
}

_POOL_CODES = {
    'selected': "SELECT stock_code FROM stock_pool WHERE pool_type='selected' AND is_active=1",
    'pit': "SELECT DISTINCT stock_code FROM pool_membership",
}


def pool_join_sql(alias: str = 'sd', pool: Optional[str] = None) -> str:
    """返回「限制到当前生效池」的 SQL 片段（JOIN + WHERE），供数据查询拼接。"""
    pool = pool or CURRENT_POOL
    if pool not in _POOL_JOIN:
        raise ValueError(f'未知股票池 {pool!r}，可选 {sorted(_POOL_JOIN)}')
    return _POOL_JOIN[pool].format(alias=alias)


def load_stock_codes(db_path: str = DEFAULT_DB, pool: Optional[str] = None) -> List[str]:
    """返回当前生效池的股票代码列表（审计、PIT、自检共用）。

    与 `pool_join_sql` 是同一个口径的两种表达，自检会断言两者一致。
    """
    pool = pool or CURRENT_POOL
    if pool not in _POOL_CODES:
        raise ValueError(f'未知股票池 {pool!r}，可选 {sorted(_POOL_CODES)}')
    con = sqlite3.connect(f'file:{db_path.replace(chr(92), "/")}?mode=ro', uri=True)
    try:
        return [r[0] for r in con.execute(_POOL_CODES[pool])]
    finally:
        con.close()


def pool_scope_label(pool: Optional[str] = None) -> str:
    """写进 provenance 的池标识，如 'selected' 或 'pit'。"""
    return pool or CURRENT_POOL
