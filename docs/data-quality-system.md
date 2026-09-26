# 数据质量体系（Data Quality System）

> 建立于 2026-09-25。本文档描述 `data_maintenance` 的数据正确性保障机制。

## 1. 这套东西解决什么问题

原 `data_maintenance` 的问题不是「脏数据多」，而是**没有守门员**：

- 清洗逻辑只有 `utils.normalize_stock_df` 一个入口，`database.py` 的
  `upsert_daily_data` / `bulk_upsert_daily` 不经过它 —— 任何直接调用都能灌脏数据
- 数据入库后**没有任何校验**，问题只能靠事后人工发现
- 没有 point-in-time 概念，`select.py` 用「今天」的市值/ST 状态筛出股票池，
  再套用到十年历史上 —— 幸存者偏差
- 没有数据血统，无法追溯某一行来自哪次更新

现在的做法：**写入时拦截 + 离线审计 + 污染区间下发**。

## 2. 三个环节

### 2.1 写入门禁（`quality.gate_dataframe`）

接入点：`update.py` 的 `_apply_ingest_gate`，在 Baostock / AKShare 两条抓取路径上
入库前执行。

**硬拦截**（直接丢弃，这些行在任何数据源下都不该存在）：

| 类型 | 判定 |
|---|---|
| `ohlc_inconsistent` | `high<low`、`close` 在 `[low,high]` 外、`open` 在 `[low,high]` 外 |
| `nonpositive_price` | `open/high/low/close` 任一 ≤ 0 |
| `zero_volume` | `volume ≤ 0` 或 `amount ≤ 0`（停牌快照，不是 K 线） |
| `placeholder_row` | `amount ≤ 1` 或 `volume ≤ 1`（占位垃圾行） |

**不拦截**（只记录，交人工决策）：

- `vwap_out_of_range`：数据源量纲差异的指纹，可能是真实数据
- `price_anomaly`：不复权下的除权日，属真实事件而非错误

### 2.2 离线审计（`data_maintenance/audit.py`）

```bash
python -m data_maintenance.audit                     # 只报告
python -m data_maintenance.audit --write             # 同时落库
python -m data_maintenance.audit --pool all          # 扫全量池
```

或走交互菜单 `python data_maintenance.py` → 选项 6。

不联网、不修改任何行情数据，只读 `stock_daily`。2307 只 / 678 万行约 15 秒。

识别 7 类问题：OHLC 不自洽、价格非正、零成交量、占位垃圾行、vwap 越界、
价格异常跳变（除权/错价）、僵尸价格（停牌填充）。

**涨跌幅上限按板块与日期判定**（`quality.price_limit`）：
科创板 688xxx 全程 ±20%；创业板 300xxx 在 2020-08-24 起 ±20%，此前 ±10%；
主板 ±10%。容差 1.15 倍。新股上市前 5 个交易日豁免（注册制下无涨跌幅限制）。

### 2.3 污染区间下发（`data_issues` → 消费侧展开）

审计产出的**只有「哪天有问题」**（`data_issues`），不产出「哪些样本不能用」。

展开是**索引空间**的事，只在一处发生：`src/data.py:apply_quality_exclusions` 用
`contract.SAMPLING.excluded_starts(p, T)` 把问题日的位次 `p` 映射成被污染的采样起始索引
`s ∈ [p-48, p]`。规则由 `contract.SAMPLING` 唯一定义（左侧 4 = 标签侧 + buffer，
右侧 44 = 上下文长度 - 1），每个问题日展开 49 个位置。

> **为什么不再有 `sample_exclusion` 表**：那曾是该语义的第二份表达 ——
> 审计侧自己也展开一遍并落库。结果两份表达不一致：审计侧把序号空间的 ±4/±44
> 直接加在日期整数上（`20150911 + 44 = 20150955` 不是合法日期，`searchsorted`
> 于是把右界定位到「问题日所在月的月末」），实测 **74.5% 的污染采样位置被漏排**，
> 3,961/3,961 个区间日期非法。同一语义两处表达，必然分叉。

现在 `sample_exclusion` 表已删除（`audit --write` 的 schema 脚本里显式 `DROP`），
`audit_scan` 台账记录「本次扫过哪些股票」（否则干净的股票与没扫过的股票在库里长得一样）。

## 3. 新增的表

| 表 | 内容 | 更新方式 |
|---|---|---|
| `data_issues` | 逐条问题：`(stock_code, date, issue_type, detail)` | `audit --write` |
| `audit_scan` | 扫描台账：`(stock_code, n_rows, n_issues)` —— 「扫过」与「有问题」是两件事 | `audit --write` |
| `pool_membership` | PIT 股票池：`(stock_code, start_date, end_date, market_cap_median)` | `pit_pool --write` |
| `stock_status` | 逐日外部状态：`(stock_code, date, tradestatus, is_st)` | `fetch_status --fetch` |
| `data_meta` | 派生状态血缘：`(key, scope, source_max_date, rows, generated_at)` | 各产出模块自动登记 |

`pool_membership` 的 `end_date = 99999999` 表示「其数据末段仍符合条件」。

> 注意：退市股若在退市前仍符合条件，也会被记为 `99999999`。
> 这一列只用于下游的**日期区间 JOIN**，不要拿它判断「今天该选谁」。

`data_meta.source_max_date` 是**新鲜度时钟**：某份派生状态生成时看到的
`stock_daily` 最新日期。源数据之后又更新过 → 该状态过期（`selfcheck` 会报）。
用数据本身的事实而不是机器时间戳，避免时区/时钟污染。

### 3.1 契约：唯一事实来源

采样几何、样本列语义、池口径、特征基准、并行度上限全部定义在
**`data_maintenance/contract.py`** 一处。`src/config.py` 只引用不复述，并带启动断言。

这条纪律是有代价换来的：`CONTEXT_LENGTH/FUTURE_DAYS/BUFFER_DAY` 曾在
`src/config.py` 与 `data_maintenance/quality.py` 里各写一份，改动任一处会让污染区间
**静默错位**（且方向是「排漏」——脏数据照常进训练集，不报错）。

## 4. 下游如何消费

### 4.1 跳过污染样本

`src/config.py`：

```python
EXCLUDE_DATA_ISSUE_SAMPLES = True    # 默认开启
```

`src/train.py` 在 `compute_label_distance_exclusions` 之后调用
`apply_quality_exclusions(train_stock_info)`，从 `data_issues` 读问题日，
按 `contract.SAMPLING` 展开成采样起始位置并入 `stock_info['excluded_positions']`。
对下游采样器完全透明，`_vectorized_process_stock` 无需改动。

### 4.1.1 训练前的自检（关键步骤不能靠人记得）

`src/train.py` 在排除之前调用 `data_maintenance.selfcheck.prerequisites()`，
校验：审计范围是否等于训练用的池、派生状态是否比源数据旧、派生表是否只有一份表达。
`DataConfig.STRICT_SELFCHECK = True` 时发现 error **直接中止训练**。

**表不存在时自动跳过并打印提示，不影响训练。**

换算：样本起始索引 `s` 的上下文末日 = `times[s + C - 1]`，
故末日区间 `[l, r]` 对应 `s ∈ [pos(l) - (C-1), pos_after(r) - 1 - (C-1)]`。

### 4.2 使用 PIT 池

切换点是 `contract.CURRENT_POOL`（一处），训练与审计同时生效：

```python
CURRENT_POOL = 'pit'   # 'selected'（今日口径） | 'pit'（逐日口径）
```

`selected` 与 `pit` 的 JOIN/取码 SQL 都由 `contract.pool_join_sql()` /
`load_stock_codes()` 生成，**训练与审计不可能各用各的池**（这是曾经的静默失效来源）。

> **切到 PIT 会改变训练数据**，与所有历史实验不可比，需要重跑基线。
> 实测规模：`selected` 2,307 只 / 678 万行 → `pit` 3,338 只 / 827 万行（1.22×）。
> 内存影响按 1.22 倍线性估算；全量加载本来就会让父进程峰值上到数 GB，
> 所以并行度受 `contract.MAX_WORKERS`（=4）限制。

## 5. 实测数据（2026-09-25，训练池 2307 只 / 678 万行）

### 结构性指标

| 检查项 | 结果 |
|---|---|
| OHLC 逻辑违规 | 0 |
| 主键重复 | 0 |
| 训练池特征缺失（m5/m10/m20/macd_hist_diff） | 0 |
| 交易日历（4025 天） | 仅 4 天残缺，全在 2015-07 股灾停牌期 |

### 检出问题

**2026-09-25 首版口径**（除权日未豁免 + 展开有 bug，已废弃，保留作对照）：
`价格异常跳变（除权/错价混在一起）` 3,491、vwap 越界 919、零成交量 50、
占位垃圾行 50、僵尸价格 37；展开为 3,961 个区间，覆盖 1,729 / 2,307 只股票。

**2026-09-26 当前口径**（`audit --write` 实跑，2,307 只 / 678 万行，13.2s）：

| 类型 | 条数 | 说明 |
|---|---|---|
| `vwap_out_of_range` | 919 | 数据源量纲指纹，交人工判断 |
| `price_anomaly` | **131** | 真错价。原 3,491 条里 **3,360 条（96.2%）当天恰有除权事件**，后复权后已无害，属白排 |
| `zero_volume` | 50 | 停牌快照存量，口径为「不应存在」（见 §8.8） |
| `placeholder_row` | 50 | 占位垃圾行存量 |
| `zombie_price` | 37 | 停牌填充，全部属于 `600636` 的 2026-04-30~05-29 连续 19 日（经 `stock_status` 外部确认 `tradestatus=0`） |

受影响股票从 1,729 只降到 **540 只**；展开后实际会排除 **11,255 个采样起始位置**
（占已扫行数 0.166%）—— 这个数由扫描侧用与消费侧同一个 `SAMPLING.excluded_starts`
算出，因此是「真的会排除多少」，不再是只在纸上成立的数字。

> **口径变化的教训**：后复权体系落地后，`check_price_anomaly` 必须豁免除权日，
> 否则审计会持续把 3,360 条合法除权报成问题 —— 既白排一批本来干净的样本，
> 又把真正需要人看的 131 条错价淹在噪声里（实际上就等于没人会去看）。
> **审计规则必须跟着数据口径演进**，这也是 `selfcheck` 存在的理由之一。

除权跳变在各年份均匀分布（约 150-300 次/年），**不是某一年的数据出了问题**。
典型样本：`000153 20240607` 前收 7.74 → 5.61（-27.5%），量能正常，是 10 送 4 一类的除权。

### PIT 池 vs 旧池（两种偏差，不要混成一个数）

| | 股票数 |
|---|---|
| 旧 `selected` 池 | 2,307 |
| PIT 池覆盖 | 3,338 |
| 仅 PIT 有 | 1,031 |
| 　├─ 数据已停止更新（退市/长停）→ **真·幸存者偏差** | ~190（约 8.2% of 旧池） |
| 　└─ 仍在交易 → 「用今天的市值/ST 套历史」的**前视偏差** | ~815 |
| 仅旧池有 | 0 |

> ⚠️ **不要把「仅 PIT 有 1,031 只 = 44.7%」当成幸存者偏差。**
> 只有退市/长停那约 190 只属于幸存者偏差；其余是市值与 ST 口径的前视偏差。
> 混在一起会导致误判优先级（例如以为该先补 ST 历史，实际市值前视是大头）。
> `pit_pool.main()` 现在会把两类分开打印。

## 6. 待决策

### 6.1 是否切换到 PIT 池（已决定：切）

切换点是 **`contract.CURRENT_POOL`**（一处），训练与审计同时生效，见 §4.2。
切换前必须按顺序：`fetch_status --fetch` → `pit_pool --write`
→ `audit --write --pool pit` → `selfcheck`。

> 下面的 SQL 是**历史切换点**，保留以说明它为什么被换掉：原来训练与审计
> 各有一段写死的池 SQL，靠「现在恰好一致」维持。一旦切换池，审计覆盖不到的
> 股票会静默失去质量筛查 —— 所以池口径必须收敛到一处。

原 `src/data.py:load_and_preprocess_data` 的 SQL：

```sql
JOIN stock_pool sp ON sd.stock_code = sp.stock_code
WHERE sp.pool_type='selected' AND sp.is_active=1
```

改为

```sql
JOIN pool_membership pm ON sd.stock_code = pm.stock_code
                       AND sd.date BETWEEN pm.start_date AND pm.end_date
```

**注意**：这会改变训练数据，与所有历史实验不可比，需要重跑基线。

### 6.2 ST 历史状态（已解决）

`fetch_status.py` 采集逐日 `isST` 到 `stock_status` 表；
`pit_pool._pit_one` 已把它接进入池条件（`~is_st`），且 `stock_status` 缺失时
**直接拒绝运行**而不是降级 —— 降级的结果就是把「当时是 ST」的股票当成正常股票放进池里。

`stock_metadata` 仍然是空表。它是「当前状态的快照」，按日历史状态归 `stock_status` 管，
两者职责不要混。

### 6.3 评估集是否也排除

当前 `apply_quality_exclusions` 只作用于 `train_stock_info`。

`_vectorized_process_stock` 里 `if excluded and not start_min_override` 这行守卫
保证了评估集自动免疫 —— 是靠结构，不是靠调用顺序。

但从正确性看，测试集里的真错价同样会污染输入与标签。
改成同时作用于评估集是几行的事，但会改变测试指标。

### 6.4 数据停更

库内最新数据 2026-07-31，已停更约 2 个月（原负责更新的同学离职）。
恢复更新后按 §7 的顺序重跑。**`selfcheck` 会直接指出哪一步没跟上**
（靠 `data_meta.source_max_date` 判断，不靠人记得）。

### 6.5 `stock_metadata` 表是空的

整个元数据子系统从未被写入过（0 行）。
`select.py` 的 ST 过滤只能靠运行时拉取股票名做文本匹配 ——
**这条路径已被证明不可靠**：实测 `600636` 是 ST 股（`stock_status.is_st=1`），
却出现在 `selected` 训练池里（见 §8.9）。

## 7. 日常运维流程

```bash
# 数据更新后（有依赖顺序，selfcheck 会校验）
python data_maintenance.py      # 选项 1：增量更新
                                #   门禁自动生效；末尾自动补复权/特征并登记 provenance
python data_maintenance.py      # 选项 6：离线质量审计（重扫 data_issues）
python src/market_index.py      # 重建市场宽度数据

# 股票池相关
python data_maintenance.py      # 选项 2：筛选股票（select.py，今日口径）
python data_maintenance.py      # 选项 7：采集逐日 tradestatus/isST（首次或增量）
python data_maintenance.py      # 选项 8：生成 PIT 池（依赖上一项）
python data_maintenance.py      # 选项 6：重新审计（池变了范围要跟着变）

# 训练
python src/train.py             # 启动即自检；自动读 data_issues 跳过污染样本

# 随时
python data_maintenance.py      # 选项 9：系统自检
```

**顺序不是靠背的。** `provenance.DEPENDENCIES` 声明依赖关系，
`selfcheck` 用 `data_meta.source_max_date` 判定每份派生状态是否比源数据旧。
漏跑任何一步都会显示为 `stale`，而不是静默地让训练用过期结论。

> **历史教训**：这一段原来是一句「顺序很重要」的叮嘱，而且**漏了复权物化**。
> 更糟的是门禁挂在 `update.py` 的两条抓取函数上，`check.py` 的修复流程直接调
> `db.upsert_daily_data` 就绕过去了 —— 恰恰是最需要门禁的时候（修数据时）。
> 现在门禁是 `DatabaseManager` 的唯一收口，依赖关系是可断言的表。

> **README 待更正**：README「数据管理」表格声称菜单有「4. 计算特征」，
> 但实际没有独立的计算特征入口 —— 特征只在 `update.py` 的更新流程里顺带执行
> （`compute_features_for_stock`），且 `close_adj` 未物化时会抛
> `FeatureBasisMissing` 而不是静默退回不复权价。
>
> 这解释了一个现象：全库 `macd_hist_diff` 有 50.64% 为 NULL，
> 且分布恰好是「不在 selected 池 = 100% 空，在 selected 池 = 0% 空」——
> 说明历史上曾用 `compute_features(pool_type='selected')` 只补算过训练池。
> 训练只查 selected 池，因此**不影响训练**；但换池后必须重算特征。

---

# 8. 后复权价格体系（2026-09-25 新增）

## 8.1 为什么用后复权

三种口径的对比：

| 口径 | 前视偏差 | 除权跳空 | 增量更新 |
|---|---|---|---|
| 不复权（原方案） | 无 | **有**（序列断裂） | 安全 |
| 前复权 qfq | **有**（历史价被未来除权回溯改写） | 无 | 每次除权要重算全历史 |
| **后复权 hfq（现方案）** | **无**（基准固定在最早日，历史价永不改变） | 无 | **安全** |

「复权有信息泄露风险」是**前复权**的缺陷，不是复权的缺陷。
项目只消费涨跌幅与相对特征（`open_rel` / `close_rel` / MA 偏离度），
都是比值 —— 后复权导致的绝对价格膨胀对它们无影响。

## 8.2 关键实测结论（`tmp_tests/probe_hfq.py`）

对 `sz.000153`（除权日 2024-06-07）同时拉 `adjustflag=1` 与 `adjustflag=3`：

| 列 | 后复权下是否调整 |
|---|---|
| `close` | ✅ 调整（比值 = 复权因子） |
| `volume` / `amount` / `turn` | ❌ **完全不调整**（比值恒 = 1.0） |

**因此：后复权价 = 不复权价 × 复权因子。不需要重拉 K 线**，
只需拉因子表即可在现有 1,374 万行上派生。零重拉风险。

除权跳空确实消失：`000153` 在 2024-06-07 的日涨跌从 **-27.52% 变为 +3.51%**（真实涨幅）。

## 8.3 因子推导规则（`tmp_tests/probe_factor_rule.py` 实测确认）

```
对日期 d： k(d) = 「dividOperateDate <= d 的最近一条 backAdjustFactor」
若 d 早于所有除权日，则 k(d) = 1.0
```

验证：`600519` 全历史 6,086 个交易日、`601003` 4,763 个交易日，
逐日比对 `close_hfq / close_unadj` 与推导值，**零不符**（最大相对误差 2e-16）。

> ⚠️ **必须从 `1990-01-01` 开始查询因子**。实测 `000001`：
> - start=1990-01-01 → 42 条，首条 `1991-04-03 / k=1.000000` ✓
> - start=2000-01-01 → 25 条，首条 `2000-11-06 / k=28.953` ✗ 历史被截断
>
> 代码里已固化为常量 `FACTOR_START = '1990-01-01'`。

## 8.4 产出

**表 `adjust_factor`**（事件级，64,368 行，覆盖 5,455 只股票）

```
(stock_code, divid_operate_date, fore_adjust_factor, back_adjust_factor, updated_at)
```

**`stock_daily` 新增 6 列**

| 列 | 含义 |
|---|---|
| `adj_factor` | 该行生效的后复权因子 |
| `open_adj` / `high_adj` / `low_adj` / `close_adj` | 后复权价 = 原始价 × `adj_factor` |
| `vwap_adj` | **必须一并换算**：`vwap = amount/volume` 用的是不复权的量纲，而 close 是后复权价，不换算会让 `(vwap - close)/close` 特征彻底失准 |

原始列（`open/high/low/close/vwap`）**保持不变** —— raw 层不可变，可回滚、可对照。

## 8.5 操作

```bash
python -m data_maintenance.adjust_factor --fetch --materialize --verify
python -m data_maintenance.adjust_factor --fetch --only-missing   # 断点续拉
```

- `--fetch`：拉全市场因子。实测单只 0.19s，4 进程全市场约 4 分钟。
  baostock 会话会在长批量查询中途失效（返回「用户未登录」），
  代码已内置自动重登 + 重试 + `--only-missing` 断点续拉。
- `--materialize`：填充 6 个新列。实测 5,456 只 / 13.7M 行约 98 秒，51,534 条 UPDATE。
- `--verify`：抽样比对派生值与 baostock 直拉的后复权 close。

## 8.6 验证结果（2026-09-25）

抽样 5 只股票共 18,508 个交易日：**18,507 一致，1 不一致**。

唯一的不一致是 `000001 20260430` —— 该行 OHLC 全为 `99.0`、`amount=1.0`、`volume=1.0`，
是一行**伪造数据**（当日真实价约 11.5）。这是源数据本身的错，不是因子推导的错；
反过来说明该行的 `close_adj` 会被算成 11,972，必须清除。

**已清除**：`('000001', 20260430)` 与 `('__TEST__', 99999998)` 共 2 行
（13,745,808 → 13,745,806）。全库备份见 `data_maintenance/backup/equinet_pre_hfq_20260925.db`。

清除后 `000001` 序列恢复连续（20260429 close=11.52 → 20260506 close=11.36，无 99.0 尖峰）。

## 8.7 下游切换（2026-09-25 已完成并验证）

三项改动全部落地：

1. **`src/data.py` 已切到复权列**（`load_and_preprocess_data` 的 SQL）
   - SELECT 语句现由 `contract.SAMPLE_SELECT_SQL` **生成**（列名→物理列的映射
     只在 `contract._SAMPLE_SOURCE` 一处定义），因此不存在「改了 SELECT
     忘了改 `cols`」这种会让模型静默读错列的错位
   - 价格列以**原列名**别名取出（`open_adj AS open` …），下游索引不用改
   - 末尾追加 `sd.close AS close_raw`（`CLOSE_RAW_IDX` = 16），供高价股过滤使用
2. **两处硬编码绝对价格过滤已改用不复权价**（下标改用具名常量，不再写 `16`）
   - `src/data.py` `normalize_and_validate_context_window`：`input_seq_raw[:, CLOSE_RAW_IDX][-1] > 40`
   - `src/data.py` `_vectorized_process_stock`：`raw_windows[:, -1, CLOSE_RAW_IDX] <= 40`
3. **9 个衍生特征已在后复权价上重算**
   - `features.py:compute_features_for_stock(..., price_col='close_adj')`
   - `database.py:_STOCK_DAILY_COLUMNS` 加入复权列以支持读取
   - 全库 5,455 只、13.7M 行重算完毕（8 进程 168 秒，0 错误）
   - **顺带修复**：`macd_hist_diff` 历史 50.64% NULL → 0%

**同时新增两个可选参数**（默认行为不变，训练路径不受影响）：

```python
load_and_preprocess_data(max_stocks=None, num_workers=None)
```

用于低内存冒烟验证。背景：全量加载 678 万行时父进程峰值可达数 GB；
2026-09-25 用默认 8 进程跑全量验证时把用户机器（31.7 GB）压到死机。
`max_stocks=100` 时峰值仅 **829 MB**。

### 验证结果（`tmp_tests/e2e_lowmem.py`，100 只股票 / 2 进程）

| # | 检查项 | 结果 |
|---|---|---|
| a | 低内存加载（10.3s） | 通过 |
| b | `stock_data` 列数 = 17 | 通过 (3900, 17) |
| c/d | 与 DB 逐位对账 **18,926 行** | 第3列 vs `close_adj` 错 **0**；第16列 vs `close` 错 **0**；比值 vs `adj_factor` 错 **0** |
| e | 高价股过滤口径差异 | 不复权口径 1.93% / 复权口径 **40.11%**（差值 **38.18 个百分点**） |
| f | 评估集构建 | `(8703, 45, 19)`，无 NaN/inf |
| g | 质量排除 | 100 只中 68 只命中，排除 1,463 个采样位置 |
| h | 除权日标签修正 | 见下 |

**e 是这次改动最大的实际收益**：如果不改那两处过滤，用后复权价判 `> 40`
会误杀 **40%** 的样本（远高于早前按复权因子 ×2/×3 估的 18%/35%）。

**h 抓到一个教科书级案例** —— `000517`，除权日 2015-09-11，复权因子跳变 ×3.0086：

```
原始价:  20150910  14.05 -> 4.46   = -68.26%   <- 假的暴跌
复权价:  20150910  55.507 -> 53.011 = -4.50%   <- 真实跌幅
```

样本末日 `20150908`（除权日落在 T+3）：

| 口径 | day3 | 标签 |
|---|---|---|
| 不复权 | **-68.26%** | **0**（判为无强势信号） |
| 后复权 | **-4.50%** | **1**（判为强势信号） |

**标签翻转了。** 不复权数据把一个真实的强势信号教成了负样本 —— 这就是"除权污染"
在标签层的实证，比看任何 AUC 数字都直接。

### 实测状态（2026-09-26 更新）

- 加载路径（`max_stocks` 子集）已在 2026-09-25 逐位对账验证；列语义此后收敛到
  `contract.SAMPLE_COLUMNS` / `SAMPLE_SELECT_SQL`，SELECT 由契约生成。
- 污染区间的换算已用**真实函数**（`data.apply_quality_exclusions`，非复刻）
  交叉验证：1,729 只命中、**175,467 个采样位置**，与独立复算完全一致、越界 0。
- `train.py` 全量路径（不传 `max_stocks`）仍未实测。首次全量训练前跑一次
  `python src/train.py` 并观察内存；`MAX_WORKERS=4`（原 8）。

## 8.8 「僵尸价格」与停牌行口径的更正

早期审计用「OHLC 完全相等」判僵尸价格，把**连续一字涨停板**也算了进去，
导致「12 只股票有 57~94 天零变化」的误导性描述。

核对后：
- `600666` 的 85 天「零变化」实际是连续一字涨停（日涨幅 +9.95%~10.03%、volume 逐日不同）→ **合法**
- `002052` 同理
- `audit.py:find_zombie_runs` 的判定（要求**连续且收盘价完全相同**）是**正确的**，
  只标了 **37 行，全部属于 `600636` 的 2026-04-30 ~ 2026-05-29 连续 19 日**（收盘价 4.51 不变）
- 该 37 行已由 `stock_status` 外部确认：这 19 天 `tradestatus=0`（**真停牌**）→ 判定正确

### 停牌行口径（2026-09-26 统一）

库内 **4,763 行** `amount=0 / volume=0` 的记录。

**旧口径**：「价格是真实的（如 `688121 20260602` close 6.48 → close_adj 6.538），
不是垃圾，予以保留。」← **这个口径与门禁冲突**：写入门禁一直硬拒
`volume<=0 | amount<=0`，也就是说同样的数据重拉一次就会被丢掉，
而存量却留着 —— 库里的状态和门禁的策略长期对不上。

**现口径**：统一为**拒收**（停牌不是 K 线），依据是 `quality.find_zombie_runs`
早已写下的原则 ——「真正的停牌应该是「没有记录」，而不是「复制一条记录」」。
这 4,763 行是历史遗留（写入时还没有门禁），**待清理**；
`selfcheck` 会统计它们并用 `stock_status.tradestatus` 判定其性质，
若出现「大量零成交行并非停牌」则报 `error`（说明是另一种数据源噪声）。

## 8.9 `stock_status` 带出的新问题：训练池里有 ST 股

`600636`（上述僵尸段的主角）经 `stock_status` 确认 `is_st=1`，
但它**出现在 `selected` 训练池里**。

原因：`select.py` 的 ST 过滤走 AKShare 当前名称做文本匹配（`'ST' in name`），
只反映「今天叫什么」，且依赖联网成功。它既不按日、也不可靠。

这正是切到 PIT 池的价值：`pool_membership` 现在按日读取 `is_st`，
`pit_pool._pit_one` 要求 `~is_st` 才入池，且 `stock_status` 缺失时直接拒绝运行。
