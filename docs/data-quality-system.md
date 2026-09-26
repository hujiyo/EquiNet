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

### 2.3 污染区间下发（`sample_exclusion` 表）

审计产出的不是「脏行清单」，而是**「哪些样本不能用」**。

样本 `t` 的定义（`src/data.py`）：输入 = `[t-44, t]` 共 45 天，标签 = `[t+1, t+4]`。

若日期 `g` 的数据有问题，会污染的样本位置是 `t ∈ [g-4, g+44]`：
- 输入被污染 ⟺ `t-44 ≤ g ≤ t` ⟺ `t ∈ [g, g+44]`
- 标签被污染 ⟺ `t+1 ≤ g ≤ t+4` ⟺ `t ∈ [g-4, g-1]`

`sample_exclusion` 存的就是合并后的 `[g-4, g+44]` 闭区间。

## 3. 新增的表

| 表 | 内容 | 更新方式 |
|---|---|---|
| `data_issues` | 逐条问题：`(stock_code, date, issue_type, detail)` | `audit --write` |
| `sample_exclusion` | 污染样本区间：`(stock_code, start_date, end_date, reason)` | `audit --write` |
| `pool_membership` | PIT 股票池：`(stock_code, start_date, end_date, market_cap_median)` | `pit_pool --write` |

`pool_membership` 的 `end_date = 99999999` 表示「存续至今」，
其他值表示该股票在此时点之后已退出（退市 / 市值越界 / 长期停牌）。

## 4. 下游如何消费

### 4.1 跳过污染样本

`src/config.py`：

```python
EXCLUDE_DATA_ISSUE_SAMPLES = True    # 默认开启
```

`src/train.py` 在 `compute_label_distance_exclusions` 之后调用
`apply_quality_exclusions(train_stock_info)`，把区间换算成采样起始位置并并入
`stock_info['excluded_positions']`。对下游采样器完全透明，`_vectorized_process_stock`
无需改动。

**表不存在时自动跳过并打印提示，不影响训练。**

换算：样本起始索引 `s` 的上下文末日 = `times[s + C - 1]`，
故末日区间 `[l, r]` 对应 `s ∈ [pos(l) - (C-1), pos_after(r) - 1 - (C-1)]`。

### 4.2 使用 PIT 池（尚未切换）

`pool_membership` 已生成但**下游还没接**。切换方法与影响见第 6 节。

## 5. 实测数据（2026-09-25，训练池 2307 只 / 678 万行）

### 结构性指标

| 检查项 | 结果 |
|---|---|
| OHLC 逻辑违规 | 0 |
| 主键重复 | 0 |
| 训练池特征缺失（m5/m10/m20/macd_hist_diff） | 0 |
| 交易日历（4025 天） | 仅 4 天残缺，全在 2015-07 股灾停牌期 |

### 检出问题

| 类型 | 条数 | 占行数 |
|---|---|---|
| 价格异常跳变（除权/错价） | 3,491 | 0.051% |
| vwap 越界（量纲指纹） | 919 | 0.014% |
| 零成交量 | 50 | 0.001% |
| 占位垃圾行 | 50 | 0.001% |
| 僵尸价格（停牌填充） | 37 | 0.001% |

展开为 3,961 个污染区间，覆盖 1,729 / 2,307 只股票。
对候选样本的实际排除率约 **0.67%**（抽样 300 只实测）。

除权跳变在各年份均匀分布（约 150-300 次/年），**不是某一年的数据出了问题**。
典型样本：`000153 20240607 前收 7.74 → 5.61 (-27.5%)`，量能正常，是 10 送 4 一类的除权。

### PIT 池 vs 旧池（幸存者偏差量化）

| | 股票数 |
|---|---|
| 旧 `selected` 池 | 2,307 |
| PIT 池覆盖 | 3,338 |
| **仅 PIT 有**（历史曾符合条件、现已退出） | **1,031（44.7%）** |
| 仅旧池有 | 0 |

**旧池只覆盖了历史上曾符合条件股票的 55%。模型从未见过这 1,031 只的走势。**

## 6. 待决策

### 6.1 是否切换到 PIT 池

切换点：`src/data.py:load_and_preprocess_data` 的 SQL，把

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
建议新开一个 run 做 A/B，而不是直接替换。

### 6.2 ST 历史状态缺失

`pool_membership` 目前只做了市值与可交易性判定，
**没有历史 ST 标记**（`stock_metadata` 是空表，且无历史快照）。

Baostock 的 `isST` 字段可按日取到，需要重拉全历史才能补齐。
在补齐之前，PIT 池里会混入当时是 ST 的股票。

### 6.3 评估集是否也排除

当前 `apply_quality_exclusions` 只作用于 `train_stock_info`，
与既有 `excluded_positions` 的语义一致（只影响训练）。

但从正确性看，测试集里的除权日同样会污染输入与标签。
改成同时作用于评估集是几行的事，但会改变测试指标，需与 6.1 一起决策。

### 6.4 数据停更

库内最新数据 2026-07-31，已停更约 2 个月（原负责更新的同学离职）。
恢复更新后需要重跑：`选项1 增量更新` → `选项6 离线质量审计` → `选项7 PIT 池`。

### 6.5 `stock_metadata` 表是空的

整个元数据子系统从未被写入过（0 行）。
`select.py` 的 ST 过滤只能靠运行时拉取股票名做文本匹配。
需要决定：回填还是废弃这张表。

## 7. 日常运维流程

```bash
# 数据更新后
python data_maintenance.py      # 选项 1：增量更新（写入门禁自动生效，特征亦在此时重算）
python data_maintenance.py      # 选项 6：离线质量审计（重算污染区间）
python src/market_index.py      # 重建市场宽度数据

# 股票池重筛后（会改变入池股票）
python data_maintenance.py      # 选项 2：筛选股票
python data_maintenance.py      # 选项 6：重新审计（新入池股票的特征需确认已算）
python data_maintenance.py      # 选项 7：重算 PIT 池

# 训练
python src/train.py             # 自动读取 sample_exclusion 并跳过污染样本
```

**顺序很重要**：审计必须在数据更新之后、训练之前。
否则 `sample_exclusion` 不包含新数据里的问题日。

> **已知文档不一致**：README「数据管理」表格声称菜单有「4. 计算特征」，
> 但 `data_maintenance.py` 的选项 4 实际是「数据库状态」，且没有独立的计算特征入口。
> 特征计算目前只在 `update.py:update_single_stock` 内部顺带执行
> （`compute_features_for_stock`）。README 待更正。
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
   - `sd.open_adj/high_adj/low_adj/close_adj/vwap_adj` 以**原列名**别名取出，
     下游索引全部不用改
   - 末尾追加 `sd.close AS close_raw`（第 17 列），供高价股过滤使用
   - `cols` 追加 `'close_raw'`；`stock_data` 由 `[N,16]` 变 `[N,17]`
2. **两处硬编码绝对价格过滤已改用不复权价**
   - `src/data.py` `normalize_and_validate_context_window`：`input_seq_raw[:, 16][-1] > 40`
   - `src/data.py` `_vectorized_process_stock`：`raw_windows[:, -1, 16] <= 40`
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

### 仍未实测的部分

`train.py` 走的是**全量路径**（不传 `max_stocks`）。该分支的 SQL 是原始
`JOIN stock_pool ...` 子句未改动，只有 SELECT 列名变了（列名已在子集路径验证过），
所以风险低，但**全量路径本身尚未实测**。首次全量训练前建议先跑一次
`python src/train.py` 并观察内存。

## 8.8 关于「僵尸价格」的更正

早期审计用「OHLC 完全相等」判僵尸价格，把**连续一字涨停板**也算了进去，
导致报告里出现「12 只股票有 57~94 天零变化」的误导性描述。

实测核对后：
- `600666` 的 85 天「零变化」实际是连续一字涨停，日涨幅 +9.95%~10.03%、volume 逐日不同 → **合法**
- `002052` 同理
- `audit.py:find_zombie_runs` 的判定（要求**连续且收盘价完全相同**）是**正确的**，
  它只标了 **37 行，全部属于 `600636` 的 2026-04-30 ~ 2026-05-29 连续 19 日**（收盘价 4.51 不变）→ 真僵尸

另有 4,764 行 `amount=0 / volume=0` 的停牌记录，**价格是真实的**（如 `688121 20260602`
close 6.48 → close_adj 6.538），不是垃圾，予以保留。
