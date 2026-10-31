# 特征工程与归一化完整指南

## 全局数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Baostock / AKShare API                                                 │
│  返回原始行情数据（OHLC价格、成交量/额、换手率）                          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  data_maintenance/update.py                                              │
│  写入数据库9个原始列: date, open, high, low, close,                       │
│                       amount, volume, exchange, vwap                     │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  data_maintenance/features.py                                            │
│  在数据库中追加8个衍生列: m5, m10, m20, dif, dea, macd_hist,              │
│                          bb_upper, bb_lower                               │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  SQLite 数据库 stock_daily 表 (26列)                                     │
│  OHLC、amount / volume / exchange / vwap、m5/m10/m20、MACD、布林带、     │
│  后复权价（*_adj）、close_raw、adj_factor 等                             │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  src/data.py · 粗处理 (coarse normalization)                             │
│  从 16 列样本（SAMPLE_COLUMNS）中取 15 列（去掉 close_raw；date 单独     │
│  用于切分），转换为去量纲的变化率/偏离度                                 │
│  输出 [45, 18]                                                           │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  src/data.py · 细处理 (FeatureNormalizer)                                │
│  18组独立 QuantileTransformer + StandardScaler                           │
│  输出 [45, 18]                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 第一步：API 原始数据

### 数据源

项目使用 **Baostock**（默认）或 **AKShare** 作为数据源，定义在 [update.py](../data_maintenance/update.py)。

### API 返回的原始字段

| 数据库列名 | 含义 | 单位/说明 | Baostock 字段 | AKShare 字段 |
|-----------|------|----------|--------------|-------------|
| `open` | 开盘价 | 元 | `open` | `开盘` |
| `high` | 最高价 | 元 | `high` | `最高` |
| `low` | 最低价 | 元 | `low` | `最低` |
| `close` | 收盘价 | 元 | `close` | `收盘` |
| `amount` | 成交额 | 元 | `amount` | `成交额` |
| `volume` | 成交量 | 股 | `volume` | `成交量` |
| `exchange` | 换手率 | %（百分比数值，如 2.5 表示 2.5%） | `turn` | `换手率` |
| `vwap` | 成交量加权平均价 | 元 | **计算得出** `amount/volume` | **计算得出** `amount/volume` |

### VWAP 的计算

VWAP（Volume Weighted Average Price，成交量加权平均价）不在 API 中直接提供，由 `update.py` 从成交额和成交量计算：

```python
df['vwap'] = df['amount'] / df['volume'].replace(0, float('nan'))
df['vwap'] = df['vwap'].fillna(df['close'])  # 成交量为0时（停牌等）用收盘价替代
```

> **含义**：假设当天所有成交按金额加权后的"真实平均成交价"。如果 VWAP > Close，说明大部分成交发生在高位（盘中强势但尾盘回落）；反之亦然。

### 数据库中的 volume 列

`volume`（成交量，单位：股）是模型第 5 列输入（相对 10 日均量的变化率），同时用于计算 VWAP 与停牌校验。换手率 `exchange` 是它的派生量（`volume / 流通股本 × 100`，保留两位小数），两者在窗口内的差别只来自股本变动和那位小数取整。

---

## 第二步：衍生特征计算

原始 9 列入库后，由 [features.py](../data_maintenance/features.py) 计算并回写 8 个衍生特征列。所有衍生特征都做了**无量纲化**（除以 close 或 MA），确保不同价位的股票特征值在同一尺度上。

### 2.1 均线偏离度 (MA Deviation)

#### 含义

衡量当前收盘价偏离 N 日均线有多远。正值表示价格在均线上方（多头），负值表示在下方（空头）。

#### 公式

$$m_N = \frac{\text{close} - \text{MA}_N}{\text{MA}_N}$$

其中 $\text{MA}_N$ 是 N 日简单移动平均（Simple Moving Average）：

$$\text{MA}_N = \frac{1}{N} \sum_{i=0}^{N-1} \text{close}_{t-i}$$

#### 项目中的三个窗口

| 列名 | 窗口 N | 作用 |
|------|--------|------|
| `m5` | 5 | 周线级别偏离（超短期趋势） |
| `m10` | 10 | 双周线级别偏离（短期趋势） |
| `m20` | 20 | 月线级别偏离（中期趋势） |

#### 前期数据不足时的处理

当 $t < N$（数据前期不足一个窗口），不简单地截断，而是**向右借数据**补齐窗口：

```python
# 例：t=2, N=5 → 左侧只有2天数据，需要向右借3天
left_part  = closes[0:t]           # [day0, day1]
right_part = closes[t+1:t+1+deficit]  # [day3, day4, day5]
ma[t] = mean(concat(left_part, right_part))
```

> **为什么向右借而非向前填充？** A股数据是从上市日开始的连续序列，向前无数据可用。向右借是一种合理的近似，与 MA 在右侧边界的行为对称。

#### 代码参考

`features.py` → `compute_ma_features(closes, window)`

---

### 2.2 MACD 衍生特征

#### 含义

MACD（Moving Average Convergence Divergence，指数平滑异同移动平均线）是趋势跟踪指标。项目使用三个无量纲化分量。

#### 原始 MACD 计算

1. **快线 EMA₁₂** 和 **慢线 EMA₂₆**：

$$\text{EMA}_N(t) = \alpha \cdot \text{price}_t + (1-\alpha) \cdot \text{EMA}_N(t-1), \quad \alpha = \frac{2}{N+1}$$

2. **DIF（差离值）**：

$$\text{DIF}_{\text{raw}} = \text{EMA}_{12} - \text{EMA}_{26}$$

3. **DEA（信号线）**：

$$\text{DEA}_{\text{raw}} = \text{EMA}_9(\text{DIF}_{\text{raw}})$$

即对 DIF 再做一次 9 日指数移动平均。

4. **MACD 柱状图**：

$$\text{MACD\_hist}_{\text{raw}} = 2 \times (\text{DIF}_{\text{raw}} - \text{DEA}_{\text{raw}})$$

#### 无量纲化

原始 MACD 值与股价量级相关（高价股的 DIF 可能是几块钱，低价股只有几分钱），因此全部除以收盘价：

| 列名 | 公式 | 含义 |
|------|------|------|
| `dif` | $\frac{\text{EMA}_{12} - \text{EMA}_{26}}{\text{close}}$ | 快慢线偏离度（正值=上升趋势，负值=下降趋势） |
| `dea` | $\frac{\text{EMA}_9(\text{DIF})}{\text{close}}$ | 信号线偏离度（DIF 的平滑版） |
| `macd_hist` | $\frac{2 \times (\text{DIF} - \text{DEA})}{\text{close}}$ | 柱状图偏离度（>0红柱多头加速，<0绿柱空头加速） |

#### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| MACD_FAST | 12 | 快线周期 |
| MACD_SLOW | 26 | 慢线周期 |
| MACD_SIGNAL | 9 | 信号线周期 |

#### EMA 初始化与前期处理

EMA 需要初始值。项目使用窗口内 SMA 作为起点：

```python
ema[0] = mean(prices[:window])  # 前 window 个点的简单平均
```

前期数据不足时，与 MA 特征一致，采用**向右借数据**策略。

#### 代码参考

`features.py` → `compute_ema(prices, window)`, `compute_macd_features(closes)`

---

### 2.3 布林带偏离特征 (Bollinger Bands)

#### 含义

布林带衡量价格波动的"异常程度"。价格接近或突破上轨 = 相对高位，接近或突破下轨 = 相对低位。

#### 原始布林带计算

1. **中轨** = MA₂₀（20 日简单移动平均）
2. **上轨** = 中轨 + $k \times \sigma_{20}$（$k=2$）
3. **下轨** = 中轨 - $k \times \sigma_{20}$

其中 $\sigma_{20}$ 是 20 日收盘价的标准差（ddof=0）。

#### 无量纲化

| 列名 | 公式 | 含义 |
|------|------|------|
| `bb_upper` | $\frac{\text{close} - \text{UPPER}}{\text{close}}$ | 上轨偏离度（>0 表示突破上轨，<0 表示在上轨下方） |
| `bb_lower` | $\frac{\text{LOWER} - \text{close}}{\text{close}}$ | 下轨偏离度（>0 表示跌破下轨，<0 表示在下轨上方） |

> **注意符号约定**：`bb_upper` 正值=价格在上轨之上（超买），`bb_lower` 正值=价格在下轨之下（超卖）。两者通常一正一负。

#### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| BB_WINDOW | 20 | 布林带周期 |
| BB_STD_MULT | 2 | 标准差倍数 |

#### 代码参考

`features.py` → `compute_bb_features(closes)`

---

## 第三步：模型输入列（契约 `SAMPLE_COLUMNS`）

数据库 `stock_daily` 表本身有 26 列（含后复权价、`adj_factor`、`updated_at` 等）；
下游样本矩阵只取其中 16 列，列名与顺序由 `data_maintenance/contract.py` 的
`SAMPLE_COLUMNS` 唯一固定 —— 价格列取后复权（`*_adj`），`close_raw` 取不复权收盘价：

| 下标 | 列名 | 数据库来源 | 说明 |
|------|------|-----------|------|
| 0 | `open` | `open_adj` | 后复权开盘价 |
| 1 | `high` | `high_adj` | 后复权最高价 |
| 2 | `low` | `low_adj` | 后复权最低价 |
| 3 | `close` | `close_adj` | 后复权收盘价（模型输入基准） |
| 4 | `vwap` | `vwap_adj` | 后复权成交量加权均价 |
| 5 | `volume` | `volume` | 成交量（股），模型第 5 列输入 |
| 6 | `m5` | `m5` | (close-MA5)/MA5 |
| 7 | `m10` | `m10` | (close-MA10)/MA10 |
| 8 | `m20` | `m20` | (close-MA20)/MA20 |
| 9 | `dif` | `dif` | (EMA12-EMA26)/close |
| 10 | `dea` | `dea` | EMA9(DIF)/close |
| 11 | `macd_hist` | `macd_hist` | 2*(DIF-DEA)/close |
| 12 | `macd_hist_diff` | `macd_hist_diff` | hist[t]-hist[t-1] |
| 13 | `bb_upper` | `bb_upper` | (close-UPPER)/close |
| 14 | `bb_lower` | `bb_lower` | (LOWER-close)/close |
| 15 | `close_raw` | `close` | **不复权**收盘价，仅供高价股过滤 |

> **注意**：模型输入取下标 0–14 的 15 列，再追加 3 列 K 线形态占比
> （wick_up / wick_dn / body_ratio），共 18 维。`date` 不在样本矩阵里，单独用于数据分割和
> 标签计算；`close_raw` 只作过滤基准；`amount` / `exchange` 留在库里（市值、换手率口径等），
> 但不再进模型输入 —— 换手率是成交量的派生量。

---

## 第四步：粗处理 (Coarse Normalization)

粗处理将数据库中的原始数值转换为去量纲的变化率/偏离度，定义在 [data.py: `normalize_and_validate_context_window()`](../src/data.py)。

### 各特征的处理方式

#### OHLC（列 0-3）：日环比变化率

$$\text{OHLC}_t = \frac{\text{price}_t - \text{close}_{t-1}}{\text{close}_{t-1}}, \quad \text{clip}[-0.1, 0.1]$$

以**前一日收盘价**为基准计算变化率，并 clip 到 ±10%。

> **为什么 clip？** A股涨跌停限制为 ±10%（ST股 ±5%），超过这个范围的数据通常是复权异常或停牌恢复，属于噪声。

#### VWAP（列 4）：相对收盘价偏离

$$\text{VWAP}_t = \frac{\text{vwap}_t - \text{close}_t}{\text{close}_t}, \quad \text{clip}[-0.1, 0.1]$$

VWAP 与当日收盘价的偏离度：
- **正值**（VWAP > Close）：盘中均价高于收盘价 → 盘中强势但尾盘回落（抛压）
- **负值**（VWAP < Close）：盘中均价低于收盘价 → 盘中弱势但尾盘拉升（买盘强）

#### 成交量（列 5）：相对 N 日均值变化率

$$\text{Volume}_t = \frac{\text{volume}_t - \overline{\text{volume}}_{N}}{\overline{\text{volume}}_{N}}$$

其中 $\overline{\text{volume}}_{N}$ 是过去 N=10 天成交量的简单移动平均（不含当日）。

> **为什么不做 clip？** 成交量的变化没有硬性上界（放量可以翻数倍），clip 会丢失有价值的放量信号。

#### MA偏离度（列 6-8）：直传

$$m_{N,t} = \text{input\_raw}[:, 6+N]$$

已在 `features.py` 中预计算为 `(close-MA)/MA` 格式，本身就是无量纲偏离度，直接传入。

#### MACD（列 9-11）：直传

$$\text{dif}_t, \text{dea}_t, \text{macd\_hist}_t = \text{input\_raw}[:, 9:12]$$

已在 `features.py` 中预计算并除以 close，无量纲，直接传入。

#### 布林带（列 12-13）：直传

$$\text{bb\_upper}_t, \text{bb\_lower}_t = \text{input\_raw}[:, 12:14]$$

已在 `features.py` 中预计算，直接传入。

#### 影线占比（列 15-17）：K 线解剖

$$w_{\text{up}} = \frac{h - \max(o,c)}{h - l},\quad w_{\text{dn}} = \frac{\min(o,c) - l}{h - l},\quad b = 1 - w_{\text{up}} - w_{\text{dn}}$$

三者都落在 [0,1] 且构造处已裁剪，无需再 clip；一字板（h≈l）守卫填 0。

### 粗处理后的数据范围

| 特征 | 范围 | 均值 | 潜在问题 |
|------|------|------|---------|
| OHLC | [-0.1, 0.1] | ≈ 0 | 范围小，零均值 |
| VWAP | [-0.1, 0.1] | ≈ 0 | 范围小，零均值 |
| 成交量 | 不定 | ≈ 0 | 长尾分布（放量日变化率可达数倍） |
| MA偏离度 | 不定 | ≈ 0 | 偏态分布 |
| MACD | 不定 | ≈ 0 | DIF/DEA/柱状图分布各异 |
| 布林带 | 不定 | 不定 | 上/下轨偏离度分布各异 |

这些特征的范围和方差差异很大，这正是需要细处理的原因。

---

## 第五步：细处理 (Fine Normalization)

### 目的

将粗处理后范围各异的特征统一为**均值≈0、方差≈1**的标准化数据，使 Transformer 中所有维度在相同尺度上工作。

### 方法：QuantileTransformer + StandardScaler

#### QuantileTransformer

**核心思想**：将任意分布映射到正态分布。

**步骤**：
1. 对数据排序，计算每个值的**累积分布函数（CDF）**值
2. 对 CDF 值应用逆正态 CDF（Probit 变换）

$$x_{\text{normal}} = \Phi^{-1}(\text{CDF}(x))$$

**优势**：
- 自动处理偏态分布（成交量的长尾会被压缩）
- 对异常值鲁棒（极端值被映射到正态的尾部，不会被放大）
- 保证输出单调性（原始排序不变）

#### StandardScaler

QuantileTransformer 输出虽然是正态分布形状，但均值和标准差不一定精确为 0 和 1，所以追加 StandardScaler：

$$x_{\text{final}} = \frac{x_{\text{normal}} - \mu}{\sigma}$$

**输出**：严格保证均值 = 0，标准差 = 1。

### 特征分组与独立 Pipeline

不同特征的分布特性不同（价格类 vs 量能类 vs 技术指标类），为每组建立独立的 pipeline：

| Pipeline | 覆盖特征 | 设计原因 |
|----------|---------|---------|
| `open` / `high` / `low` / `close` | OHLC (col 0-3) | 价格变化率，范围 [-0.1, 0.1]，逐列独立拟合 |
| `vwap` | VWAP (col 4) | 均价偏离度，分布独立于 OHLC |
| `volume` | 成交量 (col 5) | 量能特征，长尾分布 |
| `m5` / `m10` / `m20` | 均线偏离度 (col 6-8) | 三个窗口分布相似，仍逐列拟合 |
| `dif` / `dea` / `macd_hist` / `macd_hist_diff` | MACD (col 9-12) | 四列分布形态各不相同 |
| `bb_upper` / `bb_lower` | 布林带 (col 13-14) | 上/下轨偏离度分布不同 |
| `wick_up` / `wick_dn` / `body_ratio` | 影线占比 (col 15-17) | 有 [0,1] 硬边界，必须单独拟合 |

**每列一个 pipeline**，共 18 组。逐列拟合意味着某一列的分布变化不会影响其它列的分位数估计。

每个 pipeline 结构：

```python
Pipeline([
    ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000)),
    ('scaler', StandardScaler())
])
```

> **为什么分组而非用一个全局 pipeline？** 不同特征组的分布形态差异大。OHLC 集中在 [-0.1, 0.1] 且近似对称，成交量则是长尾分布。如果混在一起拟合 QuantileTransformer，各自的分位数估计会互相干扰。

---

## 特征汇总表

下表是从 API 原始数据到模型输入的完整链路：

| 模型列号 | 最终特征名 | API 原始字段 | 衍生计算 | 粗处理 | 细处理 Pipeline |
|---------|-----------|-------------|---------|--------|----------------|
| 0 | Open | `open` | — | (open-close₁)/close₁, clip | open |
| 1 | High | `high` | — | (high-close₁)/close₁, clip | high |
| 2 | Low | `low` | — | (low-close₁)/close₁, clip | low |
| 3 | Close | `close` | — | (close-close₁)/close₁, clip | close |
| 4 | VWAP | `vwap` | amount/volume | (vwap-close)/close, clip | vwap |
| 5 | 成交量 | `volume` | — | (volume-MA₁₀)/MA₁₀ | volume |
| 6 | MA5 | `close` | (close-MA5)/MA5 | 直传 | m5 |
| 7 | MA10 | `close` | (close-MA10)/MA10 | 直传 | m10 |
| 8 | MA20 | `close` | (close-MA20)/MA20 | 直传 | m20 |
| 9 | DIF | `close` | (EMA12-EMA26)/close | 直传 | dif |
| 10 | DEA | `close` | EMA9(DIF)/close | 直传 | dea |
| 11 | MACD_hist | `close` | 2*(DIF-DEA)/close | 直传 | macd_hist |
| 12 | MACD_hist_diff | `close` | hist[t]-hist[t-1] | 直传 | macd_hist_diff |
| 13 | BB_upper | `close` | (close-UPPER)/close | 直传 | bb_upper |
| 14 | BB_lower | `close` | (LOWER-close)/close | 直传 | bb_lower |
| 15 | wick_up | `open·high·low·close` | (h-max(o,c))/(h-l) | 直传 | wick_up |
| 16 | wick_dn | `open·high·low·close` | (min(o,c)-l)/(h-l) | 直传 | wick_dn |
| 17 | body_ratio | `open·high·low·close` | 1-wick_up-wick_dn | 直传 | body_ratio |

> `close₁` 表示前一日收盘价。MA₁₀ 表示过去10日的简单移动平均。

---

## 使用方法

### 拟合归一化器

```bash
python src/data.py
```

**拟合样本**：进入训练集的全部股票（与训练用同一批股票）在训练区间
（2016-01-01 ~ 2022-12-31）内的窗口。
全池按统一比例抽样，默认目标 100 万个窗口（每个窗口 45 天 × 18 列），
实际取 `NORMALIZER_FIT_WINDOWS` 与内存上限（`NORMALIZER_FIT_MEMORY_GB`）中的较小者。
每只股票按自身可用的训练窗口数等比例贡献，保底 1 个，保证每只有效股票都参与拟合，
预算也不会被代码序靠前的少数股票吃光。

采样方式为**时间分层 + 箱内随机**：把一只股票的名额对应的窗口序列切成等宽的箱，
每个箱内随机取一个窗口。箱保证窗口在训练时间轴上均匀铺开，避免相邻窗口 44/45 天
重叠带来的冗余；箱内随机保证没有固定相位，不会与星期几、月初月末、季度等日历周期
对齐。可用训练窗口总量不足预算时直接使用全部窗口。

`QuantileTransformer` 的 `subsample` 默认为 `None`（用收集到的全部值估分位）。
若设成整数，它会在拟合时再随机抽这么多个**值**，把上层的分层采样效果抹掉。

**关键**：只在训练集上拟合，避免数据泄漏。

**输出**：`./src/normalizer.pkl`

分位数数量、输出分布、窗口预算、内存上限、文件路径等参数在 [config.py: `DataConfig`](../src/config.py) 中修改（见下节）。

### 配置参数

归一化相关配置在 [config.py: `DataConfig`](../src/config.py) 中：

```python
NORMALIZER_OUTPUT_DISTRIBUTION = 'normal'   # 'normal' (标准正态) 或 'uniform' (均匀分布)
NORMALIZER_N_QUANTILES = 1000               # 分位数数量
NORMALIZER_PATH = './src/normalizer.pkl'    # 归一化器文件路径
NORMALIZER_FIT_WINDOWS = 1_000_000          # 拟合窗口预算
NORMALIZER_FIT_MEMORY_GB = 6.0              # 拟合峰值内存上限（GB）
NORMALIZER_SUBSAMPLE = None                 # None=用全部值估分位
MA_WINDOW = 10                              # 成交量相对均值的滑动窗口大小
```

### 训练时自动加载

[train.py](../src/train.py) 训练时会自动检测并加载归一化器：

```python
normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
```

---

## FAQ

### Q1: `output_distribution` 选 'normal' 还是 'uniform'？

**A**: 推荐 `'normal'`（默认值）。大多数深度学习模型假设输入近似正态分布。

### Q2: 为什么 VWAP 和 OHLC 一起 clip 到 [-0.1, 0.1]？

**A**: VWAP 的偏离度与 OHLC 变化率在同一量级（正常情况下不超过 ±5%）。clip 到 ±10% 只是过滤异常值，不影响正常数据。代码中 `np.clip(input_seq[:, :5], -0.1, 0.1)` 是前 5 列一起 clip。

### Q3: 成交量为什么不做 clip？

**A**: 成交量没有自然的上界。放量日成交量可以是均值的数倍，这是重要的交易信号。如果 clip，会丢失这些关键信息。

### Q4: 衍生特征为什么在数据库中预计算而非在线计算？

**A**: 两个原因：
1. **性能**：EMA、MA 等计算需要完整历史序列，每次在线计算代价高
2. **一致性**：预计算确保训练和推理使用完全相同的特征值

### Q5: 旧模型怎么办？

**A**: 需要重新训练。细处理后输入分布改变，旧模型权重不适用。

---

## 参考资料

1. [Bollinger Bands - Investopedia](https://www.investopedia.com/terms/b/bollingerbands.asp)
2. [MACD - Investopedia](https://www.investopedia.com/terms/m/macd.asp)
3. [Scikit-learn QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)
4. [FT-Transformer Paper](https://arxiv.org/abs/2106.11959)
5. [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556)
