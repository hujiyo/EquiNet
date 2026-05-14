# 特征归一化完整指南

## 数据概览

### 特征组成（10维）

模型输入为 `[batch_size, 45, 10]`，包含以下特征：

| 列号 | 特征 | 粗处理方法 | 粗处理后范围 |
|------|------|-----------|-------------|
| 0-3 | OHLC | 日环比变化率，clip [-0.1, 0.1] | [-0.1, 0.1] |
| 4 | VWAP | 相对收盘价偏离，clip [-0.1, 0.1] | [-0.1, 0.1] |
| 5 | Amount (成交额) | 相对N日均值变化率（N=10） | 无固定范围 |
| 6 | Exchange (换手率) | 相对N日均值变化率（N=10） | 无固定范围 |
| 7-9 | MA偏离度 (m5, m10, m20) | 直接使用预计算值 (close-MA)/MA | 无固定范围 |

数据存储在 SQLite 数据库中（[database.py](../data_maintenance/database.py)），通过 [data.py](../src/data.py) 加载和预处理。

---

## 两阶段归一化架构

### 设计思路

由于特征性质不同（价格类、量能类、技术指标类），项目采用**两阶段归一化**：

```
原始数据 ──→ 粗处理 ──→ 细处理 ──→ 模型输入
(CSV/DB)    (去量纲化)   (标准化)
```

### 阶段1: 粗处理 (Coarse Normalization)

将不同量纲的原始数据转换为可比的变化率/偏离度，定义在 [data.py: `coarse_normalize_context_window()`](../src/data.py)。

```python
# OHLC: 日环比变化率 → clip [-0.1, 0.1]
input_seq[:, :4] = (prices - prev_close) / prev_close
np.clip(input_seq[:, :4], -0.1, 0.1)

# VWAP: 相对收盘价偏离 → clip [-0.1, 0.1]
input_seq[:, 4] = (vwaps - closes) / closes
np.clip(input_seq[:, :4], -0.1, 0.1)

# Amount: 相对N日均值变化率（无clip）
input_seq[:, 5] = (amounts - MA_N) / MA_N

# Exchange: 相对N日均值变化率（无clip）
input_seq[:, 6] = (exchanges - MA_N) / MA_N

# MA偏离度: 直接使用预计算值
input_seq[:, 7:10] = input_seq_raw[:, 7:10]
```

**粗处理后的问题**：

| 特征 | 范围 | 均值 | 问题 |
|------|------|------|------|
| OHLC | `[-0.1, 0.1]` | ≈ 0 | 范围小，零均值 |
| VWAP | `[-0.1, 0.1]` | ≈ 0 | 范围小，零均值 |
| Amount | 不定 | ≈ 0 | 范围不定，可能存在长尾异常值 |
| Exchange | 不定 | ≈ 0 | 范围不定，可能存在长尾异常值 |
| MA偏离度 | 不定 | ≈ 0 | 范围不定，分布可能偏态 |

### 阶段2: 细处理 (Fine Normalization)

使用 `QuantileTransformer + StandardScaler` 将粗处理结果统一为均值≈0、方差≈1的标准化数据。

详见 [data.py: `FeatureNormalizer`](../src/data.py)。

---

## QuantileTransformer 原理

### 核心思想

**将任意分布映射到均匀分布或正态分布**

### 数学原理

对于数据点 `x`，QuantileTransformer 计算其在数据中的**累积分布函数（CDF）值**：

```
CDF(x) = (小于 x 的样本数) / 总样本数
```

然后应用逆变换：

- **`output_distribution='uniform'`**: 直接使用 CDF 值 → 输出在 `[0, 1]` 均匀分布
- **`output_distribution='normal'`**: 对 CDF 值应用逆正态 CDF（Probit）→ 输出符合标准正态分布

### 示例

假设 Amount 数据分布如下：
```
原始数据: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8, 0.9]
         ←──── 90% 集中在这里 ────→  ←─ 异常值 ─→
```

变换后：
```
Uniform: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 均匀分布
Normal:  [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]  # 标准正态
```

## StandardScaler

QuantileTransformer 的输出虽然是正态分布，但均值和标准差不一定为 0 和 1，所以需要 StandardScaler 来标准化。

```python
scaler = StandardScaler()
final = scaler.fit_transform(transformed)
```

**输出**：严格保证均值 = 0，标准差 = 1，所有特征在同一尺度。

---

## 特征分组与独立 Pipeline

由于不同特征的分布特性不同，项目为每个特征组创建了独立的归一化 pipeline：

| Pipeline | 覆盖特征 | 设计原因 |
|----------|---------|---------|
| `ohl_pipeline` | OHLC (col 0-3) | 价格类特征，分布相似 |
| `vwap_pipeline` | VWAP (col 4) | 均价偏离度，分布独立 |
| `amount_pipeline` | Amount (col 5) | 量能特征，长尾分布 |
| `exchange_pipeline` | Exchange (col 6) | 换手率特征，分布独立 |
| `ma_pipeline` | MA偏离度 (col 7-9) | 技术指标，三类MA偏离度分布相似 |

每个 pipeline 的结构：
```python
Pipeline([
    ('quantile', QuantileTransformer(output_distribution='normal', n_quantiles=1000)),
    ('scaler', StandardScaler())
])
```

---

## 使用方法

### 步骤1: 拟合归一化器

```bash
python src/data.py
```

可选参数：
```bash
python src/data.py --output-distribution uniform   # 使用均匀分布
python src/data.py --n-quantiles 500               # 调整分位数数量
```

**关键**：只在训练集上拟合，避免数据泄漏。

**输出**：`./src/normalizer.pkl`

### 步骤2: 配置参数

归一化相关配置在 [config.py: `DataConfig`](../src/config.py) 中：

```python
# 归一化器配置
NORMALIZER_OUTPUT_DISTRIBUTION = 'normal'   # 'normal' (标准正态) 或 'uniform' (均匀分布)
NORMALIZER_N_QUANTILES = 1000               # 分位数数量
NORMALIZER_PATH = './src/normalizer.pkl'    # 归一化器文件路径

# 粗处理相关配置
MA_WINDOW = 10                              # 量能/换手率相对均值的滑动窗口大小
```

### 步骤3: 训练时自动加载

[train.py](../src/train.py) 训练时会自动检测并加载归一化器：

```python
normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
```

---

## API 参考

### FeatureNormalizer

```python
# 拟合（仅在训练集上调用一次）
normalizer = FeatureNormalizer(output_distribution='normal', n_quantiles=1000)
normalizer.fit(train_stock_info)

# 单样本转换
normalized = normalizer.transform(input_seq)          # [context_length, 10] → [context_length, 10]

# 批量转换（比逐个调用快 10-100 倍）
normalized = normalizer.transform_batch(input_seqs)    # [batch, context_length, 10] → [batch, context_length, 10]

# 保存/加载
normalizer.save('normalizer.pkl')
normalizer = FeatureNormalizer.load('normalizer.pkl')
```

### 粗处理函数

```python
# 仅粗处理（用于归一化器拟合阶段收集数据）
input_seq = coarse_normalize_context_window(data, start_idx, context_length)

# 粗处理 + 细处理（完整流程）
input_seq = normalize_and_validate_context_window(data, start_idx, context_length, feature_normalizer=normalizer)
```

---

## 效果对比

### 预期结果

| 方案 | OHLC 均值 | OHLC 标准差 | Amount 均值 | Amount 标准差 | 说明 |
|------|-----------|-------------|-------------|--------------|------|
| **仅粗处理** | ≈ 0 | ≈ 0.03 | ≈ 0 | 不定 | 范围/尺度不统一 |
| **粗处理 + 细处理** | 0 | 1 | 0 | 1 | 所有特征标准化 |

### 推荐方案

当前项目默认使用**粗处理 + 细处理**的组合方案。模型中的 LayerNorm 作为网络内部的额外稳定机制保留（双重保险），不因使用了细处理而移除。

---

## FAQ

### Q1: `output_distribution` 选 'normal' 还是 'uniform'？

**A**: 推荐 `'normal'`（默认值）

| 选项 | 输出分布 | 适用场景 |
|------|---------|---------|
| `'normal'` | 标准正态 N(0,1) | 大多数深度学习模型（当前默认） |
| `'uniform'` | 均匀分布 U(0,1) | 某些特定模型 |

### Q2: `n_quantiles` 多大合适？

**A**: 默认 1000 即可

| 值 | 精度 | 速度 | 内存 |
|----|------|------|------|
| 100 | 低 | 快 | 低 |
| 1000 | 高 | 中 | 中（默认） |
| 10000 | 极高 | 慢 | 高 |

### Q3: 旧模型怎么办？

**A**: 需要重新训练。细处理后输入分布改变，旧模型权重不适用。

### Q4: 可以只归一化部分特征吗？

**A**: 当前实现按特征组独立归一化（OHLC 一组、VWAP 独立、Amount 独立、Exchange 独立、MA 一组），不建议跳过任何组。

### Q5: MA偏离度特征为什么不做粗处理？

**A**: MA偏离度在数据库中已预计算为 `(close - MA) / MA` 格式（[features.py](../data_maintenance/features.py)），本身就是无量纲的偏离度比值，直接使用即可。

---

## 参考资料

1. [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
2. [FT-Transformer Paper](https://arxiv.org/abs/2106.11959)
3. [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556)
