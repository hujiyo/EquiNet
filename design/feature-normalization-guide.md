# 特征归一化完整指南

## 📋 目录

1. [问题诊断](#问题诊断)
2. [Quantile Transformation 原理](#quantile-transformation-原理)
3. [为什么需要两步操作](#为什么需要两步操作)
4. [使用方法](#使用方法)
5. [效果对比](#效果对比)
6. [常见问题](#常见问题)

---

## 🔍 问题诊断

### 现状分析

当前数据预处理代码（[data.py:870-884](../src/data.py#L870-L884)）：

```python
# OHLC: (price - prev) / prev → clip to [-0.1, 0.1]
input_seq[:, :4] = np.clip((input_seq[:, :4] - prev_close) / prev_close, -0.1, 0.1)

# Volume: (volume - prev) / prev → clip to [-5, 5] → transform to [0, 1]
input_seq[:, 4] = np.clip((input_seq[:, 4] - prev_volume) / prev_volume, -5.0, 5.0) / 10.0 + 0.5

# Exchange: exchange / 100 → [0, 1]
input_seq[:, 5] = input_seq_raw[:, 5] / 100.0
```

### 存在的问题

| 特征 | 范围 | 均值 | 问题 |
|------|------|------|------|
| Open, High, Low, Close | `[-0.1, 0.1]` | ≈ 0 | ✅ 范围小，零均值 |
| Volume | `[0, 1]` | ≈ 0.5 | ⚠️ 范围是 OHLC 的 5 倍，有偏置 |
| Exchange | `[0, 1]` | ≈ ? | ⚠️ 范围是 OHLC 的 5 倍，可能有偏置 |

**问题1: 范围不同**
- Volume/Exchange 的范围是 OHLC 的 **5 倍**
- 导致第一层 Linear 输出中，Volume/Exchange 主导

**问题2: 集中度不同**
- Volume 可能 99% 集中在 `[0.2, 0.4]`
- Exchange 可能集中在某个小范围
- 导致特征利用率低

---

## 🎯 Quantile Transformation 原理

### 核心思想

**将任意分布映射到均匀分布或正态分布**

### 数学原理

对于数据点 `x`，Quantile Transformation 计算其在数据中的**累积分布函数（CDF）值**：

```
CDF(x) = (小于 x 的样本数) / 总样本数
```

然后应用逆变换：

- **`output_distribution='uniform'`**: 直接使用 CDF 值 → 输出在 `[0, 1]` 均匀分布
- **`output_distribution='normal'`**: 对 CDF 值应用逆正态 CDF（Probit）→ 输出符合标准正态分布

### 示例

假设 Volume 数据分布如下：
```
原始数据: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8, 0.9]
         ←──── 90% 集中在这里 ────→  ←─ 异常值 ─→
```

变换后：
```
Uniform: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # 均匀分布
Normal:  [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28]  # 标准正态
```

**神奇之处**：无论原始数据多么偏态，变换后都变成均匀分布！

---

## StandardScaler

QuantileTransformer的输出虽然是正态分布，但均值和标准差不一定为0和1，所以需要StandardScaler来标准化。

```python
scaler = StandardScaler()
final = scaler.fit_transform(transformed)
```

**输出**：
- 严格保证：均值 = 0，标准差 = 1
- 所有特征在同一尺度

---

## 📖 使用方法

### 步骤1: 拟合归一化器（首次）

```bash
python src/data.py --fit-normalizer
```

**关键**：只在训练集上拟合，避免数据泄漏！
**输出**：
- `./feature_normalizer.pkl` - 归一化器文件

### 步骤2: 开启USE_FEATURE_NORMALIZER

```python
# config.py
USE_FEATURE_NORMALIZER = True    # 是否启用特征归一化器
```

### 步骤3: 验证效果

```bash
python src/eval_normalization.py
```

**输出**：
- `normalization_comparison.png` - 分布对比图
- 终端打印统计信息

---

## 📊 效果对比

### 预期结果

| 方案 | OHLC 均值 | OHLC 标准差 | Volume 均值 | Volume 标准差 | 优点 | 缺点 |
|------|-----------|-------------|-------------|--------------|------|------|
| **原始归一化** | ≈ 0 | ≈ 0.05 | ≈ 0.5 | ≈ 0.2 | 简单 | 范围不均，集中度问题 |
| **+ LayerNorm** | 0 | 1 | 0 | 1 | 网络内稳定 | 依赖模型层 |
| **Quantile** | 0 | 1 | 0 | 1 | 预处理稳定 | 需要拟合 |

### 推荐方案

**阶段1: 立即采用**
- ✅ 使用 Quantile Transformation
- ✅ 保留 LayerNorm（双重保险）

**阶段2: 验证后优化**
- 如果效果良好，可移除 LayerNorm
- 让模型更轻量

---

### Q1: `output_distribution` 选 'normal' 还是 'uniform'？

**A**: 推荐 `'normal'`

| 选项 | 输出分布 | 适用场景 |
|------|---------|---------|
| `'normal'` | 标准正态 N(0,1) | ✅ 大多数深度学习模型 |
| `'uniform'` | 均匀分布 U(0,1) | 某些特定模型 |

### Q2: `n_quantiles` 多大合适？

**A**: 默认 1000 即可

| 值 | 精度 | 速度 | 内存 |
|----|------|------|------|
| 100 | 低 | 快 | 低 |
| 1000 | 高 | 中 | 中 |
| 10000 | 极高 | 慢 | 高 |

**建议**：从 1000 开始，如果效果不好再增加。

### Q3: 旧模型怎么办？

**A**: 需要重新训练

- 添加归一化后，输入分布改变
- 旧模型权重不适用
- 但训练会更稳定，效果可能更好

### Q5: 可以只归一化部分特征吗？

**A**: 可以，但不推荐

当前实现：
- OHLC 共用同一个 pipeline
- Volume 独立 pipeline
- Exchange 独立 pipeline

**原因**：Volume 和 Exchange 范围相近（都是 [0,1]），但分布可能不同

---

## 🚀 高级话题

### FT-Transformer 的 Feature Tokenizer

业界 SOTA 方案：为每个特征创建独立的嵌入层

```python
class FeatureTokenizer(nn.Module):
    def __init__(self, num_features=6, d_model=48):
        super().__init__()

        # 每个特征独立的 Linear
        self.embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

    def forward(self, x):
        # x: [batch, seq_len, 6]
        tokens = []

        for i in range(6):
            feature = x[:, :, i:i+1]  # [batch, seq_len, 1]
            tokens.append(self.embeddings[i](feature))  # [batch, seq_len, d_model]

        return torch.stack(tokens, dim=2)  # [batch, seq_len, 6, d_model]
```

**优点**：
- ✅ 完全消除特征尺度问题
- ✅ 模型自动学习特征权重
- ✅ 业界 SOTA 方案

## 📚 参考资料

1. [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
2. [FT-Transformer Paper](https://arxiv.org/abs/2106.11959)
3. [On Embeddings for Numerical Features](https://arxiv.org/abs/2203.05556)
4. [Power Transforms with Scikit-learn](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)
5. [Comprehensive Normalization Guide](https://mbrenndoerfer.com/writing/normalization-feature-scaling-min-max-machine-learning-guide)