# 模型末端信息压缩机制演进分析

## 概述

本文档记录 EquiNet 模型序列信息压缩机制的三次迭代：最后时间步选择 → 单向量点积注意力 → 多头注意力池化。

---

## 版本对比

| 版本 | 提交 | 机制 | 参数量 | 备注 |
|------|------|------|--------|------|
| V1 | `878deed5^` | Last Hidden State | 0 | 直接取最后时间步 |
| V2 | `878deed5` | Single-Vector Attention | d_model | 单向量点积 + softmax |
| V3 | `435c94e` | Multi-Head Attention Pooling | ~66K | 可学习 query + cross-attention |

---

## V1: Last Hidden State

**提交**: `878deed5^`（父提交）

```
[batch, seq_len, d_model]  →  x[:, -1, :]  →  Linear(d_model→1)  →  [batch, 1]
```

取 Transformer 编码后序列的最后一个时间步作为整个序列的表示。

- 零参数，无额外计算
- 信息丢失：丢弃了前 seq_len-1 个时间步
- Transformer 中各时间步是并行编码的，最后位置并不天然包含前面所有信息

---

## V2: Single-Vector Attention

**提交**: `878deed5`
**提交信息**: "新增注意力聚合机制，自适应加权所有时间步特征，替代原来仅使用最后时间步的机制"

```
[batch, seq_len, d_model]
       ↓ matmul(query)     ← query 是 [d_model] 可学习向量，初始化为 randn/√d_model
[batch, seq_len]           ← 注意力分数
       ↓ softmax
[batch, d_model]           ← 加权求和
       ↓ Linear(d_model→1)
[batch, 1]
```

初始化使用 `1/√d_model` 缩放，防止点积方差过大导致 softmax 退化为 one-hot。

- 自适应加权所有时间步，仅增加 d_model 个参数
- 单个查询向量只能学习一种聚合模式，无法同时关注不同时间尺度的模式

---

## V3: Multi-Head Attention Pooling（当前方案）

**提交**: `435c94e`（"将单向量点积注意力聚合替换为多头注意力池化模块"）

当前参数：d_model=128, nhead=4 (head_dim=32), seq_len=45

```
[batch, 45, 128]              ← Transformer 编码后的序列
       ↓                +
[batch, 1, 128]               ← 可学习 query token, N(0, 0.15²)
       ↓ Cross-Attention (4 heads)
[batch, 1, 128]
       ↓ Residual + LayerNorm (Post-Norm)
[batch, 128]
       ↓ head_norm (LayerNorm) + Linear(128→1)
[batch, 1]
```

### 关键设计

**Cross-Attention**：Query 是单个可学习 token `[1, d_model]`，Key/Value 是完整序列。每个头学习不同的聚合模式。

**Post-Norm + 残差**：`pooled = LayerNorm(query + Dropout(CrossAttn(query, x)))`，与主 Transformer 层保持一致。

**Query 初始化**：`N(0, 0.15²)`，匹配 cross-attn 初始输出的 std 量级，使残差连接在训练初期有实际意义。

### 参数量

| 组件 | 数量 |
|------|------|
| Query token | 128 |
| LayerNorm (norm_q) | 128 × 2 = 256 |
| Cross-Attn (Q/K/V/O) | 4 × 128 × 128 = 65,536 |
| **合计** | **65,920** |

> nn.MultiheadAttention 内部包含 Q/K/V 三个投影矩阵和 O 输出投影。

### 与 V2 的区别

V2 的 query 是固定向量，所有时间步与同一个向量做点积。V3 使用多头 cross-attention，每个头独立学习 query-key-value 投影，表达能力更强，但参数量从 d_model 增至 ~66K。

---

## 架构一致性

V3 的池化层与主 Transformer 架构风格统一：

```
编码器层:   Self-Attention(Q=K=V=X)  →  时间步之间互相注意
池化层:     Cross-Attention(Q=query, K=V=X)  →  query 注意所有时间步
```

这使得 warm-up、learning rate schedule 等训练技巧可以直接复用。

---

## 迭代原因

| 升级 | 动机 |
|------|------|
| V1 → V2 | Transformer 各时间步并行编码，最后位置不包含前面信息，需要显式聚合 |
| V2 → V3 | 单查询向量只有一种聚合策略，多头机制可以同时学习短期趋势、长期周期等不同模式 |
