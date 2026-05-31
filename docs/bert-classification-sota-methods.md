# BERT 分类模型 Pooling 方案调研

## 概述

BERT 用于分类任务的流程：`输入 → Encoder → Pooling Layer → Classification Head → 预测`

Encoder 是预训练的 Transformer 编码器，Classification Head 是下游任务层。Pooling Layer 将不定长序列压缩为固定长度向量，是本文档的重点。

调研时间：2025-03-17（EquiNet 对比部分更新于 2026-05-31）

---

## Pooling 方案对比

### [CLS] Token（BERT 原始方案）

来源：[BERT 原论文 (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

取最后一层 `[CLS]` token 的输出作为序列表示。零参数，BERT 原始设计。但单个 token 承担了整个序列的信息压缩压力，2024 年的研究表明 CLS 在长序列上表现不如 Mean pooling。

适用于短文本（< 128 tokens）或资源受限场景。

### Mean Pooling

来源：Sentence-Transformers 库

对所有 token 嵌入取平均。零参数，利用了全部 token 信息，在长序列任务中优于 CLS，是语义相似度任务的默认方案。缺点是无法区分不同 token 的重要性，重要信号可能被平均稀释。

### Max Pooling

对每个维度取所有 token 的最大值。能捕获每个维度的"最强信号"，对噪声相对鲁棒。适用于关键词检测场景，常与其他 pooling 方法组合使用。

### Weighted Mean Pooling

来源：Sentence-Transformers (`PoolingModeWeightedMean`)

根据 attention mask 或学习到的权重进行加权平均。能区分 token 重要性，但依赖预训练 attention 模式，可能不适应下游任务。

### Multi-Head Attention Pooling

来源：Set Transformer (Lee et al., 2019), Perceiver (Jaegle et al., 2021)

使用可学习的 query token 通过多头 cross-attention 聚合序列，每个头学习不同的聚合模式。表达能力最强，但参数量和计算量也最大。

**EquiNet 当前使用此方案。**

---

## 2024-2025 研究进展

### MaxPoolBERT (2025)

来源：[MaxPoolBERT 论文](https://arxiv.org/html/2505.15696v2)

通过层间聚合 + token 间聚合 + 最大池化 + 注意力精炼来增强 `[CLS]` 表示。轻量级扩展，在 GLUE/SuperGLUE 上有提升。

### Multi-CLS BERT (2024)

使用多个 `[CLS]` token 代表序列，每个 CLS 学习不同"视角"，作为模型集成的高效替代。减少了 GLUE/SuperGLUE 上的校准误差。

### Layer-wise Aggregation（层间聚合）

不仅使用最后一层输出，而是融合多层 BERT 表示。研究表明中间层往往包含更丰富的语义信息，SBERT-WK（带层注意力的方法）性能优异。

---

## 性能对比

| 方法 | 长序列 | 短序列 | 参数量 | 复杂度 |
|------|--------|--------|--------|--------|
| CLS Token | 中 | 好 | 0 | O(1) |
| Mean Pooling | 好 | 好 | 0 | O(T) |
| Max Pooling | 中 | 中 | 0 | O(T) |
| Multi-Head Attn | 好 | 好 | ~10K | O(Td²/h) |

结论：
1. Mean Pooling 是最稳健的基线方法
2. CLS 在长序列上表现不佳
3. Multi-Head Attention Pooling 表达能力最强，但需要更多资源
4. 层间聚合可以进一步提升性能

---

## EquiNet 对比

EquiNet 使用 Multi-Head Attention Pooling（V3），当前参数 d_model=128, nhead=4。

| | EquiNet | BERT (CLS) | BERT (Mean) | BERT (Multi-Head) |
|--|---------|------------|-------------|-------------------|
| d_model | 128 | 768 | 768 | 768 |
| Pooling 参数量 | ~66K | 0 | 0 | ~15K |
| 序列长度 | 45 | 512 | 512 | 512 |
| 头数 | 4 | N/A | N/A | 8-12 |

### 可能的改进方向

**层间聚合**：当前只用最后一层输出。BERT 研究表明中间层包含更丰富的语义信息，可以聚合最后多层。

**辅助 Pooling 集成**：将 Attention Pooling 与 Mean Pooling 的输出拼接后融合，可以提升鲁棒性。

---

## 参考资料

1. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
2. [MaxPoolBERT (2025)](https://arxiv.org/html/2505.15696v2)
3. [ModernBERT (2024)](https://arxiv.org/pdf/2412.13663)
4. [LMK > CLS Landmark Pooling (2024)](https://arxiv.org/html/2601.21525v1)
5. [Exploring Pooling Strategies (OpenReview)](https://openreview.net/pdf?id=JPjj4GClBr)
6. [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers)
7. [HuggingFace Transformers](https://github.com/huggingface/transformers)
