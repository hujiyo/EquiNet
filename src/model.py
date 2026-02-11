"""
EquiNet 模型定义文件

包含所有模型架构相关的类：
- PositionalEncoding: 位置编码
- MultiHeadAttention: 多头注意力机制
- TransformerLayer: Transformer层
- EnhancedStockTransformer: 主模型类
"""

import torch
import torch.nn as nn
import math
from config import ModelConfig, DataConfig


def init_weights(module):
    """
    Xavier (Glorot) 初始化 - 适合 Transformer 模型

    初始化范围:
    - Linear层权重: uniform[-a, +a], a = sqrt(6 / (fan_in + fan_out))
    - Linear层偏置: 0
    - Norm层权重: 1
    - Norm层偏置: 0
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform 初始化
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PositionalEncoding(nn.Module):
    """
    标准的正弦位置编码
    让 Transformer 自己学习时间依赖关系，不加人为规则
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()

        # 创建标准的正弦/余弦位置编码
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # 直接添加位置编码，LayerNorm在后续层中使用
        seq_len = x.size(1)
        pe_slice = self.pe[:seq_len, :].unsqueeze(0)
        return x + pe_slice


class MultiHeadAttention(nn.Module):
    """
    标准的多头注意力机制（Pre-Norm架构）
    让模型自动学习每个头应该关注什么特征，不人为干预
    """
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        assert d_model % nhead == 0

        # 使用标准的MultiheadAttention
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Pre-Norm: 在注意力之前进行归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)

    def forward(self, x, attn_mask=None):
        # Pre-Norm架构：先归一化，再计算注意力，最后残差连接
        # 输出 = 输入 + Dropout(Attention(LayerNorm(输入)))

        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)

        # Pre-Norm: 先对输入进行归一化
        normalized_x = self.norm(x)

        # 计算注意力
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x, attn_mask=mask)

        # 残差连接（注意这里是加到原始输入x上，而不是normalized_x）
        output = x + self.dropout(attn_output)
        return output


class TransformerLayer(nn.Module):
    """
    标准的 Transformer 层（Pre-Norm架构）
    设计理念：让模型自动学习应该关注什么特征，不加人为干预
    Pre-Norm相比Post-Norm有更好的训练稳定性
    """
    def __init__(self, d_model, nhead, use_ffn=True):
        super(TransformerLayer, self).__init__()

        self.use_ffn = use_ffn

        # 使用Pre-Norm多头注意力
        self.attention = MultiHeadAttention(d_model, nhead)

        if self.use_ffn:
            # 前馈网络，用于进一步处理注意力的输出
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, 160),  # 80 → 160
                nn.GELU(),                 # GELU激活
                nn.Dropout(ModelConfig.DROPOUT_RATE),  # 防过拟合
                nn.Linear(160, d_model),   # 160 → 80
            )

            # Pre-Norm: 在前馈网络之前进行归一化
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

    def forward(self, x):
        # x的shape: [batch_size, seq_len, d_model]

        # Pre-Norm架构的注意力子层（MultiHeadAttention内部已经实现了Pre-Norm）
        # 输出 = 输入 + Dropout(Attention(LayerNorm(输入)))
        x = self.attention(x, attn_mask=None)

        if self.use_ffn:
            # Pre-Norm架构的前馈网络子层
            # 输出 = 输入 + Dropout(FFN(LayerNorm(输入)))
            normalized_x = self.norm(x)
            ff_out = self.feed_forward(normalized_x)
            x = x + self.dropout(ff_out)

        return x


class EnhancedStockTransformer(nn.Module):
    """
    改进的 Transformer 模型（Pre-Norm架构 + 统一Embedding）

    核心改进1：统一Embedding - 端到端学习特征融合
    - 6个输入特征(OHLC + Volume + Exchange) -> 统一映射到 d_model 维
    - 让模型自己学习如何组合和表达不同类型的特征
    - 相比分离embedding，减少了人为的结构假设

    核心改进2：统一Transformer层架构
    - 所有层都使用 Attention + FFN 结构
    - 简化模型设计，降低结构复杂度
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_len):
        super(EnhancedStockTransformer, self).__init__()

        # 统一Embedding：两阶段FFN结构，让特征在进入Transformer前充分混合
        self.embedding = nn.Sequential(
            nn.Linear(ModelConfig.INPUT_DIM, ModelConfig.EMBED_HIDDEN_DIM),  # 6维 → 40维（扩展）
            nn.GELU(),                                                          # GELU激活，对负值有梯度
            nn.Linear(ModelConfig.EMBED_HIDDEN_DIM, d_model)                  # 40维 → 80维
        )

        # 使用标准位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 统一架构：所有层都使用 Attention + FFN
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, use_ffn=True)
            for i in range(num_layers)
        ])

        # Pre-Norm架构：在最后添加一个LayerNorm
        # 因为Pre-Norm的最后一层没有归一化输出
        self.final_norm = nn.LayerNorm(d_model)

        # 注意力聚合：学习每个时间步的重要性权重
        # 相比只用最后一个时间步，能更充分利用所有历史信息
        # 使用缩放初始化，避免点积方差过大导致softmax接近one-hot
        self.attention_query = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))

        # 简化输出层，减少过拟合
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 降维
            nn.GELU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(d_model // 2, output_dim)  # 最终输出
        )

        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        # 应用初始化
        self.apply(init_weights)

    def forward(self, x):
        # x: [batch_size, seq_len, 6] (OHLC + volume + exchange)

        # 1. 统一Embedding：6个特征一起映射到d_model维
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # 2. 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 3. Transformer层（Pre-Norm架构）
        for layer in self.layers:
            x = layer(x)

        # 4. Pre-Norm架构需要在最后进行归一化
        #    因为每层的输出没有经过归一化
        x = self.final_norm(x)

        # 5. 注意力聚合：自适应加权所有时间步
        # attn_scores: [batch_size, seq_len]
        attn_scores = torch.matmul(x, self.attention_query)  # 每个时间步与query的相似度
        attn_weights = torch.softmax(attn_scores, dim=1)     # 归一化为权重
        # aggregated: [batch_size, d_model] - 加权求和所有时间步
        aggregated = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        output = self.output_projection(aggregated)  # [batch_size, output_dim]
        return output
