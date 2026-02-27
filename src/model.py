"""
EquiNet 模型定义文件

包含所有模型架构相关的类：
- PositionalEncoding: 位置编码
- TwoDimensionalPositionalEncoding: 二维位置编码（用于Token化模型）
- MultiHeadAttention: 多头注意力机制
- TransformerLayer: Transformer层
- AttentionPooling: 多头注意力聚合（可学习query token + cross-attention）
- EnhancedStockTransformer: 连续值模型（原始模型）
- TokenizedStockTransformer: Token化模型（离散化输入）
- create_model(): 工厂函数，根据配置创建对应模型
"""

import torch
import torch.nn as nn
from config import ModelConfig, DataConfig

OUTPUT_LAYER_GAIN = ModelConfig.OUTPUT_LAYER_GAIN

def init_weights(module):
    """
    Xavier (Glorot) 初始化 - 适合 Transformer 模型

    初始化范围:
    - Linear层权重: uniform[-a, +a], a = sqrt(6 / (fan_in + fan_out))
    - Linear层偏置: 0
    - Norm层权重: 1
    - Norm层偏置: 0
    
    特殊处理:
    - 输出层使用更大的gain，确保logits有足够大的范围
    """
    if isinstance(module, nn.Linear):
        gain = OUTPUT_LAYER_GAIN if module.out_features == 1 else 1.0
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learned Positional Embedding）
    类似 BERT / GPT 的做法：每个位置对应一个可训练的向量
    让模型自己学习最优的位置表示，而非使用固定的正弦公式
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()

        # 可学习的位置嵌入：每个位置一个d_model维向量
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        #添加位置编码，LayerNorm在后续层中可能使用
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


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
            # 前馈网络，使用配置的扩展比例
            ffn_hidden_dim = int(d_model * ModelConfig.FFN_EXPAND_RATIO)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, ffn_hidden_dim),
                nn.GELU(),
                nn.Dropout(ModelConfig.DROPOUT_RATE),
                nn.Linear(ffn_hidden_dim, d_model),
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


class AttentionPooling(nn.Module):
    """
    多头注意力聚合（Multi-Head Attention Pooling）

    使用一个可学习的 query token 通过多头 cross-attention 聚合序列信息。
    相比单向量点积聚合，每个注意力头可以学到不同的时间聚合模式，
    表达能力更强，且与 Transformer 架构风格一致。

    参考: Set Transformer (Lee et al., 2019), Perceiver (Jaegle et al., 2021)
    """
    def __init__(self, d_model, nhead):
        super(AttentionPooling, self).__init__()

        # 可学习的 query token: [1, 1, d_model]
        self.query = nn.Parameter(torch.empty(1, 1, d_model))

        # Pre-Norm: 分别对 query 和 key-value 进行归一化
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # 多头 cross-attention: query 关注序列所有位置
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        # 初始化 query token
        nn.init.xavier_uniform_(self.query)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] - Transformer 编码后的序列

        Returns:
            pooled: [batch_size, d_model] - 聚合后的表示向量
        """
        batch_size = x.size(0)

        # 将 query 扩展到 batch 维度: [1, 1, d_model] -> [batch_size, 1, d_model]
        query = self.query.expand(batch_size, -1, -1)

        # Pre-Norm
        query_normed = self.norm_q(query)
        kv_normed = self.norm_kv(x)

        # Cross-attention: query 关注序列所有位置
        # attn_output: [batch_size, 1, d_model]
        attn_output, _ = self.cross_attn(query_normed, kv_normed, kv_normed)

        # 残差连接 + 去掉 seq_len=1 的维度
        pooled = (query + self.dropout(attn_output)).squeeze(1)  # [batch_size, d_model]

        return pooled


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

        # 多头注意力聚合：通过 cross-attention 聚合序列信息
        # 相比单向量点积，每个注意力头可以学到不同的时间聚合模式
        self.attention_pooling = AttentionPooling(d_model, nhead)

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

        # 5. 多头注意力聚合
        aggregated = self.attention_pooling(x)  # [batch_size, d_model]

        output = self.output_projection(aggregated)  # [batch_size, output_dim]
        return output


# ==================== Token化模型 ====================

class TwoDimensionalPositionalEncoding(nn.Module):
    """
    可学习的二维位置编码：时间步 + 特征类型

    设计原理：
    - 60×6的时间序列被展平成360个token
    - 每个token有两个结构信息：
      1. temporal_id: 属于第几个时间步（0-59）
      2. feature_id: 属于哪个特征（0-5，OHLC+volume+exchange）

    编码方案：
    - 时间步编码：nn.Embedding(num_timesteps, d_model)，可学习
    - 特征类型编码：nn.Embedding(num_features, d_model)，可学习
    - 两者直接相加（而非拼接），保持完整的d_model表达能力
    """
    def __init__(self, d_model, num_timesteps=DataConfig.CONTEXT_LENGTH, num_features=ModelConfig.INPUT_DIM):
        super(TwoDimensionalPositionalEncoding, self).__init__()

        self.num_features = num_features     # 6

        # 可学习的时间步编码：每个时间步一个d_model维向量
        self.temporal_pe = nn.Embedding(num_timesteps, d_model)

        # 可学习的特征类型编码：每个特征类型一个d_model维向量
        self.feature_pe = nn.Embedding(num_features, d_model)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] 其中 seq_len = 360 (60*6)

        Returns:
            x + 二维位置编码
        """
        seq_len = x.size(1)
        device = x.device

        # 为每个token计算其时间步ID和特征ID
        # token位置 0-359
        token_positions = torch.arange(seq_len, device=device)

        # temporal_id = token_pos // 6 (0-59)
        temporal_ids = token_positions // self.num_features

        # feature_id = token_pos % 6 (0-5)
        feature_ids = token_positions % self.num_features

        # 获取对应的可学习位置编码并相加
        # [seq_len, d_model] + [seq_len, d_model]
        positional_encodings = self.temporal_pe(temporal_ids) + self.feature_pe(feature_ids)

        # 添加batch维度并加到输入上
        return x + positional_encodings.unsqueeze(0)


class TokenizedTransformerLayer(nn.Module):
    """Token化模型的 Transformer 层（Pre-Norm架构）"""
    def __init__(self, d_model, nhead, use_ffn=True, dropout_rate=0.1):
        super(TokenizedTransformerLayer, self).__init__()

        self.use_ffn = use_ffn
        self.attention = MultiHeadAttention(d_model, nhead)

        if self.use_ffn:
            # FFN扩展比例：使用配置的扩展比例
            ffn_hidden_dim = int(d_model * ModelConfig.FFN_EXPAND_RATIO)
            self.feed_forward = nn.Sequential(
                nn.Linear(d_model, ffn_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(ffn_hidden_dim, d_model),
            )

            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.attention(x, attn_mask=None)

        if self.use_ffn:
            normalized_x = self.norm(x)
            ff_out = self.feed_forward(normalized_x)
            x = x + self.dropout(ff_out)

        return x


class TokenizedStockTransformer(nn.Module):
    """
    Token化版本的股票预测Transformer

    核心设计：
    1. Token Embedding: 176个token → d_model维向量（查表）
    2. 二维位置编码: 时间步编码(0-59) + 特征类型编码(0-5)
    3. Transformer: 学习token间的关系（逐层递减dropout）
    4. 输出: 聚合所有token信息进行预测
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, output_dim, max_seq_len):
        super(TokenizedStockTransformer, self).__init__()

        # Token Embedding层：将离散token ID映射到连续向量空间
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 二维位置编码：时间步 + 特征类型
        self.pos_encoding = TwoDimensionalPositionalEncoding(d_model)

        # Transformer层：逐层递减dropout
        dropout_rates = [max(0.05, 0.1 - i * 0.015) for i in range(num_layers)]
        self.layers = nn.ModuleList([
            TokenizedTransformerLayer(d_model, nhead, use_ffn=True, dropout_rate=dropout_rates[i])
            for i in range(num_layers)
        ])

        # Pre-Norm架构的最终归一化
        self.final_norm = nn.LayerNorm(d_model)

        # 多头注意力聚合
        self.attention_pooling = AttentionPooling(d_model, nhead)

        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(ModelConfig.DROPOUT_RATE),
            nn.Linear(d_model // 2, output_dim)
        )

        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        # 应用初始化
        self.apply(init_weights)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 60, 6] 连续值 或 [batch_size, 360] token ID

        Returns:
            output: [batch_size, 1] 预测logits
        """
        # 如果输入是连续值，先进行token化
        if x.dim() == 3 and x.size(-1) == ModelConfig.INPUT_DIM:
            from tokenizer import tokenize_batch_torch
            x = tokenize_batch_torch(x, flatten=True)

        # 确保token ID是long类型
        if x.dtype != torch.long:
            x = x.long()

        # Token Embedding查表
        x = self.token_embedding(x)

        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer层
        for layer in self.layers:
            x = layer(x)

        # 最终归一化
        x = self.final_norm(x)

        # 多头注意力聚合
        aggregated = self.attention_pooling(x)  # [batch_size, d_model]

        # 输出投影
        output = self.output_projection(aggregated)
        return output


# ==================== 工厂函数 ====================

def create_model(input_dim=None, d_model=None, nhead=None, num_layers=None,
                 output_dim=None, seq_len=None):
    """
    根据配置创建模型（工厂函数）

    根据 ModelConfig.MODEL_TYPE 自动选择：
    - 'continuous': EnhancedStockTransformer（连续值模型）
    - 'tokenized': TokenizedStockTransformer（Token化模型）

    Args:
        参数均为可选，如果不提供则使用 ModelConfig 中的默认值

    Returns:
        model: 对应类型的模型实例
    """
    # 使用配置中的默认值
    input_dim = input_dim or ModelConfig.INPUT_DIM
    d_model = d_model or ModelConfig.D_MODEL
    nhead = nhead or ModelConfig.NHEAD
    num_layers = num_layers or ModelConfig.NUM_LAYERS
    output_dim = output_dim or ModelConfig.OUTPUT_DIM
    seq_len = seq_len or DataConfig.CONTEXT_LENGTH

    model_type = ModelConfig.MODEL_TYPE.lower()

    if model_type == 'continuous':
        # 创建连续值模型
        model = EnhancedStockTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=output_dim,
            seq_len=seq_len
        )

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'='*50}")
        print(f"连续值模型架构 (EnhancedStockTransformer)")
        print(f"{'='*50}")
        print(f"输入维度: {input_dim}")
        print(f"序列长度: {seq_len}")
        print(f"Embedding维度: {d_model}")
        print(f"注意力头数: {nhead}")
        print(f"Transformer层数: {num_layers}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"{'='*50}\n")

    elif model_type == 'tokenized':
        # 创建Token化模型
        model = TokenizedStockTransformer(
            vocab_size=ModelConfig.VOCAB_SIZE,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=output_dim,
            max_seq_len=ModelConfig.TOKEN_SEQ_LEN
        )

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'='*50}")
        print(f"Token化模型架构 (TokenizedStockTransformer)")
        print(f"{'='*50}")
        print(f"词表大小: {ModelConfig.VOCAB_SIZE}")
        print(f"Token序列长度: {ModelConfig.TOKEN_SEQ_LEN}")
        print(f"位置编码: 二维 (时间步+特征类型)")
        print(f"Embedding维度: {d_model}")
        print(f"注意力头数: {nhead}")
        print(f"Transformer层数: {num_layers}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"{'='*50}\n")

    else:
        raise ValueError(f"未知的模型类型: {ModelConfig.MODEL_TYPE}。"
                        f"请选择 'continuous' 或 'tokenized'")

    return model
