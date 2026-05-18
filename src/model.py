"""
EquiNet 模型定义文件

包含所有模型架构相关的类：
- PositionalEncoding: 位置编码
- MultiHeadAttention: 多头注意力机制
- TransformerLayer: Transformer层
- AttentionPooling: 多注意力聚合（可学习query token + cross-attention）
- StockTransformer: 连续值模型
- create_model(): 工厂函数，创建模型
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, DataConfig

def init_weights(module):
    """
    当代主流Transformer初始化策略

    设计原则：
    1. FFN-Embedding层: 第一层gain=0.53(线性投影), 第二层gain=1.7(补偿GELU压缩)
    2. SwiGLU w1/w3层: Xavier初始化，gain=1.7
    3. SwiGLU w2层: Xavier初始化，gain=1.0（无激活函数）
    4. 输出层: 小增益，避免sigmoid饱和
    5. LayerNorm: weight=1, bias=0

    Embedding初始化计算（目标std=0.2）：
    - Linear层: 输出std = σ_input × gain × sqrt(2×fan_in/(fan_in+fan_out))
    - Embedding层: 输出std = gain × sqrt(2/(vocab_size+embedding_dim))

    各层gain计算结果：
    - FFN-Embedding第一层 (Linear 10→128): gain=0.53, 输出std≈0.2
    - FFN-Embedding GELU: 有效增益≈0.588, 输出std≈0.118
    - FFN-Embedding第二层 (Linear 128→128): gain=1.7, 输出std≈0.2 (由StockTransformer.__init__显式设置)
    - Position Embedding (Embedding 45→128): gain=1.86
    - Query Token (Parameter 128): gain=2.26
    """
    ffn_hidden_dim = ModelConfig.D_MODEL * ModelConfig.FFN_EXPAND_RATIO

    if isinstance(module, nn.Linear):
        if module.out_features == 1:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.OUTPUT_LAYER_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ModelConfig.INPUT_DIM and module.out_features == ModelConfig.D_MODEL:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.EMBEDDING_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ModelConfig.D_MODEL and module.out_features == ffn_hidden_dim:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.FFN_INIT_GAIN)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif module.in_features == ffn_hidden_dim and module.out_features == ModelConfig.D_MODEL:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Position Embedding初始化
        # vocab_size = CONTEXT_LENGTH = 45
        if module.weight.shape[0] == DataConfig.CONTEXT_LENGTH:
            nn.init.xavier_uniform_(module.weight, gain=ModelConfig.POSITION_EMBEDDING_INIT_GAIN)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1.0)


class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learned Positional Embedding）
    类似 BERT / GPT 的做法：每个位置对应一个可训练的向量
    让模型自己学习最优的位置表示，而非使用固定的正弦公式
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        #添加位置编码，LayerNorm在后续层中可能使用
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions).unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """
    多头自注意力模块（不含残差连接和归一化）
    仅负责注意力计算，残差连接由上层TransformerLayer统一管理
    """
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0
        
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)

    def forward(self, x, attn_mask=None):
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)
        
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        return self.dropout(attn_output)


class TransformerLayer(nn.Module):
    """
    标准 Transformer 层（Pre-Norm架构，SwiGLU前馈网络）
    SwiGLU: w2(SiLU(w1(x)) * w3(x))，门控机制提供选择性信息流动
    参考: Shazeer, "GLU Variants Improve Transformer" (2020)
    """
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()

        # 注意力子层
        self.attn = MultiHeadAttention(d_model, nhead)
        self.attn_norm = nn.LayerNorm(d_model)

        # SwiGLU前馈网络: w2(SiLU(w1(x)) * w3(x))
        # bias=False: Shazeer (2020) 原始设计，LLaMA/PaLM/DeepSeek/Qwen 均不使用 bias
        # GLU 逐元素乘法中 bias 会产生交叉项，导致信号偏移被平方级放大
        ffn_hidden_dim = int(d_model * ModelConfig.FFN_EXPAND_RATIO)
        self.ffn_w1 = nn.Linear(d_model, ffn_hidden_dim, bias=False)
        self.ffn_w3 = nn.Linear(d_model, ffn_hidden_dim, bias=False)
        self.ffn_w2 = nn.Linear(ffn_hidden_dim, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

    def forward(self, x):
        # 注意力子层: x = x + Dropout(Attention(LayerNorm(x)))
        x = x + self.attn(self.attn_norm(x), attn_mask=None)

        # SwiGLU前馈网络: x = x + Dropout(w2(SiLU(w1(h)) * w3(h)))
        h = self.ffn_norm(x)
        x = x + self.ffn_dropout(self.ffn_w2(F.silu(self.ffn_w1(h)) * self.ffn_w3(h)))

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

        # 初始化 query token（使用Xavier初始化）
        nn.init.xavier_uniform_(self.query, gain=ModelConfig.QUERY_INIT_GAIN)

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


class StockTransformer(nn.Module):
    """
    Transformer 模型（Pre-Norm 架构 + FFN-Embedding）

    核心设计：
    - FFN-Embedding: Linear(9→128) → GELU → Linear(128→128)
      相比纯线性映射，GELU在中间层提供非线性特征组合能力
      让第一层Transformer就能访问到特征间的非线性交互（如上影线、实体大小等）
    - Transformer 层：标准 Attention + FFN 结构，专注于跨时间模式识别
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_len):
        super(StockTransformer, self).__init__()

        # FFN-Embedding：线性投影 + 残差MLP（非线性特征交互）
        # Linear(9→128): 基础线性映射，保底传递原始特征信息
        # GELU + Linear(128→128): 残差分支，专注学习非线性交互（K线形态翻转等）
        # 残差连接: 线性映射永远保底，MLP只负责"加增量"
        self.embed_proj = nn.Linear(input_dim, d_model)
        self.embed_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # 使用标准位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        # 统一架构：所有层都使用 Attention + FFN
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead)
            for i in range(num_layers)
        ])

        # Pre-Norm架构：在最后添加一个LayerNorm
        # 因为Pre-Norm的最后一层没有归一化输出
        self.final_norm = nn.LayerNorm(d_model)

        # 多头注意力聚合：通过 cross-attention 聚合序列信息
        # 相比单向量点积，每个注意力头可以学到不同的时间聚合模式
        self.attention_pooling = AttentionPooling(d_model, nhead)

        # Pooling输出归一化：与backbone的Pre-Norm风格一致，稳定分类头输入
        self.head_norm = nn.LayerNorm(d_model)

        # 单层线性分类头（主流做法），让backbone承担特征学习
        self.output_projection = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

        # 应用初始化
        self.apply(init_weights)

        # FFN-Embedding残差MLP第二层显式初始化：补偿GELU压缩
        nn.init.xavier_uniform_(self.embed_mlp[1].weight, gain=ModelConfig.FFN_INIT_GAIN)
        nn.init.zeros_(self.embed_mlp[1].bias)

    def forward(self, x):
        # x: [batch_size, seq_len, 10] (OHLC + vwap + volume + exchange + m5 + m10 + m20)

        # 1. FFN-Embedding：线性投影 + 残差MLP
        x = self.embed_proj(x)          # 线性映射保底
        x = x + self.embed_mlp(x)       # MLP学非线性交互，残差保护线性映射

        # 位置编码
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

        # 6. Pooling归一化 + 分类头
        aggregated = self.head_norm(aggregated)
        output = self.output_projection(aggregated)  # [batch_size, output_dim]
        return output

    def load_pretrained_embedding(self, path):
        """
        加载预训练 embedding 权重（来自 pretrain_embedding.py）

        Args:
            path: 预训练权重 .pth 文件路径
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        self.embed_proj.weight.data.copy_(checkpoint['embed_proj_weight'])
        self.embed_proj.bias.data.copy_(checkpoint['embed_proj_bias'])
        # embed_mlp 结构为 Sequential(GELU, Linear)，通过类型定位 Linear 层
        linear_layer = next(m for m in self.embed_mlp if isinstance(m, nn.Linear))
        linear_layer.weight.data.copy_(checkpoint['embed_mlp_1_weight'])
        linear_layer.bias.data.copy_(checkpoint['embed_mlp_1_bias'])

        print(f"  已加载预训练 Embedding: {path}")

    def freeze_embedding(self, freeze=True):
        """
        冻结或解冻 FFN-Embedding 参数

        Args:
            freeze: True=冻结，False=解冻
        """
        for param in self.embed_proj.parameters():
            param.requires_grad = not freeze
        for param in self.embed_mlp.parameters():
            param.requires_grad = not freeze

        n_frozen = (sum(p.numel() for p in self.embed_proj.parameters()) +
                    sum(p.numel() for p in self.embed_mlp.parameters()))
        status = "冻结" if freeze else "解冻"
        print(f"  Embedding {status}: {n_frozen:,} 参数")

# ==================== 工厂函数 ====================

def create_model(input_dim=ModelConfig.INPUT_DIM, d_model=ModelConfig.D_MODEL,
                 nhead=ModelConfig.NHEAD, num_layers=ModelConfig.NUM_LAYERS,
                 output_dim=ModelConfig.OUTPUT_DIM, seq_len=DataConfig.CONTEXT_LENGTH,
                 model_arch=None):
    """
    Args:
        参数均为可选，如果不提供则使用 ModelConfig 中的默认值
        model_arch: 可选的元数据字典（来自 .pth 内的 'model_arch' 键），
                    若提供则优先使用其中的参数覆盖默认值，
                    用于 run.py 自动重建与训练时架构一致的模型

    Returns:
        model: StockTransformer 模型实例
    """
    if model_arch is not None:
        input_dim  = model_arch.get('input_dim',  input_dim)
        d_model    = model_arch.get('d_model',    d_model)
        nhead      = model_arch.get('nhead',      nhead)
        num_layers = model_arch.get('num_layers', num_layers)
        output_dim = model_arch.get('output_dim', output_dim)
        seq_len    = model_arch.get('context_length', seq_len)

    model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_dim=output_dim,
        seq_len=seq_len,
    )

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"模型架构 (StockTransformer)")
    print(f"{'='*50}")
    print(f"输入维度: {input_dim}")
    print(f"序列长度: {seq_len}")
    print(f"Embedding维度: {d_model}")
    print(f"注意力头数: {nhead}")
    print(f"Transformer层数: {num_layers}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"{'='*50}\n")

    return model
