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


class PositionalEncoding(nn.Module):
    """
    可学习位置编码（Learned Positional Embedding）
    类似 BERT / GPT 的做法：每个位置对应一个可训练的向量
    让模型自己学习最优的位置表示，而非使用固定的正弦公式
    """
    def __init__(self, d_model, seq_len=DataConfig.CONTEXT_LENGTH):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Embedding(seq_len, d_model)
        # gain推导: 使position embedding输出std匹配FFN-Embedding输出std≈0.25
        # xavier_uniform_输出std = gain × √(2/(45+128)) = gain × 0.10752
        # 令其=0.25 → gain = 0.25/0.10752 ≈ 2.32
        nn.init.xavier_uniform_(self.pe.weight, gain=2.32)

    def forward(self, x):
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
        
        self.attention = nn.MultiheadAttention(d_model, nhead, bias=False, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.ATTENTION_DROPOUT)

    def forward(self, x, attn_mask=None, return_attn=False):
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)

        attn_output, attn_weights = self.attention(
            x, x, x, attn_mask=mask,
            need_weights=return_attn,
            average_attn_weights=False
        )
        attn_output = self.dropout(attn_output)
        if return_attn:
            return attn_output, attn_weights
        return attn_output


class TransformerLayer(nn.Module):
    """
    标准 Transformer 层（Post-Norm架构，SwiGLU前馈网络）
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

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_w = self.attn(x, attn_mask=None, return_attn=True)
            x = self.attn_norm(x + attn_out)
        else:
            x = self.attn_norm(x + self.attn(x))

        x = self.ffn_norm(x + self.ffn_dropout(self.ffn_w2(F.silu(self.ffn_w1(x)) * self.ffn_w3(x))))

        if return_attn:
            return x, attn_w
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

        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        # 匹配cross-attn初始输出std≈0.15，使残差连接在训练初期有意义
        with torch.no_grad():
            self.query.data.normal_(std=0.15)

        self.norm_q = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, bias=False, batch_first=True)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

    def forward(self, x, return_attn=False):
        batch_size = x.size(0)

        query = self.query.expand(batch_size, -1, -1)

        attn_output, attn_weights = self.cross_attn(
            query, x, x,
            need_weights=return_attn,
            average_attn_weights=False
        )

        pooled = self.norm_q((query + self.dropout(attn_output)).squeeze(1))

        if return_attn:
            return pooled, attn_weights
        return pooled


class StockTransformer(nn.Module):
    """
    Transformer 模型（Post-Norm 架构 + FFN-Embedding）

    核心设计：
    - FFN-Embedding: MLP(input_dim→d_model→hidden→GELU→d_model)
      纯 MLP 结构，3层网络无需残差连接即可充分学习非线性特征交互。
    - Transformer 层：标准 Attention + FFN 结构，专注于跨时间模式识别
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, seq_len,
                 embed_expand_ratio=2):
        super(StockTransformer, self).__init__()

        # FFN-Embedding：MLP（非线性特征交互）
        # Linear(input_dim→d_model): 维度扩展
        # MLP(d_model→hidden→GELU→d_model): 学习非线性交互（K线形态翻转等）
        hidden_dim = d_model * embed_expand_ratio
        self.embed_proj = nn.Linear(input_dim, d_model, bias=False)
        self.embed_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model, bias=False)
        )

        # 使用标准位置编码
        self.pos_encoding = PositionalEncoding(d_model, seq_len)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead)
            for i in range(num_layers)
        ])

        self.attention_pooling = AttentionPooling(d_model, nhead)

        self.head_norm = nn.LayerNorm(d_model)

        self.output_projection = nn.Linear(d_model, output_dim, bias=True)
        self.dropout = nn.Dropout(ModelConfig.DROPOUT_RATE)

    def forward(self, x, return_attn=False):
        x = self.embed_proj(x)
        x = self.embed_mlp(x)

        x = self.pos_encoding(x)
        x = self.dropout(x)

        attn_weights = [] if return_attn else None
        for layer in self.layers:
            if return_attn:
                x, w = layer(x, return_attn=True)
                attn_weights.append(w)
            else:
                x = layer(x)

        if return_attn:
            aggregated, pool_w = self.attention_pooling(x, return_attn=True)
            attn_weights.append(pool_w)
        else:
            aggregated = self.attention_pooling(x)

        aggregated = self.head_norm(aggregated)
        output = self.output_projection(aggregated)

        if return_attn:
            return output, attn_weights
        return output

    def load_pretrained_embedding(self, path):
        """
        加载预训练 embedding 权重（来自 pretrain_embedding.py）

        Args:
            path: 预训练权重 .pth 文件路径
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)

        self.embed_proj.weight.data.copy_(checkpoint['embed_proj_weight'])
        self.embed_mlp[0].weight.data.copy_(checkpoint['embed_mlp_0_weight'])
        self.embed_mlp[2].weight.data.copy_(checkpoint['embed_mlp_2_weight'])

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

    # 输出层初始化：小 gain 防止 sigmoid 饱和，bias 设为先验 logit
    nn.init.xavier_uniform_(model.output_projection.weight, gain=ModelConfig.OUTPUT_LAYER_GAIN)
    # sigmoid(logit(prior)) = prior, 先验≈0.25 → bias = log(0.25/0.75) ≈ -1.1
    prior = 0.25
    model.output_projection.bias.data.fill_(math.log(prior / (1.0 - prior)))

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
