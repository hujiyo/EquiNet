'''
优化版训练脚本---
主要改进：
1. 增加了时间感知的位置编码，结合了标准的正弦余弦位置编码，让每个时间位置都有独特标识，
让模型知道哪些数据是近期的，哪些是远期的：使用指数衰减：最新的第60天权重为1.0，往前每天权重递减
2. 设计了专业化的多头注意力机制，不同的头关注不同类型的市场信号：将8个注意力头分成4类：价格趋势头、成交量头、波动率头、综合模式头
每类头专门学习特定类型的市场信号
用可学习权重自动融合不同类型的输出
3. 添加了多尺度注意力，捕获短期、中期、长期的不同模式：同时捕获短期(5-10天)、中期(15-30天)、长期(整个60天)模式
不同尺度的信息通过可学习权重进行融合
加入了残差连接和层归一化，提升训练稳定性

评分制度保持不变：
提供1000次预测机会，预测正确加一分
预测错误则按下面策略处理：
1.上涨的股票预测为下跌：-1分 
2.下跌的股票预测为上涨：-2分 
3.其余情况不加分也不扣分。
'''

import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
from tqdm import tqdm
import random
import math
import torch.nn.functional as F

# Focal Loss实现
class FocalLoss(nn.Module):
    """
    Focal Loss专门用于处理类别不平衡问题
    FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    
    参数说明：
    - alpha: 类别权重，用于平衡正负样本
    - gamma: 聚焦参数，减少易分类样本的权重
    - reduction: 损失的归约方式
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = [1.5, 2.0, 1.0]
        # 注册为buffer，会自动跟随模型移动到相应设备
        self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        inputs: [batch_size, num_classes] 模型输出的logits
        targets: [batch_size] 真实类别标签
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        targets = targets.to(self.alpha.device)
        # 现在alpha自动在正确的设备上，不用检查
        alpha_t = self.alpha[targets]  # 直接用，不用担心设备问题

        # 保证 alpha_t, pt, ce_loss 在同一设备
        device = inputs.device
        alpha_t = alpha_t.to(device)
        pt = pt.to(device)
        ce_loss = ce_loss.to(device)
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 动态类别权重调整器
class DynamicClassWeightAdjuster:
    """
    动态调整类别权重，根据训练过程中的类别分布实时调整
    """
    def __init__(self, num_classes=3, window_size=1000):
        self.num_classes = num_classes
        self.window_size = window_size
        self.class_counts = np.zeros(num_classes)
        self.total_samples = 0
        
    def update(self, targets):
        """更新类别计数"""
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        unique, counts = np.unique(targets, return_counts=True)
        for cls, count in zip(unique, counts):
            self.class_counts[cls] += count
            self.total_samples += count
    
    def get_weights(self):
        """计算当前的类别权重"""
        if self.total_samples == 0:
            return [1.0, 1.0, 1.0]
        
        # 计算每个类别的频率
        frequencies = self.class_counts / self.total_samples
        
        # 使用逆频率作为权重，并平滑处理
        weights = 1.0 / (frequencies + 1e-6)  # 加小数防止除零
        
        # 归一化权重
        weights = weights / np.mean(weights)
        
        # 限制权重范围，避免过度不平衡
        weights = np.clip(weights, 0.5, 3.0)
        
        return weights.tolist()

# 时间感知的位置编码类
class TimeAwarePositionalEncoding(nn.Module):
    """
    时间感知的位置编码，有两个作用：
    1. 让模型知道每个数据点在时间序列中的位置（第1天、第2天...第60天）
    2. 给近期数据分配更高的权重，因为近期数据对未来预测更重要
    """
    def __init__(self, d_model, max_seq_len=100, decay_factor=0.1):
        super(TimeAwarePositionalEncoding, self).__init__()
        
        # 创建标准的正弦余弦位置编码，让模型理解序列顺序
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # 使用不同频率的正弦余弦函数，让每个位置都有独特的编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用cos
        
        # 添加时间衰减权重：距离当前时间越近，权重越大
        # 比如第60天(最新)权重为1.0，第59天权重为0.9，第58天为0.81...
        time_weights = torch.exp(-decay_factor * torch.arange(max_seq_len - 1, -1, -1, dtype=torch.float))
        pe = pe * time_weights.unsqueeze(1)  # 将时间权重应用到位置编码上
        
        self.register_buffer('pe', pe)  # 注册为buffer，不参与梯度更新但会保存到模型中
        
    def forward(self, x):
        # x的shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        # 将对应长度的位置编码加到输入上
        return x + self.pe[:seq_len, :].unsqueeze(0)

# 专业化的多头注意力机制
class SpecializedMultiHeadAttention(nn.Module):
    """
    专业化的多头注意力，不同的头关注不同类型的市场信息：
    - 价格趋势头：主要看开盘、收盘价的变化趋势
    - 成交量头：分析量价关系，成交量的变化模式  
    - 波动头：关注价格波动率，最高最低价的变化
    - 综合头：学习其他复杂的组合模式
    """
    def __init__(self, d_model, nhead):
        super(SpecializedMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 确保d_model能被nhead整除
        assert d_model % nhead == 0
        
        # 为不同类型的头分配数量
        self.price_heads = max(1, nhead // 4)      # 价格趋势头
        self.volume_heads = max(1, nhead // 4)     # 成交量分析头  
        self.volatility_heads = max(1, nhead // 4) # 波动率分析头
        self.pattern_heads = nhead - self.price_heads - self.volume_heads - self.volatility_heads  # 综合模式头
        
        # 为每种类型的头创建独立的注意力机制
        self.price_attention = nn.MultiheadAttention(d_model, self.price_heads, batch_first=True)
        self.volume_attention = nn.MultiheadAttention(d_model, self.volume_heads, batch_first=True) 
        self.volatility_attention = nn.MultiheadAttention(d_model, self.volatility_heads, batch_first=True)
        self.pattern_attention = nn.MultiheadAttention(d_model, self.pattern_heads, batch_first=True)
        
        # 用于融合不同类型注意力输出的权重，让模型学会如何组合这些信息
        self.fusion_weights = nn.Parameter(torch.ones(4) / 4)  # 初始化为平均权重
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_mask=None):
        # x的shape: [batch_size, seq_len, d_model]
        # attn_mask: [seq_len, seq_len] or None

        # 处理mask：nn.MultiheadAttention 需要float mask，且要在同一设备
        mask = None
        if attn_mask is not None:
            mask = attn_mask.to(dtype=x.dtype, device=x.device)

        price_out, _ = self.price_attention(x, x, x, attn_mask=mask)
        volume_out, _ = self.volume_attention(x, x, x, attn_mask=mask)
        volatility_out, _ = self.volatility_attention(x, x, x, attn_mask=mask)
        pattern_out, _ = self.pattern_attention(x, x, x, attn_mask=mask)

        weights = torch.softmax(self.fusion_weights, dim=0)
        fused_output = (weights[0] * price_out +
                        weights[1] * volume_out +
                        weights[2] * volatility_out +
                        weights[3] * pattern_out)
        return self.fusion_norm(fused_output)

# 多尺度注意力层
class MultiScaleAttentionLayer(nn.Module):
    """
    多尺度注意力层，同时捕获短期、中期、长期的市场模式：
    - 短期：关注最近5-10天的快速变化
    - 中期：关注15-30天的中等趋势  
    - 长期：关注整个60天序列的大趋势
    """
    def __init__(self, d_model, nhead):
        super(MultiScaleAttentionLayer, self).__init__()
        
        # 为不同时间尺度创建专门的注意力机制
        self.short_term_attention = SpecializedMultiHeadAttention(d_model, nhead)  # 短期模式
        self.medium_term_attention = SpecializedMultiHeadAttention(d_model, nhead) # 中期模式  
        self.long_term_attention = SpecializedMultiHeadAttention(d_model, nhead)   # 长期模式
        
        # 前馈网络，用于进一步处理注意力的输出
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 先扩展维度
            nn.ReLU(),                        # 激活函数
            nn.Dropout(0.1),                  # 防过拟合
            nn.Linear(d_model * 4, d_model),  # 再压缩回原维度
        )
        
        # 层归一化，帮助训练稳定
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(0.1)
        
        # 多尺度融合的权重
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
    def create_scale_mask(self, seq_len, scale_type):
        """
        为不同尺度创建注意力掩码：
        - 短期：只能看到最近10天
        - 中期：只能看到最近30天  
        - 长期：可以看到全部数据，但远期数据权重衰减
        """
        mask = torch.zeros(seq_len, seq_len)
        
        if scale_type == 'short':
            # 短期：只关注最近10天的数据
            window = min(10, seq_len)
            mask[-window:, -window:] = 1
        elif scale_type == 'medium':
            # 中期：关注最近30天的数据
            window = min(30, seq_len)  
            mask[-window:, -window:] = 1
        else:  # long term
            # 长期：可以看全部数据，但距离越远权重越小
            for i in range(seq_len):
                for j in range(seq_len):
                    # 距离越远，权重越小（时间衰减）
                    distance = abs(i - j)
                    mask[i, j] = math.exp(-0.1 * distance)
                    
        return mask
        
    def forward(self, x):
        # x的shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)

        # 创建不同尺度的掩码，并转为float32和输入同设备
        short_mask = self.create_scale_mask(seq_len, 'short').to(dtype=x.dtype, device=x.device)
        medium_mask = self.create_scale_mask(seq_len, 'medium').to(dtype=x.dtype, device=x.device)
        long_mask = self.create_scale_mask(seq_len, 'long').to(dtype=x.dtype, device=x.device)

        short_out = self.short_term_attention(x, attn_mask=short_mask)
        medium_out = self.medium_term_attention(x, attn_mask=medium_mask)
        long_out = self.long_term_attention(x, attn_mask=long_mask)
        # 短期、中期、长期注意力输出
        
        # 使用可学习权重融合不同尺度的输出
        scale_weights = torch.softmax(self.scale_weights, dim=0)
        multi_scale_out = (scale_weights[0] * short_out + 
                          scale_weights[1] * medium_out + 
                          scale_weights[2] * long_out)
        
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(multi_scale_out))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x

# 增强版的Transformer模型
class EnhancedStockTransformer(nn.Module):
    """
    增强版的股票预测Transformer模型，主要改进：
    1. 加入了时间感知的位置编码
    2. 使用专业化的多头注意力机制
    3. 采用多尺度注意力层
    4. 增加了更好的正则化机制
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, max_seq_len=100):
        super(EnhancedStockTransformer, self).__init__()
        
        # 输入特征嵌入层：将8维输入特征映射到d_model维度
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 时间感知的位置编码
        self.pos_encoding = TimeAwarePositionalEncoding(d_model, max_seq_len)
        
        # 多个多尺度注意力层堆叠
        self.layers = nn.ModuleList([
            MultiScaleAttentionLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        
        # 输出层：从d_model维度映射到3个类别（上涨/下跌/震荡）
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 先降维
            nn.ReLU(),                         # 激活
            nn.Dropout(0.2),                   # 防过拟合
            nn.Linear(d_model // 2, output_dim) # 最终输出
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x的shape: [batch_size, seq_len, input_dim]
        
        # 1. 特征嵌入：将原始特征映射到更高维度的表示空间
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # 2. 加入位置编码：让模型知道时间顺序和远近关系
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 3. 通过多个多尺度注意力层进行特征学习
        for layer in self.layers:
            x = layer(x)
        
        # 4. 取最后一个时间步的输出用于预测（因为我们要预测未来）
        # x[:, -1, :] 表示取每个样本的最后一个时间步
        last_hidden = x[:, -1, :]  # [batch_size, d_model]
        
        # 5. 输出层：得到3个类别的logits（注意：这里不使用softmax，让损失函数来处理）
        output = self.output_projection(last_hidden)  # [batch_size, output_dim]
        
        return output

# 生成单个样本
def generate_single_sample(all_data):
    """
    从股票数据中随机生成一个训练样本
    输入：60天的历史数据
    输出：根据未来3天收益率确定的类别标签
    """
    for _ in range(100):  # 最多尝试100次生成有效样本
        stock_index = np.random.randint(0, len(all_data))
        stock_data = all_data[stock_index]
        context_length = 60  # 使用60天历史数据
        required_length = context_length + 3  # 需要额外3天来计算未来收益
        
        if len(stock_data) < required_length:
            continue
            
        start_index = np.random.randint(0, len(stock_data) - required_length + 1)
        input_seq = stock_data[start_index:start_index + context_length]  # 60天历史数据
        target_seq = stock_data[start_index + context_length:start_index + required_length]  # 未来3天
        
        # 计算收益率：(未来价格 - 当前价格) / 当前价格
        start_price = input_seq[-1, 3]  # 当前收盘价（第3列是end收盘价）
        end_price = target_seq[-1, 3]   # 3天后的收盘价
        
        if start_price == 0:  # 避免除零错误
            continue
            
        cumulative_return = (end_price - start_price) / start_price
        
        # 根据收益率确定类别标签
        if cumulative_return >= 0.03:      # 涨幅≥3%：大涨
            target = 0
        elif cumulative_return <= -0.02:   # 跌幅≥2%：大跌  
            target = 1
        else:                              # 其他情况：震荡
            target = 2
            
        return input_seq, target
    
    raise ValueError("无法生成有效样本：股票数据长度不足或收盘价为0")

def generate_batch_samples(all_data, batch_size):
    """
    批量生成训练样本
    返回: (batch_inputs, batch_targets)
    batch_inputs: numpy array, shape [batch_size, context_length, 8]  
    batch_targets: numpy array, shape [batch_size]
    """
    batch_inputs = []
    batch_targets = []
    
    attempts = 0
    max_attempts = batch_size * 10  # 防止无限循环
    
    while len(batch_inputs) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            input_seq, target = generate_single_sample(all_data)
            batch_inputs.append(input_seq)
            batch_targets.append(target)
        except ValueError:
            continue
    
    if len(batch_inputs) < batch_size:
        raise ValueError(f"无法生成足够的样本，只生成了 {len(batch_inputs)}/{batch_size} 个")
    
    return np.array(batch_inputs), np.array(batch_targets)

def create_evaluation_dataset(test_data, num_samples=1000):
    eval_inputs = []
    eval_targets = []
    
    for _ in tqdm(range(num_samples), desc='生成评估样本'):
        input_seq, target = generate_single_sample(test_data)
        eval_inputs.append(input_seq)
        eval_targets.append(target)
    
    return np.array(eval_inputs), np.array(eval_targets)

# 数据预处理函数
def load_and_preprocess_data(data_dir, test_ratio=0.1, seed=42):
    """
    改进的数据加载和预处理函数
    确保训练集和测试集完全独立，没有数据泄露
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    random.seed(seed)
    
    # 按股票文件划分训练集和测试集
    test_size = max(1, int(len(all_files) * test_ratio))
    test_files = set(random.sample(all_files, test_size))
    train_files = [f for f in all_files if f not in test_files]
    
    print(f"训练股票文件: {len(train_files)} 个")
    print(f"测试股票文件: {len(test_files)} 个")

    def process_files(file_list):
        data_list = []
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path)
            try:
                data = df[['start', 'max', 'min', 'end', 'volume', 'marketvolume', 'marketlimit', 'marketrange']].values
                
                # 每只股票单独标准化
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                if np.any(std == 0):
                    raise ValueError(f"文件 {file} 包含标准差为0的列")
                normalized_data = (data - mean) / std
                data_list.append(normalized_data)
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")
        return data_list

    train_data = process_files(train_files)
    test_data = process_files(test_files)
    return train_data, test_data

# 创建固定的评估数据集
def create_fixed_evaluation_dataset(test_data, num_samples=1000, seed=42):
    """
    创建固定的评估数据集，确保每次评估使用相同的样本
    这样可以准确衡量模型的进步情况
    """
    print("正在创建固定的评估数据集...")
    np.random.seed(seed)  # 固定随机种子
    random.seed(seed)
    
    eval_inputs = []
    eval_targets = []
    
    # 预先生成所有可能的样本
    all_possible_samples = []
    context_length = 60
    required_length = context_length + 3
    
    for stock_idx, stock_data in enumerate(test_data):
        if len(stock_data) < required_length:
            continue
            
        # 为每只股票生成所有可能的时间窗口样本
        for start_idx in range(len(stock_data) - required_length + 1):
            input_seq = stock_data[start_idx:start_idx + context_length]
            target_seq = stock_data[start_idx + context_length:start_idx + required_length]
            
            start_price = input_seq[-1, 3]  # 当前收盘价
            end_price = target_seq[-1, 3]   # 3天后收盘价
            
            if start_price == 0:
                continue
                
            cumulative_return = (end_price - start_price) / start_price
            
            # 确定类别标签
            if cumulative_return >= 0.03:
                target = 0  # 大涨
            elif cumulative_return <= -0.02:
                target = 1  # 大跌
            else:
                target = 2  # 震荡
                
            all_possible_samples.append((input_seq, target, stock_idx, start_idx))
    
    print(f"总共可用样本: {len(all_possible_samples)} 个")
    
    # 随机选择固定的评估样本
    if len(all_possible_samples) < num_samples:
        print(f"警告: 可用样本数 ({len(all_possible_samples)}) 少于请求的样本数 ({num_samples})")
        selected_samples = all_possible_samples
    else:
        selected_samples = random.sample(all_possible_samples, num_samples)
    
    # 分离输入和标签
    for input_seq, target, stock_idx, start_idx in selected_samples:
        eval_inputs.append(input_seq)
        eval_targets.append(target)
    
    eval_inputs = np.array(eval_inputs)
    eval_targets = np.array(eval_targets)
    
    # 打印类别分布
    unique, counts = np.unique(eval_targets, return_counts=True)
    class_names = ['大涨', '大跌', '震荡']
    print("评估集类别分布:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names[cls]}: {count} 个样本 ({count/len(eval_targets)*100:.1f}%)")
    
    return eval_inputs, eval_targets

# 批量评估函数
def evaluate_model_batch(model, eval_inputs, eval_targets, device, batch_size=100):
    """
    使用批处理进行快速评估
    """
    model.eval()
    score = 0
    total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc='批量评估中'):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # 批量处理
            batch_inputs = torch.tensor(eval_inputs[start_idx:end_idx], 
                                      dtype=torch.float32).to(device)
            batch_targets = eval_targets[start_idx:end_idx]
            
            # 批量推理
            batch_outputs = model(batch_inputs)  # [batch_size, 3]
            batch_predictions = torch.argmax(batch_outputs, dim=1).cpu().numpy()
            
            # 批量计算得分
            for j in range(len(batch_targets)):
                target = batch_targets[j]
                prediction = batch_predictions[j]
                
                class_total[target] += 1
                
                # 应用特殊的评分规则
                if prediction == target:
                    score += 1
                    class_correct[target] += 1
                elif target == 0 and prediction == 1:  # 上涨预测为下跌
                    score -= 1
                elif target == 1 and prediction == 0:  # 下跌预测为上涨  
                    score -= 2
                
                total += 1
    
    return score, total, class_correct, class_total

# 改进的训练函数
def train_model(model, train_data, test_data, epochs, learning_rate, device, batch_size, batches_per_epoch):
    """
    使用固定评估集的训练函数
    """
    # 创建固定的评估数据集（训练开始前创建一次）
    eval_inputs, eval_targets = create_fixed_evaluation_dataset(test_data, num_samples=1000)
    
    criterion = FocalLoss(alpha=[1.5, 2.0, 1.0], gamma=2.0)
    weight_adjuster = DynamicClassWeightAdjuster()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_score = float('-inf')  # 改用得分而不是准确率
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_targets = []
        
        # 训练阶段
        pbar = tqdm(range(batches_per_epoch), 
                   desc=f'Epoch {epoch + 1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        for step in pbar:
            batch_inputs, batch_targets_np = generate_batch_samples(train_data, batch_size)
            batch_targets.extend(batch_targets_np.tolist())
            
            batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
            batch_targets_tensor = torch.tensor(batch_targets_np, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            output = model(batch_inputs)
            loss = criterion(output, batch_targets_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': total_loss / (step + 1)})
        
        # 更新权重
        weight_adjuster.update(batch_targets)
        current_weights = weight_adjuster.get_weights()
        criterion.alpha = torch.tensor(current_weights, device=next(model.parameters()).device)
        scheduler.step()
        
        # 固定评估集评估
        print("正在评估模型性能...")
        score, total, class_correct, class_total = evaluate_model_batch(
            model, eval_inputs, eval_targets, device, batch_size=100
        )
        
        # 打印详细结果
        class_names = ['大涨', '大跌', '震荡']
        print(f"\nEpoch {epoch + 1} 评估结果:")
        for i in range(3):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f'  {class_names[i]}: {class_correct[i]}/{class_total[i]} = {acc:.3f}')
            else:
                print(f'  {class_names[i]}: 0/0 = 0.000 (无样本)')
        
        overall_acc = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
        avg_score = score / total if total > 0 else 0
        
        print(f'  总体准确率: {overall_acc:.3f}')
        print(f'  评估得分: {score} / {total} = {avg_score:.3f}')
        
        # 保存最佳模型
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), './out/EnhancedEquiNet_focal_best.pth')
            print(f'  ✓ 发现更好的模型！得分提升到: {score}')
        
        print("-" * 50)

if __name__ == "__main__":
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA 不可用，将使用 CPU 进行训练，训练速度可能较慢。")
    else:
        print(f"使用 GPU 进行训练: {torch.cuda.get_device_name()}")

    # 创建输出目录
    os.makedirs('./out', exist_ok=True)
    
    # 使用改进的数据加载函数
    print("正在加载和预处理数据...")
    train_data, test_data = load_and_preprocess_data('./data')
    print(f"训练数据: {len(train_data)} 只股票")
    print(f"测试数据: {len(test_data)} 只股票")

    # 模型超参数
    d_model = 128        # 模型维度（更高的维度通常能捕获更复杂的模式）
    input_dim = 8        # 输入特征维度（OHLCV + 市值相关特征）
    nhead = 8           # 注意力头数（会被分配给不同类型的专业化头）
    num_layers = 4      # Transformer层数
    output_dim = 3      # 输出类别数（上涨/下跌/震荡）

    # 训练超参数  
    epochs = 80                    # 训练轮数
    learning_rate = 0.001          # 初始学习率
    batch_size = 50  # GPU每次并行训练的样本数
    batches_per_epoch = 20  # 相当于每轮（epochs） 20*50=1000 个样本

    print("正在创建 Transformer 模型...")
    model = EnhancedStockTransformer(
        input_dim=input_dim, 
        d_model=d_model, 
        nhead=nhead, 
        num_layers=num_layers, 
        output_dim=output_dim
    ).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")

    print("开始训练...")
    # 使用带固定评估集的训练函数
    train_model(model, train_data, test_data, epochs, learning_rate, device, batch_size, batches_per_epoch)
    
    
    # 保存最终模型
    final_model_path = f'./out/EnhancedEquiNet_{d_model}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"训练完成！最终模型已保存到: {final_model_path}")
    print("最佳模型已保存到: ./out/EnhancedEquiNet_best.pth")