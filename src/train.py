import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
from tqdm import tqdm

# 定义Transformer模型
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(StockTransformer, self).__init__() # 调用父类的初始化方法
        self.embedding = nn.Linear(input_dim, d_model) # 输入维度到模型维度的线性变换,简单来说就是使用nn.Linear将输入的特征维度的数据映射到模型维度的空间
        # 编码器,将输入序列转换为一个固定长度的向量表示
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),num_layers=num_layers)             
        self.fc = nn.Linear(d_model, output_dim) # 模型维度到输出维度的线性变换
    def forward(self, x): # 前向传播 x: 输入序列
        x = self.embedding(x) # 将输入序列映射到模型维度的空间
        x = self.transformer_encoder(x) # 将输入序列通过编码器转换为一个固定长度的向量表示
        x = self.fc(x[:, -1, :]) # 取最后一个时间步的输出作为模型的输出
        return x

# 数据预处理
def load_and_preprocess_data(data_dir):
    all_data = []
    for file in os.listdir(data_dir):
        if file.endswith('.xlsx'):
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path)            
            try:                
                data = df[['start', 'max', 'min', 'end', 'volume', 'marketvolume', 'marketlimit', 'marketrange']].values # 提取特征列（共8个特征）                
                mean = np.mean(data, axis=0)# # 归一化并保存为独立股票数据
                std = np.std(data, axis=0)
                if np.any(std == 0):
                    raise ValueError(f"文件 {file} 包含标准差为0的列")   
                normalized_data = (data - mean) / std
                all_data.append(normalized_data)  # 每个元素是一个股票的二维数组                
            except Exception as e:
                print(f"处理文件 {file} 时出错: {e}")    
    return all_data  # 返回列表，每个元素是一个股票的(时间步×特征)数组

# 生成单个样本
def generate_single_sample(all_data):
    for _ in range(100):# 尝试最多100次选择有效样本（防止无限循环）
        stock_index = np.random.randint(0, len(all_data))
        stock_data = all_data[stock_index]
        context_length = 60  # 固定时间窗口为60天
        required_length = context_length + 3  # 输入序列+目标序列总长度        
        if len(stock_data) < required_length:continue # 如果当前股票数据不足，重新选择
        # 在有效范围内随机选择起始点
        start_index = np.random.randint(0, len(stock_data) - required_length + 1)        
        # 生成输入序列和目标序列
        input_seq = stock_data[start_index:start_index + context_length]
        target_seq = stock_data[start_index + context_length:start_index + required_length]        
        # 计算目标序列整体趋势（3天累计收益率）
        start_price = input_seq[-1, 3]  # 输入序列最后一个收盘价
        end_price = target_seq[-1, 3]    # 目标序列最后一个收盘价
        cumulative_return = (end_price - start_price) / start_price
        # 基于3天累计收益率进行标签分类           
        if cumulative_return >= 0.03:    target = [1, 0, 0]  # 上涨
        elif cumulative_return <= -0.02: target = [0, 1, 0]  # 下跌
        else:                            target = [0, 0, 1]  # 平稳
        return input_seq,target
    raise ValueError("无法生成有效样本：股票数据长度不足")# 单只股票数据长度小于60

# 训练模型
def train_model(model, all_data, epochs, learning_rate, device, num_samples_per_epoch):
    criterion = nn.CrossEntropyLoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) # 学习率调度器，每10个epoch衰减10倍
    for epoch in range(epochs):
        model.train() # 设置模型为训练模式
        total_loss = 0 # 记录总损失
        pbar = tqdm(range(num_samples_per_epoch), desc=f'Epoch {epoch + 1}/{epochs}, LR: {scheduler.get_last_lr()[0]}')
        for _ in pbar:
            input_seq, target = generate_single_sample(all_data) # 生成单个样本
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device) # 转换为张量并添加批次维度
            target = torch.tensor(np.argmax(target), dtype=torch.long).unsqueeze(0).to(device) # 转换为张量并添加批次维度
            optimizer.zero_grad() # 清空梯度
            output = model(input_seq) # 前向传播
            loss = criterion(output, target) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_loss += loss.item() # 累加损失
            pbar.set_postfix({'Loss': total_loss / (pbar.n + 1)}) # 更新进度条
        scheduler.step() # 更新学习率
        # 评估模型
        model.eval()
        correct = 0 # 记录正确预测数量
        total = 0 # 记录总预测数量
        for _ in range(100): # 评估100个样本
            input_seq, target = generate_single_sample(all_data)
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            target = np.argmax(target)
            output = model(input_seq)
            prediction = torch.argmax(output, dim=1).item()
            if prediction == target:
                correct += 1
            total += 1
        accuracy = correct / total # 计算准确率
        print(f'Epoch {epoch + 1} evaluation accuracy: {accuracy * 100:.2f}%')


if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():# 检查设备
        print("CUDA 不可用，将使用 CPU 进行训练，训练速度可能较慢。")
    
    all_data = load_and_preprocess_data('./data')# 加载数据（返回列表形式：每只股票独立存储）
    
    # 模型参数调整
    d_model = 256 # 模型维度
    epochs = 10 # 训练轮数
    input_dim = 8  # 原始数据包含8个特征:输入特征维度
    nhead = 8 # 注意力头数
    num_layers = 4 # 编码器层数
    output_dim = 3 # 输出维度（3个分类）
    
    model = StockTransformer(input_dim, d_model, nhead, num_layers, output_dim).to(device)
    
    # 训练参数
    num_samples_per_epoch = 1000 # 每个epoch训练样本数
    learning_rate = 0.0001
    
    # 训练模型
    train_model(model, all_data, epochs, learning_rate, device, num_samples_per_epoch)
    
    # 保存模型
    torch.save(model.state_dict(), './out/EquiNet'+ d_model +'.pth')