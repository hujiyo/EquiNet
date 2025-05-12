'''
训练脚本---
使用得分制来评估模型的性能
具体如下：
提供1000次预测机会，预测正确加一分
预测错误则按下面策略处理：
1.上涨的股票预测为下跌：-1分 
2.下跌的股票预测为上涨：-2分 
3.其余情况不加分也不扣分。
'''

import os,torch,torch.nn as nn,torch.optim as optim,pandas as pd,numpy as np
from tqdm import tqdm
import random

# 定义Transformer模型
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(StockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, output_dim)
        self.softmax = nn.Softmax(dim=1)  # 新增softmax层
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        x = self.softmax(x)  # 输出概率分布
        return x

# 数据预处理
def load_and_preprocess_data(data_dir, test_ratio=0.1, seed=42):
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    random.seed(seed)
    test_size = max(1, int(len(all_files) * test_ratio))
    test_files = set(random.sample(all_files, test_size))
    train_files = [f for f in all_files if f not in test_files]

    def process_files(file_list):
        data_list = []
        for file in file_list:
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path)
            try:
                data = df[['start', 'max', 'min', 'end', 'volume', 'marketvolume', 'marketlimit', 'marketrange']].values
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

# 生成单个样本
def generate_single_sample(all_data):
    for _ in range(100):
        stock_index = np.random.randint(0, len(all_data))
        stock_data = all_data[stock_index]
        context_length = 60
        required_length = context_length + 3
        if len(stock_data) < required_length:
            continue
        start_index = np.random.randint(0, len(stock_data) - required_length + 1)
        input_seq = stock_data[start_index:start_index + context_length]
        target_seq = stock_data[start_index + context_length:start_index + required_length]
        start_price = input_seq[-1, 3]
        end_price = target_seq[-1, 3]
        if start_price == 0:
            continue
        cumulative_return = (end_price - start_price) / start_price
        if cumulative_return >= 0.03:
            target = [1, 0, 0]
        elif cumulative_return <= -0.02:
            target = [0, 1, 0]
        else:
            target = [0, 0, 1]
        return input_seq, target
    raise ValueError("无法生成有效样本：股票数据长度不足或收盘价为0")

# 训练模型
def train_model(model, train_data, test_data, epochs, learning_rate, device, num_samples_per_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(range(num_samples_per_epoch), desc=f'Epoch {epoch + 1}/{epochs}, LR: {scheduler.get_last_lr()[0]}')
        for _ in pbar:
            input_seq, target = generate_single_sample(train_data)
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            target = torch.tensor(np.argmax(target), dtype=torch.long).unsqueeze(0).to(device)
            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'Loss': total_loss / (pbar.n + 1)})
        scheduler.step()
        # 评估模型（得分制）
        model.eval()
        score = 0
        total = 0
        for _ in range(1000):
            input_seq, target = generate_single_sample(test_data)
            input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            target_idx = np.argmax(target)
            output = model(input_seq)
            prediction = torch.argmax(output, dim=1).item()
            # 评分规则
            if prediction == target_idx:
                score += 1
            elif target_idx == 0 and prediction == 1:
                score -= 1  # 上涨预测为下跌
            elif target_idx == 1 and prediction == 0:
                score -= 2  # 下跌预测为上涨
            # 其余情况不加分也不扣分
            total += 1
        print(f'Epoch {epoch + 1} evaluation score: {score} (out of {total})')
        if score > best_accuracy:
            best_accuracy = score
            torch.save(model.state_dict(), './out/EquiNet_best.pth')

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA 不可用，将使用 CPU 进行训练，训练速度可能较慢。")

    train_data, test_data = load_and_preprocess_data('./data')

    d_model = 128
    input_dim = 8
    nhead = 8
    num_layers = 4
    output_dim = 3

    epochs = 80
    num_samples_per_epoch = 1000
    learning_rate = 0.001

    model = StockTransformer(input_dim, d_model, nhead, num_layers, output_dim).to(device)

    train_model(model, train_data, test_data, epochs, learning_rate, device, num_samples_per_epoch)

    torch.save(model.state_dict(), './out/EquiNet' + str(d_model) + '.pth')