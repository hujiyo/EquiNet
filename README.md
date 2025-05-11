# EquiNet

## 项目背景

在金融量化领域，如何利用历史数据预测股票未来走势一直是核心难题。传统方法往往难以捕捉复杂的时序与市场联动特征。EquiNet 基于 Transformer 架构，结合多只股票与大盘数据，致力于提升未来3天涨跌概率预测的准确性，助力量化投资与学术研究。

## 项目简介

EquiNet 是一个基于 PyTorch 的深度学习项目，支持灵活参数配置和高效GPU训练。通过对319只股票420天的历史数据建模，输出未来3天内上涨、下跌、平稳的概率分布。

## 主要特性

- **Transformer 架构**：强大的时序建模能力，适合金融数据。
- **灵活参数配置**：支持自定义模型维度、层数、训练轮数等，适应不同硬件与需求。
- **完全随机采样**：每轮训练/评估均从全体股票数据中随机采样，提升泛化能力。
- **归一化处理**：每只股票独立归一化，消除价格量级影响。
- **GPU加速**：自动检测CUDA，充分利用GPU资源。
- **训练过程评估**：每轮训练后自动评估模型当前预测准确率。
- **断点续训**：自动保存最佳模型权重，便于后续加载与继续训练。

## 目录结构

```
EquiNet/
├── data/                # 存放319个.xlsx股票数据文件
├── out/                 # 训练输出的模型权重
├── src/
│   ├── train.py         # 主训练脚本
│   ├── test_best_model.py # 模型评估脚本
│   └── ...              # 其他源码
├── README.md
└── ...
```

## 数据格式说明

- 数据目录：`./data`
- 每个`.xlsx`文件对应一只股票，包含420天数据，共319只股票。
- 字段说明（共9列）：
  - `time`：日期（如`2023/06/27`）
  - `start`：开盘价
  - `max`：最高价
  - `min`：最低价
  - `end`：收盘价
  - `volume`：股票成交量
  - `marketvolume`：市场总成交量
  - `marketlimit`：大盘涨跌幅（如-1%）
  - `marketrange`：大盘指数波动宽度

## 安装与环境

- Python 3.8+
- PyTorch
- pandas、numpy
- tqdm

安装依赖：
```bash
pip install torch pandas numpy tqdm
```

## 使用流程

1. **准备数据**
   - 将319个`.xlsx`股票数据文件放入`./data`目录下。

2. **运行训练脚本**
   ```bash
   python src/train.py
   ```
   - 按提示输入模型参数（如模型维度、训练轮数等），程序会说明参数对模型效果和参数量的影响。
   - 训练过程中显示进度、损失、学习率及每轮评估准确率。

3. **模型评估**
   - 训练完成后，模型权重保存在`./out/EquiNet{d_model}.pth`。
   - 可使用`src/test_best_model.py`对训练集、测试集、全集进行准确率评估：
     ```bash
     python src/test_best_model.py
     ```

4. **模型输出说明**
   - 输出为对未来3天的概率预测：
     - 上涨3%及以上的概率
     - 下跌2%及以下的概率
     - 保持在-2%~3%区间的概率
   - 示例输出：
     ```
     涨：21%
     跌：54%
     稳：25%
     ```

## 实验结果展示

- 支持多组参数实验，详见`src/test-result.CSV`。
- 主要指标为不同参数下的训练集、测试集、全集准确率，便于横向对比模型表现。

## 常见问题 FAQ

- **Q: 数据文件格式有要求吗？**
  - A: 需为Excel `.xlsx` 格式，字段顺序与上述一致，且每只股票420天数据。
- **Q: 如何调整模型大小？**
  - A: 运行时输入提示参数即可，模型维度越大，效果和参数量越高，但训练更慢。
- **Q: 支持多GPU吗？**
  - A: 当前版本默认单GPU，可自行扩展`DataParallel`等方式。
- **Q: 训练慢怎么办？**
  - A: 建议使用CUDA GPU，或减少模型参数/训练轮数。

## 当前方向

目标是增强其对股票上涨的敏感度，如果股票大概率下跌，那他就不准出现预测上涨，最多也只准预测为“平稳”，但是又担心这会抑制其预测上涨的能力，对于极高可能出现上涨的股票，应当预测上涨而非下跌......

即既要提升对上涨的敏感度又要避免误判下跌为上涨，可以采用以下分阶段解决方案：

### 一、类别重构与损失函数改进
1. **不对称类别重构**：
   ```python
   # 将原始三分类改为四分类：
   # 0: 强下跌 (cumulative_return <= -0.03)
   # 1: 弱下跌 (-0.03 < cumulative_return <= -0.01)
   # 2: 平稳 (-0.01 < cumulative_return <= 0.01)
   # 3: 上涨 (cumulative_return > 0.01)
   ```
2. **自定义损失函数**：
   ```python
   class AsymmetricFocalLoss(nn.Module):
       def __init__(self, gamma=2, alpha=None):
           super().__init__()
           self.gamma = gamma
           self.alpha = alpha  # 上调上涨类的alpha值(如1.5)，降低下跌类的alpha(如0.5)
           
       def forward(self, inputs, targets):
           CE_loss = F.cross_entropy(inputs, targets, reduction='none')
           pt = torch.exp(-CE_loss)
           # 对上涨类降低惩罚力度，对下跌类增加惩罚
           at = self.alpha.gather(0, targets.data)  
           loss = at * (1 - pt)**self.gamma * CE_loss
           return loss.mean()
   ```

### 二、置信度门控机制
1. **动态分类阈值**：
   ```python
   def dynamic_threshold_predict(output, threshold_dict=None):
       if threshold_dict is None:
           threshold_dict = {'up': 0.6, 'down': 0.7}  # 默认阈值
       probs = F.softmax(output, dim=1)
       result = torch.zeros_like(probs[:,0], dtype=torch.long)
       # 上涨条件：上涨概率超过阈值且高于下跌概率2倍
       up_mask = (probs[:,0] > threshold_dict['up']) & (probs[:,0] > probs[:,1]*2)
       # 下跌条件：下跌概率超过阈值且高于上涨概率3倍
       down_mask = (probs[:,1] > threshold_dict['down']) & (probs[:,1] > probs[:,0]*3)
       result[up_mask] = 0  # 上涨
       result[down_mask] = 1  # 下跌
       # 其余情况标记为平稳
       return result
   ```

### 三、多任务学习框架
```python
class EnhancedStockTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = ...  # 主干网络
        # 主任务：趋势预测
        self.trend_head = nn.Sequential(...)
        # 辅助任务1：波动率预测
        self.volatility_head = nn.Sequential(...)
        # 辅助任务2：动量强度预测
        self.momentum_head = nn.Sequential(...)

    def forward(self, x):
        features = self.transformer(self.embedding(x))
        trend = self.trend_head(features)
        volatility = self.volatility_head(features)
        momentum = self.momentum_head(features)
        return trend, volatility, momentum
```

### 四、对抗性训练策略
1. **样本加权策略**：
   ```python
   def generate_weighted_sample(all_data):
       # 对上涨样本增加重复采样权重
       # 对下跌样本增加噪声扰动
       # 对平稳样本进行时间弹性变形
       ...
   ```

2. **渐进式难度训练**：
   ```python
   # 第一阶段：只训练平稳 vs 非平稳
   # 第二阶段：训练平稳 vs 上涨+下跌（冻结底层参数）
   # 第三阶段：联合训练所有类别
   ```

### 五、验证指标优化
```python
# 新增关键指标监控：
def calculate_metrics(outputs, targets):
    metrics = {}
    # 上涨召回率：预测为上涨的样本中实际上涨的比例
    metrics['up_recall'] = true_positive_up / (actual_up + 1e-8)  
    # 下跌误判率：实际下跌但预测为上涨的比例
    metrics['down_misjudge'] = down_to_up / (actual_down + 1e-8)
    # 极端上涨捕获率：top 5%涨幅样本的预测准确率
    metrics['extreme_up_capture'] = extreme_up_correct / (extreme_up_total + 1e-8)
    return metrics
```

### 实施路线图建议：
1. **第一阶段（1-2周）**：
   - 实现类别重构和动态阈值机制
   - 添加波动率辅助任务
   - 调整损失函数权重（上涨类权重x1.5）

2. **第二阶段（3-4周）**：
   - 引入对抗样本生成
   - 实施渐进式训练策略
   - 优化验证指标体系

3. **第三阶段（5-6周）**：
   - 添加多头注意力可视化模块
   - 实现基于市场状态的动态阈值调整
   - 开发回测验证系统

### 关键平衡点控制：
1. **阈值校准器**：
   ```python
   # 根据市场波动率动态调整阈值：
   dynamic_threshold = base_threshold * (1 + market_volatility_factor)
   ```

2. **安全边界机制**：
   ```python
   # 当检测到市场处于极端下跌状态时：
   if market_condition == 'extreme_down':
       threshold_dict['up'] *= 1.2  # 提高上涨判定门槛
       threshold_dict['down'] *= 0.8  # 降低下跌判定门槛
   ```

3. **置信度衰减函数**：
   ```python
   # 对长期横盘股票降低上涨预测置信度：
   confidence_decay = 1 / (1 + days_since_last_breakout)
   ```

这种方法体系既保持了模型对真实上涨机会的识别能力，又通过多层过滤机制防止在下跌行情中的误判。建议采用A/B测试框架，逐步验证每个改进模块的效果。

## 参与贡献

1. Fork 本仓库
2. 新建分支（如 `feat_xxx`）
3. 提交代码
4. 新建 Pull Request

## 联系方式

- Issues: [Gitee Issues](https://gitee.com/yourname/EquiNet/issues)
- 邮箱: your_email@example.com

## 致谢

感谢所有开源社区贡献者及金融数据支持者。更多开源项目可参考 [Gitee Explore](https://gitee.com/explore)。
