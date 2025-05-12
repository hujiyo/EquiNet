# EquiNet

## 项目背景

在金融量化领域，如何利用历史数据预测股票未来走势一直是核心难题。传统方法往往难以捕捉复杂的时序与市场联动特征。EquiNet 基于 Transformer 架构，结合多只股票与大盘数据，致力于专注提升未来3天涨跌概率预测的准确性，本项为学习偏教程项目，任何人可以随意使用。

## 项目简介

 - EquiNet 旨在从0开始构建一个深度学习模型，通过对319只A股股票420天的历史数据进行建模，仅仅使用不到半小时的时间，即可训练出一个具备一定预测能力的模型EquiNet
 - EquiNet 基于 PyTorch 的深度学习项目，支持灵活参数配置和高效GPU训练。通过历史数据建模输出未来3天内上涨、下跌、平稳的概率分布。

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
│   ├── train_mark.py    # 使用积分制评估模型的训练脚本
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
- 数据来源：通达信

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

### 快速开始
1. **克隆项目**
   ```bash
   git clone https://gitee.com/hujiyo/EquiNet.git
   ```

2. **运行训练脚本**
   ```bash
   python src/train.py
   ```

3. **模型评估**
   - 训练完成后，模型权重保存在`./out/EquiNet{d_model}.pth`。
   - 可使用`src/test_best_model.py`对训练集、测试集、全集进行准确率评估：
     ```bash
     python src/test_best_model.py
     ```

### 常规训练流程
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

### 模型输出
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

## 实验结果

- 详见`src/test-result.CSV`。
- 主要指标为不同参数下的训练集、测试集、全集准确率，便于横向对比模型表现。
- 为了更好地评估模型的实际能力，最新版本提供积分制评估模型的训练脚本`src/train_mark.py`。

## 常见问题 FAQ

- **Q: 数据文件格式有要求吗？**
  - A: 需为Excel `.xlsx` 格式，字段顺序与上述一致，且每只股票提供地数据量要大于60天。
- **Q: 如何调整模型大小？**
  - A: 直接修改`src/trainXXX.py`中的`d_model`参数即可.
- **Q: 支持多GPU吗？**
  - A: 当前版本默认单GPU，可自行扩展`DataParallel`等方式。
- **Q: 训练慢怎么办？**
  - A: 建议使用CUDA GPU，或减少模型参数/训练轮数。
- **Q: 模型的参数如何选择？**
  - A: 目前正在尝试不同参数组合，具体的测试数据也会放在`src/test-result.xlsx`中。

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

### 四、验证指标优化
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

### 五、dropout层策略防止过拟合
```python
self.embedding = nn.Sequential(
    nn.Linear(input_dim, d_model),
    nn.Dropout(0.3)
)
```

### 六、增加位置编码
```python
self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
def forward(self, x):
    seq_len = x.size(1)
    x = self.embedding(x) + self.positional_encoding[:, :seq_len]
    ...
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

## 新增解决方案

### 矛盾点一：
- 问题描述：
  - 当前的评估机制是一个自定义评分系统 ，它对正确预测加分；对错误预测（尤其是代价高的错误）扣分。然而这种评分规则并未反映在训练损失函数中 ，因此模型在训练时并不知道“哪些错误更严重”。简单来说，这个测评程序其实本质上就是一个马后炮，它是独立于模型训练这个过程的，我最初的想法是如何让模型与这个惩罚机制结合。
- 目标：
  - 将惩罚机制融入模型训练中，希望模型在训练时就知道：把下跌预测为上涨比把上涨预测为下跌更严重；把横盘预测为上涨不如预测为下跌好；某些误判需要被特别惩罚。
- 解决方案：
  - **引入代价矩阵**
   ```python
   # 定义代价矩阵
   cost_matrix = torch.tensor([[0, 1, 1, 1],  # 上涨
                               [1, 0, 1, 1],  # 下跌
                               [1, 1, 0, 1],  # 平稳
                               [1, 1, 1, 0]]) # 强下跌
   # 计算代价
   cost = cost_matrix[targets, predictions]
   ```


## 参与贡献

1. Fork 本仓库
2. 新建分支（如 `dev_yourname`）
3. 提交代码
4. 新建 Pull Request

## 联系方式

- 邮箱: hj18914255909@outlook.com

## 致谢

1. 感谢通达信提供的股票数据（虽然我是买了初级会员才获得的数据的~QvQ~）。
2. 项目当前以及未来的所有代码均会是在QwQ-32b、豆包、deepseek、ChatGPT等等大佬的帮助下完成的，在此表示感谢。
最初对QwQ的prompt：
```
背景介绍：现在你在进行一个利用python进行模型训练的2030年编程比赛，参赛者有中国昔日之光：deepseek-r1:671b满血版、openAI最新编程大神：ChatGPT6.0:9600b世界版、千问推理模型QwQ-32b（你）......注意，你可以无限长时间循环分步骤思考，但是你的机会只有一次！
现在是试卷的最后一题：在./下写一个.py模型训练程序，模型数据存放在./data下，那里有着319个.xlsx文件，每个文件对应一只股票，每个文件的有效行数为421行，其中第一行也就是表头字段分别有time	start	max	min	end	volume	marketvolume	marketlimit	marketrange这9个字段，2-421行都是数据，也就是说每个文件实际上都是包含一只股票420天的基本情况和当天大盘的情况。每个文件的time都是420天并且初始日期都是对照相同的，start/end	是开/收盘价，max/min是最高/低价，volume是股票量能，marketvolume是市场量能，marketlimit是大盘涨跌幅，marketrange是大盘指数的波动宽度，（比如marketlimit为-1%，marketrange为50，则暗示大盘下跌30个点，宽度在50个点，波动很大）	模型的输出结果是对未来3天的情况进行概率预测，分别是上涨3%的概率，下跌2%的概率，保持-2%~3%的概率，例如输出：涨5%：\n跌3%：54%，\n稳：25%。
提示：
1.建议使用transformer架构，程序开始需要提示用户输入想要的模型训练的大小参数和训练轮次。为避免用户不懂，你还要有提示信息（比如对最终模型效果的影响、对模型最终参数量的影响等等）
2.训练过程中，使用随机上下文长度（20天-100天），这种随机效果可以变相降低数据量有限的弊端，然后对下面3天进行预测
3.训练过程中要给出进度信息（包括学习率），每轮训练结束后要增加一个效果环节，具体如下：用相同的方法从数据中随机获得片段作为输入然后，将概率大的视为模型的选择，循环多次即可计算出当前模型预测成功的概率并打印。
4.不同股票的价格不同，模型的目标是掌握其中更深层次的规律，所以训练数据要进行统一的归一化
5.使用支持CUDA的gpu进行训练
6.time字段的格式为'2023/06/27'
7.不要提前确定好所有的数据然后轮流开始训练，这违背了我随机思想的初衷，我要的是训练输入上的完全随机，每轮1000组随机输入
```

