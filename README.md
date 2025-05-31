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
- **常规化预测**：只关注平常的规律，不关注市场极端状态时的规律。

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
1. **正负样本数量极不平衡问题**：
   ```笔记
   Focal Loss损失函数介绍：Focal Loss的引入主要是为了解决one-stage目标检测中正负样本数量极不平衡问题。
   比如：在一张图像中能够匹配到目标的候选框（正样本）个数一般只有十几个或几十个，而没有匹配到的候选框
   （负样本）则有10000~100000个。这么多的负样本不仅对训练网络起不到什么作用，反而会淹没掉少量但有助于训练的样本。
   对于股票来说，样本不平衡问题是常态。
   ```

2. **引入代价矩阵**：
    目标：解决训练目标与评估目标不一致的问题
    加权交叉熵：给"下跌→上涨"这种错误分配更高的权重，但这种方法不如自定义损失函数精确


    ```python
    # 定义代价矩阵
    cost_matrix = torch.tensor([[0, 1, 1, 1],  # 上涨
                                [1, 0, 1, 1],  # 下跌
                                [1, 1, 0, 1],  # 平稳
                                [1, 1, 1, 0]]) # 强下跌
    # 计算代价
    cost = cost_matrix[targets, predictions]
    ```
    - 或者设计损失函数
    核心思路：让训练损失函数直接反映你的评分规则，这样模型在训练过程中就知道"哪些错误更致命"。
    具体做法是替换标准的CrossEntropyLoss，设计一个非对称惩罚损失函数：
    python# 伪代码示意
    class AsymmetricPenaltyLoss:
        def forward(self, predictions, targets):
            if prediction == target:
                loss = 0  # 正确预测无损失
            elif target == 上涨 and prediction == 下跌:
                loss = 1.0  # 轻度惩罚
            elif target == 下跌 and prediction == 上涨:
                loss = 2.0  # 重度惩罚（这种错误在投资中损失最大）
            else:
                loss = 0.5  # 震荡预测错误，中等惩罚
    



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



### 关键平衡点控制：
1. **阈值校准器**：
   ```python
   # 根据市场波动率动态调整阈值：
   dynamic_threshold = base_threshold * (1 + market_volatility_factor)
   ```



3. **置信度衰减函数**：
   ```python
   # 对长期横盘股票降低上涨预测置信度：
   confidence_decay = 1 / (1 + days_since_last_breakout)
   ```

这种方法体系既保持了模型对真实上涨机会的识别能力，又通过多层过滤机制防止在下跌行情中的误判。建议采用A/B测试框架，逐步验证每个改进模块的效果。  

## 项目修改LOG

- 2025.5.31:积分制成为默认机制，增加时间感知位置编码、Focal Loss损失函数、结合标准正弦余弦位置编码、指数衰减机制、种类差异化多头注意力机制、多尺度注意力，加入了残差连接和层归一化。
- 2025.5.12:增加mark积分制判别最优模型,但保留原判别机制
- 2025.5.1:项目start ~

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

## 术语解释（持续更新）

- `残差连接`: 简单理解就是把输入直接"跳过"一些层加到输出上，对解决`梯度消失`问题有帮助，让训练更稳定（即使某一层学坏了，至少原始信息还能通过"跳跃连接"传递下去），确保重要信息不易丢失，更容易优化（网络可以学习"在原有基础上做什么改进"，而不是"从零开始学一切"）

- `层归一化`: 简单理解就是把每一层的输出"标准化"，让数值分布更稳定（防止某些神经元输出过大或过小），加速收敛（让梯度更平滑，模型收敛更快？）

- `Focal Loss`:一种解决one-stage目标检测中正负样本数量极不平衡问题的损失函数，比如：股票中上涨的样本远远小于稳定的样本，这个时候稳定的样本的规律信息就可能会淹没上涨样本提供的信息，对于股票来说，样本不平衡问题是常态。

