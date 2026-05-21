"""
EquiNet 模型配置文件
"""

import os
import sys
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # SQLite 数据库配置
    DB_PATH = os.path.join(PROJECT_ROOT, 'data_maintenance', 'equinet.db')
    DB_BACKUP_DIR = os.path.join(PROJECT_ROOT, 'data_maintenance', 'backup')

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')

    # 数据源配置
    DATA_SOURCE = 'akshare'  # 'baostock' 或 'akshare'
    MARKET_CAP_MAX = 200e8  # 市值上限（元），200亿
    MARKET_CAP_MIN = 10e8   # 市值下限（元），10亿
    VALID_STOCK_PREFIXES = ['600', '601', '603', '605', '000', '001', '002', '003']  # 主板股票代码前缀

    # 数据分割参数（按时间划分）
    TRAIN_START_DATE = 20160101      # 训练集起始日期（含）
    TRAIN_END_DATE = 20241231        # 训练集截止日期（含）
    VAL_START_DATE = 20250101        # 验证集起始日期（含），用于训练时模型选择
    VAL_END_DATE = 20251231          # 验证集截止日期（含）
    TEST_START_DATE = 20260101       # 测试集起始日期（含），截止日期为数据库最新日期，训练结束后仅评估一次
    RANDOM_SEED = 42                 # 随机种子
    
    # 样本生成参数
    CONTEXT_LENGTH = 45              # 历史数据长度（这是核心参数，其他地方应引用这个值）
    FUTURE_DAYS = 3                  # 未来预测天数
    BUFFER_DAY = True                # 额外采集1天安全余量（用于跌停推迟判断）
    REQUIRED_LENGTH = CONTEXT_LENGTH + FUTURE_DAYS + (1 if BUFFER_DAY else 0)

    # 量能/换手率归一化配置
    MA_WINDOW = 10                   # 量能与换手率相对均值的滑动窗口大小（N日均值基准）

    # 涨跌停配置
    LIMIT_THRESHOLD = 0.095          # 涨跌停判断阈值（9.5%，覆盖普通板±10%）
    LIMIT_CHECK_MODE = 'ohlc'        # 'simple': 仅看涨跌幅 | 'ohlc': 通过OHLC判断是否开板

    # 采样策略配置
    # 'temporal': 时间顺序采样（指针在训练集上循环滑动）
    # 'random': 随机采样（每次随机选择股票和位置）
    SAMPLING_STRATEGY = 'random'

    # 是否过滤上下文最后一天接近涨停的样本
    # True: 过滤最后一天涨停的样本（防止模型过度依赖涨停信号，避免追涨策略）
    # False: 保留所有样本（让模型自己学习涨停后的走势规律）
    FILTER_CONTEXT_LAST_DAY_LIMIT_UP = True

    # 正样本距离保护参数
    # 如果位置i是正样本，则i-LABEL_DISTANCE到i-1的负样本不参与训练（排除）
    # distance=0时不排除任何样本（等价于原始行为）
    # 目的：消除正样本左侧特征高度重叠但标签相反的矛盾训练信号
    LABEL_DISTANCE = 3

    # Day1标签基准选择
    # True: day1使用开盘到收盘的日内涨幅 (close[T+1]-open[T+1])/open[T+1]，对齐实际买入价
    #       消除跳空缺口对标签的干扰，标签只反映投资者能赚到的部分
    # False: day1使用收盘到收盘的涨跌幅 (close[T+1]-close[T])/close[T]，包含隔夜跳空
    LABEL_DAY1_USE_OPEN = True

    # 评估参数
    EVAL_BATCH_SIZE = 4096            # 评估批处理大小（分批处理，减少显存占用）

    # ========== 特征归一化配置 ==========
    # 使用 QuantileTransformer + StandardScaler 进行高级特征归一化
    # 优点：
    #   1. 统一所有特征到均值0、标准差1的分布
    #   2. 自动处理特征范围不同（Amount/Exchange vs OHLC）
    #   3. 自动处理特征集中度不同（Amount 99%集中在小范围）
    #   4. 自动处理异常值和偏态分布

    # 归一化器配置
    NORMALIZER_OUTPUT_DISTRIBUTION = 'normal'  # 'normal' (标准正态) 或 'uniform' (均匀分布)
    NORMALIZER_N_QUANTILES = 1000            # 分位数数量（越大越精确但越慢）
    NORMALIZER_PATH = os.path.join(SRC_DIR, 'normalizer.pkl')

    TOP_K = 1                   # 排序收益评估的百分比（取预测概率前N%的样本）
    TOP_N_PER_DAY = 0                 # 实战收益率：每天选股数量（0表示使用全局阈值模式）
    MAX_SELECT_PER_DAY = 4             # 全局阈值模式下每天最多选股数量（0表示不限制）
    MAX_HOLDINGS = 4                  # 最大并发持仓数（由用户实际资金体量决定）
    MAX_BUY_PER_DAY = 0                # 每天最多买入数量（0表示不限制，填满所有空位）

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""
    # 基础模型参数
    INPUT_DIM = 10                   # 输入特征维度数（OHLC + vwap + volume + exchange + m5 + m10 + m20）
    D_MODEL = 128                    # 模型维度（Transformer 内部维度）
    FFN_EXPAND_RATIO = 4             # FFN 隐藏层扩展比例（hidden_dim = d_model * FFN_EXPAND_RATIO）
    NHEAD = 4                        # 注意力头数
    NUM_LAYERS = 6                   # Transformer 层数
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1 之间）

    # 注意力机制参数
    DROPOUT_RATE = 0                 # Dropout比率设置为0降低欠拟合
    ATTENTION_DROPOUT = 0            # 注意力Dropout比率设置为0降低欠拟合

    # ========== 初始化策略 ==========
    # 与 LLaMA/MiniMind 一致，全部使用 PyTorch 默认初始化
    # - nn.Linear: kaiming_uniform_(a=sqrt(5))，等效 gain≈0.58
    # - nn.Embedding: 正态分布 N(0,1)
    # - 所有层 bias=False（LLaMA/MiniMind/DeepSeek 均不使用 bias）

    # 输出层参数（当代最佳实践：避免sigmoid饱和）
    # - 输出层使用sigmoid，如果logits范围太大会导致饱和、梯度消失
    # - 目标值在[0,1]范围，初始输出应接近先验概率
    # - gain=0.1: 很小范围 (±0.06), 让初始预测logits接近0
    # - bias初始化为 log(prior/(1-prior))，prior≈0.25 → bias≈-1.1
    OUTPUT_LAYER_GAIN = 0.1          # 输出层权重初始化增益（xavier_uniform的gain参数）

# ==================== Embedding预训练参数 ====================
class EmbeddingConfig:
    """Embedding层参数（src/pretrain_embedding.py 使用）"""
    # 架构（必须与 ModelConfig 一致）
    INPUT_DIM = ModelConfig.INPUT_DIM     # 10
    D_MODEL = ModelConfig.D_MODEL         # 128

    # 训练超参数
    EPOCHS = 100                          # 预训练轮数
    BATCH_SIZE = 2560                     # 大batch，对比学习需要充足负样本
    LEARNING_RATE = 3e-3                  # 预训练学习率
    WEIGHT_DECAY = 1e-4                   # 权重衰减
    WARMUP_EPOCHS = 5                    # 预热轮数
    COSINE_ETA_MIN = 3e-5                 # 余弦退火最小学习率

    # 损失权重
    BETA = 1.0                            # 重建损失 (MSE) 权重

    # 数据采集
    MAX_SAMPLES = 200_000               # 每个epoch的训练样本数
    DEDUP_PRECISION = 3                 # 去重时特征量化精度（小数位数）

    # SIGReg 几何正则 (Balestriero & LeCun, 2025)
    # 约束嵌入分布趋向各向同性高斯 N(0, target_std²)
    SIGREG_WEIGHT = 10.0                # SIGReg 损失权重
    SIGREG_NUM_SLICES = 32              # 随机投影方向数
    SIGREG_T_MAX = 3                    # Epps-Pulley 积分上限
    SIGREG_N_POINTS = 17                # Epps-Pulley 积分节点数（奇数）
    TARGET_STD = 0.2                    # 目标标准差（缩放后 SIGReg 检验 N(0,1)）

    # 训练稳定性
    GRADIENT_CLIP_NORM = 1.0              # 梯度裁剪范数

    # 输出
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out', 'embedding_pretrain')
    BEST_EMBEDDING_PATH = os.path.join(OUTPUT_DIR, 'best_embedding.pth')

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""
    # 基础训练参数（优化训练策略）
    EPOCHS = 80                     # 训练轮数
    WARMUP_EPOCHS = 8               # 预热轮数
    COSINE_ANNEAL_EPOCHS = 14       # 余弦退火轮数（之后学习率固定在ETA_MIN）

    # 余弦退火调度器参数,学习率预热参数
    COSINE_ETA_MIN = 1e-4          # 余弦退火最小学习率 / 固定阶段学习率
    WARMUP_START_LR = 1e-4           # 预热起始学习率（提高起始值，减少过于保守的预热）

    LEARNING_RATE = 0.001            # AdamW/Adam基础学习率
    WEIGHT_DECAY = 1e-5              # AdamW/Adam权重衰减

    # 训练批处理
    BATCH_SIZE = 512                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 240            # 每轮训练的批次数（调低以适配时间序采样）

    # 优化器选择（字符串，互斥）
    # 'adamw':    标准AdamW
    # 'lion':     Lion（符号动量），内存省、泛化好（lion-pytorch库）
    # 'muon':     Muon（Newton-Schulz正交化），收敛快（KellerJordan/Muon库）
    # 'mano':     Mano混合优化器
    OPTIMIZER_TYPE = 'mano'

    # 自动混合精度（AMP）
    USE_AMP = True                   # 启用BF16自动混合精度（矩阵乘法用BF16，归一化/激活/损失保持FP32）
    # 训练和推理均受此开关控制，sigmoid始终在FP32下执行，不影响选股排名精度

    # 通用优化器参数
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # AdamW 参数（OPTIMIZER_TYPE='adamw'时生效）
    ADAMW_LR = 0.001                 # AdamW 学习率
    ADAMW_WEIGHT_DECAY = 1e-5        # AdamW 权重衰减

    # Lion 参数（OPTIMIZER_TYPE='lion'时生效）
    LION_LR = 0.0003                # Lion 学习率（AdamW的~1/3，论文推荐1/3~1/10）
    LION_WEIGHT_DECAY = 0.01        # Lion 权重衰减（比AdamW大约1000倍，论文推荐1e-2量级）
    LION_BETAS = (0.9, 0.99)        # Lion 动量系数

    # Mano 参数（OPTIMIZER_TYPE='mano'时生效）
    MANO_LR = 0.001                 # Mano 学习率
    MANO_WEIGHT_DECAY = 1e-5        # Mano 权重衰减
    MANO_MOMENTUM = 0.95             # Mano动量系数
    MANO_ADAMW_BETAS = (0.9, 0.95)   # 混合优化器中AdamW部分的beta参数
    MANO_NESTEROV = True             # 是否使用Nesterov动量（v2默认True）
    MANO_DUAL_DIM_PROJECTION = True  # 是否使用双维度投影（v2新功能，默认True）

    # Muon 参数（OPTIMIZER_TYPE='muon'时生效）
    # Muon通过Newton-Schulz迭代对2D权重矩阵的梯度动量进行正交化，加速收敛
    # 2D权重(Linear/Conv)走Muon，1D参数(bias/LayerNorm)走AdamW
    # 使用KellerJordan/Muon官方库的SingleDeviceMuonWithAuxAdam
    # 参考: https://github.com/KellerJordan/Muon
    MUON_LR = 0.02                  # Muon 学习率（官方默认0.02，示例用0.05）
    MUON_WEIGHT_DECAY = 0           # Muon 权重衰减（官方默认0，正交化本身已提供正则化）
    MUON_MOMENTUM = 0.95            # Muon动量系数（官方默认0.95）
    MUON_ADAMW_LR_RATIO = 0.8       # AdamW部分学习率 = MUON_LR × 此比例（官方示例: scalar lr=0.04 / muon lr=0.05）
    MUON_ADAMW_BETAS = (0.8, 0.95)  # AdamW部分的beta参数（官方示例: 0.8, 0.95）

    OPEN_EARLY_STOPPING = False       # 是否开启早停机制

    @staticmethod
    def get_base_lr():
        """返回当前优化器对应的默认学习率"""
        opt = TrainingConfig.OPTIMIZER_TYPE.lower()
        if opt.startswith('lion'):
            return TrainingConfig.LION_LR
        if opt == 'mano':
            return TrainingConfig.MANO_LR
        if opt == 'muon':
            return TrainingConfig.MUON_LR
        return TrainingConfig.ADAMW_LR

    @staticmethod
    def get_base_wd():
        """返回当前优化器对应的默认权重衰减"""
        opt = TrainingConfig.OPTIMIZER_TYPE.lower()
        if opt.startswith('lion'):
            return TrainingConfig.LION_WEIGHT_DECAY
        if opt == 'mano':
            return TrainingConfig.MANO_WEIGHT_DECAY
        if opt == 'muon':
            return TrainingConfig.MUON_WEIGHT_DECAY
        return TrainingConfig.ADAMW_WEIGHT_DECAY

# ==================== 损失函数配置 ====================
class LossConfig:
    """损失函数相关配置"""
    # 'dynamic_bce':批权重动态平衡 | 'pairwise_bce':BCE+Pairwise排序 | 'standard_bce':标准二元交叉熵
    LOSS_TYPE = 'dynamic_bce'

    POS_WEIGHT = 4.0  # DynamicWeightedBCE 的正样本权重

    # --- Pairwise排序损失（LOSS_TYPE='pairwise_bce'时生效）---
    PAIRWISE_WEIGHT = 0.5           # Pairwise损失权重系数（总损失 = BCE + PAIRWISE_WEIGHT * Pairwise）
    PAIRWISE_TOP_K = 0.10           # Top K%预测区域（构建pair的样本范围）
    PAIRWISE_POS_WEIGHT = 2.0       # 正负对的额外权重（放大排序梯度）
    PAIRWISE_WARMUP_EPOCHS = 8      # 前N轮纯BCE训练，之后引入Pairwise
    PAIRWISE_SIGMA = 1.0            # RankNet温度参数（控制排序信号的锐度）
    PAIRWISE_NUM_NEG = 2            # 每个正样本配对的负样本数

# ==================== 用户自定义标签生成函数 ====================
def generate_label(day1_change, day2_change, day3_change):
    """
    此函数定义什么是"强势买入信号"。
    只要返回 0（无信号）或 1（有信号）即可。

    ========== 核心概念区分 ==========
    【涨跌幅】vs【收益率】：
    - 涨跌幅：基准是前一日收盘价，用于判断股票走势强弱
      * Day1涨跌幅 = (T+1收盘 - T日收盘) / T日收盘
      * Day2涨跌幅 = (T+2收盘 - T+1收盘) / T+1收盘
      * Day3涨跌幅 = (T+3收盘 - T+2收盘) / T+2收盘
    - 收益率：基准是买入价（T+1开盘价），用于计算投资回报
      * 不在此函数中使用，仅用于评估模型表现

    ========== 可用变量说明 ==========
    Args:
        day1_change: Day1 涨跌幅，范围约 [-0.10, 0.10]
        day2_change: Day2 涨跌幅，范围约 [-0.10, 0.10]
        day3_change: Day3 涨跌幅，范围约 [-0.10, 0.10]
    派生变量（根据需要自行计算）：
    - cum_2day = day1_change + day2_change（Day1+Day2 累计涨跌幅）
    - cum_3day = day1_change + day2_change + day3_change（三天累计涨跌幅）
    - max_day = max(day1_change, day2_change, day3_change)（最大涨跌幅）
    - min_day = min(day1_change, day2_change, day3_change)（最小涨跌幅）

    ========== 规则说明 ==========
    当前实现使用五条件规则，满足任一即返回1（强势信号）：
    1. 单日爆发：Day1≥5% 且 累计≥3%
    2. 双日接力：Day1+Day2≥6% 且 Day1,Day2>1% 且 累计≥3%
    3. 稳健上涨：Day1,Day2,Day3≥1% 且 累计≥5%
    4. 爆发后延续：任意一天≥8% 且 累计≥6% 且 Day1≥-2%
    5. 累计达标：3天累计≥8% 且 Day1≥-2%    

    Returns:
        int: 1 表示强势信号（正样本），0 表示无信号（负样本）
    """
    # ========== 派生变量计算 ==========
    cum_2day = day1_change + day2_change
    cum_3day = day1_change + day2_change + day3_change
    max_day = max(day1_change, day2_change, day3_change)

    # ========== 五条件规则（默认实现，可完全自定义）==========
    # 满足任一条件即返回1（强势信号）

    # 规则1：单日爆发 + 累计兜底
    if day1_change >= 0.05 and cum_3day >= 0.03:
        return 1

    # 规则2：双日接力 + 累计兜底
    if cum_2day >= 0.06 and day1_change > 0.01 and day2_change > 0.01 and cum_3day >= 0.03:
        return 1

    # 规则3：稳健上涨（天然安全）
    if day1_change >= 0.01 and day2_change >= 0.01 and day3_change >= 0.01 and cum_3day >= 0.06:
        return 1

    # 规则4：爆发后延续 + Day1保护
    if max_day >= 0.08 and cum_3day >= 0.06 and day1_change >= -0.02:
        return 1

    # 规则5：累计达标 + Day1保护
    if cum_3day >= 0.08 and day1_change >= -0.02:
        return 1

    return 0

# ==================== 用户自定义收益率计算函数 ====================
def calculate_returns(t1_open, t1_close, t2_open=None, t2_close=None, t3_close=None,
                      day1_change=None, day2_change=None, day3_change=None):
    """
    此函数定义如何计算投资收益率，用于评估模型表现。
    支持 1～3 天的可用数据，t2_open/t2_close/t3_close 为可选参数。

    ========== 核心概念区分 ==========
    【涨跌幅】vs【收益率】：
    - 涨跌幅：基准是前一日收盘价，用于判断股票走势强弱
    - 收益率：基准是买入价（T+1开盘价），用于计算投资回报

    ========== 智能止损策略 ==========
    模型预测的是"未来3天满足5条规则的强势信号"，如果走势严重偏离预期，应提前止损：

    1. Day1 ≤ -3%  → 第二天开盘止损（大跌，开盘立刻跑）
    2. Day1+Day2 < -2% → 第二天收盘止损（累计亏2%，收盘走人）
    3. Day1,Day2 都 < 1% → 第二天收盘止损（都不符合预期，收盘走人；释放资金用于第2天买入其他股票）

    ========== 收益率计算公式 ==========
    买入价 = t1_open

    - 第二天开盘卖: (t2_open - t1_open) / t1_open
    - 第二天收盘卖: (t2_close - t1_open) / t1_open = Day1 + Day2
    - 第三天收盘卖: (t3_close - t1_open) / t1_open = Day1 + Day2 + Day3

    Args:
        t1_open: T+1 开盘价（买入价），不能为零
        t1_close: T+1 收盘价（必填）
        t2_open: T+2 开盘价（可选，用于第二天开盘止损）
        t2_close: T+2 收盘价（可选）
        t3_close: T+3 收盘价（可选）
        day1_change: Day1 涨跌幅（必填，用于止损判断）
        day2_change: Day2 涨跌幅（可选）
        day3_change: Day3 涨跌幅（可选）

    Returns:
        tuple: (cumulative_return, daily_returns)
            - cumulative_return: 累计收益率（考虑止损，调用方应优先使用此值）
            - daily_returns: 每日收益率列表，注意：触发止损时长度可能小于3
    """
    # Day1 收益率（日内）
    day1_return = (t1_close - t1_open) / t1_open
    daily_returns = [day1_return]

    # ========== 止损判断 ==========

    # 止损条件1: Day1 大跌（≤ -3%）→ 第二天开盘止损
    if day1_change is not None and day1_change <= -0.03:
        if t2_open is not None:
            # 第二天开盘卖: 收益 = (t2_open - t1_open) / t1_open
            # 更新 daily_returns[0] 为实际收益（第二天开盘卖）
            actual_return = (t2_open - t1_open) / t1_open
            daily_returns = [actual_return]
            return actual_return, daily_returns
        # 如果没有 t2_open，用 Day1 收益作为近似
        return day1_return, daily_returns

    # 如果没有 Day2 数据，直接返回
    if t2_close is None:
        return day1_return, daily_returns

    # Day2 收益率贡献
    day2_return = (t2_close - t1_close) / t1_open
    daily_returns.append(day2_return)

    # ========== 第二天收盘止损判断 ==========

    # 止损条件2: Day1+Day2 < -2% → 第二天收盘止损
    if day1_change is not None and day2_change is not None:
        if day1_change + day2_change < -0.02:
            cumulative_return = day1_return + day2_return
            return cumulative_return, daily_returns

    # 止损条件3: Day1,Day2 都 < 1% → 第二天收盘止损
    if day1_change is not None and day2_change is not None:
        if day1_change < 0.01 and day2_change < 0.01:
            cumulative_return = day1_return + day2_return
            return cumulative_return, daily_returns

    # 如果没有 Day3 数据，返回前两天收益
    if t3_close is None:
        cumulative_return = day1_return + day2_return
        return cumulative_return, daily_returns

    # 没有触发止损，持有满3天
    day3_return = (t3_close - t2_close) / t1_open
    daily_returns.append(day3_return)
    cumulative_return = day1_return + day2_return + day3_return
    return cumulative_return, daily_returns


# ==================== 设备配置 ====================
class DeviceConfig:
    @staticmethod
    def get_device():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            return device
        else:
            print("ERROR:CUDA 不可用，程序退出")
            sys.exit(1)

# ==================== 配置打印函数 ====================
def print_config_summary():
    print("=" * 50)

    print(f"模型参数:")
    print(f"  输入维度: {ModelConfig.INPUT_DIM}")
    print(f"  模型维度: {ModelConfig.D_MODEL}")
    print(f"  注意力头数: {ModelConfig.NHEAD}")
    print(f"  层数: {ModelConfig.NUM_LAYERS}")
    print(f"  输出维度: {ModelConfig.OUTPUT_DIM}")
    print(f"  序列长度: {DataConfig.CONTEXT_LENGTH}")

    optimizer_names = {'adamw': 'AdamW', 'lion': 'Lion', 'mano': 'Mano'}
    optimizer_display = optimizer_names.get(TrainingConfig.OPTIMIZER_TYPE.lower(), TrainingConfig.OPTIMIZER_TYPE)

    print(f"训练参数:")
    print(f"  训练轮数: {TrainingConfig.EPOCHS}")
    print(f"  学习率: {TrainingConfig.get_base_lr()}")
    print(f"  权重衰减: {TrainingConfig.get_base_wd()}")
    print(f"  批处理大小: {TrainingConfig.BATCH_SIZE}")
    print(f"  每轮批次数: {TrainingConfig.BATCHES_PER_EPOCH}")
    print(f"  优化器: {optimizer_display}")
    print(f"  混合精度(AMP): {'BF16' if TrainingConfig.USE_AMP else '关闭(FP32)'}")
    print(f"  预热轮数: {TrainingConfig.WARMUP_EPOCHS}")
    print(f"  预热起始学习率: {TrainingConfig.WARMUP_START_LR}")
    main_epochs = TrainingConfig.EPOCHS - TrainingConfig.WARMUP_EPOCHS
    print(f"  余弦退火轮数: {TrainingConfig.COSINE_ANNEAL_EPOCHS} (后{main_epochs - TrainingConfig.COSINE_ANNEAL_EPOCHS}轮固定在ETA_MIN)")

    print(f"数据参数:")
    print(f"  数据库: {DataConfig.DB_PATH}")
    print(f"  采样策略: {DataConfig.SAMPLING_STRATEGY} ({'时间顺序采样' if DataConfig.SAMPLING_STRATEGY == 'temporal' else '随机采样'})")
    print(f"  训练集范围: {DataConfig.TRAIN_START_DATE} ~ {DataConfig.TRAIN_END_DATE}")
    print(f"  验证集范围: {DataConfig.VAL_START_DATE} ~ {DataConfig.VAL_END_DATE}")
    print(f"  测试集起始: {DataConfig.TEST_START_DATE} ~ 最新")
    print(f"  上下文长度: {DataConfig.CONTEXT_LENGTH}")
    print(f"  涨停过滤: {'开启' if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP else '关闭'}")
    print(f"  评估批处理大小: {DataConfig.EVAL_BATCH_SIZE}")
    print(f"  最大持仓: {DataConfig.MAX_HOLDINGS}  每日买入上限: {DataConfig.MAX_BUY_PER_DAY if DataConfig.MAX_BUY_PER_DAY > 0 else '不限'}")
    print(f"标签参数:")
    print(f"  正样本距离保护: {DataConfig.LABEL_DISTANCE}")
    print(f"  Day1基准: {'开盘价(日内涨幅)' if DataConfig.LABEL_DAY1_USE_OPEN else '前日收盘(含跳空)'}")

    print("=" * 50)
