"""
EquiNet 模型配置文件
"""

import sys
import torch

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # 数据路径
    DATA_DIR = './data'              # 数据目录
    OUTPUT_DIR = './out'             # 输出目录

    # 数据分割参数（按时间划分）
    TEST_DAYS = 110                   # 测试集天数（每只股票的最近N天作为测试集）
    RANDOM_SEED = 42                 # 随机种子
    
    # 训练集时间范围限制
    TRAIN_START_YEAR = 2016          # 训练集起始年份（2020年及以前的数据不参与训练）
    
    # 样本生成参数
    CONTEXT_LENGTH = 30              # 历史数据长度（这是核心参数，其他地方应引用这个值）,TEST_DAYS - CONTEXT_LENGTH = 80
    FUTURE_DAYS = 3                  # 未来预测天数
    REQUIRED_LENGTH = CONTEXT_LENGTH + FUTURE_DAYS  # 每样本总需求长度

    # 强势信号检测参数（用于二分类标签设置）
    SIGNAL_DAY1_BURST = 0.05         # 单日爆发阈值：Day1涨幅≥5%
    SIGNAL_TWO_DAY_CUM = 0.06        # 双日累计阈值：Day1+Day2≥6%
    SIGNAL_DAY_MIN = 0.01            # 单日最小涨幅：≥1%
    SIGNAL_THREE_DAY_CUM = 0.05      # 三日累计阈值：Day1+Day2+Day3≥5%
    SIGNAL_ANY_BURST = 0.08          # 任意一天爆发阈值：≥8%
    SIGNAL_BURST_CUM = 0.06          # 爆发后累计阈值：累计≥6%
    UPRISE_THRESHOLD = 0.08          # 上涨阈值（8%，涨幅≥8%视为上涨）
    # 风险控制参数（用于二分类标签设置）
    SIGNAL_MIN_CUM_RETURN = 0.03     # 最低累计收益：≥3%（过滤累计亏损或微利样本）
    SIGNAL_DAY1_MAX_DROP = -0.02     # Day1最大跌幅：≥-2%（避免买入当天就亏）

    # 采样策略配置
    # 'temporal': 时间顺序采样（指针在训练集上循环滑动）
    # 'random': 随机采样（每次随机选择股票和位置）
    SAMPLING_STRATEGY = 'temporal'

    # 是否过滤上下文最后一天接近涨停的样本
    # True: 过滤最后一天涨停的样本（防止模型过度依赖涨停信号，避免追涨策略）
    # False: 保留所有样本（让模型自己学习涨停后的走势规律）
    FILTER_CONTEXT_LAST_DAY_LIMIT_UP = True

    # 评估参数
    EVAL_BATCH_SIZE = 256            # 评估批处理大小（分批处理，减少显存占用）
    TOP_K = 1                   # 排序收益评估的百分比（取预测概率前N%的样本）
    TOP_N_PER_DAY = 0                 # 实战收益率：每天选股数量（0表示使用全局阈值模式）
    MAX_SELECT_PER_DAY = 4             # 全局阈值模式下每天最多选股数量（0表示不限制）
# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""

    # ========== 模型类型选择 ==========
    # 'continuous': 连续值模型 (6维连续输入)
    # 'tokenized': Token化模型 (将输入离散化为token ID)
    MODEL_TYPE = 'continuous'  # 可选: 'continuous' 或 'tokenized'

    # 基础模型参数
    INPUT_DIM = 6                    # 输入特征维度数（OHLC + volume + exchange）
    D_MODEL = 48                     # 模型维度（Transformer内部维度）
    EMBED_HIDDEN_DIM = 48            # Embedding中间层维度（两阶段FFN：INPUT_DIM → EMBED_HIDDEN_DIM → D_MODEL）
    FFN_EXPAND_RATIO = 4             # FFN隐藏层扩展比例（hidden_dim = d_model * FFN_EXPAND_RATIO）
    NHEAD = 4                        # 注意力头数
    NUM_LAYERS = 6                   # Transformer层数
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1之间）

    # 注意力机制参数
    DROPOUT_RATE = 0                 # Dropout比率设置为0降低欠拟合
    ATTENTION_DROPOUT = 0            # 注意力Dropout比率设置为0降低欠拟合

    # Token化参数（仅当 MODEL_TYPE='tokenized' 时使用）
    # 词表大小 = 4*20(OHLC) + 36(volume) + 60(exchange) = 176
    VOCAB_SIZE = 176
    TOKEN_SEQ_LEN = DataConfig.CONTEXT_LENGTH * INPUT_DIM  # Token序列长度 = 60 * 6 = 360

    # 输出层参数
    OUTPUT_LAYER_GAIN = 3.0          # 输出层初始化增益（增大logits范围）

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""
    # 基础训练参数（优化训练策略）
    EPOCHS = 400                     # 训练轮数（增加轮数以充分训练小模型）
    LEARNING_RATE = 0.001            # 初始学习率（提高学习率）

    # 训练批处理
    BATCH_SIZE = 512                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 1            # 每轮训练的批次数（调低以适配时间序采样）

    # 优化器参数
    USE_ADAMW = True                 # 是否使用AdamW优化器
    USE_MANO = True                  # 是否使用Mano优化器（与AdamW/Adam互斥，优先级最高）
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # Mano优化器参数（当USE_MANO=True时生效）
    MANO_MOMENTUM = 0.95             # Mano动量系数
    MANO_ADAMW_BETAS = (0.9, 0.95)   # 混合优化器中AdamW部分的beta参数

    # 余弦退火调度器参数,学习率预热参数
    COSINE_ETA_MIN = 5e-6            # 余弦退火最小学习率（训练末期的精细微调学习率）
    WARMUP_RATIO = 0.1               # 预热轮数占比（总轮数的10%）
    WARMUP_START_LR = 1e-4           # 预热起始学习率（提高起始值，减少过于保守的预热）

# ==================== 损失函数配置 ====================
class LossConfig:
    """损失函数相关配置"""

    LOSS_TYPE = 'task_aligned'  #'task_aligned':任务对齐损失 | 'dynamic_bce':批权重动态平衡 | 'standard_bce':标准二元交叉熵
    
    POS_WEIGHT = 4.0  # DynamicWeightedBCE 的正样本权重（同时用于 TaskAlignedLoss 的基础BCE组件）

    # ========== TaskAlignedLoss 参数 ==========
    # 各子损失的权重
    RANK_LOSS_WEIGHT = 0.3       # 排序损失权重（确保高收益样本排在前面）
    RETURN_LOSS_WEIGHT = 0.2     # 收益加权损失权重（收益越高/亏损越大，梯度越强）
    TOPK_LOSS_WEIGHT = 0.1       # Top-K聚焦损失权重（只关注模型预测最高的那部分）

    # 排序损失参数
    RANK_MARGIN = 0.1            # 排序损失的margin（要求正负样本对的分数差距至少这么大）
    RANK_NUM_PAIRS = 64          # 每个batch采样的正负样本对数量

    # 收益加权参数
    RETURN_ALPHA = 5.0           # 正样本收益率放大系数（收益越高权重越大）
    RETURN_BETA = 3.0            # 负样本亏损放大系数（亏损越多权重越大）
    RETURN_CLIP = 0.30           # 收益率裁剪范围（防止极端值主导梯度）

    # Top-K聚焦参数
    TOPK_RATIO = 0.10            # 每个batch中关注的Top比例（10%）

    @staticmethod
    def use_dynamic_bce():
        return LossConfig.LOSS_TYPE.lower() == 'dynamic_bce'

    @staticmethod
    def use_task_aligned():
        return LossConfig.LOSS_TYPE.lower() == 'task_aligned'

# ==================== 设备配置 ====================
class DeviceConfig:
    """设备相关配置"""
    @staticmethod
    def get_device():
        """获取训练设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def print_device_info():
        """打印设备信息"""
        device = DeviceConfig.get_device()
        if device.type == "cuda":
            print(f"使用{torch.cuda.get_device_name()}进行训练")
        else:
            print("ERROR:CUDA 不可用，程序退出")
            sys.exit(1)
        return device

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

    print(f"训练参数:")
    print(f"  训练轮数: {TrainingConfig.EPOCHS}")
    print(f"  学习率: {TrainingConfig.LEARNING_RATE}")
    print(f"  批处理大小: {TrainingConfig.BATCH_SIZE}")
    print(f"  每轮批次数: {TrainingConfig.BATCHES_PER_EPOCH}")
    warmup_epochs = max(1, int(TrainingConfig.EPOCHS * TrainingConfig.WARMUP_RATIO))
    print(f"  预热轮数: {warmup_epochs} (总轮数的{TrainingConfig.WARMUP_RATIO*100:.0f}%)")
    print(f"  预热起始学习率: {TrainingConfig.WARMUP_START_LR}")

    print(f"数据参数:")
    print(f"  数据目录: {DataConfig.DATA_DIR}")
    print(f"  采样策略: {DataConfig.SAMPLING_STRATEGY} ({'时间顺序采样' if DataConfig.SAMPLING_STRATEGY == 'temporal' else '随机采样'})")
    print(f"  训练集起始年份: {DataConfig.TRAIN_START_YEAR}年（过滤{DataConfig.TRAIN_START_YEAR-1}年及以前的数据）")
    print(f"  测试集天数: {DataConfig.TEST_DAYS}天")
    print(f"  上下文长度: {DataConfig.CONTEXT_LENGTH}")
    print(f"  上涨阈值: {DataConfig.UPRISE_THRESHOLD*100}%")
    print(f"  涨停过滤: {'开启' if DataConfig.FILTER_CONTEXT_LAST_DAY_LIMIT_UP else '关闭'}")

    print(f"标签机制: 强势信号检测（0/1二分类）")
    print(f"  1.单日爆发: Day1涨幅 ≥ {DataConfig.SIGNAL_DAY1_BURST*100:.0f}%")
    print(f"  2.双日接力: Day1+Day2 ≥ {DataConfig.SIGNAL_TWO_DAY_CUM*100:.0f}% 且 Day1,Day2 > {DataConfig.SIGNAL_DAY_MIN*100:.0f}%")
    print(f"  3.稳健上涨: Day1,Day2,Day3 ≥ {DataConfig.SIGNAL_DAY_MIN*100:.0f}% 且 累计 ≥ {DataConfig.SIGNAL_THREE_DAY_CUM*100:.0f}%")
    print(f"  4.爆发后延续: 任意一天 ≥ {DataConfig.SIGNAL_ANY_BURST*100:.0f}% 且 累计 ≥ {DataConfig.SIGNAL_BURST_CUM*100:.0f}%")
    print(f"  5.累计达标: 3天累计涨幅 ≥ {DataConfig.UPRISE_THRESHOLD*100:.0f}%（基础条件）")

    print(f"评估参数:")
    print(f"  评估批处理大小: {DataConfig.EVAL_BATCH_SIZE}")
    print("=" * 50)
