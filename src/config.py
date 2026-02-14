"""
EquiNet 模型配置文件
统一管理模型参数、训练参数和评估参数
"""

import math
import torch

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # 数据路径
    DATA_DIR = './data'              # 数据目录
    OUTPUT_DIR = './out'             # 输出目录

    # 数据分割参数（按时间划分）
    TEST_DAYS = 140                   # 测试集天数（每只股票的最近N天作为测试集）
    RANDOM_SEED = 42                 # 随机种子
    
    # 训练集时间范围限制
    TRAIN_START_YEAR = 2019          # 训练集起始年份（2020年及以前的数据不参与训练）
    
    # 样本生成参数
    CONTEXT_LENGTH = 60              # 历史数据长度（这是核心参数，其他地方应引用这个值）
    FUTURE_DAYS = 3                  # 未来预测天数
    REQUIRED_LENGTH = CONTEXT_LENGTH + FUTURE_DAYS  # 总需求长度（上下文 + 未来天数）

    # 上涨阈值（二分类）
    UPRISE_THRESHOLD = 0.08          # 上涨阈值（8%，涨幅≥8%视为上涨）

    # 强势信号检测参数
    SIGNAL_DAY1_BURST = 0.05         # 单日爆发阈值：Day1涨幅≥5%
    SIGNAL_TWO_DAY_CUM = 0.06        # 双日累计阈值：Day1+Day2≥6%
    SIGNAL_DAY_MIN = 0.01            # 单日最小涨幅：≥1%
    SIGNAL_THREE_DAY_CUM = 0.05      # 三日累计阈值：Day1+Day2+Day3≥5%
    SIGNAL_ANY_BURST = 0.08          # 任意一天爆发阈值：≥8%
    SIGNAL_BURST_CUM = 0.06          # 爆发后累计阈值：累计≥6%

    # 评估参数
    EVAL_BATCH_SIZE = 100             # 评估批处理大小
    TOP_PERCENT = 1                   # 排序收益评估的百分比（取预测概率前N%的样本）
    TOP_N_PER_DAY = 4                 # 实战收益率：每天选股数量（0表示使用全局阈值模式）
    
    # 模型保存条件
    MIN_AUC = 0.65                    # 最低AUC要求（按时间划分后的真实性能基线）

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""

    # ========== 模型类型选择 ==========
    # 'continuous': 连续值模型 (master分支，使用6维连续输入)
    # 'tokenized': Token化模型 (test分支，将输入离散化为token ID)
    MODEL_TYPE = 'continuous'  # 可选: 'continuous' 或 'tokenized'

    # 基础模型参数
    INPUT_DIM = 6                    # 输入特征维度数（OHLC + volume + exchange）
    D_MODEL = 24                     # 模型维度（Transformer内部维度）
    EMBED_HIDDEN_DIM = 48            # Embedding中间层维度（两阶段FFN：6→40→D_MODEL）
    FFN_EXPAND_RATIO = 4             # FFN隐藏层扩展比例（hidden_dim = d_model * FFN_EXPAND_RATIO）
    NHEAD = 2                        # 注意力头数
    NUM_LAYERS = 6                   # Transformer层数
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1之间）
    SEQ_LEN = DataConfig.CONTEXT_LENGTH  # 最大序列长度（直接引用CONTEXT_LENGTH，确保一致性）

    # 注意力机制参数（为小模型调整）
    DROPOUT_RATE = 0.1                 # Dropout比率设置为0降低欠拟合
    ATTENTION_DROPOUT = 0.1            # 注意力Dropout比率设置为0降低欠拟合

    # Token化参数（仅当 MODEL_TYPE='tokenized' 时使用）
    # 词表大小 = 4*20(OHLC) + 36(volume) + 60(exchange) = 176
    VOCAB_SIZE = 176
    TOKEN_SEQ_LEN = DataConfig.CONTEXT_LENGTH * INPUT_DIM  # Token序列长度 = 60 * 6 = 360

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""

    # 基础训练参数（优化训练策略）
    EPOCHS = 400                     # 训练轮数（增加轮数以充分训练小模型）
    LEARNING_RATE = 0.001            # 初始学习率（提高学习率）

    # 训练批处理
    BATCH_SIZE = 1024                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 2            # 每轮训练的批次数（调低以适配时间序采样）
    # BATCHES_PER_EPOCH*EPOCHS=800

    # 优化器参数
    USE_ADAMW = True                 # 是否使用AdamW优化器
    USE_MANO = False                  # 是否使用Mano优化器（与AdamW/Adam互斥，优先级最高）
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # Mano优化器参数（当USE_MANO=True时生效）
    MANO_MOMENTUM = 0.95             # Mano动量系数
    MANO_ADAMW_BETAS = (0.9, 0.95)   # 混合优化器中AdamW部分的beta参数

    # 学习率调度器参数
    SCHEDULER_STEP_SIZE = 10         # 学习率调度步长
    SCHEDULER_GAMMA = 0.5            # 学习率衰减因子
    
    # 余弦退火调度器参数
    USE_COSINE_ANNEALING = True      # 是否使用余弦退火调度器
    COSINE_ETA_MIN = 5e-6            # 余弦退火最小学习率（训练末期的精细微调学习率）
    
    # 学习率预热参数
    WARMUP_RATIO = 0.1               # 预热轮数占比（总轮数的10%）
    WARMUP_START_LR = 1e-4           # 预热起始学习率（提高起始值，减少过于保守的预热）

# ==================== 损失函数配置 ====================
class LossConfig:
    """损失函数相关配置"""

    # 可选值: 'dynamic_bce'（使用DynamicWeightedBCE）或 'standard_bce'（使用BCEWithLogitsLoss）
    LOSS_TYPE = 'dynamic_bce'

    # DynamicWeightedBCE 的正样本权重
    POS_WEIGHT = 4.0

    @staticmethod
    def use_dynamic_bce():
        return LossConfig.LOSS_TYPE.lower() == 'dynamic_bce'

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
            print(f"使用 GPU 进行训练: {torch.cuda.get_device_name()}")
            # 检查BF16支持
            if torch.cuda.is_bf16_supported():
                print("✓ GPU 支持 BF16 加速训练")
            else:
                print("⚠ GPU 不支持 BF16，训练可能较慢或出错（建议使用RTX 30系及以上显卡）")
        else:
            print("CUDA 不可用，将使用 CPU 进行训练，训练速度可能较慢。")
            print("⚠ CPU 模式下 BF16 性能可能不如 FP32")
        return device

# ==================== 模型保存配置 ====================
class ModelSaveConfig:
    """模型保存相关配置"""

    # 模型文件名
    BEST_MODEL_NAME = 'EnhancedEquiNet_focal_best.pth'
    FINAL_MODEL_NAME = 'EnhancedEquiNet_final.pth'

    @staticmethod
    def get_best_model_path():
        """获取最佳模型路径"""
        return f'./out/{ModelSaveConfig.BEST_MODEL_NAME}'

    @staticmethod
    def get_final_model_path(d_model):
        """获取最终模型路径"""
        return f'./out/EnhancedEquiNet_{d_model}.pth'

# ==================== 配置打印函数 ====================
def print_config_summary():
    """打印配置摘要"""
    print("=" * 50)
    print("EquiNet 模型配置摘要")
    print("=" * 50)

    print(f"模型架构:")
    print(f"  输入维度: {ModelConfig.INPUT_DIM}")
    print(f"  模型维度: {ModelConfig.D_MODEL}")
    print(f"  注意力头数: {ModelConfig.NHEAD}")
    print(f"  层数: {ModelConfig.NUM_LAYERS}")
    print(f"  输出维度: {ModelConfig.OUTPUT_DIM}")
    print(f"  序列长度: {DataConfig.CONTEXT_LENGTH} (由CONTEXT_LENGTH统一控制)")

    print(f"\n训练参数:")
    print(f"  训练轮数: {TrainingConfig.EPOCHS}")
    print(f"  学习率: {TrainingConfig.LEARNING_RATE}")
    print(f"  批处理大小: {TrainingConfig.BATCH_SIZE}")
    print(f"  每轮批次数: {TrainingConfig.BATCHES_PER_EPOCH}")
    warmup_epochs = max(1, int(TrainingConfig.EPOCHS * TrainingConfig.WARMUP_RATIO))
    print(f"  预热轮数: {warmup_epochs} (总轮数的{TrainingConfig.WARMUP_RATIO*100:.0f}%)")
    print(f"  预热起始学习率: {TrainingConfig.WARMUP_START_LR}")
    
    print(f"\n学习率调度:")
    if TrainingConfig.USE_COSINE_ANNEALING:
        total_main_epochs = TrainingConfig.EPOCHS * (1 - TrainingConfig.WARMUP_RATIO)
        print(f"  调度策略: 余弦退火")
        print(f"  预热阶段: 第1-{int(TrainingConfig.EPOCHS * TrainingConfig.WARMUP_RATIO)}轮")
        print(f"  退火周期: {int(total_main_epochs)}轮 (主训练全程)")
        print(f"  最小学习率: {TrainingConfig.COSINE_ETA_MIN}")
    else:
        print(f"  调度策略: 阶梯衰减")
        print(f"  衰减步长: {TrainingConfig.SCHEDULER_STEP_SIZE}轮")
        print(f"  衰减因子: {TrainingConfig.SCHEDULER_GAMMA}")

    print(f"\n数据参数:")
    print(f"  数据目录: {DataConfig.DATA_DIR}")
    print(f"  训练集起始年份: {DataConfig.TRAIN_START_YEAR}年（过滤{DataConfig.TRAIN_START_YEAR-1}年及以前的数据）")
    print(f"  测试集天数: {DataConfig.TEST_DAYS}天")
    print(f"  上下文长度: {DataConfig.CONTEXT_LENGTH}")
    print(f"  上涨阈值: {DataConfig.UPRISE_THRESHOLD*100}%")
    print(f"\n标签机制: 强势信号检测（0/1二分类）")
    print(f"  1. 单日爆发: Day1涨幅 ≥ {DataConfig.SIGNAL_DAY1_BURST*100:.0f}%")
    print(f"  2. 双日接力: Day1+Day2 ≥ {DataConfig.SIGNAL_TWO_DAY_CUM*100:.0f}% 且 Day1,Day2 > {DataConfig.SIGNAL_DAY_MIN*100:.0f}%")
    print(f"  3. 稳健上涨: Day1,Day2,Day3 ≥ {DataConfig.SIGNAL_DAY_MIN*100:.0f}% 且 累计 ≥ {DataConfig.SIGNAL_THREE_DAY_CUM*100:.0f}%")
    print(f"  4. 爆发后延续: 任意一天 ≥ {DataConfig.SIGNAL_ANY_BURST*100:.0f}% 且 累计 ≥ {DataConfig.SIGNAL_BURST_CUM*100:.0f}%")
    print(f"  5. 累计达标: 3天累计涨幅 ≥ {DataConfig.UPRISE_THRESHOLD*100:.0f}%（基础条件）")

    print(f"\n评估参数:")
    print(f"  评估批处理大小: {DataConfig.EVAL_BATCH_SIZE}")
    
    print(f"\n模型保存条件:")
    print(f"  最低AUC要求: {DataConfig.MIN_AUC}")

    print("=" * 50)

if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary() 