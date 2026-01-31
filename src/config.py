"""
EquiNet 模型配置文件
统一管理模型参数、训练参数和评估参数
"""

import torch

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # 数据路径
    DATA_DIR = './data'              # 数据目录
    OUTPUT_DIR = './out'             # 输出目录

    # 数据分割参数（按时间划分）
    TEST_DAYS = 80                   # 测试集天数（每只股票的最近N天作为测试集）
    RANDOM_SEED = 42                 # 随机种子
    
    # 训练集时间范围限制
    TRAIN_START_YEAR = 2021          # 训练集起始年份（2020年及以前的数据不参与训练）
    
    # 样本生成参数
    CONTEXT_LENGTH = 60              # 历史数据长度（这是核心参数，其他地方应引用这个值）
    FUTURE_DAYS = 3                  # 未来预测天数
    REQUIRED_LENGTH = CONTEXT_LENGTH + FUTURE_DAYS  # 总需求长度（上下文 + 未来天数）

    # 上涨阈值（二分类）
    UPRISE_THRESHOLD = 0.08          # 上涨阈值（8%，涨幅≥8%视为上涨）

    # 评估参数
    EVAL_BATCH_SIZE = 100             # 评估批处理大小
    TOP_PERCENT = 1                   # 排序收益评估的百分比（取预测概率前N%的样本）
    
    # 模型保存条件
    MIN_AUC = 0.65                    # 最低AUC要求（按时间划分后的真实性能基线）

# ==================== 模型架构参数 ====================
class ModelConfig:
    """模型架构相关参数"""
    # 基础模型参数
    INPUT_DIM = 7                    # 输入特征维度数（OHLC + volume + exchange + rate）
    D_MODEL = 28                     # 模型维度（Transformer内部维度）
    EMBED_HIDDEN_DIM = 40            # Embedding中间层维度（两阶段FFN：7→40→80）
    NHEAD = 2                        # 注意力头数
    NUM_LAYERS = 6                   # Transformer层数
    OUTPUT_DIM = 1                   # 输出维度（上涨概率，0-1之间）
    SEQ_LEN = DataConfig.CONTEXT_LENGTH  # 最大序列长度（直接引用CONTEXT_LENGTH，确保一致性）

    # 注意力机制参数（为小模型调整）
    DROPOUT_RATE = 0.1                 # Dropout比率设置为0降低欠拟合
    ATTENTION_DROPOUT = 0.1            # 注意力Dropout比率设置为0降低欠拟合
    
    # Token化参数
    # 词表大小 = 4*20(OHLC) + 36(volume) + 60(exchange) + 24(rate) = 200
    VOCAB_SIZE = 200
    TOKEN_SEQ_LEN = DataConfig.CONTEXT_LENGTH * INPUT_DIM  # Token序列长度 = 60 * 7 = 420

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
    USE_ADAMW = False                 # 是否使用AdamW优化器
    WEIGHT_DECAY = 1e-5              # 权重衰减
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # 学习率调度器参数
    SCHEDULER_STEP_SIZE = 10         # 学习率调度步长
    SCHEDULER_GAMMA = 0.5            # 学习率衰减因子
    
    # 余弦退火调度器参数
    USE_COSINE_ANNEALING = True      # 是否使用余弦退火调度器
    COSINE_T_MAX = 30                # 余弦退火周期（30轮完成衰减，更快收敛）
    COSINE_ETA_MIN = 1e-5            # 余弦退火最小学习率（提高到1e-5，避免学习率过小）
    
    # 学习率预热参数
    WARMUP_EPOCHS = 5                # 预热轮数（前5轮逐步达到最高学习率）
    WARMUP_START_LR = 1e-4           # 预热起始学习率（提高起始值，减少过于保守的预热）

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
    print(f"  预热轮数: {TrainingConfig.WARMUP_EPOCHS}")
    print(f"  预热起始学习率: {TrainingConfig.WARMUP_START_LR}")
    
    print(f"\n学习率调度:")
    if TrainingConfig.USE_COSINE_ANNEALING:
        print(f"  调度策略: 余弦退火")
        print(f"  退火周期: {TrainingConfig.COSINE_T_MAX}轮")
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
    print(f"\n标签机制: 二分类（{DataConfig.UPRISE_THRESHOLD*100:.0f}%为阈值）")
    print(f"  涨幅≥{DataConfig.UPRISE_THRESHOLD*100:.0f}% → 标签1.0 (上涨)")
    print(f"  涨幅<{DataConfig.UPRISE_THRESHOLD*100:.0f}% → 标签0.0 (不上涨)")

    print(f"\n评估参数:")
    print(f"  评估批处理大小: {DataConfig.EVAL_BATCH_SIZE}")
    
    print(f"\n模型保存条件:")
    print(f"  最低AUC要求: {DataConfig.MIN_AUC}")

    print("=" * 50)

if __name__ == "__main__":
    # 打印配置摘要
    print_config_summary() 