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
    DATA_SOURCE = 'baostock'  # 'baostock' 或 'akshare'
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
    MA_WINDOW = 10                   # 量能与换手率相对均值的滑动窗口大小（N日均值基准）,左侧数据不够时自动向右侧借数据为预期机制

    # 涨跌停配置
    LIMIT_THRESHOLD = 0.095          # 涨跌停判断阈值（9.5%，覆盖普通板±10%）
    LIMIT_CHECK_MODE = 'ohlc'        # 'simple': 仅看涨跌幅 | 'ohlc': 通过OHLC判断是否开板

    # T+1开盘涨停过滤开关
    # 开盘即封死涨停价，散户/程序实际无法买入，是"不可交易"样本。
    # True: 训练样本排除这些样本，评估阶段始终排除（不受此开关控制，
    #       因评估应反映可交易的真实表现）
    # False: 训练样本保留开盘涨停样本，仅评估阶段过滤
    FILTER_LIMIT_UP_OPEN = False

    # 正负样本比例配置
    # 每个 batch 中正样本的占比，用于应对正负样本极度不平衡（正样本约 3-5%）
    # 设为 0.25 即每 batch 中 25% 正样本 + 75% 负样本，通过上采样正样本实现类别平衡
    POSITIVE_RATIO = 0.25

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
    # distance=0时不排除任何样本
    # 目的：消除正样本左侧特征高度重叠但标签相反的矛盾训练信号
    LABEL_DISTANCE = 3

    # Day1标签基准选择
    # True: day1使用开盘到收盘的日内涨幅 (close[T+1]-open[T+1])/open[T+1]，对齐实际买入价
    #       消除跳空缺口对标签的干扰，标签只反映投资者能赚到的部分
    # False: day1使用收盘到收盘的涨跌幅 (close[T+1]-close[T])/close[T]，包含隔夜跳空
    LABEL_DAY1_USE_OPEN = True

    # ========== 极端行情过滤 ==========
    # 市场普涨/普跌日的标签由 beta 驱动而非个股主力运作，属于噪声标签。
    # 涨跌比（上涨家数/下跌家数）超过 EXTREME_UP_DOWN_RATIO 的日期视为极端行情日，
    # 未来窗口落在这些日期的样本既不作正样本也不作负样本，直接剔除。
    # 使用前需先运行 src/market_index.py 生成 out/market_index.json
    MARKET_BREADTH_PATH = os.path.join(OUTPUT_DIR, 'market_index.json')
    EXCLUDE_EXTREME_MARKET = True     # 是否启用极端行情过滤
    EXTREME_UP_DOWN_RATIO = 50.0      # 涨跌比阈值（上涨家数/下跌家数 ≥ 此值视为极端）

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
    INPUT_DIM = 16                   # 输入特征维度数（OHLC[open/close相对昨收,high/low相对当日open] + vwap + volume + exchange + m5 + m10 + m20 + dif + dea + macd_hist + macd_hist_diff + bb_upper + bb_lower）
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
    # 训练超参数
    EPOCHS = 100                          # 预训练轮数
    BATCH_SIZE = 2560                     # 大batch，对比学习需要充足负样本
    LEARNING_RATE = 3e-3                  # 预训练学习率
    WEIGHT_DECAY = 0.1                   # 权重衰减
    WARMUP_EPOCHS = 10                   # 预热轮数
    COSINE_ETA_MIN = 1e-4                 # 余弦退火最小学习率

    # 数据采集
    MAX_SAMPLES = 200_000               # 每个epoch的训练样本数
    DEDUP_PRECISION = 1                 # batch内去重精度（小数位数），值越小去重越激进
    BATCH_DEDUP_OVERSAMPLE = 2.0        # batch内去重过采样倍数（先多采再去重，保证多样性）

    # 与 Recon MSE 同量级、batch size 不变（论文 3.2 似乎说明）。
    VISREG_WEIGHT = 0.6                # λ, 正则项权重
    VISREG_NUM_SLICES = 256             # 随机投影方向数 K (论文建议 K>C·D, 此处 D=128, K/D=2)
    VISREG_W_SCALE = 1.0                # 尺度项权重 (逐维 std→1, VICReg 方差项)
    VISREG_W_SHAPE = 0.6                # 形状项权重 (SWD 对齐高斯分位数)。
                                        # 0.6 由单变量实验臂 shape06 定版：削弱"单峰高斯"
                                        # 反压后，语义沉淀在 z 本体的晚期衰退消失——
                                        # ep50→100 的 CKA/zK1/zr 由同步下滑转为稳定，
                                        # 终值 zK1 0.355→0.389 / zr 0.532→0.568 /
                                        # CKA 0.535→0.569，D= 0.084→0.080 同步小幅改善；
                                        # 尺度(std=TARGET_STD)与中心化契约由独立条款保护不受影响。
                                        # 对照: w_shape=1.0 基线与 dw16/lam45 臂数据见 git 历史
    VISREG_W_CENTER = 1.0               # 中心化项权重 (批均值→0)
    TARGET_STD = 0.2                    # 目标标准差（缩放后 VISReg 尺度项 target std依旧=1）

    # 衍生特征解码：线性解码器额外解码 N 个"16维的跨域非线性函数"，
    # 强迫 embedding 编码维度间关系（特征融合），而非逐维独立存储。
    # 详见 pretrain_embedding.py:compute_derived_features
    N_DERIVED_FEATURES = 11             # 衍生特征数量
    DERIVED_EPS = 1e-6                  # 衍生特征除法防零 epsilon
    DERIVED_WEIGHT = 1.0                # 衍生重建损失权重 (剪尾标准化后等权=1.0，可调防辅助任务主导)
    DERIVED_WINSORIZE_PCT = (1.0, 99.0) # 衍生特征剪尾分位数：剔除极少数极端行情样本
                                        # 对损失的支配（"极端值霸凌"），(0,100)=不剪尾

    # 核加权类感知对比学习（几何轴主损失，SupCon；Khosla et al., 2020）：
    # 正对权重 = 多头一致度核 K_ij（全粗+细分头中标签一致的比例），
    # 施加在 projector 输出 g(z) 上（SimCLR/DINOv2 路线：projector 吸收
    # "收紧"压力，z 保住 O/D 线性可读性，训练后丢弃）。
    # 重建/VISReg 只保证"每条K线的信息可读出"，不管样本间相似度结构——
    # 而下游 attention 消费的恰恰是 z 的点积相似度，该几何必须显式训练。
    # 锚点设计：单标签（vpa-only 硬标签版）锚点贫乏——3 簇填不满空间、
    # 簇内聚拢无终点（同=0.968）、与锚点弱相关的语义被挤出几何（vws
    # 退化实测）；一致度核把锚点增殖为头组合胞元（~百级），全头一致
    # 全力度聚拢、部分一致分级聚拢、全不一致斥远——样本按丰富语义
    # 分级"沉淀"。正对核=CKA 目标核（同一数学对象），损失由 batch 内
    # 全部成对点积构成，梯度触及 z 的每个方向。
    # z 不坍缩由 VISReg（各向同性高斯散度）+ Recon（细粒度信息）保证。
    # 历史注记：掩码视图 InfoNCE（SCARF式）已删除——其前提"被掩列可从
    # 剩余列恢复"对本特征表示不成立（技术列依赖历史不在当日输入、
    # 逐列分位数归一化破坏跨列代数关系）：掩技术列 → 模型学会忽略它们
    # （alignment→0 任务空转）；掩源列 → 语义真空（伪方向视图）。
    # SCARF 是无标签时代的拐杖，有确定性标签时直接用标签教几何。
    SUPCON_ENABLED = True               # 总开关，False=退回纯线性探针模式
    SUPCON_WEIGHT = 0.2                 # w_c：SupCon 权重（冷启动≈ln(B)~7.8，
                                        # 与另两项 O(1) 相比 0.2 使加权贡献同数量级）；
                                        # 剩余权重按 (1-w_c) 分配给 VISReg(λ)/Recon(1-λ)，
                                        # λ=VISREG_WEIGHT 相对语义不变，w_c=0 时公式退回旧版
    SUPCON_TAU = 0.2                    # 温度：控制不一致对推开力度（数千负样本常规 0.1~0.2）

    # 分类头（粗粒度类别监督，Kronos 分层监督的"粗"端；头定义见
    # pretrain_embedding.py:CLS_HEAD_SPEC）：
    # 符号类衍生目标本质是类别 {-1,0,1}，被 MSE 当连续数训时存在
    # "猜0.7也算对"的盲区，模型没有动力让同类K线在 embedding 空间靠拢；
    # 交叉熵没有"差不多"，必须选边，逼 embedding 出现可线性读出的
    # 类别可分结构（同类聚拢/异类分开）——下游 attention 消费点积
    # 相似度，此几何直接可用。
    # 注意与 VISReg 相克：可分性在 z 分布上产生多峰，高斯形状项拉单峰，
    # 权重别开大（默认 0.1：4头平均CE≈0.5~0.7，加权贡献≈0.05~0.07，
    # 与另三项 O(1) 量级相当但偏轻）。
    CLS_ENABLED = True                   # 粗头总开关，False=退回纯回归监督
    CLS_WEIGHT = 0.1                     # 粗头分类权重（4头 CE 等权平均后 × 该权重）
    # "平"类区间：|v| ≤ ε 判平（ε 为每个头独立的分位数阈值，见
    # pretrain_embedding.py:compute_cls_stats）。不用"恰好=0"（连续分布
    # 上测度为零、平类退化）；用区间保证三类都非空。
    CLS_FLAT_PCT = 15.0                  # 每个头 |v| 最小的百分之几判"平"
                                         # （在 pre_sampled 池上预计算，确定性）

    # ---- 多粒度细头（粗头之上的分辨率层） ----
    # 3 类粗头只有"分得开"的压力，类内几何是平的（涨0.5%与涨9.8%标签同为
    # "涨"，CE 施加完全相同的拉力）；细头把同一判别值等频分桶，在桶粒度上
    # 继续施加同类聚拢压力，给类内几何加分辨率（Kronos 粗细token思想的
    # 判别式移植）。等频分桶天然类平衡；训练后期仍有梯度（3类头 Acc 平台
    # 化后细头继续供压）。注意 CE 不保证"相邻桶更近"（桶被当无序类别），
    # 相邻结构只是 v 连续性的副作用，要严格有序需 CORAL 类损失（未采用）。
    CLS_FINE_ENABLED = True              # 细头开关（仅 direction/vol_price_align 两族）
    CLS_FINE_BUCKETS = 5                 # 桶数（大涨/小涨/平/小跌/大跌；vws 细版与
                                         # dir/vpa 强相关属冗余不设，drv 新锚点先粗验证）
    CLS_FINE_WEIGHT = 0.1                # 细头 CE 权重（ln5≈1.61 起步 vs 粗头 ln3≈1.1，
                                         # 同乘 0.1 后量级相当）

    # ---- CKA 几何监控（Kornblith et al., ICML 2019） ----
    # 逐 epoch 计算 CKA(z, 4粗头标签核)：z 的 Gram 矩阵与"同类比例"标签核
    # 的中心化相关——衡量类别成对几何与标签的一致性（"z 里挨得近的样本对
    # 是否同类"）。probe B/C 测信息轴（单样本线性读出），CKA 测几何轴
    # （样本对相对位置），两者正交。用于决策而非仅监控：
    # 高=类别几何已饱和（SupCon 无油水），低=有空间（SupCon 该上的证据）。
    CKA_LOG_ENABLED = True               # CKA 逐 epoch 日志开关

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

    # 训练批处理
    BATCH_SIZE = 512                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 120            # 每轮训练的批次数（调低以适配时间序采样）

    # 优化器选择（字符串，互斥）
    # 'adamw':    标准AdamW
    # 'lion':     Lion（符号动量），内存省、泛化好（lion-pytorch库）
    # 'muon':     Muon（Newton-Schulz正交化），收敛快（KellerJordan/Muon库）
    OPTIMIZER_TYPE = 'adamw'

    # 自动混合精度（AMP）
    USE_AMP = True                   # 启用BF16自动混合精度（矩阵乘法用BF16，归一化/激活/损失保持FP32）
    # 训练和推理均受此开关控制，sigmoid始终在FP32下执行，不影响选股排名精度

    # 通用优化器参数
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # AdamW 参数（OPTIMIZER_TYPE='adamw'时生效）
    ADAMW_LR = 0.001                 # AdamW 学习率
    ADAMW_WEIGHT_DECAY = 0.1         # AdamW 权重衰减

    # Lion 参数（OPTIMIZER_TYPE='lion'时生效）
    LION_LR = 0.0003                # Lion 学习率（AdamW的~1/3，论文推荐1/3~1/10）
    LION_WEIGHT_DECAY = 0.01        # Lion 权重衰减（比AdamW大约1000倍，论文推荐1e-2量级）
    LION_BETAS = (0.9, 0.99)        # Lion 动量系数

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
        if opt == 'muon':
            return TrainingConfig.MUON_LR
        return TrainingConfig.ADAMW_LR

    @staticmethod
    def get_base_wd():
        """返回当前优化器对应的默认权重衰减"""
        opt = TrainingConfig.OPTIMIZER_TYPE.lower()
        if opt.startswith('lion'):
            return TrainingConfig.LION_WEIGHT_DECAY
        if opt == 'muon':
            return TrainingConfig.MUON_WEIGHT_DECAY
        return TrainingConfig.ADAMW_WEIGHT_DECAY

# ==================== 损失函数配置 ====================
class LossConfig:
    """损失函数相关配置"""
    # 'dynamic_bce':批权重动态平衡 | 'pairwise_bce':BCE+Pairwise排序 | 'balanced_bce':正负各半损失量纲不变
    LOSS_TYPE = 'dynamic_bce'

    POS_WEIGHT = 4.0  # DynamicWeightedBCE 的正样本权重

    # Pairwise排序损失配置（LOSS_TYPE='pairwise_bce'时生效）
    PAIRWISE_WEIGHT = 0.5           # Pairwise损失权重系数（总损失 = BCE + PAIRWISE_WEIGHT * Pairwise）
    PAIRWISE_TOP_K = 0.10           # Top K%预测区域（构建pair的样本范围）
    PAIRWISE_POS_WEIGHT = 2.0       # 正负对的额外权重（放大排序梯度）
    PAIRWISE_WARMUP_EPOCHS = 8      # 前N轮纯BCE训练，之后引入Pairwise
    PAIRWISE_SIGMA = 1.0            # RankNet温度参数（控制排序信号的锐度）
    PAIRWISE_NUM_NEG = 2            # 每个正样本配对的负样本数

# ==================== 用户自定义标签生成函数 ====================
def generate_label(day1_change, day2_change, day3_change):
    """
    此函数定义什么是"强势买入信号":返回 0=无强势信号,1=有强势信号

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

    optimizer_names = {'adamw': 'AdamW', 'lion': 'Lion', 'muon': 'Muon'}
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
