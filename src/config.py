"""
EquiNet 模型配置文件
"""

import os
import sys
import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# ==================== 数据契约 ====================
# 采样几何 / 样本列语义 / 股票池口径 / 特征基准 / 并行度上限的**唯一事实来源**
# 是 data_maintenance/contract.py。这里只做引用与断言，绝不复述数值。
# 原因：这些数曾经在 config.py 与 quality.py 各写一份，改动任一处会让污染区间
# 静默错位（且方向是「排漏」——脏数据照常进训练集，不报错）。
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_maintenance.contract import (  # noqa: E402
    SAMPLING, SAMPLE_COLUMNS, SAMPLE_SELECT_SQL, CLOSE_IDX, CLOSE_RAW_IDX,
    MODEL_INPUT_DIM, FEATURE_PRICE_COL, MAX_WORKERS, CURRENT_POOL,
    INFORMATIONAL_ISSUE_TYPES, is_exclusion_type,
    pool_join_sql, load_stock_codes, pool_scope_label,
)

# ==================== 数据参数 ====================
class DataConfig:
    """数据相关参数"""
    # SQLite 数据库配置
    DB_PATH = os.path.join(PROJECT_ROOT, 'data_maintenance', 'equinet.db')
    DB_BACKUP_DIR = os.path.join(PROJECT_ROOT, 'data_maintenance', 'backup')

    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'out')

    # 数据源配置
    DATA_SOURCE = 'baostock'  # 'baostock' 或 'akshare'
    # 池准入三项硬条件（市值区间/代码前缀）的定义处在 contract，消费方是
    # pit_pool.py 与 select.py —— 这里**不再转载**：曾写成
    # `MARKET_CAP_MAX = MARKET_CAP_MAX` 这种自赋值（RHS 落到模块全局，能跑，
    # 但极易被误读成 no-op、或被「清理」回字面量从而重新造成两处定义）。

    # 数据分割参数（按时间划分）
    #
    # 2026-09-27 定稿（hzy 确认），三段严格不重叠、角色不同：
    #   训练 2016-01-01 ~ 2022-12-31
    #   验证 2023-01-01 ~ 2025-12-31（每个 epoch 都评，选模/早停；约 730 个交易日）
    #   测试 2026-01-01 ~ 2026-07-31（训练结束后只评一次）
    # 历史教训：曾为「测试数据更好看」把测试窗口从三年缩到一年——
    # 那是用测试集挑测试集。窗口一旦定下，不得按结果回选。
    # 验证集/测试集角色曾配反过一次（2026-09-27 上午），已纠正：验证=选模，测试=终评。
    TRAIN_START_DATE = 20160101      # 训练集起始日期（含）
    TRAIN_END_DATE = 20221231        # 训练集截止日期（含）
    VAL_START_DATE = 20230101        # 验证集起始日期（含），用于训练时模型选择
    VAL_END_DATE = 20251231          # 验证集截止日期（含）
    TEST_START_DATE = 20260101       # 测试集起始日期（含）
    TEST_END_DATE = 20260731         # 测试集截止日期（含）；0 或 None = 到数据库最新日期
    RANDOM_SEED = 42                 # 随机种子

    # ⚠ 测试集窗口不得按结果回选。
    # 测试集=2026-最新 是数据可用范围决定的（2026-07-31 停更），不是按结果挑的。
    # 注意：该窗口仅约 136 个预测日，会触发 MIN_EVAL_DAYS 警告——这是已知且接受的
    # 取舍（验收口径与 2026H1 标定表一致），解读其收益数字时必须带上 ±1.39% 的噪声尺度。

    # 评估窗口下限（预测日数）。低于此值时，日均超额的标准误（约 ±0.5%）
    # 与要分辨的差异同量级，结论不可信。写进流程而不是文档，见 train.py。
    MIN_EVAL_DAYS = 250
    
    # 样本生成参数（定义处：data_maintenance/contract.py:SAMPLING）
    CONTEXT_LENGTH = SAMPLING.context_length   # 历史数据长度（核心参数）
    FUTURE_DAYS = SAMPLING.future_days         # 未来预测天数
    BUFFER_DAY = SAMPLING.buffer_days > 0      # 额外采集1天安全余量（用于跌停推迟判断）
    REQUIRED_LENGTH = SAMPLING.required_length
    # 污染半径：脏数据日 g 会污染的样本起始索引为 [p-48, p]（p = g 在序列中的位次）
    CONTAMINATE_LEFT = SAMPLING.label_backspan + SAMPLING.context_length - 1
    CONTAMINATE_RIGHT = 0

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

    # ========== 数据质量排除 ==========
    # 离线审计（python -m data_maintenance.audit --write）产出的 `data_issues`
    # 表记录「哪天有什么问题」（事实层）；训练加载时由
    # src/data.py:apply_quality_exclusions 用 contract.SAMPLING 把问题日
    # 换算成被污染的采样起始位置（决策层）。没有预先展开好的区间表 ——
    # 曾经的 sample_exclusion 就是那种副本，因单位错位漏排 74.5%，已删除。
    # 区间语义：问题日 g 在该股序列中的位次为 p 时，采样起始索引
    # s ∈ [p-48, p] 的样本被排除（45 天输入或 3+1 天标签包含 g）。
    #
    # True : 启用（推荐）。表不存在时自动跳过并打印提示，不影响训练
    # False: 忽略质量排除（退回旧行为）
    #
    # 重新生成问题日：python -m data_maintenance.audit --write
    EXCLUDE_DATA_ISSUE_SAMPLES = True

    # ========== 训练前自检 ==========
    # DataConfig.STRICT_SELFCHECK: 训练启动前跑 data_maintenance.selfcheck 的快速档，
    # 校验「数据前提」是否还成立（审计范围=训练池、派生状态未过期、派生表只有一份表达）。
    #   True : 发现 error 直接中止训练（推荐）—— 这些错误都会静默污染训练数据
    #   False: 只打印，继续训练
    # 单项检查失败但确定为无害时，改这里而不是删断言。
    STRICT_SELFCHECK = True

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
    INPUT_DIM = 19                   # 输入特征维度数（16 原始列 + wick_up/wick_dn/body_ratio 三列 K线形态占比）
    D_MODEL = 128                    # 模型维度（Transformer 内部维度）
    FFN_EXPAND_RATIO = 4             # FFN 隐藏层扩展比例（hidden_dim = d_model * FFN_EXPAND_RATIO）
    NHEAD = 4                        # 注意力头数
    NUM_LAYERS = 4                   # Transformer 层数
    # 2026-09-27：6 → 4。理由不是「更深更好」，而是相反 ——
    # 本任务真正的独立观测只有约 2000~7000 个交易日（相邻样本 context 重叠 44/45），
    # 6 层约 170 万参数相对这个样本量偏大。先做容量缩减实验，再谈加深。
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

# ==================== 表示正则参数 ====================
class RegularizationConfig:
    """单阶段训练中作用于 Embedding 的表示正则参数"""
    # 目标标准差：同时用于位置编码初始化与 VISReg 的尺度归一化，
    # 保证 embedding 的边际尺度与位置编码的初始尺度一致。
    TARGET_STD = 0.2

    # VISReg：主训练每个 step 独立采样单日 K 线，对其 embedding 施加
    # 尺度/形状(SWD)/中心化三项解耦正则；不参与任务判决，只约束表征分布。
    VISREG_MAIN_WEIGHT = 0.4         # λ：相对 BCE 的权重
    VISREG_AUX_BATCH_SIZE = 4096     # 每步独立采样的单日 K 线数
    VISREG_NUM_SLICES = 256          # 随机投影方向数 K
    VISREG_W_SCALE = 1.0             # 尺度项权重（逐维 std→1）
    VISREG_W_SHAPE = 0.6             # 形状项权重（SWD 对齐高斯分位数）
    VISREG_W_CENTER = 1.0            # 中心化项权重（批均值→0）

    # Embedding 与主干联合训练时的学习率比例（相对主干学习率）
    EMBED_LR_RATIO = 0.2

# ==================== 训练参数 ====================
class TrainingConfig:
    """训练相关参数"""
    # 训练批处理
    BATCH_SIZE = 512                 # GPU每次并行训练的样本数（增加批大小）
    BATCHES_PER_EPOCH = 120            # 每个测评周期内的参数更新步数

    # 基础训练参数
    EPOCHS = 30                     # 测评周期数（每个周期结束做一次验证集评估与选模）
    # 总步数 = EPOCHS × BATCHES_PER_EPOCH，学习率按"步"更新

    # ===== 两段式学习率调度（预热 → 退火），每一步更新一次 =====
    # 预热：LR_START 线性升到 LR_PEAK，占 LR_WARMUP_STEPS 步
    # 退火：LR_PEAK 按 LR_DECAY_SHAPE 降到 LR_FINAL，占 LR_DECAY_STEPS 步
    # 约束：LR_WARMUP_STEPS + LR_DECAY_STEPS == EPOCHS × BATCHES_PER_EPOCH
    LR_START = 1e-6                          # 初始学习率（预热起点，保证第一步也有更新）
    LR_WARMUP_STEPS = BATCHES_PER_EPOCH*1    # 预热步数
    LR_PEAK = 1e-4                           # 最大学习率（同时是优化器的基准学习率）
    LR_DECAY_STEPS = BATCHES_PER_EPOCH*29    # 退火步数
    LR_FINAL = 5e-6                          # 最终学习率（退火终点）
    LR_DECAY_SHAPE = 'cosine'                # 退火曲线：'cosine'（余弦）或 'linear'（线性）

    # 自动混合精度（AMP）
    USE_AMP = True                   # 启用BF16自动混合精度（矩阵乘法用BF16，归一化/激活/损失保持FP32）
    # 训练和推理均受此开关控制，sigmoid始终在FP32下执行，不影响选股排名精度

    # 通用优化器参数
    GRADIENT_CLIP_NORM = 1.0         # 梯度裁剪范数

    # AdamW 参数（唯一优化器）
    ADAMW_WEIGHT_DECAY = 0.1         # 权重衰减

    OPEN_EARLY_STOPPING = True        # 是否开启早停机制

    @staticmethod
    def get_base_lr():
        """优化器的基准学习率 = 学习率调度的最大学习率"""
        return TrainingConfig.LR_PEAK

    @staticmethod
    def total_steps():
        """总优化步数 = 测评周期数 × 每周期步数"""
        return TrainingConfig.EPOCHS * TrainingConfig.BATCHES_PER_EPOCH

    @staticmethod
    def validate_lr_schedule():
        """校验两段式学习率配置，配错时直接报错而不是悄悄跑偏"""
        cfg = TrainingConfig
        total = cfg.total_steps()
        planned = cfg.LR_WARMUP_STEPS + cfg.LR_DECAY_STEPS
        if planned != total:
            raise ValueError(
                f"学习率调度与总步数不一致: 预热 {cfg.LR_WARMUP_STEPS} + 退火 "
                f"{cfg.LR_DECAY_STEPS} = {planned}，但总步数 = EPOCHS "
                f"{cfg.EPOCHS} × BATCHES_PER_EPOCH {cfg.BATCHES_PER_EPOCH} = {total}"
            )
        if cfg.LR_DECAY_SHAPE not in ('cosine', 'linear'):
            raise ValueError(f"LR_DECAY_SHAPE 只支持 'cosine' / 'linear'，当前 {cfg.LR_DECAY_SHAPE!r}")
        if cfg.LR_START < 0:
            raise ValueError("LR_START 不能为负数（0 表示预热从 0 起步）")
        if min(cfg.LR_PEAK, cfg.LR_FINAL) <= 0:
            raise ValueError("LR_PEAK / LR_FINAL 必须为正数")
        if cfg.LR_START > cfg.LR_PEAK or cfg.LR_FINAL > cfg.LR_PEAK:
            raise ValueError("LR_START 与 LR_FINAL 都不能超过 LR_PEAK")
        return total

    @staticmethod
    def get_base_wd():
        """返回权重衰减"""
        return TrainingConfig.ADAMW_WEIGHT_DECAY

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


def _generate_label_batch(day1_change, day2_change, day3_change):
    """
    generate_label 的向量化版本（逐位等价：比较/加减均为元素级 IEEE 运算，
    NaN 与标量版行为一致——所有比较为 False → 标签 0）

    由 data._vectorized_process_stock 等批量路径通过 generate_label.batch 调用，
    消除百万级样本上逐样本 Python 函数调用的解释器开销。
    """
    day1 = np.asarray(day1_change)
    day2 = np.asarray(day2_change)
    day3 = np.asarray(day3_change)
    cum_2day = day1 + day2
    cum_3day = day1 + day2 + day3          # 左结合，与标量版求值顺序一致
    max_day = np.maximum(np.maximum(day1, day2), day3)
    cond = (
        ((day1 >= 0.05) & (cum_3day >= 0.03))
        | ((cum_2day >= 0.06) & (day1 > 0.01) & (day2 > 0.01) & (cum_3day >= 0.03))
        | ((day1 >= 0.01) & (day2 >= 0.01) & (day3 >= 0.01) & (cum_3day >= 0.06))
        | ((max_day >= 0.08) & (cum_3day >= 0.06) & (day1 >= -0.02))
        | ((cum_3day >= 0.08) & (day1 >= -0.02))
    )
    return cond.astype(np.float32)


# 向量化快路径挂载：data 侧 getattr(label_fn, 'batch') 探测使用，
# 自定义标签函数（无 .batch 属性）自动回退逐样本循环
generate_label.batch = _generate_label_batch

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


def _calculate_returns_batch(t1_open, t1_close, t2_open, t2_close, t3_close,
                             day1_change, day2_change, available_days):
    """
    calculate_returns 的向量化版本（逐位等价）

    与标量版的参数对应关系：
    - 标量版对 Day2/Day3 缺失传 None；此处传全量数组（无效位置为越界
      clip 出的垃圾值），由 available_days >= 2 / >= 3 掩码承担 None 语义，
      垃圾车道先安全化（置 1/0）再统一运算，杜绝垃圾值影响结果
    - day3_change 标量版未使用，此处同样不需要
    - day1_change 由调用方保证恒为数组（与标量调用点一致：从不为 None）

    Returns:
        (cumulative [N] float64, daily [N,3] float64 右侧以 0 填充, n_days [N] int)
        daily/n_days 组合还原出与标量版相同的变长逐日收益列表
    """
    day1_change = np.asarray(day1_change)
    day2_change = np.asarray(day2_change)
    n = len(day1_change)
    has2 = available_days >= 2
    has3 = available_days >= 3

    # 无效车道安全化：除法分母置 1、分子置 0，结果被掩码丢弃，仅避免
    # 垃圾值产生 inf/NaN 告警（errstate 双保险）
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_t1_open = np.where(has2, t1_open, 1.0)
        safe_t2_open = np.where(has2, t2_open, 1.0)
        safe_t2_close = np.where(has2, t2_close, 0.0)
        safe_t3_close = np.where(has3, t3_close, 0.0)

        day1_return = (t1_close - t1_open) / t1_open
        ret_stop1 = (safe_t2_open - t1_open) / safe_t1_open   # 止损1: 第二天开盘卖
        day2_return = (safe_t2_close - t1_close) / safe_t1_open
        day3_return = (safe_t3_close - safe_t2_close) / safe_t1_open

        # 止损条件（标量版的 None 判断 → 掩码化）：
        # day1_change 恒非 None；止损2/3 仅 has2（day2_change 非 None）时判断
        stop1 = day1_change <= -0.03
        stop1_exec = stop1 & has2                 # 止损1 且有 Day2 开盘 → 第二天开盘卖
        m2 = has2 & ~stop1                        # 走到 Day2 的车道
        stop2 = m2 & (day1_change + day2_change < -0.02)
        stop3 = m2 & (day1_change < 0.01) & (day2_change < 0.01)
        stop23 = stop2 | stop3                    # 第二天收盘止损（收益= d1+d2）
        full3 = m2 & ~stop23 & has3               # 持有满 3 天
        till2 = m2 & ~stop23 & ~has3              # 只有两天数据（收益= d1+d2）
        ret2_lanes = stop23 | till2               # 两天收益车道（均 ⊂ m2）

        two_day = day1_return + day2_return
        three_day = two_day + day3_return

        cumulative = day1_return.copy()
        cumulative[stop1_exec] = ret_stop1[stop1_exec]
        cumulative[ret2_lanes] = two_day[ret2_lanes]
        cumulative[full3] = three_day[full3]

        daily = np.zeros((n, 3), dtype=np.float64)
        daily[:, 0] = day1_return
        daily[stop1_exec, 0] = ret_stop1[stop1_exec]
        daily[m2, 1] = day2_return[m2]
        daily[full3, 2] = day3_return[full3]

        n_days = np.ones(n, dtype=np.int64)
        n_days[ret2_lanes] = 2
        n_days[full3] = 3

    return cumulative, daily, n_days


# 向量化快路径挂载（同 generate_label.batch）：data 侧探测使用，
# 自定义收益函数（无 .batch 属性）自动回退逐样本循环
calculate_returns.batch = _calculate_returns_batch


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

# ==================== 契约一致性断言 ====================
# 这些都是「两个地方各自定义、可能悄悄跑偏」的点。启动即校验，宁可启动失败
# 也不要带着错位的口径跑完一整个训练。
def _assert_contract() -> None:
    assert MODEL_INPUT_DIM == ModelConfig.INPUT_DIM, (
        f'契约 MODEL_INPUT_DIM={MODEL_INPUT_DIM} 与 '
        f'ModelConfig.INPUT_DIM={ModelConfig.INPUT_DIM} 不一致')
    assert SAMPLE_COLUMNS[CLOSE_IDX] == 'close', f'CLOSE_IDX={CLOSE_IDX} 不是 close'
    assert SAMPLE_COLUMNS[CLOSE_RAW_IDX] == 'close_raw', \
        f'CLOSE_RAW_IDX={CLOSE_RAW_IDX} 不是 close_raw'
    assert len(SAMPLE_COLUMNS) == 17, f'样本列应为 17（16 输入列 + close_raw），实际 {len(SAMPLE_COLUMNS)}'
    assert DataConfig.CONTAMINATE_LEFT == SAMPLING.label_backspan + SAMPLING.context_length - 1
    assert DataConfig.CONTAMINATE_RIGHT == 0, '污染区间右界就是问题日自身（序号空间）'


_assert_contract()


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

    total_steps = TrainingConfig.validate_lr_schedule()
    print(f"训练参数:")
    print(f"  测评周期数: {TrainingConfig.EPOCHS}")
    print(f"  总步数: {total_steps} ({TrainingConfig.EPOCHS} × {TrainingConfig.BATCHES_PER_EPOCH})")
    print(f"  学习率: {TrainingConfig.get_base_lr()}")
    print(f"  权重衰减: {TrainingConfig.get_base_wd()}")
    print(f"  批处理大小: {TrainingConfig.BATCH_SIZE}")
    print(f"  每周期步数: {TrainingConfig.BATCHES_PER_EPOCH}")
    print(f"  优化器: AdamW")
    print(f"  混合精度(AMP): {'BF16' if TrainingConfig.USE_AMP else '关闭(FP32)'}")
    print(f"  学习率调度(按步更新): 预热 {TrainingConfig.LR_WARMUP_STEPS} 步 "
          f"{TrainingConfig.LR_START:g}→{TrainingConfig.LR_PEAK:g}，"
          f"退火 {TrainingConfig.LR_DECAY_STEPS} 步 "
          f"{TrainingConfig.LR_PEAK:g}→{TrainingConfig.LR_FINAL:g}"
          f"（{TrainingConfig.LR_DECAY_SHAPE}）")
    print(f"  早停: {'开启' if TrainingConfig.OPEN_EARLY_STOPPING else '关闭'}")
    print(f"  训练模式: 单阶段端到端 (Embedding 不冻结)")
    print(f"  Embedding 学习率比例: {RegularizationConfig.EMBED_LR_RATIO}")
    print(f"  VISReg: λ={RegularizationConfig.VISREG_MAIN_WEIGHT}, "
          f"单日辅助batch={RegularizationConfig.VISREG_AUX_BATCH_SIZE}")

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
