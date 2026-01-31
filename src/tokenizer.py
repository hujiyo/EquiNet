"""
Token化模块

将连续的股价特征离散化为token ID，用于后续的embedding查表。

Token化方案（每个特征独立token空间，范围与连续版一致）：
- open  (特征0): 20个token (ID: 0-19)，范围[-0.1, 0.1]，步长1%
- high  (特征1): 20个token (ID: 20-39)，范围[-0.1, 0.1]，步长1%
- low   (特征2): 20个token (ID: 40-59)，范围[-0.1, 0.1]，步长1%
- close (特征3): 20个token (ID: 60-79)，范围[-0.1, 0.1]，步长1%
- volume成交量变化 (特征4): 36个token (ID: 80-115)，范围[0, 1]，非均匀分桶
    - 区间一 [0, 0.6]: 20个桶，步长3%（原始-100%到+100%，步长10%）
    - 区间二 [0.6, 1.0]: 16个桶，步长2.5%（原始+100%到+500%，步长25%）
- exchange换手率 (特征5): 60个token (ID: 140-199)，范围[0, 1]，非均匀分桶
    - 区间一 [0, 0.2]: 40个桶，步长0.5%（原始值0-20%）
    - 区间二 [0.2, 1.0]: 20个桶，步长4%（原始值20-100%）
- rate量比 (特征6): 24个token (ID: 176-199)，范围[0, 1]，非均匀分桶
    - 区间一 [0, 0.12]: 12个桶，步长0.01（原始值0-1.2，步长0.1）
    - 区间二 [0.12, 0.24]: 4个桶，步长0.03（原始值1.2-2.4，步长0.3）
    - 区间三 [0.24, 1.0]: 8个桶，步长0.095（原始值2.4-10，步长0.95）

总词表大小: 200个token
输入: [seq_len, 7] 连续值
输出: [seq_len * 7] token ID (展平后)
"""

import numpy as np
import torch
from config import DataConfig, ModelConfig


# Token化配置
class TokenConfig:
    """Token化相关参数 - 每个特征独立设计"""
    
    # ========== 特征0-3: OHLC价格特征 ==========
    # 范围[-0.1, 0.1]对应±10%涨跌幅，20个桶，步长1%
    OHLC_NUM_BUCKETS = 20
    OHLC_MIN = -0.1
    OHLC_MAX = 0.1
    # 每个OHLC特征独立的token空间
    OPEN_OFFSET = 0      # token 0-19
    HIGH_OFFSET = 20     # token 20-39
    LOW_OFFSET = 40      # token 40-59
    CLOSE_OFFSET = 60    # token 60-79
    
    # ========== 特征4: volume成交量变化（非均匀分桶，两段式） ==========
    # 归一化后范围[0, 1]，其中0.5对应0%变化
    # 原始相对变化：(V_t - V_{t-1}) / V_{t-1}，clip到[-5, 5]后映射为 x/10+0.5
    # 区间一[0, 0.6]: 20个桶，步长3%（原始-100%到+100%，步长10%）
    # 区间二[0.6, 1.0]: 16个桶，步长2.5%（原始+100%到+500%，步长25%）
    VOLUME_NUM_BUCKETS = 36  # 20 + 16
    VOLUME_MIN = 0.0
    VOLUME_MAX = 1.0
    VOLUME_ZONE1_MAX = 0.6   # 区间一边界（原始+100%变化）
    VOLUME_ZONE1_BUCKETS = 20  # 区间一桶数（-100%到+100%，步长10%）
    VOLUME_ZONE2_BUCKETS = 16  # 区间二桶数（+100%到+500%，步长25%）
    VOLUME_OFFSET = 80   # token 80-115
    
    # ========== 特征5: exchange换手率（非均匀分桶） ==========
    # 原始值0-100%，归一化后0-1，与连续版范围一致
    # 区间一[0, 0.2]: 40个桶，步长0.5%（原始值0-20%）
    # 区间二[0.2, 1.0]: 20个桶，步长4%（原始值20-100%）
    EXCHANGE_NUM_BUCKETS = 60  # 40 + 20
    EXCHANGE_MIN = 0.0
    EXCHANGE_MAX = 1.0   # 与连续版一致
    EXCHANGE_ZONE1_MAX = 0.2   # 区间一边界（原始值20%）
    EXCHANGE_ZONE1_BUCKETS = 40  # 区间一桶数（0-20%，步长0.5%）
    EXCHANGE_ZONE2_BUCKETS = 20  # 区间二桶数（20-100%，步长4%）
    EXCHANGE_OFFSET = 116  # token 116-175
    
    # ========== 特征6: rate量比（非均匀分桶，三段式） ==========
    # 原始值0-10，归一化后0-1，与连续版范围一致
    # 区间一[0, 0.12]: 12个桶，步长0.01（原始值0-1.2，步长0.1）
    # 区间二[0.12, 0.24]: 4个桶，步长0.03（原始值1.2-2.4，步长0.3）
    # 区间三[0.24, 1.0]: 8个桶，步长0.095（原始值2.4-10，步长0.95）
    RATE_NUM_BUCKETS = 24  # 12 + 4 + 8
    RATE_MIN = 0.0
    RATE_MAX = 1.0       # 与连续版一致
    RATE_ZONE1_MAX = 0.12   # 区间一边界（原始值1.2）
    RATE_ZONE2_MAX = 0.24   # 区间二边界（原始值2.4）
    RATE_ZONE1_BUCKETS = 12  # 区间一桶数（0-1.2，步长0.1）
    RATE_ZONE2_BUCKETS = 4   # 区间二桶数（1.2-2.4，步长0.3）
    RATE_ZONE3_BUCKETS = 8   # 区间三桶数（2.4-10，步长0.95）
    RATE_OFFSET = 176    # token 176-199
    
    # ========== 汇总 ==========
    # 总词表大小 = 4*20 + 60 + 60 + 34 = 234
    VOCAB_SIZE = (4 * OHLC_NUM_BUCKETS + VOLUME_NUM_BUCKETS + 
                  EXCHANGE_NUM_BUCKETS + RATE_NUM_BUCKETS)  # 234个token
    
    # token化后的序列长度
    TOKEN_SEQ_LEN = DataConfig.CONTEXT_LENGTH * ModelConfig.INPUT_DIM  # 60 * 7 = 420


def _value_to_bucket(value: float, min_val: float, max_val: float, num_buckets: int) -> int:
    """
    将连续值映射到桶索引
    
    Args:
        value: 连续值
        min_val: 最小值（超出截断）
        max_val: 最大值（超出截断）
        num_buckets: 桶数量
    
    Returns:
        桶索引 [0, num_buckets-1]
    """
    # 截断到范围内
    value = max(min_val, min(max_val, value))
    
    # 归一化到[0, 1)，注意边界情况
    normalized = (value - min_val) / (max_val - min_val)
    
    # 映射到桶索引：[0, num_buckets)
    # 使用floor确保一致性，并限制在有效范围
    bucket = int(normalized * num_buckets)
    
    # 双重保护：确保不超出范围
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    if bucket < 0:
        bucket = 0
    
    return bucket


def _bucket_to_value(bucket: int, min_val: float, max_val: float, num_buckets: int) -> float:
    """
    将桶索引映射回连续值（桶中心值）
    
    Args:
        bucket: 桶索引
        min_val: 最小值
        max_val: 最大值
        num_buckets: 桶数量
    
    Returns:
        桶中心对应的连续值
    """
    bucket_width = (max_val - min_val) / num_buckets
    return min_val + (bucket + 0.5) * bucket_width


def _nonuniform_bucket_volume(value: float) -> int:
    """
    volume成交量变化的非均匀分桶（两段式）
    范围[0, 1]，其中0.5对应0%变化
    - 区间一[0, 0.6]: 20个桶，步长3%（原始-100%到+100%，步长10%）
    - 区间二[0.6, 1.0]: 16个桶，步长2.5%（原始+100%到+500%，步长25%）
    """
    value = max(0.0, min(1.0, value))  # 截断到[0, 1]

    if value <= TokenConfig.VOLUME_ZONE1_MAX:  # 区间一 [0, 0.6]
        # 20个桶覆盖[0, 0.6]
        normalized = value / TokenConfig.VOLUME_ZONE1_MAX
        bucket = int(normalized * TokenConfig.VOLUME_ZONE1_BUCKETS)
        bucket = min(bucket, TokenConfig.VOLUME_ZONE1_BUCKETS - 1)
    else:  # 区间二 [0.6, 1.0]
        # 16个桶覆盖[0.6, 1.0]
        normalized = (value - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX)
        bucket = TokenConfig.VOLUME_ZONE1_BUCKETS + int(normalized * TokenConfig.VOLUME_ZONE2_BUCKETS)
        bucket = min(bucket, TokenConfig.VOLUME_NUM_BUCKETS - 1)

    return bucket


def _nonuniform_bucket_exchange(value: float) -> int:
    """
    exchange换手率的非均匀分桶（两段式）
    范围[0, 1]，与连续版一致
    - 区间一[0, 0.2]: 40个桶，步长0.5%（原始值0-20%）
    - 区间二[0.2, 1.0]: 20个桶，步长4%（原始值20-100%）
    """
    value = max(0.0, min(1.0, value))  # 截断到[0, 1]
    
    if value <= TokenConfig.EXCHANGE_ZONE1_MAX:  # 区间一 [0, 0.2]
        # 40个桶覆盖[0, 0.2]
        normalized = value / TokenConfig.EXCHANGE_ZONE1_MAX
        bucket = int(normalized * TokenConfig.EXCHANGE_ZONE1_BUCKETS)
        bucket = min(bucket, TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1)
    else:  # 区间二 [0.2, 1.0]
        # 20个桶覆盖[0.2, 1.0]
        normalized = (value - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX)
        bucket = TokenConfig.EXCHANGE_ZONE1_BUCKETS + int(normalized * TokenConfig.EXCHANGE_ZONE2_BUCKETS)
        bucket = min(bucket, TokenConfig.EXCHANGE_NUM_BUCKETS - 1)
    
    return bucket


def _nonuniform_bucket_rate(value: float) -> int:
    """
    rate量比的非均匀分桶（三段式）
    范围[0, 1]，与连续版一致
    - 区间一[0, 0.12]: 12个桶，步长0.01（原始值0-1.2，步长0.1）
    - 区间二[0.12, 0.24]: 4个桶，步长0.03（原始值1.2-2.4，步长0.3）
    - 区间三[0.24, 1.0]: 8个桶，步长0.095（原始值2.4-10，步长0.95）
    """
    value = max(0.0, min(1.0, value))  # 截断到[0, 1]

    if value <= TokenConfig.RATE_ZONE1_MAX:  # 区间一 [0, 0.12]
        # 12个桶覆盖[0, 0.12]
        normalized = value / TokenConfig.RATE_ZONE1_MAX
        bucket = int(normalized * TokenConfig.RATE_ZONE1_BUCKETS)
        bucket = min(bucket, TokenConfig.RATE_ZONE1_BUCKETS - 1)
    elif value <= TokenConfig.RATE_ZONE2_MAX:  # 区间二 [0.12, 0.24]
        # 4个桶覆盖[0.12, 0.24]
        normalized = (value - TokenConfig.RATE_ZONE1_MAX) / (TokenConfig.RATE_ZONE2_MAX - TokenConfig.RATE_ZONE1_MAX)
        bucket = TokenConfig.RATE_ZONE1_BUCKETS + int(normalized * TokenConfig.RATE_ZONE2_BUCKETS)
        bucket = min(bucket, TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS - 1)
    else:  # 区间三 [0.24, 1.0]
        # 8个桶覆盖[0.24, 1.0]
        normalized = (value - TokenConfig.RATE_ZONE2_MAX) / (1.0 - TokenConfig.RATE_ZONE2_MAX)
        bucket = TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS + int(normalized * TokenConfig.RATE_ZONE3_BUCKETS)
        bucket = min(bucket, TokenConfig.RATE_NUM_BUCKETS - 1)

    return bucket


def tokenize_features(input_seq: np.ndarray, flatten: bool = True) -> np.ndarray:
    """
    将连续特征序列转换为token ID序列
    
    Args:
        input_seq: [seq_len, 7] 连续值数组（已预处理）
                   特征0-3: OHLC，范围[-0.1, 0.1]
                   特征4: volume，范围[0, 1]
                   特征5: exchange，范围[0, 1]（非均匀分桶）
                   特征6: rate，范围[0, 1]（非均匀分桶）
        flatten: 是否展平为一维数组
    
    Returns:
        token_ids: [seq_len * 7] 或 [seq_len, 7] token ID数组 (int64)
    """
    seq_len, num_features = input_seq.shape
    assert num_features == ModelConfig.INPUT_DIM, f"期望{ModelConfig.INPUT_DIM}个特征，实际{num_features}"
    
    token_ids = np.empty((seq_len, num_features), dtype=np.int64)
    
    for t in range(seq_len):
        # 特征0-3: OHLC（均匀分桶）
        ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET, 
                        TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
        for f in range(4):
            value = input_seq[t, f]
            bucket = _value_to_bucket(value, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS)
            token_ids[t, f] = ohlc_offsets[f] + bucket
        
        # 特征4: volume（非均匀分桶）
        value = input_seq[t, 4]
        bucket = _nonuniform_bucket_volume(value)
        token_ids[t, 4] = TokenConfig.VOLUME_OFFSET + bucket
        
        # 特征5: exchange（非均匀分桶）
        value = input_seq[t, 5]
        bucket = _nonuniform_bucket_exchange(value)
        token_ids[t, 5] = TokenConfig.EXCHANGE_OFFSET + bucket
        
        # 特征6: rate（非均匀分桶）
        value = input_seq[t, 6]
        bucket = _nonuniform_bucket_rate(value)
        token_ids[t, 6] = TokenConfig.RATE_OFFSET + bucket
    
    if flatten:
        return token_ids.reshape(-1)
    return token_ids


def tokenize_features_vectorized(input_seq: np.ndarray, flatten: bool = True) -> np.ndarray:
    """
    向量化版本：将连续特征序列转换为token ID序列
    
    Args:
        input_seq: [seq_len, 7] 连续值数组（已预处理）
        flatten: 是否展平为一维数组
    
    Returns:
        token_ids: [seq_len * 7] 或 [seq_len, 7] token ID数组 (int64)
    """
    seq_len, num_features = input_seq.shape
    token_ids = np.empty((seq_len, num_features), dtype=np.int64)
    
    # 特征0-3: OHLC，每个特征独立token空间（均匀分桶）
    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET, 
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = input_seq[:, i]
        clipped = np.clip(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = np.minimum((normalized * TokenConfig.OHLC_NUM_BUCKETS).astype(np.int64), 
                            TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, i] = buckets + ohlc_offsets[i]
    
    # 特征4: volume成交量变化（非均匀分桶，两段式，范围[0, 1]）
    vol = input_seq[:, 4]
    vol_clipped = np.clip(vol, 0.0, 1.0)
    vol_buckets = np.zeros(seq_len, dtype=np.int64)
    # 区间一 [0, 0.6]: 20个桶
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    # 区间二 [0.6, 1.0]: 16个桶
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX

    vol_buckets[zone1_mask] = np.minimum(
        (vol_clipped[zone1_mask] / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE1_BUCKETS - 1
    )
    vol_buckets[zone2_mask] = TokenConfig.VOLUME_ZONE1_BUCKETS + np.minimum(
        ((vol_clipped[zone2_mask] - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )
    token_ids[:, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET
    
    # 特征5: exchange换手率（非均匀分桶，两段式，范围[0, 1]）
    exc = input_seq[:, 5]
    exc_clipped = np.clip(exc, 0.0, 1.0)
    # 区间一 [0, 0.2]: 40个桶
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = np.zeros(seq_len, dtype=np.int64)
    # 区间一
    exc_buckets[zone1_mask] = np.minimum(
        (exc_clipped[zone1_mask] / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1
    )
    # 区间二 [0.2, 1.0]: 20个桶
    exc_buckets[~zone1_mask] = TokenConfig.EXCHANGE_ZONE1_BUCKETS + np.minimum(
        ((exc_clipped[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
    )
    token_ids[:, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET
    
    # 特征6: rate量比（非均匀分桶，三段式，范围[0, 1]）
    rate = input_seq[:, 6]
    rate_clipped = np.clip(rate, 0.0, 1.0)
    rate_buckets = np.zeros(seq_len, dtype=np.int64)
    # 区间一 [0, 0.15]: 15个桶
    zone1_mask = rate_clipped <= TokenConfig.RATE_ZONE1_MAX
    # 区间二 [0.15, 0.21]: 3个桶
    zone2_mask = (rate_clipped > TokenConfig.RATE_ZONE1_MAX) & (rate_clipped <= TokenConfig.RATE_ZONE2_MAX)
    # 区间三 [0.21, 1.0]: 16个桶
    zone3_mask = rate_clipped > TokenConfig.RATE_ZONE2_MAX
    
    rate_buckets[zone1_mask] = np.minimum(
        (rate_clipped[zone1_mask] / TokenConfig.RATE_ZONE1_MAX * TokenConfig.RATE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE1_BUCKETS - 1
    )
    rate_buckets[zone2_mask] = TokenConfig.RATE_ZONE1_BUCKETS + np.minimum(
        ((rate_clipped[zone2_mask] - TokenConfig.RATE_ZONE1_MAX) / (TokenConfig.RATE_ZONE2_MAX - TokenConfig.RATE_ZONE1_MAX) * TokenConfig.RATE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE2_BUCKETS - 1
    )
    rate_buckets[zone3_mask] = TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS + np.minimum(
        ((rate_clipped[zone3_mask] - TokenConfig.RATE_ZONE2_MAX) / (1.0 - TokenConfig.RATE_ZONE2_MAX) * TokenConfig.RATE_ZONE3_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE3_BUCKETS - 1
    )
    token_ids[:, 6] = rate_buckets + TokenConfig.RATE_OFFSET
    
    if flatten:
        return token_ids.reshape(-1)  # [seq_len * 7]
    return token_ids  # [seq_len, 7]


def tokenize_batch(batch_input: np.ndarray, flatten: bool = True) -> np.ndarray:
    """
    批量token化
    
    Args:
        batch_input: [batch_size, seq_len, 7] 连续值数组
        flatten: 是否展平最后两维
    
    Returns:
        token_ids: [batch_size, seq_len * 7] 或 [batch_size, seq_len, 7] token ID数组
    """
    batch_size, seq_len, num_features = batch_input.shape
    token_ids = np.empty((batch_size, seq_len, num_features), dtype=np.int64)
    
    # 特征0-3: OHLC，每个特征独立token空间
    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET, 
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = batch_input[:, :, i]
        clipped = np.clip(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = np.minimum((normalized * TokenConfig.OHLC_NUM_BUCKETS).astype(np.int64), 
                            TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, :, i] = buckets + ohlc_offsets[i]
    
    # 特征4: volume成交量变化（非均匀分桶，两段式，范围[0, 1]）
    vol = batch_input[:, :, 4]
    vol_clipped = np.clip(vol, 0.0, 1.0)
    vol_buckets = np.zeros((batch_size, seq_len), dtype=np.int64)
    # 区间一 [0, 0.6]: 20个桶
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    # 区间二 [0.6, 1.0]: 16个桶
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX

    vol_buckets[zone1_mask] = np.minimum(
        (vol_clipped[zone1_mask] / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE1_BUCKETS - 1
    )
    vol_buckets[zone2_mask] = TokenConfig.VOLUME_ZONE1_BUCKETS + np.minimum(
        ((vol_clipped[zone2_mask] - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )
    token_ids[:, :, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET
    
    # 特征5: exchange换手率（非均匀分桶，两段式，范围[0, 1]）
    exc = batch_input[:, :, 5]
    exc_clipped = np.clip(exc, 0.0, 1.0)
    # 区间一 [0, 0.2]: 40个桶
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = np.zeros((batch_size, seq_len), dtype=np.int64)
    exc_buckets[zone1_mask] = np.minimum(
        (exc_clipped[zone1_mask] / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1
    )
    # 区间二 [0.2, 1.0]: 20个桶
    exc_buckets[~zone1_mask] = TokenConfig.EXCHANGE_ZONE1_BUCKETS + np.minimum(
        ((exc_clipped[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
    )
    token_ids[:, :, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET
    
    # 特征6: rate量比（非均匀分桶，三段式，范围[0, 1]）
    rate = batch_input[:, :, 6]
    rate_clipped = np.clip(rate, 0.0, 1.0)
    rate_buckets = np.zeros((batch_size, seq_len), dtype=np.int64)
    # 区间一 [0, 0.15]: 15个桶
    zone1_mask = rate_clipped <= TokenConfig.RATE_ZONE1_MAX
    # 区间二 [0.15, 0.21]: 3个桶
    zone2_mask = (rate_clipped > TokenConfig.RATE_ZONE1_MAX) & (rate_clipped <= TokenConfig.RATE_ZONE2_MAX)
    # 区间三 [0.21, 1.0]: 16个桶
    zone3_mask = rate_clipped > TokenConfig.RATE_ZONE2_MAX
    
    rate_buckets[zone1_mask] = np.minimum(
        (rate_clipped[zone1_mask] / TokenConfig.RATE_ZONE1_MAX * TokenConfig.RATE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE1_BUCKETS - 1
    )
    rate_buckets[zone2_mask] = TokenConfig.RATE_ZONE1_BUCKETS + np.minimum(
        ((rate_clipped[zone2_mask] - TokenConfig.RATE_ZONE1_MAX) / (TokenConfig.RATE_ZONE2_MAX - TokenConfig.RATE_ZONE1_MAX) * TokenConfig.RATE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE2_BUCKETS - 1
    )
    rate_buckets[zone3_mask] = TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS + np.minimum(
        ((rate_clipped[zone3_mask] - TokenConfig.RATE_ZONE2_MAX) / (1.0 - TokenConfig.RATE_ZONE2_MAX) * TokenConfig.RATE_ZONE3_BUCKETS).astype(np.int64),
        TokenConfig.RATE_ZONE3_BUCKETS - 1
    )
    token_ids[:, :, 6] = rate_buckets + TokenConfig.RATE_OFFSET
    
    if flatten:
        return token_ids.reshape(batch_size, -1)  # [batch, seq_len * 7]
    return token_ids  # [batch, seq_len, 7]


def tokenize_batch_torch(batch_input: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    PyTorch版本的批量token化（可用于GPU加速）

    重要：本函数强制使用 FP32 进行离散化计算，避免 BF16 精度不足导致的桶边界抖动。
    BF16 只有 7 位有效数字，在桶边界附近的值会因为精度损失随机跳到相邻的 token。

    Args:
        batch_input: [batch_size, seq_len, 7] 连续值张量（任意精度）
        flatten: 是否展平最后两维

    Returns:
        token_ids: [batch_size, seq_len * 7] 或 [batch_size, seq_len, 7] token ID张量 (long)
    """
    batch_size, seq_len, num_features = batch_input.shape
    device = batch_input.device

    # 初始化输出张量
    token_ids = torch.empty((batch_size, seq_len, num_features), dtype=torch.long, device=device)

    # 关键修复：强制转换为 FP32 进行离散化计算，避免 BF16 精度问题
    # BF16 只有 7 位有效数字，在桶边界附近的值会因为精度损失随机跳到相邻的 token
    if batch_input.dtype != torch.float32:
        batch_input = batch_input.to(torch.float32)

    # 特征0-3: OHLC，每个特征独立token空间
    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = batch_input[:, :, i]
        clipped = torch.clamp(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = torch.clamp((normalized * TokenConfig.OHLC_NUM_BUCKETS).long(),
                             max=TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, :, i] = buckets + ohlc_offsets[i]
    
    # 特征4: volume成交量变化（非均匀分桶，两段式，范围[0, 1]）
    vol = batch_input[:, :, 4]
    vol_clipped = torch.clamp(vol, 0.0, 1.0)
    # 区间一 [0, 0.6]: 20个桶
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    # 区间二 [0.6, 1.0]: 16个桶
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX

    # 计算各区间的桶索引
    zone1_buckets = torch.clamp((vol_clipped / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).long(),
                                max=TokenConfig.VOLUME_ZONE1_BUCKETS - 1)
    zone2_buckets = TokenConfig.VOLUME_ZONE1_BUCKETS + torch.clamp(
        ((vol_clipped - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).long(),
        max=TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )

    # 使用where选择正确的桶
    vol_buckets = torch.where(zone1_mask, zone1_buckets, zone2_buckets)
    token_ids[:, :, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET
    
    # 特征5: exchange换手率（非均匀分桶，两段式，范围[0, 1]）
    exc = batch_input[:, :, 5]
    exc_clipped = torch.clamp(exc, 0.0, 1.0)
    # 区间一 [0, 0.2]: 40个桶
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = torch.where(
        zone1_mask,
        torch.clamp((exc_clipped / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).long(),
                   max=TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS + torch.clamp(
            ((exc_clipped - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).long(),
            max=TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
        )
    )
    token_ids[:, :, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET
    
    # 特征6: rate量比（非均匀分桶，三段式，范围[0, 1]）
    rate = batch_input[:, :, 6]
    rate_clipped = torch.clamp(rate, 0.0, 1.0)
    # 区间一 [0, 0.15]: 15个桶
    zone1_mask = rate_clipped <= TokenConfig.RATE_ZONE1_MAX
    # 区间二 [0.15, 0.21]: 3个桶
    zone2_mask = (rate_clipped > TokenConfig.RATE_ZONE1_MAX) & (rate_clipped <= TokenConfig.RATE_ZONE2_MAX)
    # 区间三 [0.21, 1.0]: 16个桶
    
    # 计算各区间的桶索引
    zone1_buckets = torch.clamp((rate_clipped / TokenConfig.RATE_ZONE1_MAX * TokenConfig.RATE_ZONE1_BUCKETS).long(),
                                max=TokenConfig.RATE_ZONE1_BUCKETS - 1)
    zone2_buckets = TokenConfig.RATE_ZONE1_BUCKETS + torch.clamp(
        ((rate_clipped - TokenConfig.RATE_ZONE1_MAX) / (TokenConfig.RATE_ZONE2_MAX - TokenConfig.RATE_ZONE1_MAX) * TokenConfig.RATE_ZONE2_BUCKETS).long(),
        max=TokenConfig.RATE_ZONE2_BUCKETS - 1
    )
    zone3_buckets = TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS + torch.clamp(
        ((rate_clipped - TokenConfig.RATE_ZONE2_MAX) / (1.0 - TokenConfig.RATE_ZONE2_MAX) * TokenConfig.RATE_ZONE3_BUCKETS).long(),
        max=TokenConfig.RATE_ZONE3_BUCKETS - 1
    )
    
    # 使用嵌套where选择正确的桶
    rate_buckets = torch.where(zone1_mask, zone1_buckets,
                               torch.where(zone2_mask, zone2_buckets, zone3_buckets))
    token_ids[:, :, 6] = rate_buckets + TokenConfig.RATE_OFFSET
    
    if flatten:
        return token_ids.reshape(batch_size, -1)  # [batch, seq_len * 7]
    return token_ids  # [batch, seq_len, 7]


def get_token_info(token_id: int) -> dict:
    """
    获取token的详细信息（用于调试）
    
    Args:
        token_id: token ID
    
    Returns:
        包含特征索引、桶索引、值范围的字典
    """
    feature_names = ['open', 'high', 'low', 'close', 'volume', 'exchange', 'rate']
    
    # 根据token_id确定特征和桶索引
    if token_id < TokenConfig.HIGH_OFFSET:  # 0-19: open
        feature_idx = 0
        bucket_idx = token_id - TokenConfig.OPEN_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.LOW_OFFSET:  # 20-39: high
        feature_idx = 1
        bucket_idx = token_id - TokenConfig.HIGH_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.CLOSE_OFFSET:  # 40-59: low
        feature_idx = 2
        bucket_idx = token_id - TokenConfig.LOW_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.VOLUME_OFFSET:  # 60-79: close
        feature_idx = 3
        bucket_idx = token_id - TokenConfig.CLOSE_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.EXCHANGE_OFFSET:  # 80-115: volume（非均匀分桶，两段式）
        feature_idx = 4
        bucket_idx = token_id - TokenConfig.VOLUME_OFFSET
        if bucket_idx < TokenConfig.VOLUME_ZONE1_BUCKETS:  # 区间一 [0, 0.6]
            bucket_width = TokenConfig.VOLUME_ZONE1_MAX / TokenConfig.VOLUME_ZONE1_BUCKETS
            bucket_start = bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        else:  # 区间二 [0.6, 1.0]
            zone2_bucket_idx = bucket_idx - TokenConfig.VOLUME_ZONE1_BUCKETS
            bucket_width = (1.0 - TokenConfig.VOLUME_ZONE1_MAX) / TokenConfig.VOLUME_ZONE2_BUCKETS
            bucket_start = TokenConfig.VOLUME_ZONE1_MAX + zone2_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.RATE_OFFSET:  # 140-199: exchange（非均匀分桶，两段式）
        feature_idx = 5
        bucket_idx = token_id - TokenConfig.EXCHANGE_OFFSET
        if bucket_idx < TokenConfig.EXCHANGE_ZONE1_BUCKETS:  # 区间一 [0, 0.2]
            bucket_width = TokenConfig.EXCHANGE_ZONE1_MAX / TokenConfig.EXCHANGE_ZONE1_BUCKETS
            bucket_start = bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        else:  # 区间二 [0.2, 1.0]
            zone2_bucket_idx = bucket_idx - TokenConfig.EXCHANGE_ZONE1_BUCKETS
            bucket_width = (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) / TokenConfig.EXCHANGE_ZONE2_BUCKETS
            bucket_start = TokenConfig.EXCHANGE_ZONE1_MAX + zone2_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
    else:  # 200-233: rate（非均匀分桶，三段式）
        feature_idx = 6
        bucket_idx = token_id - TokenConfig.RATE_OFFSET
        if bucket_idx < TokenConfig.RATE_ZONE1_BUCKETS:  # 区间一 [0, 0.15]
            bucket_width = TokenConfig.RATE_ZONE1_MAX / TokenConfig.RATE_ZONE1_BUCKETS
            bucket_start = bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        elif bucket_idx < TokenConfig.RATE_ZONE1_BUCKETS + TokenConfig.RATE_ZONE2_BUCKETS:  # 区间二 [0.15, 0.21]
            zone2_bucket_idx = bucket_idx - TokenConfig.RATE_ZONE1_BUCKETS
            bucket_width = (TokenConfig.RATE_ZONE2_MAX - TokenConfig.RATE_ZONE1_MAX) / TokenConfig.RATE_ZONE2_BUCKETS
            bucket_start = TokenConfig.RATE_ZONE1_MAX + zone2_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        else:  # 区间三 [0.21, 1.0]
            zone3_bucket_idx = bucket_idx - TokenConfig.RATE_ZONE1_BUCKETS - TokenConfig.RATE_ZONE2_BUCKETS
            bucket_width = (1.0 - TokenConfig.RATE_ZONE2_MAX) / TokenConfig.RATE_ZONE3_BUCKETS
            bucket_start = TokenConfig.RATE_ZONE2_MAX + zone3_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
    
    center_value = (bucket_start + bucket_end) / 2
    
    return {
        'token_id': token_id,
        'feature_idx': feature_idx,
        'feature_name': feature_names[feature_idx],
        'bucket_idx': bucket_idx,
        'bucket_range': (bucket_start, bucket_end),
        'center_value': center_value
    }


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("Token化模块测试")
    print("=" * 50)
    
    print(f"\n配置信息:")
    print(f"  OHLC桶数: {TokenConfig.OHLC_NUM_BUCKETS} (步长1%)")
    print(f"  Volume桶数: {TokenConfig.VOLUME_NUM_BUCKETS} (两段式: -100%到+100%用20桶步长10%, +100%到+500%用16桶步长25%)")
    print(f"  Exchange桶数: {TokenConfig.EXCHANGE_NUM_BUCKETS} (两段式: 0-20%用40桶步长0.5%, 20-100%用20桶步长4%)")
    print(f"  Rate桶数: {TokenConfig.RATE_NUM_BUCKETS} (三段式: 0-1.2用12桶步长0.1, 1.2-2.4用4桶步长0.3, 2.4-10用8桶步长0.95)")
    print(f"  总词表大小: {TokenConfig.VOCAB_SIZE}")
    print(f"  Token序列长度: {TokenConfig.TOKEN_SEQ_LEN}")
    
    # 创建测试数据（模拟实际数据分布，范围与连续版一致[0,1]）
    np.random.seed(42)
    seq_len = DataConfig.CONTEXT_LENGTH
    
    test_input = np.zeros((seq_len, 7), dtype=np.float32)
    test_input[:, :4] = np.random.uniform(-0.1, 0.1, (seq_len, 4))  # OHLC: [-10%, +10%]
    test_input[:, 4] = np.random.uniform(0, 1, seq_len)  # volume: [0, 1]
    test_input[:, 5] = np.random.uniform(0, 1, seq_len)  # exchange: [0, 1]（与连续版一致）
    test_input[:, 6] = np.random.uniform(0, 1, seq_len)  # rate: [0, 1]（与连续版一致）
    
    print(f"\n测试输入形状: {test_input.shape}")
    print(f"  OHLC范围: [{test_input[:, :4].min():.4f}, {test_input[:, :4].max():.4f}]")
    print(f"  Volume范围: [{test_input[:, 4].min():.4f}, {test_input[:, 4].max():.4f}]")
    print(f"  Exchange范围: [{test_input[:, 5].min():.4f}, {test_input[:, 5].max():.4f}]")
    print(f"  Rate范围: [{test_input[:, 6].min():.4f}, {test_input[:, 6].max():.4f}]")
    
    # 测试单样本token化
    token_ids = tokenize_features_vectorized(test_input)
    print(f"\nToken化结果形状: {token_ids.shape}")
    print(f"  Token ID范围: [{token_ids.min()}, {token_ids.max()}]")
    
    # 验证token分布
    print(f"\n前7个token (第1天的7个特征):")
    for i in range(7):
        info = get_token_info(token_ids[i])
        print(f"  {info['feature_name']:8s}: token={info['token_id']:3d}, "
              f"bucket={info['bucket_idx']:2d}, range=[{info['bucket_range'][0]:+.4f}, {info['bucket_range'][1]:+.4f}]")
    
    # 测试批量token化
    batch_size = 32
    batch_input = np.zeros((batch_size, seq_len, 7), dtype=np.float32)
    batch_input[:, :, :4] = np.random.uniform(-0.1, 0.1, (batch_size, seq_len, 4))
    batch_input[:, :, 4] = np.random.uniform(0, 1, (batch_size, seq_len))
    batch_input[:, :, 5] = np.random.uniform(0, 0.1, (batch_size, seq_len))
    batch_input[:, :, 6] = np.random.uniform(0.05, 0.2, (batch_size, seq_len))
    
    batch_tokens = tokenize_batch(batch_input)
    print(f"\n批量Token化结果形状: {batch_tokens.shape}")
    
    # 测试PyTorch版本
    batch_tensor = torch.from_numpy(batch_input)
    batch_tokens_torch = tokenize_batch_torch(batch_tensor)
    print(f"PyTorch Token化结果形状: {batch_tokens_torch.shape}")
    
    # 验证NumPy和PyTorch结果一致
    assert np.allclose(batch_tokens, batch_tokens_torch.numpy()), "NumPy和PyTorch结果不一致!"
    print("\n✓ NumPy和PyTorch结果一致")
    
    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
