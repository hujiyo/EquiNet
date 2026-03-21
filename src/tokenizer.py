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
- exchange换手率 (特征5): 60个token (ID: 116-175)，范围[0, 1]，非均匀分桶
    - 区间一 [0, 0.2]: 40个桶，步长0.5%（原始值0-20%）
    - 区间二 [0.2, 1.0]: 20个桶，步长4%（原始值20-100%）

总词表大小: 176个token
输入: [seq_len, 7] 连续值
输出: [seq_len * 7] token ID (展平后)
"""

import numpy as np
import torch
from config import DataConfig, ModelConfig


# Token化配置
class TokenConfig:
    OHLC_NUM_BUCKETS = 20
    OHLC_MIN = -0.1
    OHLC_MAX = 0.1
    OPEN_OFFSET = 0
    HIGH_OFFSET = 20
    LOW_OFFSET = 40
    CLOSE_OFFSET = 60

    VOLUME_NUM_BUCKETS = 36
    VOLUME_MIN = 0.0
    VOLUME_MAX = 1.0
    VOLUME_ZONE1_MAX = 0.6
    VOLUME_ZONE1_BUCKETS = 20
    VOLUME_ZONE2_BUCKETS = 16
    VOLUME_OFFSET = 80

    EXCHANGE_NUM_BUCKETS = 60
    EXCHANGE_MIN = 0.0
    EXCHANGE_MAX = 1.0
    EXCHANGE_ZONE1_MAX = 0.2
    EXCHANGE_ZONE1_BUCKETS = 40
    EXCHANGE_ZONE2_BUCKETS = 20
    EXCHANGE_OFFSET = 116

    INDEX_NUM_BUCKETS = 20
    INDEX_MIN = -0.1
    INDEX_MAX = 0.1
    INDEX_OFFSET = 176

    VOCAB_SIZE = (4 * OHLC_NUM_BUCKETS + VOLUME_NUM_BUCKETS +
                  EXCHANGE_NUM_BUCKETS + INDEX_NUM_BUCKETS)

    TOKEN_SEQ_LEN = DataConfig.CONTEXT_LENGTH * ModelConfig.INPUT_DIM


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


def tokenize_features(input_seq: np.ndarray, flatten: bool = True) -> np.ndarray:
    seq_len, num_features = input_seq.shape

    token_ids = np.empty((seq_len, num_features), dtype=np.int64)

    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for f in range(4):
        value = input_seq[:, f]
        clipped = np.clip(value, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = np.minimum((normalized * TokenConfig.OHLC_NUM_BUCKETS).astype(np.int64),
                            TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, f] = buckets + ohlc_offsets[f]

    vol = input_seq[:, 4]
    vol_clipped = np.clip(vol, 0.0, 1.0)
    vol_buckets = np.zeros(seq_len, dtype=np.int64)
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone1_mask] = np.minimum(
        (vol_clipped[zone1_mask] / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE1_BUCKETS - 1
    )
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone2_mask] = TokenConfig.VOLUME_ZONE1_BUCKETS + np.minimum(
        ((vol_clipped[zone2_mask] - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )
    token_ids[:, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET

    exc = input_seq[:, 5]
    exc_clipped = np.clip(exc, 0.0, 1.0)
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = np.zeros(seq_len, dtype=np.int64)
    exc_buckets[zone1_mask] = np.minimum(
        (exc_clipped[zone1_mask] / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1
    )
    exc_buckets[~zone1_mask] = TokenConfig.EXCHANGE_ZONE1_BUCKETS + np.minimum(
        ((exc_clipped[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
    )
    token_ids[:, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET

    idx = input_seq[:, 6]
    clipped = np.clip(idx, TokenConfig.INDEX_MIN, TokenConfig.INDEX_MAX)
    normalized = (clipped - TokenConfig.INDEX_MIN) / (TokenConfig.INDEX_MAX - TokenConfig.INDEX_MIN)
    buckets = np.minimum((normalized * TokenConfig.INDEX_NUM_BUCKETS).astype(np.int64),
                        TokenConfig.INDEX_NUM_BUCKETS - 1)
    token_ids[:, 6] = buckets + TokenConfig.INDEX_OFFSET

    if flatten:
        return token_ids.reshape(-1)
    return token_ids


def tokenize_features_vectorized(input_seq: np.ndarray, flatten: bool = True) -> np.ndarray:
    seq_len, num_features = input_seq.shape
    token_ids = np.empty((seq_len, num_features), dtype=np.int64)

    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = input_seq[:, i]
        clipped = np.clip(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = np.minimum((normalized * TokenConfig.OHLC_NUM_BUCKETS).astype(np.int64),
                            TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, i] = buckets + ohlc_offsets[i]

    vol = input_seq[:, 4]
    vol_clipped = np.clip(vol, 0.0, 1.0)
    vol_buckets = np.zeros(seq_len, dtype=np.int64)
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone1_mask] = np.minimum(
        (vol_clipped[zone1_mask] / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE1_BUCKETS - 1
    )
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone2_mask] = TokenConfig.VOLUME_ZONE1_BUCKETS + np.minimum(
        ((vol_clipped[zone2_mask] - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )
    token_ids[:, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET

    exc = input_seq[:, 5]
    exc_clipped = np.clip(exc, 0.0, 1.0)
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = np.zeros(seq_len, dtype=np.int64)
    exc_buckets[zone1_mask] = np.minimum(
        (exc_clipped[zone1_mask] / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1
    )
    exc_buckets[~zone1_mask] = TokenConfig.EXCHANGE_ZONE1_BUCKETS + np.minimum(
        ((exc_clipped[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
    )
    token_ids[:, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET

    idx = input_seq[:, 6]
    clipped = np.clip(idx, TokenConfig.INDEX_MIN, TokenConfig.INDEX_MAX)
    normalized = (clipped - TokenConfig.INDEX_MIN) / (TokenConfig.INDEX_MAX - TokenConfig.INDEX_MIN)
    buckets = np.minimum((normalized * TokenConfig.INDEX_NUM_BUCKETS).astype(np.int64),
                        TokenConfig.INDEX_NUM_BUCKETS - 1)
    token_ids[:, 6] = buckets + TokenConfig.INDEX_OFFSET

    if flatten:
        return token_ids.reshape(-1)
    return token_ids


def tokenize_batch(batch_input: np.ndarray, flatten: bool = True) -> np.ndarray:
    batch_size, seq_len, num_features = batch_input.shape
    token_ids = np.empty((batch_size, seq_len, num_features), dtype=np.int64)

    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = batch_input[:, :, i]
        clipped = np.clip(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = np.minimum((normalized * TokenConfig.OHLC_NUM_BUCKETS).astype(np.int64),
                            TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, :, i] = buckets + ohlc_offsets[i]

    vol = batch_input[:, :, 4]
    vol_clipped = np.clip(vol, 0.0, 1.0)
    vol_buckets = np.zeros((batch_size, seq_len), dtype=np.int64)
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone1_mask] = np.minimum(
        (vol_clipped[zone1_mask] / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE1_BUCKETS - 1
    )
    zone2_mask = vol_clipped > TokenConfig.VOLUME_ZONE1_MAX
    vol_buckets[zone2_mask] = TokenConfig.VOLUME_ZONE1_BUCKETS + np.minimum(
        ((vol_clipped[zone2_mask] - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )
    token_ids[:, :, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET

    exc = batch_input[:, :, 5]
    exc_clipped = np.clip(exc, 0.0, 1.0)
    zone1_mask = exc_clipped <= TokenConfig.EXCHANGE_ZONE1_MAX
    exc_buckets = np.zeros((batch_size, seq_len), dtype=np.int64)
    exc_buckets[zone1_mask] = np.minimum(
        (exc_clipped[zone1_mask] / TokenConfig.EXCHANGE_ZONE1_MAX * TokenConfig.EXCHANGE_ZONE1_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE1_BUCKETS - 1
    )
    exc_buckets[~zone1_mask] = TokenConfig.EXCHANGE_ZONE1_BUCKETS + np.minimum(
        ((exc_clipped[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_MAX) / (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) * TokenConfig.EXCHANGE_ZONE2_BUCKETS).astype(np.int64),
        TokenConfig.EXCHANGE_ZONE2_BUCKETS - 1
    )
    token_ids[:, :, 5] = exc_buckets + TokenConfig.EXCHANGE_OFFSET

    idx = batch_input[:, :, 6]
    clipped = np.clip(idx, TokenConfig.INDEX_MIN, TokenConfig.INDEX_MAX)
    normalized = (clipped - TokenConfig.INDEX_MIN) / (TokenConfig.INDEX_MAX - TokenConfig.INDEX_MIN)
    buckets = np.minimum((normalized * TokenConfig.INDEX_NUM_BUCKETS).astype(np.int64),
                        TokenConfig.INDEX_NUM_BUCKETS - 1)
    token_ids[:, :, 6] = buckets + TokenConfig.INDEX_OFFSET

    if flatten:
        return token_ids.reshape(batch_size, -1)
    return token_ids


def tokenize_batch_torch(batch_input: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    batch_size, seq_len, num_features = batch_input.shape
    device = batch_input.device

    token_ids = torch.empty((batch_size, seq_len, num_features), dtype=torch.long, device=device)

    if batch_input.dtype != torch.float32:
        batch_input = batch_input.to(torch.float32)

    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for i in range(4):
        col = batch_input[:, :, i]
        clipped = torch.clamp(col, TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX)
        normalized = (clipped - TokenConfig.OHLC_MIN) / (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN)
        buckets = torch.clamp((normalized * TokenConfig.OHLC_NUM_BUCKETS).long(),
                             max=TokenConfig.OHLC_NUM_BUCKETS - 1)
        token_ids[:, :, i] = buckets + ohlc_offsets[i]

    vol = batch_input[:, :, 4]
    vol_clipped = torch.clamp(vol, 0.0, 1.0)
    zone1_mask = vol_clipped <= TokenConfig.VOLUME_ZONE1_MAX

    zone1_buckets = torch.clamp((vol_clipped / TokenConfig.VOLUME_ZONE1_MAX * TokenConfig.VOLUME_ZONE1_BUCKETS).long(),
                                max=TokenConfig.VOLUME_ZONE1_BUCKETS - 1)
    zone2_buckets = TokenConfig.VOLUME_ZONE1_BUCKETS + torch.clamp(
        ((vol_clipped - TokenConfig.VOLUME_ZONE1_MAX) / (1.0 - TokenConfig.VOLUME_ZONE1_MAX) * TokenConfig.VOLUME_ZONE2_BUCKETS).long(),
        max=TokenConfig.VOLUME_ZONE2_BUCKETS - 1
    )

    vol_buckets = torch.where(zone1_mask, zone1_buckets, zone2_buckets)
    token_ids[:, :, 4] = vol_buckets + TokenConfig.VOLUME_OFFSET

    exc = batch_input[:, :, 5]
    exc_clipped = torch.clamp(exc, 0.0, 1.0)
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

    idx = batch_input[:, :, 6]
    clipped = torch.clamp(idx, TokenConfig.INDEX_MIN, TokenConfig.INDEX_MAX)
    normalized = (clipped - TokenConfig.INDEX_MIN) / (TokenConfig.INDEX_MAX - TokenConfig.INDEX_MIN)
    buckets = torch.clamp((normalized * TokenConfig.INDEX_NUM_BUCKETS).long(),
                         max=TokenConfig.INDEX_NUM_BUCKETS - 1)
    token_ids[:, :, 6] = buckets + TokenConfig.INDEX_OFFSET

    if flatten:
        return token_ids.reshape(batch_size, -1)
    return token_ids


def detokenize_features(token_ids: np.ndarray, unflatten: bool = True) -> np.ndarray:
    if token_ids.ndim == 1:
        if unflatten:
            seq_len = len(token_ids) // 7
            token_ids = token_ids.reshape(seq_len, 7)
        else:
            raise ValueError("一维token_ids需要unflatten=True")

    seq_len, num_features = token_ids.shape
    continuous_values = np.zeros((seq_len, num_features), dtype=np.float32)

    ohlc_offsets = [TokenConfig.OPEN_OFFSET, TokenConfig.HIGH_OFFSET,
                    TokenConfig.LOW_OFFSET, TokenConfig.CLOSE_OFFSET]
    for f in range(4):
        token_id_col = token_ids[:, f]
        bucket = token_id_col - ohlc_offsets[f]
        bucket_width = (TokenConfig.OHLC_MAX - TokenConfig.OHLC_MIN) / TokenConfig.OHLC_NUM_BUCKETS
        continuous_values[:, f] = TokenConfig.OHLC_MIN + (bucket + 0.5) * bucket_width

    vol_bucket = token_ids[:, 4] - TokenConfig.VOLUME_OFFSET
    zone1_mask = vol_bucket < TokenConfig.VOLUME_ZONE1_BUCKETS
    zone1_bucket_width = TokenConfig.VOLUME_ZONE1_MAX / TokenConfig.VOLUME_ZONE1_BUCKETS
    continuous_values[zone1_mask, 4] = (vol_bucket[zone1_mask] + 0.5) * zone1_bucket_width
    zone2_bucket = vol_bucket[~zone1_mask] - TokenConfig.VOLUME_ZONE1_BUCKETS
    zone2_bucket_width = (1.0 - TokenConfig.VOLUME_ZONE1_MAX) / TokenConfig.VOLUME_ZONE2_BUCKETS
    continuous_values[~zone1_mask, 4] = TokenConfig.VOLUME_ZONE1_MAX + (zone2_bucket + 0.5) * zone2_bucket_width

    exc_bucket = token_ids[:, 5] - TokenConfig.EXCHANGE_OFFSET
    zone1_mask = exc_bucket < TokenConfig.EXCHANGE_ZONE1_BUCKETS
    zone1_bucket_width = TokenConfig.EXCHANGE_ZONE1_MAX / TokenConfig.EXCHANGE_ZONE1_BUCKETS
    continuous_values[zone1_mask, 5] = (exc_bucket[zone1_mask] + 0.5) * zone1_bucket_width
    zone2_bucket = exc_bucket[~zone1_mask] - TokenConfig.EXCHANGE_ZONE1_BUCKETS
    zone2_bucket_width = (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) / TokenConfig.EXCHANGE_ZONE2_BUCKETS
    continuous_values[~zone1_mask, 5] = TokenConfig.EXCHANGE_ZONE1_MAX + (zone2_bucket + 0.5) * zone2_bucket_width

    idx_bucket = token_ids[:, 6] - TokenConfig.INDEX_OFFSET
    bucket_width = (TokenConfig.INDEX_MAX - TokenConfig.INDEX_MIN) / TokenConfig.INDEX_NUM_BUCKETS
    continuous_values[:, 6] = TokenConfig.INDEX_MIN + (idx_bucket + 0.5) * bucket_width

    return continuous_values


def get_token_info(token_id: int) -> dict:
    feature_names = ['open', 'high', 'low', 'close', 'volume', 'exchange', 'index']

    if token_id < TokenConfig.HIGH_OFFSET:
        feature_idx = 0
        bucket_idx = token_id - TokenConfig.OPEN_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.LOW_OFFSET:
        feature_idx = 1
        bucket_idx = token_id - TokenConfig.HIGH_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.CLOSE_OFFSET:
        feature_idx = 2
        bucket_idx = token_id - TokenConfig.LOW_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.VOLUME_OFFSET:
        feature_idx = 3
        bucket_idx = token_id - TokenConfig.CLOSE_OFFSET
        min_val, max_val, num_buckets = TokenConfig.OHLC_MIN, TokenConfig.OHLC_MAX, TokenConfig.OHLC_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.EXCHANGE_OFFSET:
        feature_idx = 4
        bucket_idx = token_id - TokenConfig.VOLUME_OFFSET
        if bucket_idx < TokenConfig.VOLUME_ZONE1_BUCKETS:
            bucket_width = TokenConfig.VOLUME_ZONE1_MAX / TokenConfig.VOLUME_ZONE1_BUCKETS
            bucket_start = bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        else:
            zone2_bucket_idx = bucket_idx - TokenConfig.VOLUME_ZONE1_BUCKETS
            bucket_width = (1.0 - TokenConfig.VOLUME_ZONE1_MAX) / TokenConfig.VOLUME_ZONE2_BUCKETS
            bucket_start = TokenConfig.VOLUME_ZONE1_MAX + zone2_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.INDEX_OFFSET:
        feature_idx = 5
        bucket_idx = token_id - TokenConfig.EXCHANGE_OFFSET
        if bucket_idx < TokenConfig.EXCHANGE_ZONE1_BUCKETS:
            bucket_width = TokenConfig.EXCHANGE_ZONE1_MAX / TokenConfig.EXCHANGE_ZONE1_BUCKETS
            bucket_start = bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
        else:
            zone2_bucket_idx = bucket_idx - TokenConfig.EXCHANGE_ZONE1_BUCKETS
            bucket_width = (1.0 - TokenConfig.EXCHANGE_ZONE1_MAX) / TokenConfig.EXCHANGE_ZONE2_BUCKETS
            bucket_start = TokenConfig.EXCHANGE_ZONE1_MAX + zone2_bucket_idx * bucket_width
            bucket_end = bucket_start + bucket_width
    elif token_id < TokenConfig.VOCAB_SIZE:
        feature_idx = 6
        bucket_idx = token_id - TokenConfig.INDEX_OFFSET
        min_val, max_val, num_buckets = TokenConfig.INDEX_MIN, TokenConfig.INDEX_MAX, TokenConfig.INDEX_NUM_BUCKETS
        bucket_width = (max_val - min_val) / num_buckets
        bucket_start = min_val + bucket_idx * bucket_width
        bucket_end = bucket_start + bucket_width
    else:
        raise ValueError(f"无效的token_id: {token_id}，超出词表范围[0, {TokenConfig.VOCAB_SIZE-1}]")

    center_value = (bucket_start + bucket_end) / 2

    return {
        'token_id': token_id,
        'feature_idx': feature_idx,
        'feature_name': feature_names[feature_idx],
        'bucket_idx': bucket_idx,
        'bucket_range': (bucket_start, bucket_end),
        'center_value': center_value
    }
