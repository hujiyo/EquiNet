"""
衍生特征线性探针（对照实验）

验证预训练 embedding 是否真的把跨维非线性结构编码进了内部。
用三组误差对照，回答"D= 的下降到底发生在 embedding 里还是解码器里"：

1. 训练时探针 A：加载 best_embedding.pth 中的 embedding + 线性解码器
   （解码器即训练时的线性探针），在真实K线池上算逐特征掩码 MSE。
   应与训练日志的 D= 一致（数值口径校验）。

2. 事后探针 B：冻结 embedding，在 z 上重新拟合线性回归 → 衍生特征
   （在留出集上评估）。这是"embedding 此刻内部有多少结构可线性读出"
   的最干净度量，不依赖训练过程。

3. 线性拷贝下限 C：在原始19维特征上拟合线性回归 → 衍生特征
   （留出集评估）。这是"embedding 若只是19维线性拷贝"时探针能达到的
   最佳误差（sign/绝对值/除法/乘积都不是线性组合，下限必然>0）。

判定：B 显著低于 C（如 B/C < 0.5）→ 非线性结构确实被 embedding
编码成了可线性读出的方向，衍生任务机制成立；
B ≈ C → 结构不在 embedding 里，衍生任务没起作用。

用法：
  python src/probe_derived.py                          # 默认 300 万条抽样
  python src/probe_derived.py --checkpoint <路径> --n-samples 5000000
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from config import (DataConfig, DeviceConfig, EmbeddingConfig, ModelConfig)
from data import load_and_preprocess_data, FeatureNormalizer
from pretrain_embedding import (PretrainModel, collect_kline_data,
                                build_derived_targets, DERIVED_FEATURE_NAMES)

PROBE_CHUNK = 500_000


def masked_mse_per_feature(pred, target, mask):
    """逐特征掩码 MSE（与训练损失同口径：仅有效样本、逐特征归一）"""
    diff2 = (pred - target) ** 2
    num = (diff2 * mask).sum(axis=0)
    den = mask.sum(axis=0) + 1e-8
    return num / den


def fit_and_eval_linear(X_fit, y_fit, mask_fit, X_eval, y_eval, mask_eval):
    """逐特征拟合线性回归（拟合/评估均只用在掩码有效样本上），留出集评估"""
    n_feat = y_fit.shape[1]
    errs = np.zeros(n_feat, dtype=np.float64)
    for i in range(n_feat):
        m_fit = mask_fit[:, i] > 0
        m_eval = mask_eval[:, i] > 0
        if m_fit.sum() < 1000 or m_eval.sum() < 1000:
            errs[i] = np.nan
            continue
        reg = LinearRegression().fit(X_fit[m_fit], y_fit[m_fit, i])
        pred = reg.predict(X_eval[m_eval])
        errs[i] = np.mean((pred - y_eval[m_eval, i]) ** 2)
    return errs


def probe(checkpoint_path, train_stock_info, feature_normalizer,
          device, n_samples):
    print(f"\n{'='*70}")
    print(f"衍生特征线性探针: {os.path.basename(checkpoint_path)}")
    print(f"{'='*70}")

    # 1. 与预训练相同的路径重建 K线池（池只需覆盖抽样，3× 余量即可，
    #    避免为百万级抽样构建全量 1 亿池的内存/耗时；
    #    分位数/均值统计在百万级样本已收敛，对 B/C 比值无影响）
    kline_pool = collect_kline_data(
        train_stock_info, feature_normalizer, pool_cap=n_samples * 3)

    # 2. 抽样 + 衍生目标（按行独立计算，只对抽样子集构建）
    rng = np.random.default_rng(0)
    n = min(n_samples, len(kline_pool))
    probe_idx = rng.choice(len(kline_pool), n, replace=False)
    x = kline_pool[probe_idx]
    y, m_uint8 = build_derived_targets(x, feature_normalizer,
                                       chunk_size=PROBE_CHUNK)
    m = (m_uint8 > 0).astype(np.float64)
    print(f"\n[抽样] {n:,} 条K线 (池 {len(kline_pool):,})")

    # 3. 加载 checkpoint 中的 embedding + 训练时线性解码器
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'decoder_state_dict' not in ckpt:
        print("  ✗ checkpoint 缺少 decoder_state_dict（旧格式），无法做训练时探针")
        return
    n_derived = ckpt['decoder_state_dict']['weight'].shape[0] - ModelConfig.INPUT_DIM
    print(f"  checkpoint: n_derived={n_derived}")

    model = PretrainModel(
        input_dim=ModelConfig.INPUT_DIM,
        d_model=ModelConfig.D_MODEL,
        n_derived=n_derived,
    ).to(device)
    model.embedding.embed_proj.weight.data.copy_(ckpt['embed_proj_weight'])
    model.embedding.embed_mlp[0].weight.data.copy_(ckpt['embed_mlp_0_weight'])
    model.embedding.embed_mlp[2].weight.data.copy_(ckpt['embed_mlp_2_weight'])
    model.decoder.load_state_dict(ckpt['decoder_state_dict'])
    model.eval()

    # 4. 训练时探针 A：embedding+decoder 前向，逐特征掩码 MSE
    print(f"\n[训练时探针 A] 加载的 embedding+decoder (chunk={PROBE_CHUNK:,})")
    recon_all = []
    with torch.no_grad():
        for start in range(0, n, PROBE_CHUNK):
            end = min(start + PROBE_CHUNK, n)
            batch = torch.tensor(x[start:end], dtype=torch.float32).to(device)
            z, recon = model(batch)
            recon_all.append(recon.cpu().numpy())
    recon_all = np.concatenate(recon_all, axis=0)
    probe_A_derived = masked_mse_per_feature(
        recon_all[:, ModelConfig.INPUT_DIM:], y, m)
    probe_A_orig = np.mean((recon_all[:, :ModelConfig.INPUT_DIM] - x) ** 2)

    # 5. 事后探针 B + 线性拷贝下限 C：冻结 z / 原始 x 上重新拟合线性回归
    #    打乱后对半切分拟合/留出，保证可比性
    #    （池按股票顺序拼接，若直接取前半/后半，评估集股票在拟合集中缺席，
    #     B/C 同向偏高但幅度不同，B/C 比值有偏）
    perm = rng.permutation(n)
    half = n // 2
    fit_idx = perm[:half]
    eval_idx = perm[half:]
    z_all = []
    with torch.no_grad():
        for start in range(0, n, PROBE_CHUNK):
            end = min(start + PROBE_CHUNK, n)
            batch = torch.tensor(x[start:end], dtype=torch.float32).to(device)
            z_all.append(model.embedding(batch).cpu().numpy())
    z_all = np.concatenate(z_all, axis=0)

    print(f"  [事后探针 B] z 上重新拟合线性回归 (fit={half:,} / eval={n-half:,})")
    probe_B = fit_and_eval_linear(z_all[fit_idx], y[fit_idx], m[fit_idx],
                                  z_all[eval_idx], y[eval_idx], m[eval_idx])
    print(f"  [线性拷贝下限 C] 原始19维上拟合线性回归 (同上划分)")
    probe_C = fit_and_eval_linear(x[fit_idx], y[fit_idx], m[fit_idx],
                                  x[eval_idx], y[eval_idx], m[eval_idx])

    # 6. 汇总表
    print(f"\n{'='*100}")
    print(f"{'特征':>16s} {'有效%':>7s} {'探针A(训练)':>11s} "
          f"{'探针B(事后)':>11s} {'下限C(线性拷贝)':>14s} {'B/C':>6s}  判定")
    print(f"{'-'*100}")
    for i, name in enumerate(DERIVED_FEATURE_NAMES[:n_derived]):
        valid_pct = m[:, i].mean() * 100
        a, b, c = probe_A_derived[i], probe_B[i], probe_C[i]
        ratio = b / c if c > 1e-6 else np.nan
        if np.isnan(ratio):
            verdict = '样本不足'
        elif ratio < 0.5:
            verdict = '结构已编码'
        elif ratio < 0.8:
            verdict = '部分编码'
        else:
            verdict = '未编码≈线性拷贝'
        print(f"{name:>16s} {valid_pct:6.1f}% {a:11.4f} {b:11.4f} {c:14.4f} "
              f"{ratio:6.2f}  {verdict}")
    print(f"{'-'*100}")

    valid_feat = ~np.isnan(probe_B) & ~np.isnan(probe_C)
    if valid_feat.sum() > 0:
        mean_A = np.nanmean(probe_A_derived)
        mean_B = np.nanmean(probe_B[valid_feat])
        mean_C = np.nanmean(probe_C[valid_feat])
        print(f"汇总: 探针A={mean_A:.4f}  探针B={mean_B:.4f}  "
              f"线性拷贝下限C={mean_C:.4f}  (B/C={mean_B/max(mean_C,1e-6):.2f})")
    print(f"原始19维: 探针A={probe_A_orig:.6f}  (训练日志 O= 参考)")

    if valid_feat.sum() == 0:
        print("\n结论: 无法评估（有效样本不足）")
    elif np.nanmean(probe_B[valid_feat]) < 0.5 * np.nanmean(probe_C[valid_feat]):
        print("\n结论: 非线性结构显著编码在 embedding 内部，衍生任务机制成立。")
        print("      事后探针 B 与训练探针 A 同口径时，可直接用 B/C 比值向领导汇报。")
    else:
        print("\n结论: 结构未被 embedding 编码（B ≈ C），衍生任务未起预期作用。")
        print("      建议回到 MLP 解码器或换掩码重建路线。")


def main():
    parser = argparse.ArgumentParser(
        description='衍生特征线性探针（对照实验）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help=f'embedding 权重文件 (默认 {EmbeddingConfig.BEST_EMBEDDING_PATH})')
    parser.add_argument('--n-samples', type=int, default=3_000_000,
                        help='探针抽样条数 (默认 3,000,000)')
    args = parser.parse_args()

    checkpoint_path = (args.checkpoint
                       or os.path.join(EmbeddingConfig.OUTPUT_DIR,
                                       'best_embedding.pth'))
    device = DeviceConfig.get_device()

    print("[步骤1] 加载训练数据...")
    train_stock_info, _, _ = load_and_preprocess_data()

    print("\n[步骤2] 加载特征归一化器...")
    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        print("  归一化器不存在，先运行 python src/data.py 创建")
        sys.exit(1)

    probe(checkpoint_path, train_stock_info, feature_normalizer,
          device, args.n_samples)


if __name__ == "__main__":
    main()