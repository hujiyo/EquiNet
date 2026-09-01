"""
注意力机制诊断脚本

灵感来源: "Are Sixteen Heads Really Better than One?" (Michel et al., NeurIPS 2019)

实验一 · 逐头屏蔽:
  对模型中每个注意力头，逐个将其屏蔽（设 ξ_h = 0），
  计算屏蔽后模型在验证集上的 AUC 变化率和 Loss 变化率。
  - AUC 变化越小 → 该头越不重要（"偷懒"）
  - AUC 变化越大 → 该头越关键

实验二 · 整层屏蔽:
  将某一层的全部注意力头同时屏蔽，对比整层屏蔽 Δ% 与单头最大 Δ%：
  - 整层屏蔽 Δ% ≈ 0       → 🟡 层无用（整层空转）
  - 整层 Δ% >> 单头最大 Δ% → 🔴 头趋同（冗余但不空转）
  - 整层 Δ% ≈ 单头最大 Δ%  → 🟢 层有效，头有分工

实验三 · 位置编码:
  将可学习位置编码清零，测量 AUC 和 Loss 变化，验证模型是否真正利用了时序信息。

模型选择方式与 run.py 保持一致。
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from config import ModelConfig, DataConfig, DeviceConfig, LossConfig
from model import create_model
from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,
    FeatureNormalizer,
)
from training_utils import _get_amp_context, DynamicWeightedBCE, BalancedBCE
from run import list_available_models, select_model, load_model



def _mask_mha_head(mha, head_idx, head_dim):
    """
    屏蔽指定头（等效于移除该头对最终输出的全部贡献，头输出恒为 0）

    兼容两种实现：
    - 对于nn.MultiheadAttention（标准库，如 AttentionPooling.cross_attn）：
      in_proj_weight 中该头的 Q/K/V 行置零。
    - 对于自定义 MultiHeadAttention（src/model.py，TransformerLayer.attn，
      分离的 q/k/v/o_proj、bias=False）：仅置零 v_proj 中该头对应的
      输出行即可——V_h = 0 使该头输出恒为 0，与注意力权重取值无关；
      o_proj 无需改动（该头输入恒 0），也不影响其他头的计算。

    注意：对自定义 MHA 只置零 q_proj 是错误的——Q = 0 仅使注意力
    权重均匀化（softmax 全等），头输出变为 V 的序列均值 mean(V_h)
    而非 0，头仍然向残差流注入信号，屏蔽实验的结论会失真。
    """
    with torch.no_grad():
        if hasattr(mha, 'in_proj_weight') and mha.in_proj_weight is not None:
            d_model = mha.embed_dim
            for offset in [0, d_model, 2 * d_model]:
                s = offset + head_idx * head_dim
                e = offset + (head_idx + 1) * head_dim
                mha.in_proj_weight[s:e, :] = 0
            return
        
        # 对于自定义 MultiHeadAttention: 头维度连续排列，头 h 的输出行区间
        # = [h*head_dim, (h+1)*head_dim)，置零 v_proj 对应行实现真屏蔽。
        s = head_idx * head_dim
        e = (head_idx + 1) * head_dim
        mha.v_proj.weight[s:e, :] = 0


def _create_eval_criterion(eval_targets):
    if LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        test_targets = np.array(eval_targets)
        test_pos_count = np.sum(test_targets >= 0.5)
        test_neg_count = np.sum(test_targets < 0.5)
        if test_pos_count > 0 and test_neg_count > 0:
            test_neg_weight = LossConfig.POS_WEIGHT * (test_pos_count / test_neg_count)
        elif test_pos_count == 0:
            test_neg_weight = float(LossConfig.POS_WEIGHT)
        else:
            test_neg_weight = 0.1
        criterion.weight_0_0.fill_(test_neg_weight)
    elif LossConfig.LOSS_TYPE.lower() == 'balanced_bce':
        criterion = BalancedBCE(reduction='mean')
        criterion.update_weights(np.array(eval_targets))
    else:
        raise ValueError(f"未知 LOSS_TYPE: {LossConfig.LOSS_TYPE} (支持: dynamic_bce / pairwise_bce / balanced_bce)")
    return criterion


def _compute_auc_and_loss(model, eval_inputs, eval_targets, device, criterion=None, batch_size=DataConfig.EVAL_BATCH_SIZE, tradeable_mask=None):
    model.eval()
    all_preds = []
    total_loss = 0.0
    num_samples = len(eval_inputs)
    num_batches = (num_samples + batch_size - 1) // batch_size
    amp_ctx = _get_amp_context(device)

    if criterion is None:
        criterion = _create_eval_criterion(eval_targets)

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            batch = torch.tensor(eval_inputs[start:end], dtype=torch.float32, device=device)
            with amp_ctx:
                logits = model(batch)
            scores = torch.sigmoid(logits.float())
            all_preds.append(scores.cpu().numpy().flatten())

            batch_targets = torch.tensor(eval_targets[start:end], dtype=torch.float32, device=device)
            loss = criterion(logits.squeeze(-1), batch_targets)
            total_loss += loss.item() * (end - start)
            del batch, logits, batch_targets

    all_preds = np.concatenate(all_preds)
    all_targets = np.array(eval_targets)

    # 过滤：排除T+1开盘价接近涨停的不可交易样本
    if tradeable_mask is not None and not np.all(tradeable_mask):
        all_preds = all_preds[tradeable_mask]
        all_targets = all_targets[tradeable_mask]

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    avg_loss = total_loss / num_samples
    return auc, avg_loss


def _mask_layer_all_heads(layer, nhead, head_dim):
    mha = layer.attn
    for h in range(nhead):
        _mask_mha_head(mha, h, head_dim)


def _mask_position_encoding(model):
    with torch.no_grad():
        model.pos_encoding.pe.weight.zero_()


def _print_position_table(pos_results, baseline_auc, baseline_loss):
    print(f"\n{'=' * 92}")
    print(f"  位置编码诊断 (Baseline AUC = {baseline_auc:.6f}, Loss = {baseline_loss:.6f})")
    print(f"{'=' * 92}")
    print(f"  {'测试':<24}  {'AUC':>10}  {'Δ AUC':>10}  {'Δ%':>8}  {'Loss':>10}  {'Δ Loss':>10}  {'Δ%':>8}")
    print(f"  {'─' * 88}")

    for r in pos_results:
        print(
            f"  {r['desc']:<24}  {r['auc']:>10.6f}  {r['auc_change']:>+10.6f}  {r['auc_change_pct']:>+7.3f}%  "
            f"{r['loss']:>10.6f}  {r['loss_change']:>+10.6f}  {r['loss_change_pct']:>+7.3f}%"
        )

    zero_auc_pct = pos_results[0]["auc_change_pct"]
    shuffle_auc_pct = pos_results[1]["auc_change_pct"]

    print(f"  {'─' * 88}")
    if abs(zero_auc_pct) < 0.1:
        print(f"  ⚠️  位置编码几乎无用（清零后 AUC Δ% = {zero_auc_pct:+.3f}%）")
        print(f"      模型可能没有学到时序模式，仅依赖特征值本身")
    else:
        print(f"  ✅ 位置编码有效（清零后 AUC Δ% = {zero_auc_pct:+.3f}%）")

    if abs(shuffle_auc_pct) < abs(zero_auc_pct) * 0.3:
        print(f"  ⚠️  打乱位置影响远小于清零（AUC Δ% = {shuffle_auc_pct:+.3f}%）")
        print(f"      位置编码可能仅作为偏置，而非编码真正的时序关系")
    elif shuffle_auc_pct > 0:
        print(f"  ⚠️  打乱位置后 AUC 反而上升（Δ% = {shuffle_auc_pct:+.3f}%）")
        print(f"      位置编码可能引入了过拟合，反而干扰了泛化")
    print()


def _print_layer_table(layer_results, baseline_auc):
    print(f"\n{'=' * 80}")
    print(f"  整层屏蔽实验 (Baseline AUC = {baseline_auc:.6f})")
    print(f"  区分: 头趋同（层有用但头冗余） vs 层无用（整层空转）")
    print(f"{'=' * 80}")
    print(f"  {'层':<8}  {'整层屏蔽 Δ%':>14}  {'单头最大 Δ%':>14}  {'诊断':<20}")
    print(f"  {'─' * 70}")

    for r in layer_results:
        pct = r["auc_change_pct"]
        max_single = r["max_single_pct"]
        if abs(pct) < 0.05:
            diag = "🟡 层无用（整层空转）"
        elif abs(pct) > abs(max_single) * 2:
            diag = "🔴 头趋同（冗余但不空转）"
        else:
            diag = "🟢 层有效，头有分工"

        print(
            f"  {r['desc']:<8}  {pct:>+13.3f}%  {max_single:>+13.3f}%  {diag}"
        )

    print(f"  {'─' * 70}")
    print(f"  🟡 层无用: 整层屏蔽也几乎无影响，层在空转")
    print(f"  🔴 头趋同: 整层屏蔽影响 >> 单头屏蔽影响，说明头之间高度冗余")
    print(f"  🟢 层有效: 整层屏蔽与单头屏蔽影响相当，头之间有分工\n")


def _print_heatmap_table(results, nhead, num_layers, baseline_auc, pool_results):
    header_heads = "".join(f"  H{h+1}     " for h in range(nhead))
    sep = "─" * (12 + len("  ─────── ") * nhead)

    print(f"\n{'=' * 72}")
    print(f"  注意力头重要性分析结果 (Baseline AUC = {baseline_auc:.6f})")
    print(f"{'=' * 72}")
    print(f"  {'Layer':<8}{header_heads}")
    print(f"  {sep}")

    for layer_idx in range(num_layers):
        row = f"  L{layer_idx + 1:<6}"
        for head_idx in range(nhead):
            r = results[layer_idx * nhead + head_idx]
            pct = r["auc_change_pct"]
            if abs(pct) < 0.05:
                tag = "💤"
            elif pct < -0.5:
                tag = "🔥"
            else:
                tag = "  "
            row += f" {pct:+.3f}%{tag}"
        print(row)

    if pool_results:
        print(f"  {sep}")
        row = f"  {'Pool':<8}"
        for r in pool_results:
            pct = r["auc_change_pct"]
            if abs(pct) < 0.05:
                tag = "💤"
            elif pct < -0.5:
                tag = "🔥"
            else:
                tag = "  "
            row += f" {pct:+.3f}%{tag}"
        print(row)

    print(f"  {sep}")


def _print_ranking_table(results, pool_results, baseline_auc, baseline_loss):
    all_results = results + pool_results
    all_results.sort(key=lambda r: r["auc_change_pct"])

    print(f"\n{'=' * 92}")
    print(f"  注意力头重要性排名（按 AUC 下降幅度从大到小）")
    print(f"{'=' * 92}")
    print(f"  {'排名':>4}  {'头':<18}  {'类型':<18}  {'AUC':>10}  {'Δ AUC':>10}  {'Δ%':>8}  {'Loss':>10}  {'Δ Loss':>10}  {'Δ%':>8}")
    print(f"  {'─' * 88}")

    for rank, r in enumerate(all_results, 1):
        print(
            f"  {rank:>4}  {r['desc']:<18}  {r['type']:<18}  "
            f"{r['auc']:>10.6f}  {r['auc_change']:>+10.6f}  {r['auc_change_pct']:>+7.3f}%  "
            f"{r['loss']:>10.6f}  {r['loss_change']:>+10.6f}  {r['loss_change_pct']:>+7.3f}%"
        )

    lazy = [r for r in all_results if abs(r["auc_change_pct"]) < 0.05]
    critical = [r for r in all_results if r["auc_change_pct"] < -0.5]

    print(f"\n  关键头 (AUC Δ% < -0.5%): {len(critical)} 个")
    for r in critical:
        print(f"    - {r['desc']} ({r['type']})  AUC Δ% = {r['auc_change_pct']:+.3f}%  Loss Δ% = {r['loss_change_pct']:+.3f}%")

    print()


def main():
    print("=" * 64)
    print("  EquiNet · 注意力机制诊断")
    print("  方法: 'Are Sixteen Heads Really Better than One?' (NeurIPS 2019)")
    print("=" * 64)

    device = DeviceConfig.get_device()
    if device.type == "cuda":
        print(f"  设备: GPU ({torch.cuda.get_device_name()})")
    else:
        print(f"  设备: CPU")

    models = list_available_models()
    if not models:
        print("\n  没有可用的模型，请先训练模型。")
        return

    model_idx = select_model(models)
    selected_file = models[model_idx]
    model_path = os.path.join(DataConfig.OUTPUT_DIR, selected_file)

    print(f"\n正在加载模型: {selected_file}")
    model, metadata = load_model(model_path, device)

    if metadata and metadata.get("model_arch"):
        arch = metadata["model_arch"]
        nhead = arch.get("nhead", ModelConfig.NHEAD)
        num_layers = arch.get("num_layers", ModelConfig.NUM_LAYERS)
        d_model = arch.get("d_model", ModelConfig.D_MODEL)
    else:
        nhead = ModelConfig.NHEAD
        num_layers = ModelConfig.NUM_LAYERS
        d_model = ModelConfig.D_MODEL

    head_dim = d_model // nhead
    total_transformer_heads = num_layers * nhead
    total_pooling_heads = nhead

    print(f"  架构: {num_layers} 层 × {nhead} 头 = {total_transformer_heads} 自注意力头")
    print(f"  AttentionPooling: {total_pooling_heads} 头")
    print(f"  总计: {total_transformer_heads + total_pooling_heads} 个注意力头")

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    print(f"\n正在加载数据集...")
    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    eval_inputs, eval_targets, eval_cumulative_returns, eval_day_indices, eval_daily_returns, eval_tradeable_mask = \
        create_fixed_evaluation_dataset(val_stock_info, feature_normalizer)
    print(f"  验证集样本数: {len(eval_inputs)}")

    original_state_dict = copy.deepcopy(model.state_dict())
    eval_criterion = _create_eval_criterion(eval_targets)

    print(f"\n正在计算 Baseline...")
    baseline_auc, baseline_loss = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
    print(f"  Baseline AUC = {baseline_auc:.6f}  Loss = {baseline_loss:.6f}")

    transformer_results = []
    pool_results = []

    print(f"\n{'─' * 72}")
    print(f"  开始逐头屏蔽实验（共 {total_transformer_heads + total_pooling_heads} 个头）...")
    print(f"{'─' * 72}")

    for layer_idx in range(num_layers):
        for head_idx in range(nhead):
            desc = f"L{layer_idx + 1}.H{head_idx + 1}"
            model.load_state_dict(copy.deepcopy(original_state_dict))
            _mask_mha_head(model.layers[layer_idx].attn, head_idx, head_dim)

            masked_auc, masked_loss = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
            auc_change = masked_auc - baseline_auc
            auc_change_pct = (auc_change / baseline_auc) * 100 if baseline_auc > 0 else 0
            loss_change = masked_loss - baseline_loss
            loss_change_pct = (loss_change / baseline_loss) * 100 if baseline_loss > 0 else 0

            transformer_results.append({
                "type": "Self-Attention",
                "layer": layer_idx + 1,
                "head": head_idx + 1,
                "desc": desc,
                "auc": masked_auc,
                "auc_change": auc_change,
                "auc_change_pct": auc_change_pct,
                "loss": masked_loss,
                "loss_change": loss_change,
                "loss_change_pct": loss_change_pct,
            })

            tag = "💤偷懒" if abs(auc_change_pct) < 0.05 else ("🔥关键" if auc_change_pct < -0.5 else "")
            if auc_change > 0.001 and loss_change > 0.001:
                anomaly = "❓AUC↑Loss↑"
            elif auc_change > 0.001 and abs(loss_change) < 0.0001:
                anomaly = "❓AUC↑Loss≈"
            elif auc_change < -0.001 and loss_change < -0.001:
                anomaly = "❓AUC↓Loss↓"
            elif auc_change < -0.001 and abs(loss_change) < 0.0001:
                anomaly = "❓AUC↓Loss≈"
            else:
                anomaly = ""
            gi = layer_idx * nhead + head_idx + 1
            print(
                f"  [{gi:2d}/{total_transformer_heads + total_pooling_heads}] "
                f"{desc:10s}  AUC={masked_auc:.6f}({auc_change:+.6f})  "
                f"Loss={masked_loss:.6f}({loss_change:+.6f})  {tag}  {anomaly}"
            )

    for head_idx in range(total_pooling_heads):
        desc = f"Pool.H{head_idx + 1}"
        model.load_state_dict(copy.deepcopy(original_state_dict))
        _mask_mha_head(model.attention_pooling.cross_attn, head_idx, head_dim)

        masked_auc, masked_loss = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
        auc_change = masked_auc - baseline_auc
        auc_change_pct = (auc_change / baseline_auc) * 100 if baseline_auc > 0 else 0
        loss_change = masked_loss - baseline_loss
        loss_change_pct = (loss_change / baseline_loss) * 100 if baseline_loss > 0 else 0

        pool_results.append({
            "type": "Attention-Pooling",
            "layer": "-",
            "head": head_idx + 1,
            "desc": desc,
            "auc": masked_auc,
            "auc_change": auc_change,
            "auc_change_pct": auc_change_pct,
            "loss": masked_loss,
            "loss_change": loss_change,
            "loss_change_pct": loss_change_pct,
        })

        tag = "💤偷懒" if abs(auc_change_pct) < 0.05 else ("🔥关键" if auc_change_pct < -0.5 else "")
        if auc_change > 0.001 and loss_change > 0.001:
            anomaly = "❓AUC↑Loss↑"
        elif auc_change > 0.001 and abs(loss_change) < 0.0001:
            anomaly = "❓AUC↑Loss≈"
        elif auc_change < -0.001 and loss_change < -0.001:
            anomaly = "❓AUC↓Loss↓"
        elif auc_change < -0.001 and abs(loss_change) < 0.0001:
            anomaly = "❓AUC↓Loss≈"
        else:
            anomaly = ""
        gi = total_transformer_heads + head_idx + 1
        print(
            f"  [{gi:2d}/{total_transformer_heads + total_pooling_heads}] "
            f"{desc:10s}  AUC={masked_auc:.6f}({auc_change:+.6f})  "
            f"Loss={masked_loss:.6f}({loss_change:+.6f})  {tag}  {anomaly}"
        )

    model.load_state_dict(original_state_dict)

    print(f"\n{'─' * 72}")
    print(f"  开始整层屏蔽实验（共 {num_layers} 层）...")
    print(f"{'─' * 72}")

    layer_results = []
    for layer_idx in range(num_layers):
        desc = f"L{layer_idx + 1}"
        model.load_state_dict(copy.deepcopy(original_state_dict))
        _mask_layer_all_heads(model.layers[layer_idx], nhead, head_dim)

        masked_auc, _ = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
        auc_change = masked_auc - baseline_auc
        auc_change_pct = (auc_change / baseline_auc) * 100 if baseline_auc > 0 else 0

        layer_heads = transformer_results[layer_idx * nhead:(layer_idx + 1) * nhead]
        max_single_pct = min(r["auc_change_pct"] for r in layer_heads)

        layer_results.append({
            "desc": desc,
            "auc": masked_auc,
            "auc_change": auc_change,
            "auc_change_pct": auc_change_pct,
            "max_single_pct": max_single_pct,
        })

        print(
            f"  [{layer_idx + 1}/{num_layers}] {desc:10s}  "
            f"AUC={masked_auc:.6f}  Δ={auc_change:+.6f}  Δ%={auc_change_pct:+.3f}%"
        )

    model.load_state_dict(original_state_dict)

    print(f"\n{'─' * 72}")
    print(f"  开始位置编码实验...")
    print(f"{'─' * 72}")

    pos_results = []

    model.load_state_dict(copy.deepcopy(original_state_dict))
    _mask_position_encoding(model)
    masked_auc, masked_loss = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
    auc_change = masked_auc - baseline_auc
    auc_change_pct = (auc_change / baseline_auc) * 100 if baseline_auc > 0 else 0
    loss_change = masked_loss - baseline_loss
    loss_change_pct = (loss_change / baseline_loss) * 100 if baseline_loss > 0 else 0
    pos_results.append({
        "desc": "位置编码清零",
        "auc": masked_auc,
        "auc_change": auc_change,
        "auc_change_pct": auc_change_pct,
        "loss": masked_loss,
        "loss_change": loss_change,
        "loss_change_pct": loss_change_pct,
    })
    print(f"  [1/2] 位置编码清零    AUC={masked_auc:.6f}({auc_change:+.6f})  Loss={masked_loss:.6f}({loss_change:+.6f})")

    rng = np.random.RandomState(42)
    perm = rng.permutation(45).tolist()
    model.load_state_dict(copy.deepcopy(original_state_dict))
    with torch.no_grad():
        orig_weight = model.pos_encoding.pe.weight.data.clone()
        model.pos_encoding.pe.weight.data = orig_weight[perm]
    masked_auc, masked_loss = _compute_auc_and_loss(model, eval_inputs, eval_targets, device, eval_criterion, tradeable_mask=eval_tradeable_mask)
    auc_change = masked_auc - baseline_auc
    auc_change_pct = (auc_change / baseline_auc) * 100 if baseline_auc > 0 else 0
    loss_change = masked_loss - baseline_loss
    loss_change_pct = (loss_change / baseline_loss) * 100 if baseline_loss > 0 else 0
    pos_results.append({
        "desc": "位置编码随机打乱",
        "auc": masked_auc,
        "auc_change": auc_change,
        "auc_change_pct": auc_change_pct,
        "loss": masked_loss,
        "loss_change": loss_change,
        "loss_change_pct": loss_change_pct,
    })
    print(f"  [2/2] 位置编码随机打乱  AUC={masked_auc:.6f}({auc_change:+.6f})  Loss={masked_loss:.6f}({loss_change:+.6f})")

    model.load_state_dict(original_state_dict)

    _print_heatmap_table(transformer_results, nhead, num_layers, baseline_auc, pool_results)
    _print_ranking_table(transformer_results, pool_results, baseline_auc, baseline_loss)
    _print_layer_table(layer_results, baseline_auc)
    _print_position_table(pos_results, baseline_auc, baseline_loss)


if __name__ == "__main__":
    main()
