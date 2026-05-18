"""
信号流诊断脚本 (Signal Flow Diagnosis)

在前向传播过程中，通过 PyTorch hook 在每个关键节点记录激活的统计量，
让信息在网络中的流动完全可视化。

诊断内容:
  1. 前向信号流: 每层残差流的 std/norm 变化、Attn/FFN 分支的贡献占比
  2. 梯度信号流: 反向传播时各层的梯度 std/norm，检测梯度消失/爆炸
  3. 参数权重统计: 每层权重矩阵的范数，检测参数退化

模型选择方式与 run.py 完全一致。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from config import ModelConfig, DataConfig, DeviceConfig, LossConfig
from model import create_model
from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,
    FeatureNormalizer,
)
from training_utils import DynamicWeightedBCE
from run import list_available_models, select_model, load_model


def _register_forward_hooks(model):
    hooks = []
    stats = OrderedDict()

    def _make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            if tensor.dim() == 3:
                flat = tensor.detach().reshape(-1)
            elif tensor.dim() == 2:
                flat = tensor.detach().reshape(-1)
            else:
                flat = tensor.detach().reshape(-1)
            stats[name] = {
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "norm": flat.norm().item(),
                "abs_mean": flat.abs().mean().item(),
                "max": flat.abs().max().item(),
                "shape": tuple(tensor.shape),
            }
        return hook

    hook_targets = OrderedDict()
    hook_targets["Embed.embed_proj"] = model.embed_proj

    embed_mlp_linear = next((m for m in model.embed_mlp if isinstance(m, nn.Linear)), None)
    if embed_mlp_linear is not None:
        hook_targets["Embed.mlp_linear"] = embed_mlp_linear

    hook_targets["Embed.pos_encoding"] = model.pos_encoding

    for i, layer in enumerate(model.layers):
        hook_targets[f"L{i+1}.attn_norm"] = layer.attn_norm
        hook_targets[f"L{i+1}.attn_out"] = layer.attn
        hook_targets[f"L{i+1}.ffn_norm"] = layer.ffn_norm
        hook_targets[f"L{i+1}.ffn_out"] = layer.ffn_w2

    hook_targets["FinalNorm"] = model.final_norm
    hook_targets["Pool.cross_attn"] = model.attention_pooling.cross_attn
    hook_targets["Head.norm"] = model.head_norm
    hook_targets["Output"] = model.output_projection

    for name, module in hook_targets.items():
        h = module.register_forward_hook(_make_hook(name))
        hooks.append(h)

    return hooks, stats


def _register_full_forward_hooks(model):
    hooks = []
    stats = OrderedDict()

    def _make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            flat = tensor.detach().reshape(-1)
            stats[name] = {
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "norm": flat.norm().item(),
                "abs_mean": flat.abs().mean().item(),
                "max": flat.abs().max().item(),
                "shape": tuple(tensor.shape),
            }
        return hook

    h = model.embed_proj.register_forward_hook(_make_hook("① embed_proj"))
    hooks.append(h)

    if embed_mlp_linear := next((m for m in model.embed_mlp if isinstance(m, nn.Linear)), None):
        h = embed_mlp_linear.register_forward_hook(_make_hook("② embed_mlp (残差分支)"))
        hooks.append(h)

    h = model.pos_encoding.register_forward_hook(_make_hook("③ +pos_encoding"))
    hooks.append(h)

    for i, layer in enumerate(model.layers):
        ln = layer.attn_norm
        h = ln.register_forward_hook(_make_hook(f"L{i+1} ④ attn_norm"))
        hooks.append(h)

        attn = layer.attn
        h = attn.register_forward_hook(_make_hook(f"L{i+1} ⑤ attn分支输出"))
        hooks.append(h)

        ln2 = layer.ffn_norm
        h = ln2.register_forward_hook(_make_hook(f"L{i+1} ⑥ ffn_norm"))
        hooks.append(h)

        w2 = layer.ffn_w2
        h = w2.register_forward_hook(_make_hook(f"L{i+1} ⑦ ffn分支输出"))
        hooks.append(h)

    h = model.final_norm.register_forward_hook(_make_hook("⑧ final_norm"))
    hooks.append(h)

    h = model.attention_pooling.cross_attn.register_forward_hook(_make_hook("⑨ pool_cross_attn"))
    hooks.append(h)

    h = model.head_norm.register_forward_hook(_make_hook("⑩ head_norm"))
    hooks.append(h)

    h = model.output_projection.register_forward_hook(_make_hook("⑪ output"))
    hooks.append(h)

    return hooks, stats


def _register_gradient_hooks(model):
    hooks = []
    grad_stats = OrderedDict()

    def _make_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is None:
                return
            tensor = grad_output[0]
            flat = tensor.detach().reshape(-1)
            grad_stats[name] = {
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "norm": flat.norm().item(),
                "abs_mean": flat.abs().mean().item(),
            }
        return hook

    targets = OrderedDict()
    targets["embed_proj"] = model.embed_proj
    for i, layer in enumerate(model.layers):
        targets[f"L{i+1}.attn_norm"] = layer.attn_norm
        targets[f"L{i+1}.ffn_norm"] = layer.ffn_norm
    targets["final_norm"] = model.final_norm
    targets["head_norm"] = model.head_norm

    for name, module in targets.items():
        h = module.register_full_backward_hook(_make_hook(name))
        hooks.append(h)

    return hooks, grad_stats


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
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    return criterion


def _print_signal_flow_table(stats, num_layers):
    print(f"\n{'=' * 110}")
    print(f"  前向信号流 (Forward Signal Flow)")
    print(f"{'=' * 110}")
    print(f"  {'节点':<30}  {'std':>10}  {'norm':>10}  {'|mean|':>10}  {'max':>10}  {'shape'}")
    print(f"  {'─' * 104}")

    prev_std = None
    for name, s in stats.items():
        std = s["std"]
        norm = s["norm"]
        abs_mean = s["abs_mean"]
        max_val = s["max"]
        shape = "×".join(str(d) for d in s["shape"])

        if prev_std is not None and prev_std > 0:
            ratio = std / prev_std
            if ratio < 0.5:
                marker = f"  ⚠️ ×{ratio:.2f}"
            elif ratio > 2.0:
                marker = f"  ⚠️ ×{ratio:.2f}"
            else:
                marker = f"     ×{ratio:.2f}"
        else:
            marker = ""

        print(f"  {name:<30}  {std:>10.6f}  {norm:>10.2f}  {abs_mean:>10.6f}  {max_val:>10.4f}  {shape}{marker}")
        prev_std = std

    print(f"  {'─' * 104}")


def _print_residual_analysis(model, eval_inputs, eval_targets, device, num_layers, nhead):
    print(f"\n{'=' * 110}")
    print(f"  残差流逐层分析 (Residual Stream Analysis)")
    print(f"  测量每层 Attn/FFN 分支对残差流的实际贡献占比")
    print(f"{'=' * 110}")

    records = []

    def make_residual_hook(layer_idx):
        def pre_hook(module, input):
            x = input[0]
            flat = x.detach().reshape(-1)
            records.append({
                "type": "pre_residual",
                "layer": layer_idx,
                "std": flat.std().item(),
                "norm": flat.norm().item(),
            })
        return pre_hook

    hooks = []
    for i, layer in enumerate(model.layers):
        h = layer.register_forward_pre_hook(make_residual_hook(i))
        hooks.append(h)

    model.eval()
    with torch.no_grad():
        batch = torch.tensor(eval_inputs[:32], dtype=torch.float32, device=device)
        _ = model(batch)

    for h in hooks:
        h.remove()

    print(f"  {'层':<8}  {'残差流 std':>12}  {'残差流 norm':>12}  {'Attn分支贡献':>14}  {'FFN分支贡献':>14}  {'Attn占比':>10}  {'FFN占比':>10}")
    print(f"  {'─' * 100}")

    pre_stds = {r["layer"]: r["std"] for r in records if r["type"] == "pre_residual"}
    pre_norms = {r["layer"]: r["norm"] for r in records if r["type"] == "pre_residual"}

    model.eval()
    with torch.no_grad():
        batch = torch.tensor(eval_inputs[:32], dtype=torch.float32, device=device)

        x = model.embed_proj(batch)
        x = x + model.embed_mlp(x)
        x = model.pos_encoding(x)
        x = model.dropout(x)

        for i, layer in enumerate(model.layers):
            pre_norm = x.detach().norm().item()
            pre_std = x.detach().reshape(-1).std().item()

            h_normed = layer.attn_norm(x)
            attn_out = layer.attn(h_normed, attn_mask=None)
            attn_contrib_norm = attn_out.detach().norm().item()
            attn_contrib_std = attn_out.detach().reshape(-1).std().item()

            x_after_attn = x + attn_out

            h2 = layer.ffn_norm(x_after_attn)
            ffn_gated = torch.nn.functional.silu(layer.ffn_w1(h2)) * layer.ffn_w3(h2)
            ffn_out = layer.ffn_dropout(layer.ffn_w2(ffn_gated))
            ffn_contrib_norm = ffn_out.detach().norm().item()
            ffn_contrib_std = ffn_out.detach().reshape(-1).std().item()

            x = x_after_attn + ffn_out

            attn_ratio = attn_contrib_std / pre_std if pre_std > 0 else 0
            ffn_ratio = ffn_contrib_std / pre_std if pre_std > 0 else 0

            print(
                f"  L{i+1:<6}  {pre_std:>12.6f}  {pre_norm:>12.2f}  "
                f"{attn_contrib_std:>14.6f}  {ffn_contrib_std:>14.6f}  "
                f"{attn_ratio:>9.2f}×  {ffn_ratio:>9.2f}×"
            )

    print(f"  {'─' * 100}")
    print(f"  Attn/FFN占比 = 分支输出std / 残差流std，健康范围约 0.05~0.50")
    print(f"  <0.01 → 分支几乎没贡献   >1.0 → 分支主导了残差流\n")


def _print_gradient_flow_table(grad_stats):
    print(f"\n{'=' * 80}")
    print(f"  梯度信号流 (Gradient Flow)")
    print(f"{'=' * 80}")
    print(f"  {'节点':<20}  {'grad_std':>12}  {'grad_norm':>12}  {'|grad_mean|':>12}")
    print(f"  {'─' * 68}")

    prev_std = None
    for name, s in grad_stats.items():
        std = s["std"]
        norm = s["norm"]
        abs_mean = s["abs_mean"]

        if prev_std is not None and prev_std > 0:
            ratio = std / prev_std
            if ratio < 0.1:
                marker = f"  🔻 ×{ratio:.4f}"
            elif ratio < 0.5:
                marker = f"  ⚠️ ×{ratio:.2f}"
            elif ratio > 5.0:
                marker = f"  🔺 ×{ratio:.2f}"
            else:
                marker = f"     ×{ratio:.2f}"
        else:
            marker = ""

        print(f"  {name:<20}  {std:>12.8f}  {norm:>12.6f}  {abs_mean:>12.8f}{marker}")
        prev_std = std

    print(f"  {'─' * 68}")
    print(f"  🔻 梯度消失 (<0.1×)   🔺 梯度爆炸 (>5×)   ⚠️ 轻微衰减\n")


def _print_weight_stats(model, num_layers):
    print(f"\n{'=' * 100}")
    print(f"  参数权重统计 (Weight Statistics)")
    print(f"{'=' * 100}")
    print(f"  {'参数':<35}  {'shape':<20}  {'weight_std':>12}  {'weight_norm':>12}  {'bias_mean':>12}")
    print(f"  {'─' * 95}")

    params_to_check = OrderedDict()
    params_to_check["embed_proj.weight"] = model.embed_proj.weight
    params_to_check["embed_proj.bias"] = model.embed_proj.bias

    for i, layer in enumerate(model.layers):
        mha = layer.attn.attention
        if hasattr(mha, 'in_proj_weight') and mha.in_proj_weight is not None:
            params_to_check[f"L{i+1}.in_proj.weight"] = mha.in_proj_weight
        if hasattr(mha, 'out_proj') and mha.out_proj.weight is not None:
            params_to_check[f"L{i+1}.out_proj.weight"] = mha.out_proj.weight
            if mha.out_proj.bias is not None:
                params_to_check[f"L{i+1}.out_proj.bias"] = mha.out_proj.bias

        params_to_check[f"L{i+1}.ffn_w1.weight"] = layer.ffn_w1.weight
        params_to_check[f"L{i+1}.ffn_w3.weight"] = layer.ffn_w3.weight
        params_to_check[f"L{i+1}.ffn_w2.weight"] = layer.ffn_w2.weight

    params_to_check["pool.query"] = model.attention_pooling.query
    pool_mha = model.attention_pooling.cross_attn
    if hasattr(pool_mha, 'in_proj_weight') and pool_mha.in_proj_weight is not None:
        params_to_check["pool.in_proj.weight"] = pool_mha.in_proj_weight
    if hasattr(pool_mha, 'out_proj') and pool_mha.out_proj.weight is not None:
        params_to_check["pool.out_proj.weight"] = pool_mha.out_proj.weight

    params_to_check["output.weight"] = model.output_projection.weight
    params_to_check["output.bias"] = model.output_projection.bias

    for name, param in params_to_check.items():
        w = param.data
        shape_str = "×".join(str(d) for d in w.shape)
        w_norm = w.norm().item()
        bias_mean = ""

        if name.endswith(".bias"):
            bias_mean = f"{w.mean().item():>12.6f}"
            w_std_str = "-"
            w_norm_str = "-"
        else:
            w_std_str = f"{w.std().item():>12.6f}"
            w_norm_str = f"{w_norm:>12.2f}"

            bias_name = name.replace(".weight", ".bias")
            if bias_name in params_to_check:
                b = params_to_check[bias_name].data
                bias_mean = f"{b.mean().item():>12.6f}"
            else:
                bias_mean = "          N/A"

        print(f"  {name:<35}  {shape_str:<20}  {w_std_str:>12}  {w_norm_str:>12}  {bias_mean:>12}")

    print(f"  {'─' * 95}")


def main():
    print("=" * 72)
    print("  EquiNet · 信号流诊断 (Signal Flow Diagnosis)")
    print("  监控前向/梯度信号在各层的 std、norm 变化")
    print("=" * 72)

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
    else:
        nhead = ModelConfig.NHEAD
        num_layers = ModelConfig.NUM_LAYERS

    print(f"  架构: {num_layers} 层 × {nhead} 头")

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    print(f"\n正在加载数据集...")
    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()

    eval_inputs, eval_targets, _, _, _ = create_fixed_evaluation_dataset(
        val_stock_info, feature_normalizer
    )
    print(f"  验证集样本数: {len(eval_inputs)}")

    # ===== 实验1: 前向信号流 =====
    print(f"\n{'─' * 72}")
    print(f"  [1/4] 前向信号流 (注册 hook，跑一次前向传播)...")
    print(f"{'─' * 72}")

    hooks, fwd_stats = _register_full_forward_hooks(model)
    model.eval()
    with torch.no_grad():
        sample_batch = torch.tensor(eval_inputs[:32], dtype=torch.float32, device=device)
        _ = model(sample_batch)
    for h in hooks:
        h.remove()

    _print_signal_flow_table(fwd_stats, num_layers)

    # ===== 实验2: 残差流逐层分析 =====
    print(f"\n{'─' * 72}")
    print(f"  [2/4] 残差流逐层分析 (手动展开每层的 Attn/FFN 分支贡献)...")
    print(f"{'─' * 72}")

    _print_residual_analysis(model, eval_inputs, eval_targets, device, num_layers, nhead)

    # ===== 实验3: 梯度信号流 =====
    print(f"\n{'─' * 72}")
    print(f"  [3/4] 梯度信号流 (跑一次反向传播，记录梯度统计量)...")
    print(f"{'─' * 72}")

    grad_hooks, grad_stats = _register_gradient_hooks(model)
    model.train()

    eval_criterion = _create_eval_criterion(eval_targets)
    grad_batch = torch.tensor(eval_inputs[:64], dtype=torch.float32, device=device)
    grad_targets = torch.tensor(eval_targets[:64], dtype=torch.float32, device=device)

    model.zero_grad()
    logits = model(grad_batch)
    loss = eval_criterion(logits.squeeze(-1), grad_targets)
    loss.backward()

    for h in grad_hooks:
        h.remove()

    _print_gradient_flow_table(grad_stats)

    model.eval()

    # ===== 实验4: 参数权重统计 =====
    print(f"\n{'─' * 72}")
    print(f"  [4/4] 参数权重统计 (每层权重矩阵的范数)...")
    print(f"{'─' * 72}")

    _print_weight_stats(model, num_layers)

    print(f"\n{'=' * 72}")
    print(f"  诊断完成")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
