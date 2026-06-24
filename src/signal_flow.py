"""
信号流诊断 - 各路线 std

手动展开前向传播，记录每条路线的 std。不做分析。
"""

import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from config import ModelConfig, DataConfig, DeviceConfig, LossConfig
from data import (
    load_and_preprocess_data,
    create_fixed_evaluation_dataset,
    FeatureNormalizer,
)
from training_utils import DynamicWeightedBCE
from run import list_available_models, select_model, load_model


def _s2(v):
    """2位有效数字"""
    if v == 0:
        return "      0"
    return f"{v:.2g}".rjust(7)


def _std(x):
    return x.detach().reshape(-1).std().item()


def _register_gradient_hooks(model):
    hooks = []
    grad_stats = OrderedDict()

    def _make_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is None:
                return
            grad_stats[name] = _std(grad_output[0])
        return hook

    targets = OrderedDict()
    targets["embed_proj"] = model.embed_proj
    for i, layer in enumerate(model.layers):
        targets[f"L{i+1}.attn_norm"] = layer.attn_norm
        targets[f"L{i+1}.ffn_norm"] = layer.ffn_norm
    targets["head_norm"] = model.head_norm

    for name, module in targets.items():
        h = module.register_full_backward_hook(_make_hook(name))
        hooks.append(h)

    return hooks, grad_stats


def _create_eval_criterion(eval_targets):
    if LossConfig.LOSS_TYPE.lower() == 'dynamic_bce':
        criterion = DynamicWeightedBCE(pos_weight=LossConfig.POS_WEIGHT, reduction='mean')
        test_targets = np.array(eval_targets)
        test_pos = np.sum(test_targets >= 0.5)
        test_neg = np.sum(test_targets < 0.5)
        if test_pos > 0 and test_neg > 0:
            neg_w = LossConfig.POS_WEIGHT * (test_pos / test_neg)
        elif test_pos == 0:
            neg_w = float(LossConfig.POS_WEIGHT)
        else:
            neg_w = 0.1
        criterion.weight_0_0.fill_(neg_w)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    return criterion


def _print_forward_flow(model, eval_inputs, device):
    model.eval()
    batch = torch.tensor(eval_inputs[:32], dtype=torch.float32, device=device)

    print("\n前向信号流 std")
    print("─" * 72)

    with torch.no_grad():
        # Embedding (与 model.py 一致: proj → mlp, 无残差)
        x = model.embed_proj(batch)
        proj = _std(x)
        x = model.embed_mlp(x)
        mlp = _std(x)
        x = model.pos_encoding(x)
        after_pos = _std(x)
        x = model.dropout(x)

        print(f"  proj {_s2(proj)}  mlp输出 {_s2(mlp)}  +pos {_s2(after_pos)}")

        # Transformer layers (Post-Norm: sublayer → +residual → norm)
        # Post-Norm 每层输出经 LayerNorm，std 恒≈1，无诊断价值
        # 只记录归一化前各子层的信号强度
        rows = []
        for i, layer in enumerate(model.layers):
            entry = _std(x)
            attn_out = layer.attn(x, attn_mask=None)
            attn = _std(attn_out)
            x = layer.attn_norm(x + attn_out)
            ffn_gated = torch.nn.functional.silu(layer.ffn_w1(x)) * layer.ffn_w3(x)
            ffn_out = layer.ffn_dropout(layer.ffn_w2(ffn_gated))
            ffn = _std(ffn_out)
            x = layer.ffn_norm(x + ffn_out)
            rows.append((i + 1, entry, attn, ffn))

        print()
        print(f"       {'res_in':>7}  {'attn':>7}  {'ffn':>7}")
        print(f"  {'─' * 36}")
        for (li, entry, attn, ffn) in rows:
            print(f"  L{li:<3} {_s2(entry)}  {_s2(attn)}  {_s2(ffn)}")

        # Head
        pooled = model.attention_pooling(x)
        pool = _std(pooled)
        out = model.output_projection(model.head_norm(pooled))
        output = _std(out)

        print()
        print(f"  pool {_s2(pool)}  output {_s2(output)}")

    print("─" * 72)


def _print_gradient_flow(grad_stats):
    print("\n梯度流 std")
    print("─" * 40)
    for name, std_val in grad_stats.items():
        print(f"  {name:<18} {_s2(std_val)}")
    print("─" * 40)


def _print_weight_stats(model, num_layers):
    print("\n权重 std")
    print("─" * 45)
    print(f"  embed_proj        {_s2(model.embed_proj.weight.std().item())}")
    print(f"  embed_mlp[0]      {_s2(model.embed_mlp[0].weight.std().item())}")
    print(f"  embed_mlp[2]      {_s2(model.embed_mlp[2].weight.std().item())}")
    for i, layer in enumerate(model.layers):
        mha = layer.attn
        if hasattr(mha, 'in_proj_weight') and mha.in_proj_weight is not None:
            print(f"  L{i+1}.in_proj      {_s2(mha.in_proj_weight.std().item())}")
        if hasattr(mha, 'out_proj') and mha.out_proj.weight is not None:
            print(f"  L{i+1}.out_proj     {_s2(mha.out_proj.weight.std().item())}")
        print(f"  L{i+1}.ffn_w1       {_s2(layer.ffn_w1.weight.std().item())}")
        print(f"  L{i+1}.ffn_w3       {_s2(layer.ffn_w3.weight.std().item())}")
        print(f"  L{i+1}.ffn_w2       {_s2(layer.ffn_w2.weight.std().item())}")
    pool_mha = model.attention_pooling.cross_attn
    if hasattr(pool_mha, 'in_proj_weight') and pool_mha.in_proj_weight is not None:
        print(f"  pool.in_proj      {_s2(pool_mha.in_proj_weight.std().item())}")
    if hasattr(pool_mha, 'out_proj') and pool_mha.out_proj.weight is not None:
        print(f"  pool.out_proj     {_s2(pool_mha.out_proj.weight.std().item())}")
    print(f"  output            {_s2(model.output_projection.weight.std().item())}")
    print("─" * 45)


def main():
    device = DeviceConfig.get_device()
    if device.type == "cuda":
        print(f"设备: GPU ({torch.cuda.get_device_name()})")
    else:
        print(f"设备: CPU")

    models = list_available_models()
    if not models:
        print("\n没有可用的模型，请先训练。")
        return

    model_idx = select_model(models)
    selected_file = models[model_idx]
    model_path = os.path.join(DataConfig.OUTPUT_DIR, selected_file)

    print(f"加载模型: {selected_file}")
    model, metadata = load_model(model_path, device)

    if metadata and metadata.get("model_arch"):
        arch = metadata["model_arch"]
        nhead = arch.get("nhead", ModelConfig.NHEAD)
        num_layers = arch.get("num_layers", ModelConfig.NUM_LAYERS)
    else:
        nhead = ModelConfig.NHEAD
        num_layers = ModelConfig.NUM_LAYERS

    print(f"架构: {num_layers}层 x {nhead}头")

    if os.path.exists(DataConfig.NORMALIZER_PATH):
        feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)
    else:
        raise FileNotFoundError(f"归一化器文件不存在: {DataConfig.NORMALIZER_PATH}")

    print("加载数据集...")
    train_stock_info, val_stock_info, test_stock_info = load_and_preprocess_data()
    eval_inputs, eval_targets, _, _, _ = create_fixed_evaluation_dataset(
        val_stock_info, feature_normalizer
    )
    print(f"验证集: {len(eval_inputs)} 样本")

    # 前向信号流
    _print_forward_flow(model, eval_inputs, device)

    # 梯度流
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
    _print_gradient_flow(grad_stats)
    model.eval()

    # 权重统计
    _print_weight_stats(model, num_layers)


if __name__ == "__main__":
    main()
