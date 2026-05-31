"""
EquiNet交互式注意力可视化工具

上半部分: 45天K线图 (OHLC蜡烛图 + 成交量)
下半部分: 注意力热力图 (逐层自注意力均值 / Pooling聚合注意力 / Attention Rollout)
交互操作: Space=下一个  Backspace=上一个  L=切换层/头  Q=退出

用法: python src/visualize_attention.py
"""

import os
import sys

# 设置交互式后端（Windows 下 TkAgg 最稳定）
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from config import ModelConfig, DataConfig, DeviceConfig
from data import (load_and_preprocess_data, FeatureNormalizer,
                  normalize_and_validate_context_window, generate_label,
                  calculate_returns)
from training_utils import _get_amp_context
from run import list_available_models, select_model, load_model

def load_test_samples(test_stock_info, feature_normalizer, max_samples=300):
    """
    从测试集生成可视化样本，同时保留原始OHLCV数据用于K线图。

    Returns:
        list[dict]: 每个元素包含:
            stock_code, dates, raw_ohlcv, raw_volumes,
            model_input, label, cumulative_return
    """
    samples = []
    context_len = DataConfig.CONTEXT_LENGTH
    required_len = DataConfig.REQUIRED_LENGTH

    for stock_info in test_stock_info:
        if len(samples) >= max_samples:
            break

        stock_code = stock_info['file_name']
        data = stock_info['data']
        times = stock_info['times']
        split_point = stock_info.get('test_split_point', 0)

        data_length = len(data)
        start_min = max(1, split_point)
        start_max = data_length - required_len

        if start_max < start_min:
            continue

        # 从测试集中均匀采样
        valid_indices = list(range(start_min, start_max + 1))
        step = max(1, len(valid_indices) // max(3, max_samples // len(test_stock_info)))
        sampled_indices = valid_indices[::step]

        for start_idx in sampled_indices:
            if len(samples) >= max_samples:
                break

            # 粗归一化（不含细归一化，用于生成有效样本检查）
            raw_input = normalize_and_validate_context_window(
                data, start_idx, context_len,
                check_limit_up=True, required_length=context_len)

            if raw_input is None:
                continue

            # 细归一化
            if feature_normalizer is not None:
                model_input = feature_normalizer.transform_batch(raw_input[np.newaxis])[0]
            else:
                model_input = raw_input

            # 原始OHLCV (columns: open=0, high=1, low=2, close=3)
            raw_ohlcv = data[start_idx: start_idx + context_len, :4].copy()
            raw_volumes = data[start_idx: start_idx + context_len, 5].copy()
            dates = times[start_idx: start_idx + context_len].copy()

            # 标签和收益计算
            future_days = DataConfig.FUTURE_DAYS
            if start_idx + context_len + future_days <= data_length:
                closes = data[:, 3]
                t1_open = data[start_idx + context_len, 0]
                t1_close = closes[start_idx + context_len]

                day1_change = (t1_close - t1_open) / t1_open if t1_open > 0 else 0
                day2_change = (closes[start_idx + context_len + 1] - closes[start_idx + context_len]) / closes[start_idx + context_len] if start_idx + context_len + 1 < data_length else 0
                day3_change = (closes[start_idx + context_len + 2] - closes[start_idx + context_len + 1]) / closes[start_idx + context_len + 1] if start_idx + context_len + 2 < data_length else 0

                label = generate_label(day1_change, day2_change, day3_change)

                t2_o = data[start_idx + context_len + 1, 0] if start_idx + context_len + 1 < data_length else None
                t2_c = closes[start_idx + context_len + 1] if start_idx + context_len + 1 < data_length else None
                t3_c = closes[start_idx + context_len + 2] if start_idx + context_len + 2 < data_length else None
                cumulative_return, _ = calculate_returns(t1_open, t1_close, t2_o, t2_c, t3_c)
            else:
                label = 0
                cumulative_return = 0.0

            samples.append({
                'stock_code': stock_code,
                'dates': dates,
                'raw_ohlcv': raw_ohlcv,
                'raw_volumes': raw_volumes,
                'model_input': model_input,
                'label': label,
                'cumulative_return': cumulative_return,
            })
    return samples

def run_attention_inference(model, samples, device):
    """对所有样本执行带注意力提取的推理。"""
    model.eval()
    amp_ctx = _get_amp_context(device)

    predictions = []
    all_attentions = []

    with torch.no_grad():
        for sample in samples:
            x = torch.tensor(sample['model_input'], dtype=torch.float32, device=device)
            x = x.unsqueeze(0)
            with amp_ctx:
                logits, attn_weights = model(x, return_attn=True)
            prob = torch.sigmoid(logits.float()).item()
            predictions.append(prob)
            all_attentions.append([w.float().cpu() for w in attn_weights])

    return predictions, all_attentions


# ==================== 注意力计算 ====================
def extract_attention_data(attn_weights):
    """
    从模型输出的注意力权重中提取可视化数据。

    模型返回: [self_attn_0, ..., self_attn_{N-1}, pool_attn]
      - self_attn: (1, nhead, 45, 45) — 各层自注意力
      - pool_attn:  (1, nhead, 1, 45) — 聚合层交叉注意力

    Returns:
        self_attn_per_layer: (num_layers, nhead, 45) — 各层各头每个位置平均接收到的注意力
        pool_attn: (nhead, 45) — 聚合层对各天的注意力权重（等价于原[CLS]注意力）
    """
    num_layers = len(attn_weights) - 1

    per_layer = []
    for i in range(num_layers):
        # (1, nhead, 45, 45) → mean over query positions → (nhead, 45)
        mean_received = attn_weights[i][0].mean(dim=1).numpy()
        per_layer.append(mean_received)

    # Pooling: (1, nhead, 1, 45) → (nhead, 45)
    pool = attn_weights[-1][0, :, 0, :].numpy()

    return np.array(per_layer), pool


def compute_attention_rollout(attn_weights):
    """
    计算 Attention Rollout (Abnar & Zuidema 2020)。

    在自注意力层中累积信息流（考虑残差连接），最后与聚合层注意力结合。
    聚合层的 learnable query 等价于 [CLS] 的角色，其注意力代表最终预测对各天的依赖。

    Returns:
        rollout: (45,) — 各天对最终预测的综合注意力贡献
    """
    num_layers = len(attn_weights) - 1
    seq_len = attn_weights[0].shape[-1]  # 45

    result = np.eye(seq_len, dtype=np.float32)

    for i in range(num_layers):
        # 平均所有头: (45, 45)
        attn = attn_weights[i][0].mean(dim=0).numpy()
        # 加入残差连接贡献
        augmented = 0.5 * attn + 0.5 * np.eye(seq_len, dtype=np.float32)
        result = augmented @ result

    # 聚合层注意力 (平均所有头): (45,)
    pool = attn_weights[-1][0].mean(dim=0)[0].numpy()

    # 综合信息流: pool @ rollout_matrix → (45,)
    rollout = pool @ result
    return rollout


# 暗色主题配色
DARK_BG = '#0f0f1a'
PANEL_BG = '#161625'
EDGE_COLOR = '#2a2a4a'
TEXT_COLOR = '#d0d0d0'
GRID_COLOR = '#1e1e35'
UP_COLOR = '#ff1744'    # 涨：红色（A股习惯）
DOWN_COLOR = '#00c853'  # 跌：绿色
ATTN_HIGH = '#ff9100'

# 自定义注意力色图
ATTN_CMAP = LinearSegmentedColormap.from_list(
    'attention', ['#0f0f1a', '#331a00', '#663300', '#ff6d00', '#ff9100', '#ffab40'], N=256)


class AttentionVisualizer:
    """交互式注意力可视化器。"""
    def __init__(self, samples, predictions, all_attentions, num_layers, nhead):
        self.samples = samples
        self.predictions = predictions
        self.all_attentions = all_attentions
        self.num_layers = num_layers
        self.nhead = nhead
        self.current_idx = 0
        self.total = len(samples)
        self.mode = 'per_layer'  # 'per_layer' or 'per_head'
        self.show_rollout_overlay = True

        # 预计算所有注意力数据
        self.self_attn_per_layer = []
        self.pool_attns = []
        self.rollout_attns = []
        for attn in all_attentions:
            self_attn, pool_attn = extract_attention_data(attn)
            self.self_attn_per_layer.append(self_attn)
            self.pool_attns.append(pool_attn)
            self.rollout_attns.append(compute_attention_rollout(attn))

        # 设置全局暗色主题
        plt.rcParams.update({
            'figure.facecolor': DARK_BG,
            'axes.facecolor': PANEL_BG,
            'axes.edgecolor': EDGE_COLOR,
            'text.color': TEXT_COLOR,
            'axes.labelcolor': TEXT_COLOR,
            'xtick.color': '#888888',
            'ytick.color': '#888888',
            'grid.color': GRID_COLOR,
            'grid.alpha': 0.3,
        })

        self._setup_figure()

    def _setup_figure(self):
        """创建图窗和子图布局。"""
        self.fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
        self.fig.canvas.manager.set_window_title('EquiNet Attention Visualizer')

        gs = gridspec.GridSpec(
            4, 1, height_ratios=[3, 0.8, 0.3, 1.8],
            hspace=0.08, left=0.06, right=0.97, top=0.92, bottom=0.05)

        self.ax_candle = self.fig.add_subplot(gs[0])
        self.ax_vol = self.fig.add_subplot(gs[1], sharex=self.ax_candle)
        self.ax_attn_strip = self.fig.add_subplot(gs[2], sharex=self.ax_candle)
        self.ax_heatmap = self.fig.add_subplot(gs[3])

        # Colorbar（只创建一次，避免重复叠加）
        dummy = np.zeros((1, 1))
        self._cbar_im = self.ax_heatmap.imshow(dummy, aspect='auto', cmap=ATTN_CMAP, vmin=0)
        self._cbar = self.fig.colorbar(self._cbar_im, ax=self.ax_heatmap,
                                       orientation='vertical', fraction=0.015, pad=0.01)
        self._cbar.ax.tick_params(labelsize=7, colors='#888888')
        self._cbar.outline.set_edgecolor(EDGE_COLOR)

        # 标题区域
        self.title_text = self.fig.text(
            0.5, 0.97, '', fontsize=15, fontweight='bold',
            color='#ffffff', ha='center', va='top',
            fontfamily='monospace')

        self.info_text = self.fig.text(
            0.5, 0.945, '', fontsize=11,
            color='#aaaaaa', ha='center', va='top',
            fontfamily='monospace')

        # 底部操作提示
        self.fig.text(
            0.5, 0.01,
            '[Space] Next   [Backspace] Prev   [L] Layer/Head   '
            '[R] Rollout overlay   [Q] Quit',
            fontsize=10, color='#666666', ha='center', va='bottom',
            fontfamily='monospace')

        # 键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _on_key_press(self, event):
        if event.key == ' ':
            self.current_idx = (self.current_idx + 1) % self.total
            self._update_display()
            self.fig.canvas.draw_idle()
        elif event.key == 'backspace':
            self.current_idx = (self.current_idx - 1) % self.total
            self._update_display()
            self.fig.canvas.draw_idle()
        elif event.key == 'l':
            self.mode = 'per_head' if self.mode == 'per_layer' else 'per_layer'
            self._update_display()
            self.fig.canvas.draw_idle()
        elif event.key == 'r':
            self.show_rollout_overlay = not self.show_rollout_overlay
            self._update_display()
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)

    def _update_display(self):
        """重绘当前样本的所有图表。"""
        idx = self.current_idx
        sample = self.samples[idx]

        # 清空所有轴
        self.ax_candle.clear()
        self.ax_vol.clear()
        self.ax_attn_strip.clear()
        self.ax_heatmap.clear()

        # 绘制K线图
        rollout = self.rollout_attns[idx]
        self._draw_candlestick(sample, rollout)

        # 绘制注意力条带 (K线下方的注意力强度条)
        self._draw_attn_strip(rollout)

        # 绘制注意力热力图
        self._draw_heatmap(
            self.self_attn_per_layer[idx],
            self.pool_attns[idx],
            rollout)

        # 更新标题
        self._update_title(sample, idx)

        # 恢复轴样式
        for ax in [self.ax_candle, self.ax_vol, self.ax_attn_strip, self.ax_heatmap]:
            ax.set_facecolor(PANEL_BG)
            for spine in ax.spines.values():
                spine.set_color(EDGE_COLOR)

    def _update_title(self, sample, idx):
        """更新标题和副标题。"""
        code = sample['stock_code']
        dates = sample['dates']
        pred = self.predictions[idx]
        label = sample['label']
        ret = sample['cumulative_return']

        date_start = str(dates[0])
        date_end = str(dates[-1])
        date_start_fmt = f"{date_start[:4]}.{date_start[4:6]}.{date_start[6:]}"
        date_end_fmt = f"{date_end[:4]}.{date_end[4:6]}.{date_end[6:]}"

        label_str = "POSITIVE" if label == 1 else "NEGATIVE"

        self.title_text.set_text(
            f"{code}  |  {date_start_fmt} - {date_end_fmt}  |  "
            f"Sample {idx + 1}/{self.total}")

        pred_bar = '#' * int(pred * 20) + '-' * (20 - int(pred * 20))
        ret_str = f"{ret * 100:+.2f}%" if ret != 0 else "N/A"
        self.info_text.set_text(
            f"Pred: {pred:.4f} [{pred_bar}]  |  "
            f"Label: {label_str}  |  "
            f"Return: {ret_str}")

    def _draw_candlestick(self, sample, rollout):
        """绘制K线图。"""
        ax = self.ax_candle
        ohlcv = sample['raw_ohlcv']

        opens = ohlcv[:, 0]
        highs = ohlcv[:, 1]
        lows = ohlcv[:, 2]
        closes = ohlcv[:, 3]

        n = len(closes)
        x = np.arange(n)

        # 注意力高亮背景
        if self.show_rollout_overlay:
            rollout_norm = rollout / (rollout.max() + 1e-8)
            for i in range(n):
                alpha = float(np.clip(rollout_norm[i] * 0.4, 0, 0.4))
                ax.axvspan(i - 0.4, i + 0.4, alpha=alpha, color=ATTN_HIGH, linewidth=0)

        # 绘制蜡烛图
        for i in range(n):
            is_up = closes[i] >= opens[i]
            color = UP_COLOR if is_up else DOWN_COLOR

            # 影线 (high-low)
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=0.8)

            # 实体 (open-close)
            body_bottom = min(opens[i], closes[i])
            body_height = abs(closes[i] - opens[i])
            if body_height < 0.01:
                body_height = closes[i] * 0.002
            ax.bar(x[i], body_height, bottom=body_bottom,
                   width=0.6, color=color, edgecolor=color, linewidth=0.5)

        # Y轴: 显式设置范围
        price_min, price_max = lows.min(), highs.max()
        padding = (price_max - price_min) * 0.05
        ax.set_ylim(price_min - padding, price_max + padding)

        # Y轴价格标注
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelbottom=False)

        # 网格
        ax.grid(True, alpha=0.15, axis='y')
        ax.set_xlim(-0.5, n - 0.5)

        # 绘制成交量（独立子图）
        self._draw_volume(sample)

    def _draw_volume(self, sample):
        """绘制成交量柱状图。"""
        ax = self.ax_vol
        ohlcv = sample['raw_ohlcv']
        volumes = sample['raw_volumes']
        closes = ohlcv[:, 3]
        opens = ohlcv[:, 0]

        n = len(volumes)
        x = np.arange(n)

        for i in range(n):
            is_up = closes[i] >= opens[i]
            color = UP_COLOR if is_up else DOWN_COLOR
            ax.bar(x[i], volumes[i], width=0.6, color=color, alpha=0.7, linewidth=0)

        ax.set_xlim(-0.5, n - 0.5)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelsize=7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'{v / 1e6:.0f}M' if v >= 1e6 else f'{v / 1e3:.0f}K'))
        ax.grid(True, alpha=0.1, axis='y')
        ax.set_ylabel('Vol', fontsize=7, color='#666666')

    def _draw_attn_strip(self, rollout):
        """绘制K线下方的注意力强度条。"""
        ax = self.ax_attn_strip
        rollout_norm = rollout / (rollout.max() + 1e-8)
        rollout_2d = rollout_norm[np.newaxis, :]  # (1, 45)

        ax.imshow(rollout_2d, aspect='auto', cmap=ATTN_CMAP,
                  interpolation='nearest', vmin=0, vmax=1)
        ax.set_yticks([])
        ax.set_xticks([])

    def _draw_heatmap(self, self_attn_per_layer, pool_attn, rollout):
        """绘制注意力热力图。"""
        ax = self.ax_heatmap

        if self.mode == 'per_layer':
            # 每层平均所有头 + Pooling行 + Rollout行
            layer_avg = self_attn_per_layer.mean(axis=1)  # (num_layers, 45)
            pool_row = pool_attn.mean(axis=0)[np.newaxis, :]  # (1, 45)
            rollout_row = rollout[np.newaxis, :]
            heatmap_data = np.vstack([layer_avg, pool_row, rollout_row])

            row_labels = ([f'Layer {i+1}' for i in range(self.num_layers)]
                          + ['Pooling', 'Rollout'])
        else:
            # 逐层逐头展开 + Pooling逐头 + Rollout行
            rows = []
            row_labels = []
            for l in range(self.num_layers):
                for h in range(self.nhead):
                    rows.append(self_attn_per_layer[l, h])
                    row_labels.append(f'L{l+1}H{h+1}')
            for h in range(self.nhead):
                rows.append(pool_attn[h])
                row_labels.append(f'PH{h+1}')
            rollout_row = rollout[np.newaxis, :]
            heatmap_data = np.vstack(rows + [rollout_row])
            row_labels.append('Rollout')

        im = ax.imshow(heatmap_data, aspect='auto', cmap=ATTN_CMAP,
                        interpolation='nearest', vmin=0)

        # 更新预创建的 colorbar 数据
        self._cbar.update_normal(im)

        # 行标签
        ax.set_yticks(range(len(row_labels)))
        if self.mode == 'per_head':
            ax.set_yticklabels(row_labels, fontsize=6)
        else:
            ax.set_yticklabels(row_labels, fontsize=9)

        # X轴日期标注
        n = heatmap_data.shape[1]
        dates = self.samples[self.current_idx]['dates']
        tick_positions = list(range(0, n, 5))
        tick_labels = []
        for pos in tick_positions:
            d = str(dates[pos])
            tick_labels.append(f"{d[4:6]}/{d[6:]}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8)

        # 分隔线: 自注意力与聚合层之间
        sep_idx = self.num_layers - 1 if self.mode == 'per_layer' else self.num_layers * self.nhead - 1
        ax.axhline(y=sep_idx + 0.5, color='#444466', linewidth=0.6, linestyle=':')
        # 分隔线: 聚合层与Rollout之间
        ax.axhline(y=len(row_labels) - 1.5, color='#555555', linewidth=0.8, linestyle='--')

    def show(self):
        """显示交互式图窗。"""
        self._update_display()
        plt.show()


# ==================== 主函数 ====================
def main():
    # 设备
    device = DeviceConfig.get_device()
    if device.type != "cuda":
        print("错误: 需要 CUDA GPU 才能运行可视化工具。")
        sys.exit(1)
    print(f"Device: GPU ({torch.cuda.get_device_name()})")

    # 模型选择
    models = list_available_models()
    if not models:
        print("\n  没有可用的模型，请先训练模型。")
        return

    model_idx = select_model(models)
    selected_file = models[model_idx]
    model_path = os.path.join(DataConfig.OUTPUT_DIR, selected_file)

    print(f"\n  正在加载模型: {selected_file}")
    model, metadata = load_model(model_path, device)

    # 获取模型架构信息
    arch = metadata.get('model_arch', {}) if metadata else {}
    nhead = arch.get('nhead', ModelConfig.NHEAD)
    num_layers = arch.get('num_layers', ModelConfig.NUM_LAYERS)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params:,}")
    print(f"  层数: {num_layers}  头数: {nhead}")

    # 加载归一化器
    if not os.path.exists(DataConfig.NORMALIZER_PATH):
        print(f"\n  错误: 归一化器不存在: {DataConfig.NORMALIZER_PATH}")
        print(f"  请先运行: python data.py")
        return

    feature_normalizer = FeatureNormalizer.load(DataConfig.NORMALIZER_PATH)

    # 加载数据
    print(f"\n  正在加载数据...")
    _, _, test_stock_info = load_and_preprocess_data()
    print(f"  测试集: {len(test_stock_info)} 只股票")

    # 构建样本
    print(f"\n  正在构建可视化样本...")
    samples = load_test_samples(test_stock_info, feature_normalizer, max_samples=300)
    if not samples:
        print("  没有可用的测试样本。")
        return
    print(f"  共 {len(samples)} 个样本")

    # 推理（提取注意力）
    print(f"  正在提取注意力权重...")
    predictions, all_attentions = run_attention_inference(model, samples, device)

    # 统计
    pos_count = sum(1 for s in samples if s['label'] == 1)
    neg_count = len(samples) - pos_count
    print(f"  正样本: {pos_count}  负样本: {neg_count}")

    # 启动可视化
    print(f"\n可视化窗口启动...")

    viz = AttentionVisualizer(samples, predictions, all_attentions, num_layers, nhead)
    viz.show()


if __name__ == '__main__':
    main()
