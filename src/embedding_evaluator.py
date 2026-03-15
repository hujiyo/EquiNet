"""
Embedding层评估脚本

核心思路：Embedding层是一个映射函数 f: R^6 → R^48
评估重点：映射本身的性质，而非下游任务效果

评估维度：
1. 局部敏感性分析：输入微小扰动时，输出如何变化
2. 全局敏感性分析: 不同输入范围，输出变化幅度
3. Jacobian分析: 每个输入维度对每个输出维度的影响
4. 表示多样性: 不同输入产生的输出是否足够分散
5. 饱和度分析: GELU激活是否饱和
6. 扰动传播: 各输入维度的扰动如何传播到输出
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import argparse

from model import create_model
from data import load_and_preprocess_data
from config import DataConfig, ModelConfig


class EmbeddingLayerAnalyzer:
    """Embedding层分析器 - 评估映射函数本身的性质"""
    
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.embedding_layer = None
        
    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model_arch = checkpoint.get('model_arch', {})
                self.model = create_model(model_arch=model_arch).to(self.device)
                
                # 加载权重，允许部分不匹配（新旧模型结构差异）
                current_state = self.model.state_dict()
                loaded_state = checkpoint['state_dict']
                
                # 只加载匹配的权重
                matched_state = {}
                for key in current_state:
                    if key in loaded_state:
                        if current_state[key].shape == loaded_state[key].shape:
                            matched_state[key] = loaded_state[key]
                        else:
                            print(f"  跳过不匹配的权重: {key}")
                    else:
                        print(f"  跳过缺失的权重: {key}")
                
                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型: {self.model_path}")
            else:
                self.model = create_model().to(self.device)
                # 旧格式checkpoint直接是state_dict
                current_state = self.model.state_dict()
                
                matched_state = {}
                for key in current_state:
                    if key in checkpoint:
                        if current_state[key].shape == checkpoint[key].shape:
                            matched_state[key] = checkpoint[key]
                
                current_state.update(matched_state)
                self.model.load_state_dict(current_state)
                print(f"加载训练好的模型(旧格式): {self.model_path}")
        else:
            self.model = create_model().to(self.device)
            print("使用随机初始化模型")
        self.model.eval()
        
        if hasattr(self.model, 'embedding'):
            self.embedding_layer = self.model.embedding
        else:
            raise AttributeError("模型不包含embedding层")
        
        return self.model
    
    def analyze_jacobian(self, sample_inputs=None, n_samples=100, store_matrices=False):
        """
        Jacobian矩阵分析: 计算 ∂output/∂input
        
        Jacobian矩阵 J[i,j] = ∂output_j / ∂input_i
        揭示每个输入维度对每个输出维度的影响
        
        Args:
            sample_inputs: 测试样本
            n_samples: 样本数量
            store_matrices: 是否存储完整的Jacobian矩阵(可能占用大量内存)
        """
        print("\n[Jacobian矩阵分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        results = {
            'mean_jacobian_norm': [],
            'per_input_sensitivity': [],
            'per_output_sensitivity': []
        }
        
        if store_matrices:
            results['jacobian_matrices'] = []
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        
        for i in range(min(n_samples, len(sample_inputs))):
            x = sample_inputs[i:i+1].clone().requires_grad_(True)
            
            output = self.embedding_layer(x)
            
            jacobian = torch.zeros(6, 48, device=self.device)
            
            for out_dim in range(48):
                if x.grad is not None:
                    x.grad.zero_()
                
                output[0, :, out_dim].sum().backward(retain_graph=True)
                jacobian[:, out_dim] = x.grad[0, 0, :].clone()
            
            if store_matrices:
                results['jacobian_matrices'].append(jacobian.cpu().numpy())
            results['mean_jacobian_norm'].append(torch.norm(jacobian).item())
            
            input_sensitivity = torch.norm(jacobian, dim=1)
            results['per_input_sensitivity'].append(input_sensitivity.cpu().numpy())
            
            output_sensitivity = torch.norm(jacobian, dim=0)
            results['per_output_sensitivity'].append(output_sensitivity.cpu().numpy())
        
        avg_input_sens = np.mean(results['per_input_sensitivity'], axis=0)
        avg_output_sens = np.mean(results['per_output_sensitivity'], axis=0)
        
        print(f"  平均Jacobian范数: {np.mean(results['mean_jacobian_norm']):.4f}")
        print(f"  各输入维度敏感性:")
        for i, name in enumerate(feature_names):
            print(f"    {name}: {avg_input_sens[i]:.4f}")
        
        return_dict = {
            'mean_jacobian_norm': float(np.mean(results['mean_jacobian_norm'])),
            'input_sensitivity': {name: float(avg_input_sens[i]) for i, name in enumerate(feature_names)},
            'output_sensitivity_range': [float(np.min(avg_output_sens)), float(np.max(avg_output_sens))]
        }
        
        if store_matrices:
            return_dict['jacobian_matrices'] = results['jacobian_matrices']
            
        return return_dict
    
    def analyze_local_sensitivity(self, sample_inputs=None, n_samples=50, epsilon=1e-4):
        """
        局部敏感性分析: 输入微小扰动时，输出如何变化
        
        对于每个样本，在每个输入维度上添加微小扰动，观察输出变化
        """
        print("\n[局部敏感性分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        results = {name: [] for name in feature_names}
        results['overall'] = []
        
        with torch.no_grad():
            for i in range(min(n_samples, len(sample_inputs))):
                x = sample_inputs[i:i+1]
                base_output = self.embedding_layer(x)
                
                for j, name in enumerate(feature_names):
                    x_perturbed = x.clone()
                    x_perturbed[0, :, j] += epsilon
                    
                    perturbed_output = self.embedding_layer(x_perturbed)
                    
                    diff = torch.norm(perturbed_output - base_output).item()
                    results[name].append(diff)
                
                x_all_perturbed = x.clone()
                x_all_perturbed += epsilon
                all_perturbed_output = self.embedding_layer(x_all_perturbed)
                overall_diff = torch.norm(all_perturbed_output - base_output).item()
                results['overall'].append(overall_diff)
        
        summary = {}
        print(f"  扰动幅度 ε = {epsilon}")
        print(f"  各维度扰动导致的输出变化:")
        for name in feature_names:
            mean_diff = np.mean(results[name])
            std_diff = np.std(results[name])
            summary[name] = {'mean': float(mean_diff), 'std': float(std_diff)}
            print(f"    {name}: {mean_diff:.6f} ± {std_diff:.6f}")
        
        print(f"  全维度同时扰动: {np.mean(results['overall']):.6f}")
        summary['overall'] = {'mean': float(np.mean(results['overall'])), 'std': float(np.std(results['overall']))}
        
        return summary
    
    def analyze_global_sensitivity(self, sample_inputs=None, n_samples=100):
        """
        全局敏感性分析: 不同输入范围，输出变化幅度
        
        将每个输入维度从-0.1到0.1变化，观察输出变化
        """
        print("\n[全局敏感性分析]")
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        results = {name: {'output_range': [], 'output_std': []} for name in feature_names}
        
        if sample_inputs is None:
            base_input = torch.zeros(1, 30, 6, device=self.device)
        else:
            if isinstance(sample_inputs, np.ndarray):
                base_input = torch.tensor(sample_inputs[:1], dtype=torch.float32, device=self.device)
            else:
                base_input = sample_inputs[:1]
        
        with torch.no_grad():
            for j, name in enumerate(feature_names):
                outputs = []
                
                values = np.linspace(-0.1, 0.1, 21) if name in ['Open', 'High', 'Low', 'Close'] else np.linspace(0, 1, 21)
                
                for v in values:
                    x = base_input.clone()
                    x[0, :, j] = v
                    output = self.embedding_layer(x)
                    outputs.append(output[0, 0, :].cpu().numpy())
                
                outputs = np.array(outputs)
                output_range = outputs.max(axis=0) - outputs.min(axis=0)
                results[name]['output_range'] = output_range.tolist()
                results[name]['output_std'] = float(np.std(outputs, axis=0).mean())
        
        print(f"  各输入维度变化时，输出变化范围(平均):")
        for name in feature_names:
            avg_range = np.mean(results[name]['output_range'])
            print(f"    {name}: 平均变化范围 = {avg_range:.4f}, 输出std = {results[name]['output_std']:.4f}")
        
        return results
    
    def analyze_input_output_diversity(self, sample_inputs=None, n_samples=500):
        """
        表示多样性分析: 不同输入产生的输出是否足够分散
        
        计算输入空间和输出空间的距离矩阵相关性
        """
        print("\n[表示多样性分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.embedding_layer(sample_inputs)
            
            inputs_flat = sample_inputs.reshape(len(sample_inputs), -1).cpu().numpy()
            outputs_flat = outputs.reshape(len(outputs), -1).cpu().numpy()
            
            
            n_pairs = min(1000, len(sample_inputs) * (len(sample_inputs) - 1) // 2)
            input_dists = []
            output_dists = []
            
            indices = np.random.choice(len(sample_inputs), min(100, len(sample_inputs)), replace=False)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx_i, idx_j = indices[i], indices[j]
                    input_dist = np.linalg.norm(inputs_flat[idx_i] - inputs_flat[idx_j])
                    output_dist = np.linalg.norm(outputs_flat[idx_i] - outputs_flat[idx_j])
                    input_dists.append(input_dist)
                    output_dists.append(output_dist)
            
            input_dists = np.array(input_dists)
            output_dists = np.array(output_dists)
            
            correlation, _ = pearsonr(input_dists, output_dists)
            
            output_norms = np.linalg.norm(outputs_flat, axis=1)
            output_cosine_sims = []
            for i in range(min(200, len(outputs_flat))):
                for j in range(i+1, min(200, len(outputs_flat))):
                    cos_sim = 1 - cosine(outputs_flat[i], outputs_flat[j])
                    output_cosine_sims.append(cos_sim)
            
        print(f"  输入-输出距离相关性: {correlation:.4f}")
        print(f"  输出向量平均余弦相似度: {np.mean(output_cosine_sims):.4f}")
        print(f"  输出向量范数: {np.mean(output_norms):.4f} ± {np.std(output_norms):.4f}")
        print(f"  输出向量范数范围: [{np.min(output_norms):.4f}, {np.max(output_norms):.4f}]")
        
        return {
            'input_output_correlation': float(correlation),
            'output_cosine_similarity': float(np.mean(output_cosine_sims)),
            'output_norm_mean': float(np.mean(output_norms)),
            'output_norm_std': float(np.std(output_norms)),
            'output_norm_range': [float(np.min(output_norms)), float(np.max(output_norms))]
        }
    
    def analyze_saturation(self, sample_inputs=None, n_samples=100):
        """
        饱和度分析: 分析embedding层输出的分布特征
        
        由于embedding是单层Linear，直接分析其输出分布
        """
        print("\n[饱和度分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            x = sample_inputs
            
            hidden = self.embedding_layer(x)
            
            hidden_flat = hidden.cpu().numpy().flatten()
            
            saturation_ratio = np.mean(np.abs(hidden_flat) > 3)
            
            dead_ratio = np.mean(np.abs(hidden_flat) < 0.01)
        
        print(f"  Embedding层输出:")
        print(f"    均值: {np.mean(hidden_flat):.4f}")
        print(f"    标准差: {np.std(hidden_flat):.4f}")
        print(f"    范围: [{np.min(hidden_flat):.4f}, {np.max(hidden_flat):.4f}]")
        print(f"    饱和比例(|x|>3): {saturation_ratio*100:.2f}%")
        print(f"    死神经元比例(|output|<0.01): {dead_ratio*100:.2f}%")
        
        return {
            'hidden_mean': float(np.mean(hidden_flat)),
            'hidden_std': float(np.std(hidden_flat)),
            'hidden_min': float(np.min(hidden_flat)),
            'hidden_max': float(np.max(hidden_flat)),
            'saturation_ratio': float(saturation_ratio),
            'dead_neuron_ratio': float(dead_ratio)
        }
    
    def analyze_critical_points(self, sample_inputs=None, n_samples=50):
        """
        临界点分析: 找出哪些输入区域会导致输出剧烈变化
        
        通过计算二阶导数（Hessian近似）来识别敏感区域
        """
        print("\n[临界点分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        
        second_order_sensitivity = {name: [] for name in feature_names}
        
        epsilon = 1e-3
        
        for i in range(min(n_samples, len(sample_inputs))):
            x = sample_inputs[i:i+1].clone().requires_grad_(True)
            
            output = self.embedding_layer(x)
            
            for j, name in enumerate(feature_names):
                x_plus = x.clone()
                x_plus[0, :, j] += epsilon
                x_minus = x.clone()
                x_minus[0, :, j] -= epsilon
                
                with torch.no_grad():
                    out_plus = self.embedding_layer(x_plus)
                    out_minus = self.embedding_layer(x_minus)
                    out_base = self.embedding_layer(x)
                
                second_deriv = torch.norm(out_plus - 2*out_base + out_minus) / (epsilon ** 2)
                second_order_sensitivity[name].append(second_deriv.item())
        
        print(f"  各输入维度的二阶敏感性(曲率):")
        results = {}
        for name in feature_names:
            mean_sens = np.mean(second_order_sensitivity[name])
            max_sens = np.max(second_order_sensitivity[name])
            results[name] = {'mean': float(mean_sens), 'max': float(max_sens)}
            print(f"    {name}: 平均={mean_sens:.4f}, 最大={max_sens:.4f}")
        
        return results
    
    def analyze_dimension_contribution(self, sample_inputs=None, n_samples=100):
        """
        维度贡献分析: 每个输入维度对每个输出维度的贡献
        
        通过消融实验: 将某个输入维度置零，观察各输出维度的变化
        """
        print("\n[维度贡献分析]")
        
        if sample_inputs is None:
            sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        elif isinstance(sample_inputs, np.ndarray):
            sample_inputs = torch.tensor(sample_inputs[:n_samples], dtype=torch.float32, device=self.device)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        
        with torch.no_grad():
            base_output = self.embedding_layer(sample_inputs)
            base_output_flat = base_output.reshape(-1, 48).cpu().numpy()
            
            contribution_matrix = np.zeros((6, 48))
            
            for j, name in enumerate(feature_names):
                masked_input = sample_inputs.clone()
                masked_input[:, :, j] = 0
                
                masked_output = self.embedding_layer(masked_input)
                masked_output_flat = masked_output.reshape(-1, 48).cpu().numpy()
                
                diff = np.abs(base_output_flat - masked_output_flat).mean(axis=0)
                contribution_matrix[j, :] = diff
        
        print(f"  各输入维度对输出维度的平均贡献:")
        for i, name in enumerate(feature_names):
            avg_contrib = contribution_matrix[i, :].mean()
            max_contrib = contribution_matrix[i, :].max()
            print(f"    {name}: 平均={avg_contrib:.4f}, 最大={max_contrib:.4f}")
        
        return {
            'contribution_matrix': contribution_matrix.tolist(),
            'feature_names': feature_names
        }
    
    def visualize_sensitivity(self, save_dir='embedding_eval_results'):
        """可视化敏感性分析结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']
        
        n_samples = 100
        sample_inputs = torch.randn(n_samples, 30, 6, device=self.device) * 0.1
        sample_inputs[:, :, 4] = torch.rand(n_samples, 30, device=self.device)  # Volume: [0, 1]
        sample_inputs[:, :, 5] = torch.rand(n_samples, 30, device=self.device)  # Exchange: [0, 1]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        print("\n[生成可视化图表...]")
        
        ax = axes[0, 0]
        with torch.no_grad():
            base_output = self.embedding_layer(sample_inputs)
            
            sensitivities = []
            for j, name in enumerate(feature_names):
                diffs = []
                for eps in [1e-5, 1e-4, 1e-3, 1e-2]:
                    x_perturbed = sample_inputs.clone()
                    x_perturbed[:, :, j] += eps
                    perturbed_output = self.embedding_layer(x_perturbed)
                    diff = torch.norm(perturbed_output - base_output).item()
                    diffs.append(diff)
                sensitivities.append(diffs)
            
            sensitivities = np.array(sensitivities)
            epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
            
            for i, name in enumerate(feature_names):
                ax.loglog(epsilons, sensitivities[i], 'o-', label=name)
            ax.set_xlabel('Perturbation Size (ε)')
            ax.set_ylabel('Output Change')
            ax.set_title('Local Sensitivity vs Perturbation Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        with torch.no_grad():
            base_input = torch.zeros(1, 30, 6, device=self.device)
            
            feature_indices = {'Open': 0, 'High': 1, 'Low': 2, 'Close': 3, 'Volume': 4, 'Exchange': 5}
            
            for name in ['Open', 'Close', 'Volume']:
                outputs = []
                j = feature_indices[name]
                values = np.linspace(-0.1, 0.1, 21) if name != 'Volume' else np.linspace(0, 1, 21)
                
                for v in values:
                    x = base_input.clone()
                    x[0, :, j] = v
                    output = self.embedding_layer(x)
                    norm = torch.norm(output).item()
                    outputs.append(norm)
                
                ax.plot(values, outputs, 'o-', label=name)
            
            ax.set_xlabel('Input Value')
            ax.set_ylabel('Output Norm')
            ax.set_title('Output Norm vs Input Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        ax = axes[0, 2]
        with torch.no_grad():
            test_inputs = torch.randn(500, 30, 6, device=self.device) * 0.1
            test_inputs[:, :, 4] = torch.rand(500, 30, device=self.device)
            test_inputs[:, :, 5] = torch.rand(500, 30, device=self.device)
            outputs = self.embedding_layer(test_inputs)
            outputs_flat = outputs.reshape(-1, 48).cpu().numpy()
            
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            outputs_2d = pca.fit_transform(outputs_flat)
            
            ax.scatter(outputs_2d[:, 0], outputs_2d[:, 1], alpha=0.5, s=5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            ax.set_title('Output Space Distribution (PCA)')
        
        ax = axes[1, 0]
        with torch.no_grad():
            x = sample_inputs[:50]
            
            hidden = self.embedding_layer(x)
            
            hidden_flat = hidden.cpu().numpy().flatten()
            
            ax.hist(hidden_flat, bins=50, alpha=0.7, edgecolor='black', label='Embedding Output')
            ax.axvline(x=-3, color='orange', linestyle='--', label='x=±3')
            ax.axvline(x=3, color='orange', linestyle='--')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Embedding Layer Output Distribution')
            ax.legend()
        
        ax = axes[1, 1]
        with torch.no_grad():
            base_output = self.embedding_layer(sample_inputs)
            
            contributions = []
            for j, name in enumerate(feature_names):
                masked = sample_inputs.clone()
                masked[:, :, j] = 0
                masked_output = self.embedding_layer(masked)
                diff = torch.norm(masked_output - base_output).item()
                contributions.append(diff)
            
            bars = ax.bar(feature_names, contributions, alpha=0.7)
            ax.set_ylabel('Output Change (L2 norm)')
            ax.set_title('Feature Contribution (Ablation)')
            
            for bar, contrib in zip(bars, contributions):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{contrib:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax = axes[1, 2]
        with torch.no_grad():
            test_inputs = torch.randn(200, 30, 6, device=self.device) * 0.1
            test_inputs[:, :, 4] = torch.rand(200, 30, device=self.device)
            test_inputs[:, :, 5] = torch.rand(200, 30, device=self.device)
            outputs = self.embedding_layer(test_inputs)
            
            input_norms = torch.norm(test_inputs.reshape(200, -1), dim=1).cpu().numpy()
            output_norms = torch.norm(outputs.reshape(200, -1), dim=1).cpu().numpy()
            
            ax.scatter(input_norms, output_norms, alpha=0.5)
            ax.set_xlabel('Input Norm')
            ax.set_ylabel('Output Norm')
            ax.set_title('Input-Output Norm Relationship')
            
            z = np.polyfit(input_norms, output_norms, 1)
            p = np.poly1d(z)
            ax.plot(input_norms, p(input_norms), "r--", alpha=0.8, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'embedding_sensitivity_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  可视化图表已保存到: {save_dir}/embedding_sensitivity_analysis.png")
    
    def generate_report(self, results, save_dir='embedding_eval_results'):
        """生成分析报告"""
        os.makedirs(save_dir, exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'analysis_results': results,
            'summary': self._generate_summary(results)
        }
        
        with open(os.path.join(save_dir, 'embedding_layer_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def _generate_summary(self, results):
        """生成分析总结"""
        summary = {
            'key_findings': [],
            'potential_issues': [],
            'recommendations': []
        }
        
        if 'local_sensitivity' in results:
            ls = results['local_sensitivity']
            sensitivities = [(name, ls[name]['mean']) for name in ['Open', 'High', 'Low', 'Close', 'Volume', 'Exchange']]
            sensitivities.sort(key=lambda x: x[1], reverse=True)
            
            max_sens = sensitivities[0]
            min_sens = sensitivities[-1]
            
            summary['key_findings'].append(f"最敏感输入维度: {max_sens[0]} (变化={max_sens[1]:.4f})")
            summary['key_findings'].append(f"最不敏感输入维度: {min_sens[0]} (变化={min_sens[1]:.4f})")
            
            if max_sens[1] / (min_sens[1] + 1e-10) > 10:
                summary['potential_issues'].append(f"输入维度敏感性差异过大 ({max_sens[0]}比{min_sens[0]}敏感{max_sens[1]/(min_sens[1]+1e-10):.1f}倍)")
                summary['recommendations'].append("考虑对输入特征进行归一化，使各维度敏感性均衡")
        
        if 'diversity' in results:
            div = results['diversity']
            summary['key_findings'].append(f"输出向量平均余弦相似度: {div['output_cosine_similarity']:.4f}")
            
            if div['output_cosine_similarity'] > 0.9:
                summary['potential_issues'].append("输出向量过于相似，表示多样性不足")
                summary['recommendations'].append("考虑增加embedding维度或添加正则化")
        
        if 'saturation' in results:
            sat = results['saturation']
            summary['key_findings'].append(f"GELU饱和比例: {sat['saturation_ratio']*100:.2f}%")
            summary['key_findings'].append(f"死神经元比例: {sat['dead_neuron_ratio']*100:.2f}%")
            
            if sat['saturation_ratio'] > 0.1:
                summary['potential_issues'].append(f"GELU激活饱和比例较高 ({sat['saturation_ratio']*100:.1f}%)")
                summary['recommendations'].append("考虑添加LayerNorm或调整权重初始化")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Embedding层专业评估工具')
    parser.add_argument('--model', type=str, default=None,
                        help='指定要分析的模型文件路径（例如: ./out/modelB_xxx.pth）。如果不指定，将自动使用最新的模型')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用的模型文件并退出')
    args = parser.parse_args()

    # 设置工作目录为脚本所在目录，确保所有相对路径都相对于脚本位置
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("="*70)
    print("Embedding层专业评估工具")
    print("评估映射函数 f: R^6 → R^48 的性质")
    print("="*70)

    # 列出所有可用模型
    out_dir = './out'
    model_files = []
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            if f.endswith('.pth'):
                model_files.append(os.path.join(out_dir, f))

    if args.list_models:
        print("\n可用的模型文件:")
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for i, mf in enumerate(model_files, 1):
                mtime = datetime.fromtimestamp(os.path.getmtime(mf)).strftime('%Y-%m-%d %H:%M:%S')
                size = os.path.getsize(mf) / (1024 * 1024)  # MB
                print(f"  {i}. {os.path.basename(mf)} ({mtime}, {size:.2f} MB)")
        else:
            print("  未找到任何模型文件")
        return

    # 确定要使用的模型路径
    if args.model:
        if not os.path.exists(args.model):
            print(f"\n错误: 指定的模型文件不存在: {args.model}")
            print("\n可用的模型文件:")
            if model_files:
                model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                for i, mf in enumerate(model_files, 1):
                    print(f"  {i}. {os.path.basename(mf)}")
            else:
                print("  未找到任何模型文件")
            return
        model_path = args.model
        print(f"\n使用指定的模型: {os.path.basename(model_path)}")
    else:
        # 按修改时间排序，取最新的模型
        if model_files:
            model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            print(f"\n找到 {len(model_files)} 个模型文件，使用最新的: {os.path.basename(model_files[0])}")
            model_path = model_files[0]
        else:
            print("\n未找到模型文件，将使用随机初始化模型")
            model_path = None
    
    save_dir = './embedding_eval_results'
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[步骤1] 加载数据...")
    train_stock_info, test_stock_info = load_and_preprocess_data()
    
    print("\n[步骤2] 准备测试样本...")
    all_inputs = []
    for stock in test_stock_info[:50]:
        data = stock['data']
        test_split = stock['test_split_point']
        for i in range(test_split, min(test_split + 10, len(data) - 33)):
            context = data[i:i+30]
            all_inputs.append(context)
    
    sample_inputs = np.array(all_inputs[:500])
    print(f"  准备了 {len(sample_inputs)} 个测试样本")

    print("\n[步骤3] 初始化分析器...")
    analyzer = EmbeddingLayerAnalyzer(model_path=model_path)
    analyzer.load_model()
    
    print("\n[步骤4] 执行分析...")
    results = {}
    
    results['jacobian'] = analyzer.analyze_jacobian(sample_inputs, n_samples=50)
    results['local_sensitivity'] = analyzer.analyze_local_sensitivity(sample_inputs, n_samples=50)
    results['global_sensitivity'] = analyzer.analyze_global_sensitivity(sample_inputs, n_samples=50)
    results['diversity'] = analyzer.analyze_input_output_diversity(sample_inputs, n_samples=200)
    results['saturation'] = analyzer.analyze_saturation(sample_inputs, n_samples=100)
    results['critical_points'] = analyzer.analyze_critical_points(sample_inputs, n_samples=30)
    results['dimension_contribution'] = analyzer.analyze_dimension_contribution(sample_inputs, n_samples=100)
    
    print("\n[步骤5] 生成可视化...")
    analyzer.visualize_sensitivity(save_dir)
    
    print("\n[步骤6] 生成报告...")
    report = analyzer.generate_report(results, save_dir)
    
    print("\n" + "="*70)
    print("分析总结")
    print("="*70)
    
    summary = report['summary']
    
    print("\n关键发现:")
    for finding in summary['key_findings']:
        print(f"  • {finding}")
    
    if summary['potential_issues']:
        print("\n潜在问题:")
        for issue in summary['potential_issues']:
            print(f"  ⚠ {issue}")
    
    if summary['recommendations']:
        print("\n优化建议:")
        for rec in summary['recommendations']:
            print(f"  💡 {rec}")
    
    print("\n" + "="*70)
    print(f"分析完成！结果已保存到: {save_dir}")
    print("="*70)
    
    return results, report


if __name__ == "__main__":
    results, report = main()
