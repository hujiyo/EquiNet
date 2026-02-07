"""
混合优化器：HybridManoAdamW

结合Mano和AdamW的优势：
- Mano优化2D矩阵参数（Transformer的权重矩阵）
- AdamW优化Embedding层、LayerNorm和1D参数（bias, LayerNorm weight）

这种混合策略遵循论文建议，利用不同优化器的优势：
1. Mano对矩阵参数进行流形归一化，提升训练效率
2. AdamW对1D参数和稀疏激活层进行自适应学习率调整
"""

import torch.optim as optim
from .mano import ManoOptimizer


class HybridManoAdamW(optim.Optimizer):
    """
    混合优化器：Mano用于2D矩阵，AdamW用于1D向量和Embedding

    用法：
        mano_params = [p for p in model.parameters() if p.dim() >= 2]
        adamw_params = [p for p in model.parameters() if p.dim() < 2]

        optimizer = HybridManoAdamW(
            mano_params=mano_params,
            adamw_params=adamw_params,
            lr=5e-4,
            momentum=0.95,
            weight_decay=0.01
        )
    """

    def __init__(self, mano_params, adamw_params, lr, momentum, weight_decay, betas=(0.9, 0.95)):
        """
        初始化混合优化器

        参数：
            mano_params (list): Mano优化的参数列表（2D矩阵）
            adamw_params (list): AdamW优化的参数列表（1D向量）
            lr (float): 学习率
            momentum (float): Mano动量系数
            weight_decay (float): 权重衰减
            betas (tuple): AdamW的beta参数，默认(0.9, 0.95)
        """
        # 过滤掉空的参数列表
        mano_params = [p for p in mano_params if p is not None]
        adamw_params = [p for p in adamw_params if p is not None]

        # 构建参数组用于父类初始化（这是PyTorch Optimizer要求的）
        all_params = mano_params + adamw_params
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, betas=betas)
        super().__init__(all_params, defaults)

        # Mano优化器（仅当存在2D参数时才创建）
        if len(mano_params) > 0:
            self.mano_optim = ManoOptimizer(
                mano_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            self.mano_optim = None

        # AdamW优化器（仅当存在1D参数时才创建）
        if len(adamw_params) > 0:
            self.adamw_optim = optim.AdamW(
                adamw_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
        else:
            self.adamw_optim = None

        # 重新组织param_groups，合并两个优化器的参数组
        self.param_groups = []
        if self.mano_optim:
            self.param_groups.extend(self.mano_optim.param_groups)
        if self.adamw_optim:
            self.param_groups.extend(self.adamw_optim.param_groups)

    def zero_grad(self):
        """清空所有优化器的梯度"""
        if self.mano_optim:
            self.mano_optim.zero_grad()
        if self.adamw_optim:
            self.adamw_optim.zero_grad()

    def step(self, closure=None):
        """
        执行一步优化

        参数：
            closure (callable, optional): 重新计算模型并返回loss的闭包

        返回：
            loss (Tensor or None): 闭包返回的loss值
        """
        loss_mano = None
        loss_adamw = None

        if self.mano_optim:
            loss_mano = self.mano_optim.step(closure)
        if self.adamw_optim:
            loss_adamw = self.adamw_optim.step(closure)

        return loss_mano if loss_mano is not None else loss_adamw

    def state_dict(self):
        """
        返回优化器状态

        返回：
            dict: 包含'mano'和'adamw'两个键的字典
        """
        state = {}
        if self.mano_optim:
            state['mano'] = self.mano_optim.state_dict()
        if self.adamw_optim:
            state['adamw'] = self.adamw_optim.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """
        加载优化器状态

        参数：
            state_dict (dict): 包含'mano'和'adamw'两个键的字典
        """
        if 'mano' in state_dict and self.mano_optim:
            self.mano_optim.load_state_dict(state_dict['mano'])
        if 'adamw' in state_dict and self.adamw_optim:
            self.adamw_optim.load_state_dict(state_dict['adamw'])


def create_optimizer(model, optimizer_type='adamw', lr=1e-3, momentum=0.95, weight_decay=1e-5, betas=(0.9, 0.95)):
    """
    根据类型创建优化器

    参数：
        model: PyTorch模型
        optimizer_type (str): 优化器类型，'adamw'或'mano'
        lr (float): 学习率
        momentum (float): Mano动量系数
        weight_decay (float): 权重衰减
        betas (tuple): AdamW的beta参数

    返回：
        optimizer: 创建好的优化器
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'mano':
        # 分离2D和1D参数
        mano_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.dim() >= 2:
                mano_params.append(param)
            else:
                adamw_params.append(param)

        print(f"优化器: HybridManoAdamW (Mano用于2D矩阵, AdamW用于1D参数)")
        print(f"  2D参数数量: {len(mano_params)}")
        print(f"  1D参数数量: {len(adamw_params)}")

        return HybridManoAdamW(
            mano_params=mano_params,
            adamw_params=adamw_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas
        )
    else:
        # 默认使用AdamW
        if optimizer_type == 'adamw':
            print(f"优化器: AdamW (weight_decay={weight_decay})")
            return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            print(f"优化器: Adam (weight_decay={weight_decay})")
            return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
