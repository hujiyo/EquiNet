"""
SIGReg: Sketched Isotropic Gaussian Regularization

基于 LeJEPA 论文 (Balestriero & LeCun, 2025) 实现。
通过随机投影 + Epps-Pulley 检验统计量，约束 embedding 分布趋向各向同性高斯。

参考: https://github.com/galilai-group/lejepa
"""

import torch
import torch.nn as nn


class EppsPulley(nn.Module):
    """
    Epps-Pulley 检验统计量（基于经验特征函数）

    对 1D 样本检验 H0: X ~ N(0,1)
    通过梯形积分近似: T = N * ∫ |φ_emp(t) - exp(-t²/2)|² * w(t) dt
    """

    def __init__(self, t_max=3, n_points=17):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd"

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        phi = (t.square() * 0.5).neg().exp()

        self.register_buffer('t', t)
        self.register_buffer('combined_weights', weights * phi)
        self.register_buffer('phi', phi)

    def forward(self, x):
        """
        Args:
            x: [*, N, K] -- N 个样本，K 个投影方向
        Returns:
            [*, K] -- 每个投影方向的 EP 检验统计量
        """
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_mean = torch.cos(x_t).mean(dim=-3)
        sin_mean = torch.sin(x_t).mean(dim=-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.combined_weights) * N


class SIGRegLoss(nn.Module):
    """
    SIGReg 损失：随机投影 + Epps-Pulley 检验

    将高维 embedding 投影到多个随机 1D 方向，
    对每个方向计算 EP 统计量，衡量偏离 N(0,1) 的程度。
    梯度回传推动 embedding 分布趋向各向同性高斯。

    每次 forward 用 global_step 作种子在独立 torch.Generator 上重新生成投影矩阵
    （项目自创机制，不是 LeJEPA 官方实现 —— 官方 MINIMAL.md 直接用
    `torch.randn(..., device=...)` 走全局 RNG，无 manual_seed、无 step 索引）：
    - 算法上等价：每步拿到 IID 高斯列向量，列归一化后即可累积足够多随机方向
      满足 Cramér-Wold 假设（K 步累计 K·num_slices 个不同方向）
    - 工程动机：将 SIGReg 的随机源从全局 RNG 隔离，避免别处 `torch.randn/rand`
      调用顺序变动破坏投影序列的复现性
    - 投影矩阵在 torch.no_grad() 下生成，梯度仅通过 embedding 回传
    """

    def __init__(self, d_model=128, num_slices=256, t_max=3, n_points=17):
        super().__init__()
        self.d_model = d_model
        self.num_slices = num_slices
        self.ep_test = EppsPulley(t_max=t_max, n_points=n_points)
        self.global_step = torch.zeros(1, dtype=torch.long)
        self._generators = {}

    def _get_generator(self, device, seed):
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        g = self._generators[device]
        g.manual_seed(seed)
        return g

    def forward(self, z):
        """
        Args:
            z: [batch, d_model] -- embedding 向量
        Returns:
            scalar: SIGReg 损失值
        """
        with torch.no_grad():
            seed = self.global_step.item()
            g = self._get_generator(z.device, seed)
            A = torch.randn(self.d_model, self.num_slices,
                            device=z.device, dtype=z.dtype, generator=g)
            A = A / A.norm(p=2, dim=0, keepdim=True)
            self.global_step.add_(1)

        z_proj = z @ A
        z_proj = z_proj.unsqueeze(0)
        stats = self.ep_test(z_proj)
        return stats.mean()
