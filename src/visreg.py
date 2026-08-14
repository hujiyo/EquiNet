"""
VISReg: Variance-Invariance-Sketching Regularization

基于 Wu, Balestriero & Levine, 2026 (arXiv:2606.02572) 实现
SIGReg 的改进版：将嵌入空间正则化解耦为尺度、形状、中心化三个独立分量，
用 Sliced-Wasserstein 距离替代 Epps-Pulley 检验。
参考: https://haiyuwu.github.io/visreg

---

SIGReg: Sketched Isotropic Gaussian Regularization
于 LeJEPA 论文 (Balestriero & LeCun, 2025) 中提出
参考: https://github.com/galilai-group/lejepa

---
调用方应在传入前把 z 缩放到目标尺度（即 z / target_std），如EmbeddingConfig.TARGET_STD
"""

import math
import torch
import torch.nn as nn


class VISRegLoss(nn.Module):
    """
    VISReg 损失：尺度 + 形状(SWD) + 中心化 三个解耦分量
    """

    def __init__(self, num_slices=256,
                 w_scale=1.0, w_shape=1.0, w_center=1.0):
        super().__init__()
        self.num_slices = num_slices
        self.w_scale = w_scale
        self.w_shape = w_shape
        self.w_center = w_center

        self._quantile_cache = {}  # N -> tensor(q_N, device, dtype)

    def _gaussian_quantiles(self, N, device, dtype):
        """
        1D 2-Wasserstein 闭式解的目标分位数：
            q_i = icdf_N(0,1)( i / (N+1) ),  i = 1..N
        icdf(u) = sqrt(2) * erfinv(2u - 1)

        【必须在 fp32 下计算】erfinv 在 u→1 时趋向 +inf，对精度极敏感。
        在 bf16 AMP 下 N/(N+1)（如 4096/4097≈0.99976）会被舍入到 1.0，
        导致 erfinv(2·1−1)=erfinv(1)=+inf，污染整条损失为 nan。
        因此这里禁用 autocast、强制 fp32 计算，缓存后按需 cast 回 z.dtype。
        结果按 (N, device) 缓存。
        """
        key = (N, str(device))
        if key not in self._quantile_cache:
            with torch.autocast(device_type=device.type, enabled=False):
                u = torch.arange(1, N + 1, device=device, dtype=torch.float32) / (N + 1)
                sqrt2 = torch.tensor(math.sqrt(2.0), device=device, dtype=torch.float32)
                q = sqrt2 * torch.erfinv(2 * u - 1)
            self._quantile_cache[key] = q
        return self._quantile_cache[key].to(dtype)

    def forward(self, z):
        """
        Args:
            z: [N, D] embedding（调用方应已按 target_std 缩放，
               使尺度损失 target std=1，等价于原 std=target_std）
        Returns:
            scalar: w_scale·L_scale + w_shape·L_shape + w_center·L_center
        """
        N, D = z.size(0), z.size(1)

        # 1. 中心化损失：批均值→0
        mu = z.mean(dim=0)                           # [D]
        L_center = mu.pow(2).mean()

        # 2. 尺度损失：逐维标准差→1（软 L2 双向惩罚，std<1 与 std>1 均惩罚；
        #    区别于 VICReg 单向 hinge max(0,γ-std)^2；坍塌时梯度趋于常数）
        z_cent = z - mu                              # [N, D]
        std = z_cent.std(dim=0, unbiased=False)       # [D]
        L_scale = (1.0 - std).pow(2).mean()

        # 3. 形状损失 (Sliced Wasserstein Distance)
        #    用 std.detach() 归一化（stop-gradient），形状优化不干扰尺度调节。
        #    归一化在 fp32 下进行：bf16 下小 std 可能舍入为 0 或丢精度，导致
        #    z_norm 出现 inf/nan；fp32 除法更稳，结果再 cast 回 z.dtype。
        std_sg = std.detach().unsqueeze(0).clamp_min(1e-12)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z_norm = (z_cent.float() / std_sg.float()).to(z.dtype)

        with torch.no_grad():
            W = torch.randn(D, self.num_slices,
                            device=z.device, dtype=z.dtype)
            W = W / W.norm(p=2, dim=0, keepdim=True)

        p = z_norm @ W                               # [N, K]
        p_sorted = torch.sort(p, dim=0).values       # [N, K]

        q = self._gaussian_quantiles(N, z.device, z.dtype)  # [N]
        L_shape = (p_sorted - q.unsqueeze(1)).pow(2).mean()

        return (self.w_scale * L_scale
                + self.w_shape * L_shape
                + self.w_center * L_center)
