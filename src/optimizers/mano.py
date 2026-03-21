"""
Mano: Restriking Manifold Optimization for LLM Training

Based on arXiv:2601.23000 Algorithm 1

v2 Updates (2026.3.14):
- Removed parameter normalization (p_unit), directly use p.data for tangent projection
- Changed eps handling from clamp(min=eps) to addition (+ eps)
- Nesterov momentum now defaults to True for better data scaling performance
- Implemented dual-dimension projection for improved performance

References:
- https://github.com/xie-lab-ml/Mano-Restriking-Manifold-Optimization-for-LLM-Training
"""

import torch
import torch.optim as optim
import math


class ManoOptimizer(optim.Optimizer):
    """
    Mano Optimizer: Manifold-based optimizer for 2D matrix parameters

    v2 Algorithm:
    1. Update momentum: M_t = μ * M_{t-1} + g_t
    2. Rotating manifold: k = t mod 2
    3. Tangent space projection (v2): v_t = M_t - p_t ⊙ ⟨M_t, p_t⟩_k
    4. Update vector normalization (v2): v̂_t = v_t / (‖v_t‖_{2,k} + eps)
    5. Parameter update: θ_{t+1} = θ_t * (1 - η_t*λ) - η_t * 0.2*√(n_k)*v̂_t

    Dual-dimension projection (optional, enabled by default):
    - Project on both dimensions at each step for improved performance
    - u = g - (g · p) * p  on dim 0, then on dim 1
    - Normalize on both dimensions

    Args:
        params (iterable): iterable of parameters to optimize
        lr (float): learning rate (default: 1e-3)
        momentum (float): momentum coefficient (default: 0.95)
        weight_decay (float): weight decay (default: 0.1)
        nesterov (bool): whether to use Nesterov-style momentum (default: True)
        eps (float): epsilon for numerical stability (default: 1e-8)
        dual_dim_projection (bool): whether to use dual-dimension projection (default: True)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.95,
        weight_decay=0.1,
        nesterov=True,
        eps=1e-8,
        dual_dim_projection=True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum should be in [0,1): {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            dual_dim_projection=dual_dim_projection,
            steps=0
        )
        super().__init__(params, defaults)

    def _tangent_projection_single_dim(self, g, p, dim):
        """
        Single-dimension tangent projection: v = g - p * <g, p>_dim

        Args:
            g (Tensor): gradient/momentum tensor
            p (Tensor): parameter tensor
            dim (int): dimension for projection (0 or 1)

        Returns:
            Tensor: projected tangent vector
        """
        inner_prod = torch.sum(g * p, dim=dim, keepdim=True)
        return g - p * inner_prod

    def _normalize_single_dim(self, v, dim, eps):
        """
        Single-dimension normalization: v / (||v||_dim + eps)

        Args:
            v (Tensor): vector to normalize
            dim (int): dimension for normalization
            eps (float): epsilon for numerical stability

        Returns:
            Tensor: normalized vector
        """
        norm = torch.norm(v, p=2, dim=dim, keepdim=True)
        return v / (norm + eps)

    def _dual_dim_projection_and_normalize(self, g, p, eps):
        """
        Dual-dimension projection and normalization (v2 improvement)

        Project on both dimensions at each step for improved performance.

        Args:
            g (Tensor): gradient/momentum tensor
            p (Tensor): parameter tensor
            eps (float): epsilon for numerical stability

        Returns:
            Tensor: projected and normalized update vector
        """
        u = g - (torch.sum(g * p, dim=0, keepdim=True) * p)
        u = u - (torch.sum(u * p, dim=1, keepdim=True) * p)

        u = u / (torch.norm(u, p=2, dim=0, keepdim=True) + eps)
        u = u / (torch.norm(u, p=2, dim=1, keepdim=True) + eps)

        return u

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss

        Returns:
            loss (Tensor or None): loss value from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mu = group['momentum']
            weight_decay = group['weight_decay']
            lr = group['lr']
            nesterov = group['nesterov']
            eps = group['eps']
            dual_dim = group['dual_dim_projection']

            dim = int(group['steps'] % 2)

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad.data
                assert g.ndim == 2, f"Mano only supports 2D parameters, got {g.ndim}D"

                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(mu).add_(g)

                g = g.add(buf, alpha=mu) if nesterov else buf

                if dual_dim:
                    u = self._dual_dim_projection_and_normalize(g, p.data, eps)
                else:
                    v = self._tangent_projection_single_dim(g, p.data, dim)
                    u = self._normalize_single_dim(v, dim, eps)

                p.data.mul_(1 - lr * weight_decay)

                adjusted_lr = lr * 0.2 * math.sqrt(g.shape[dim])
                p.data.add_(u, alpha=-adjusted_lr)

            group['steps'] += 1

        return loss


class HybridManoAdamW(optim.Optimizer):
    """
    Hybrid optimizer: Mano for 2D matrices, AdamW for 1D vectors and embeddings

    Combines the strengths of both optimizers:
    - Mano for matrix parameters (Transformer weight matrices)
    - AdamW for embeddings, LayerNorm, and 1D parameters (bias, LayerNorm weight)

    v2 Updates:
    - Nesterov momentum defaults to True
    - Dual-dimension projection enabled by default

    Args:
        mano_params (list): parameters for Mano optimization (2D matrices)
        adamw_params (list): parameters for AdamW optimization (1D vectors)
        lr (float): learning rate
        momentum (float): Mano momentum coefficient
        weight_decay (float): weight decay
        nesterov (bool): whether to use Nesterov-style momentum (default: True)
        eps (float): epsilon for numerical stability (default: 1e-8)
        betas (tuple): AdamW beta parameters (default: (0.9, 0.95))
        dual_dim_projection (bool): whether to use dual-dimension projection (default: True)
    """

    def __init__(
        self,
        mano_params,
        adamw_params,
        lr,
        momentum,
        weight_decay,
        nesterov=True,
        eps=1e-8,
        betas=(0.9, 0.95),
        dual_dim_projection=True
    ):
        mano_params = [p for p in mano_params if p is not None]
        adamw_params = [p for p in adamw_params if p is not None]

        all_params = mano_params + adamw_params
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            betas=betas,
            dual_dim_projection=dual_dim_projection
        )
        super().__init__(all_params, defaults)

        if len(mano_params) > 0:
            self.mano_optim = ManoOptimizer(
                mano_params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                eps=eps,
                dual_dim_projection=dual_dim_projection
            )
        else:
            self.mano_optim = None

        if len(adamw_params) > 0:
            self.adamw_optim = optim.AdamW(
                adamw_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
        else:
            self.adamw_optim = None

        self.param_groups = []
        if self.mano_optim:
            self.param_groups.extend(self.mano_optim.param_groups)
        if self.adamw_optim:
            self.param_groups.extend(self.adamw_optim.param_groups)

    def zero_grad(self):
        """Clears gradients of all optimizers"""
        if self.mano_optim:
            self.mano_optim.zero_grad()
        if self.adamw_optim:
            self.adamw_optim.zero_grad()

    def step(self, closure=None):
        """
        Performs a single optimization step

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss

        Returns:
            loss (Tensor or None): loss value from closure
        """
        loss = None

        if closure is not None:
            loss = closure()

        if self.mano_optim:
            self.mano_optim.step(None)
        if self.adamw_optim:
            self.adamw_optim.step(None)

        return loss

    def state_dict(self):
        """
        Returns optimizer state

        Returns:
            dict: dictionary containing 'mano' and 'adamw' keys
        """
        state = {}
        if self.mano_optim:
            state['mano'] = self.mano_optim.state_dict()
        if self.adamw_optim:
            state['adamw'] = self.adamw_optim.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """
        Loads optimizer state

        Args:
            state_dict (dict): dictionary containing 'mano' and 'adamw' keys
        """
        if 'mano' in state_dict and self.mano_optim:
            self.mano_optim.load_state_dict(state_dict['mano'])
        if 'adamw' in state_dict and self.adamw_optim:
            self.adamw_optim.load_state_dict(state_dict['adamw'])


def create_optimizer(
    model,
    optimizer_type='mano',
    lr=1e-3,
    momentum=0.95,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    nesterov=True,
    dual_dim_projection=True
):
    """
    Factory function to create optimizer based on type

    Args:
        model: PyTorch model
        optimizer_type (str): optimizer type, 'mano' or 'adamw'
        lr (float): learning rate
        momentum (float): Mano momentum coefficient
        weight_decay (float): weight decay
        betas (tuple): AdamW beta parameters
        nesterov (bool): whether to use Nesterov-style momentum (default: True)
        dual_dim_projection (bool): whether to use dual-dimension projection (default: True)

    Returns:
        optimizer: created optimizer
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == 'mano':
        mano_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if param.dim() == 2:
                mano_params.append(param)
            else:
                adamw_params.append(param)

        print(f"Optimizer: HybridManoAdamW (Mano for 2D matrices, AdamW for 1D params)")
        print(f"  2D params: {len(mano_params)}")
        print(f"  1D params: {len(adamw_params)}")
        print(f"  Nesterov: {nesterov}")
        print(f"  Dual-dimension projection: {dual_dim_projection}")

        return HybridManoAdamW(
            mano_params=mano_params,
            adamw_params=adamw_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            nesterov=nesterov,
            dual_dim_projection=dual_dim_projection
        )
    else:
        if optimizer_type == 'adamw':
            print(f"Optimizer: AdamW (weight_decay={weight_decay})")
            return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        else:
            print(f"Optimizer: Adam (weight_decay={weight_decay})")
            return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
