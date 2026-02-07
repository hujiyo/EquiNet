"""
EquiNet优化器模块

包含多种优化器实现：
- ManoOptimizer: 基于流形优化的LLM训练优化器（从pawlette项目移植）
- HybridManoAdamW: 混合优化器，Mano用于2D矩阵，AdamW用于1D参数

使用方法：
    from optimizers import create_optimizer
    
    # 创建Mano优化器
    optimizer = create_optimizer(model, optimizer_type='mano', lr=5e-4)
    
    # 创建AdamW优化器
    optimizer = create_optimizer(model, optimizer_type='adamw', lr=1e-3)
"""

from .mano import ManoOptimizer
from .hybrid_optimizer import HybridManoAdamW, create_optimizer

__all__ = ['ManoOptimizer', 'HybridManoAdamW', 'create_optimizer']
