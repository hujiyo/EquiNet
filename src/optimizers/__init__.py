"""
EquiNet Optimizers Module

Includes multiple optimizer implementations:
- ManoOptimizer: Manifold-based optimizer for LLM training (ported from pawlette)
- HybridManoAdamW: Hybrid optimizer, Mano for 2D matrices, AdamW for 1D parameters

Usage:
    from optimizers import create_optimizer
    
    # Create Mano optimizer
    optimizer = create_optimizer(model, optimizer_type='mano', lr=5e-4)
    
    # Create AdamW optimizer
    optimizer = create_optimizer(model, optimizer_type='adamw', lr=1e-3)
"""

from .mano import ManoOptimizer, HybridManoAdamW, create_optimizer

__all__ = ['ManoOptimizer', 'HybridManoAdamW', 'create_optimizer']
