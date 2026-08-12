"""Minimal stand-in for aurora.model.film.AdaptiveLayerNorm.

Not part of your 10 uploaded files. swin3d.py calls it as
`AdaptiveLayerNorm(dim, time_dim, scale_bias=scale_bias)` then `self.norm1(x, c)`
and separately `blk.norm1.init_weights()` from `BasicLayer3D.init_respostnorm()`.
Implemented as the standard adaLN-Zero pattern: LayerNorm(x) modulated by a
scale/shift predicted from the conditioning vector `c`, zero-initialised so the
block starts as an identity-ish residual (standard DiT/adaLN-Zero practice,
consistent with the comment in swin3d.py that says this init must run *after*
the generic `.apply(init_weights)` pass).
"""
import torch
import torch.nn as nn

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, time_dim: int, scale_bias: float = 0.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(time_dim, 2 * dim)
        self.scale_bias = scale_bias
        self.init_weights()

    def init_weights(self) -> None:
        """Zero-init so the modulation starts as scale=1, shift=0 (identity)."""
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim); c: (B, time_dim)
        scale, shift = self.to_scale_shift(c).chunk(2, dim=-1)  # each (B, dim)
        scale = scale.unsqueeze(1) + self.scale_bias  # (B, 1, dim)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift
