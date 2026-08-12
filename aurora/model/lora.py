"""Minimal stand-in for aurora.model.lora.

Not part of your 10 uploaded files. The surface-only branch always constructs
its backbone with `use_lora=False`, so `LoRARollout` is never instantiated in
that path - this only needs to exist so `swin3d.py` and `aurora_lite.py` can
import it. Kept as a real (if simple) module rather than a bare stub, in case
you later want LoRA on the surface-only backbone too.
"""
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["LoRAMode", "LoRARollout"]

LoRAMode = Literal["single", "all"]


class LoRARollout(nn.Module):
    """Low-rank adapter, optionally indexed by roll-out step.

    `lora_mode="single"` shares one (A, B) pair across all roll-out steps.
    `lora_mode="all"` keeps a separate pair per step up to `lora_steps`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
    ) -> None:
        super().__init__()
        self.lora_mode = lora_mode
        self.lora_steps = lora_steps
        self.scaling = alpha / r
        n = 1 if lora_mode == "single" else lora_steps
        self.A = nn.Parameter(torch.zeros(n, r, in_features))
        self.B = nn.Parameter(torch.zeros(n, out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rollout_step: int = 0) -> torch.Tensor:
        idx = 0 if self.lora_mode == "single" else min(rollout_step, self.lora_steps - 1)
        A, B = self.A[idx], self.B[idx]
        return self.drop(x) @ A.T @ B.T * self.scaling
