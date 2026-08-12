"""Lightweight surface-only encoder for Aurora (surface + static variables).

This module provides SurfaceOnlyEncoder which is a minimal encoder that uses the
existing LevelPatchEmbed + pos/time encodings to produce token tensors compatible
with the Swin3D backbone used elsewhere in the repo.

Designed to be small (embed_dim default 64) and to expose `.latent_levels` and
`.patch_size` attributes used by AuroraLite when computing patch_res.
"""
from datetime import timedelta
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from aurora.batch import Batch
from aurora.model.fourier import (
    absolute_time_expansion,
    lead_time_expansion,
    pos_expansion,
    scale_expansion,
)
from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.posencoding import pos_scale_enc
from aurora.model.util import init_weights

class SurfaceOnlyEncoder(nn.Module):
    """Minimal surface-only encoder.

    Produces token tensor of shape (B, L, D) where L = (H/patch_size)*(W/patch_size)
    and D = embed_dim. Exposes `latent_levels` and `patch_size` attributes for
    compatibility with the rest of the codebase.
    """

    def __init__(
        self,
        surf_vars: Tuple[str, ...],
        static_vars: Optional[Tuple[str, ...]] = None,
        patch_size: int = 4,
        embed_dim: int = 128,
        max_history_size: int = 2,
    ) -> None:
        super().__init__()
        self.surf_vars = tuple(surf_vars)
        self.static_vars = tuple(static_vars) if static_vars is not None else tuple()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.max_history_size = max_history_size

        # Surface-only operates on a single "level" (the surface), so latent_levels=1
        self.latent_levels = 1

        var_names = self.surf_vars + self.static_vars if len(self.static_vars) > 0 else self.surf_vars
        # Use LevelPatchEmbed to embed each variable (and history) into a shared embedding
        self.patch_embed = LevelPatchEmbed(var_names=var_names, patch_size=patch_size, embed_dim=embed_dim, history_size=max_history_size)

        # Embeddings reused for pos/scale/time conditioning (small linear layers)
        self.pos_embed = nn.Linear(embed_dim, embed_dim)
        self.scale_embed = nn.Linear(embed_dim, embed_dim)
        self.lead_time_embed = nn.Linear(embed_dim, embed_dim)
        self.absolute_time_embed = nn.Linear(embed_dim, embed_dim)

        self.pos_drop = nn.Dropout(p=0.0)

        self.apply(init_weights)

    def forward(self, batch: Batch, lead_time: timedelta) -> torch.Tensor:
        """Encode a batch of surface+static variables into tokens (B, L, D).

        Args:
            batch: aurora.batch.Batch (must contain surf_vars, optionally static_vars)
            lead_time: lead time (timedelta)

        Returns:
            torch.Tensor: tokens of shape (B, L, D)
        """
        surf_vars = tuple(batch.surf_vars.keys())
        static_vars = tuple(batch.static_vars.keys()) if len(batch.static_vars) > 0 else tuple()

        x_surf = torch.stack(tuple(batch.surf_vars.values()), dim=2)  # (B, T, V_surf, H, W)
        B, T, _, H, W = x_surf.shape

        if len(static_vars) > 0:
            x_static = torch.stack(tuple(batch.static_vars.values()), dim=2)  # (B, T, V_static, H, W)
            # static vars are constant over time -> expand history dimension
            x_static = x_static.expand((B, T, -1, -1, -1))
            x_surf = torch.cat((x_surf, x_static), dim=2)

        # Rearrange to (B, V, T, H, W) as expected by LevelPatchEmbed
        x_surf = rearrange(x_surf, "b t v h w -> b v t h w")
        var_names = surf_vars + static_vars

        # Patch-embed -> (B, L, D)
        tokens = self.patch_embed(x_surf, var_names)
        dtype = tokens.dtype

        # Positional and scale encodings computed from metadata lat/lon
        lat, lon = batch.metadata.lat, batch.metadata.lon
        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=pos_expansion,
            scale_expansion=scale_expansion,
        )

        # Add embeddings (keep dtype consistent)
        pos_e = self.pos_embed(pos_encode[None, ...].to(dtype=dtype))
        scale_e = self.scale_embed(scale_encode[None, ...].to(dtype=dtype))
        tokens = tokens + pos_e + scale_e

        # Lead time embedding
        lead_hours = lead_time.total_seconds() / 3600
        lead_times = lead_hours * torch.ones(B, dtype=dtype, device=tokens.device)
        lead_time_encode = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)
        tokens = tokens + lead_time_emb.unsqueeze(1)

        # Absolute time embedding (uses batch.metadata.time list)
        absolute_times_list = [t.timestamp() / 3600 for t in batch.metadata.time]
        absolute_times = torch.tensor(absolute_times_list, dtype=torch.float32, device=tokens.device)
        absolute_time_encode = absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_emb = self.absolute_time_embed(absolute_time_encode.to(dtype=dtype))
        tokens = tokens + absolute_time_emb.unsqueeze(1)

        tokens = self.pos_drop(tokens)
        return tokens