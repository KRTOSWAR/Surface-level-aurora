"""Lightweight MLP decoder for the surface-only branch.

Why this exists instead of reusing `Perceiver3DDecoderLite`:
`aurora_lite.py`'s `surface_only=True` branch currently wires
`self.decoder = Perceiver3DDecoderLite(surf_vars=surf_vars, atmos_vars=atmos_vars, ...)`.
That decoder is built around the general multi-level case: it rearranges the
backbone output into `(B, L, C, D)` and then does
`x[..., 1:, :]` to peel off "the atmospheric levels beyond the surface level"
for its `PerceiverResampler` level-deaggregation step. For the surface-only
backbone, `patch_res[0] == 1` (SurfaceOnlyEncoder.latent_levels == 1), so that
slice is empty, and if `batch.atmos_vars` is genuinely empty (as it is for a
surface-only batch) the subsequent
`torch.stack([self.atmos_heads[name](...) for name in atmos_vars], dim=-1)`
raises `RuntimeError: stack expects a non-empty TensorList`. There's no
atmospheric level structure to deaggregate here, so this decoder skips that
machinery entirely instead of trying to feed it an empty tensor.

Also fixes a real bug present in both decoder classes in decoder_lite.py:
`nn.ParameterDict` is for `nn.Parameter` values only - `register_parameter`
raises `TypeError` if you hand it an `nn.Module` (a `Linear` or an `MLP`).
Both `Perceiver3DDecoderLite.surf_heads`/`atmos_heads` and
`MLPDecoderLite.surf_heads` store `nn.Linear`/`MLP` *modules* in a
`ParameterDict`, which will fail at construction. Using `nn.ModuleDict` here
instead.

Design choice worth flagging explicitly: `predict_vars` is independent of
`batch.surf_vars.keys()`. That's what lets the encoder see a wider input set
(msl, 10u, 10v, 2t, steer_u, steer_v, shear_u, shear_v, ...) while the decoder
only builds prediction heads for a chosen subset (msl, 10u, 10v). If you
actually want the model to *also* forecast steering flow / shear as an
auxiliary next-step target rather than pure input conditioning, just add
their names to `predict_vars` - everything else (unpatchify, unnormalisation,
Batch construction) is already variable-name-agnostic.
"""
from datetime import timedelta

import torch
import torch.nn as nn

from aurora.batch import Batch, Metadata
from aurora.normalisation import unnormalise_surf_var
from aurora.model.util import init_weights

__all__ = ["SurfaceOnlyMLPDecoder"]


class SurfaceOnlyMLPDecoder(nn.Module):
    """Maps the Swin3D backbone's per-patch tokens straight to a next-step
    prediction for a chosen subset of surface variables. No level
    deaggregation, no atmospheric heads - this is the whole decoder.
    """

    def __init__(
        self,
        predict_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 256,  # NOTE: must equal the *backbone's* output width,
        # i.e. 2 * surface_embed_dim (the final decoder stage concatenates
        # with the first encoder stage's skip connection - see
        # Swin3DTransformerBackbone.forward()'s last `torch.cat` - so the
        # tokens arriving here are twice as wide as surface_embed_dim, exactly
        # mirroring why aurora_lite.py's *existing* (standard-path) code
        # constructs `Perceiver3DDecoderLite(embed_dim=embed_dim * 2, ...)`.
        hidden_dim: int = 128,
        stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        self.predict_vars = tuple(predict_vars)
        self.patch_size = patch_size
        self.stats = stats

        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, patch_size**2),
                )
                for name in self.predict_vars
            }
        )
        self.apply(init_weights)

    def _unpatchify_one(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, L, P*P) -> (B, H, W), L = (H/P)*(W/P)."""
        B, L, PP = x.shape
        p = self.patch_size
        hp, wp = H // p, W // p
        assert L == hp * wp, f"L={L} != (H/P)*(W/P)={hp * wp} - check patch_res vs. H,W."
        x = x.reshape(B, hp, wp, p, p)
        x = x.permute(0, 1, 3, 2, 4)  # B, hp, p, wp, p
        x = x.reshape(B, hp * p, wp * p)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        lead_time: timedelta,
        patch_res: tuple[int, int, int],
    ) -> tuple[Batch, torch.Tensor]:
        """
        Args:
            x: backbone output, `(B, L, embed_dim)`. For the surface-only
                backbone `patch_res[0] == 1` always, so `L == patch_res[1] * patch_res[2]`
                already - no level dimension to unwrap.
            batch: the (normalised, cropped) input batch - only used for lat/lon/time
                metadata and to echo `static_vars` through unchanged.
            lead_time: forecast lead time (adds onto `batch.metadata.time`).
            patch_res: `(C, H', W')` from `AuroraLite.forward()`. `C` is unused here
                (asserted to be 1) since there is only ever one surface level.

        Returns:
            Predicted `Batch` (unnormalised) for `predict_vars`, plus the raw
            backbone tensor `x` (kept for parity with `Perceiver3DDecoderLite`'s
            return convention, e.g. if you want it for a later distillation loss).
        """
        assert patch_res[0] == 1, (
            f"SurfaceOnlyMLPDecoder expects a single surface level (patch_res[0]==1), "
            f"got patch_res={patch_res}. Perceiver3DDecoderLite is what you want for "
            f"a model with real atmospheric levels."
        )

        lat, lon = batch.metadata.lat, batch.metadata.lon
        H, W = lat.shape[0], lon.shape[-1]

        preds: dict[str, torch.Tensor] = {}
        for name in self.predict_vars:
            out = self.heads[name](x)  # (B, L, P*P)
            img = self._unpatchify_one(out, H, W)  # (B, H, W)
            preds[name] = unnormalise_surf_var(img, name, stats=self.stats)

        pred_batch = Batch(
            surf_vars=preds,
            static_vars=batch.static_vars,
            atmos_vars={},
            metadata=Metadata(
                lat=lat,
                lon=lon,
                time=tuple(t + lead_time for t in batch.metadata.time),
                atmos_levels=batch.metadata.atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )
        return pred_batch, x
