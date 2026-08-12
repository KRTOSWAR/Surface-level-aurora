"""Minimal stand-ins for aurora.model.util.

NOT part of your 10 uploaded files - these were referenced by swin3d.py /
surface_encoder.py (`from aurora.model.util import ...`) but never provided.
Implementations below satisfy the documented call sites exactly (same
signatures, same shapes in/out); the *numeric* init scheme is a standard
choice (truncated-normal / zeros, the usual ViT/Swin convention) since the
call sites never depend on Microsoft's exact constants for correctness.
"""
import torch
import torch.nn as nn

__all__ = ["init_weights", "maybe_adjust_windows", "check_lat_lon_dtype", "unpatchify"]


def init_weights(m: nn.Module) -> None:
    """Applied via `.apply(init_weights)` - standard ViT/Swin-style init."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        # `AdaptiveLayerNorm.norm` is `nn.LayerNorm(..., elementwise_affine=False)`,
        # which has `weight=None, bias=None` - guard both, same as the Linear/Conv3d
        # branches above, or `.apply(init_weights)` crashes on any module containing one.
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        if m.weight is not None:
            nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv3d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def maybe_adjust_windows(
    window_size: tuple[int, ...],
    shift_size: tuple[int, ...],
    res: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Standard Swin clamp: if the input resolution is smaller than the configured
    window along some axis, use the whole resolution as the window for that axis
    (and disable shifting for it, since you can't shift a window equal to the
    full extent)."""
    window_size = list(window_size)
    shift_size = list(shift_size)
    for i in range(len(res)):
        if res[i] <= window_size[i]:
            window_size[i] = res[i]
            shift_size[i] = 0
    return tuple(window_size), tuple(shift_size)


def check_lat_lon_dtype(lat: torch.Tensor, lon: torch.Tensor) -> None:
    assert lat.dtype in (torch.float32, torch.float64), f"Unexpected lat dtype {lat.dtype}."
    assert lon.dtype in (torch.float32, torch.float64), f"Unexpected lon dtype {lon.dtype}."


def unpatchify(x: torch.Tensor, v: int, h: int, w: int, p: int) -> torch.Tensor:
    """Inverse of splitting an (h, w) image into non-overlapping p x p patches
    and flattening each patch's `v` per-variable pixel blocks into one vector.

    Args:
        x: `(B, L, C, v * p * p)` where `L = (h // p) * (w // p)`.
        v: number of variables packed into the last dim.
        h, w: full output spatial resolution.
        p: patch size.

    Returns:
        `(B, v, C, h, w)`.
    """
    B, L, C, VPP = x.shape
    hp, wp = h // p, w // p
    assert L == hp * wp, f"L={L} does not match (h//p)*(w//p)={hp * wp}."
    assert VPP == v * p * p, f"Last dim {VPP} does not match v*p*p={v * p * p}."
    x = x.reshape(B, hp, wp, C, v, p, p)
    x = x.permute(0, 4, 3, 1, 5, 2, 6)  # B, v, C, hp, p, wp, p
    x = x.reshape(B, v, C, hp * p, wp * p)
    return x
