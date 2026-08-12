"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from functools import partial
from typing import Optional

import torch

__all__ = [
    "normalise_surf_var",
    "normalise_atmos_var",
    "unnormalise_surf_var",
    "unnormalise_atmos_var",
]


def normalise_surf_var(
    x: torch.Tensor,
    name: str,
    stats: Optional[dict[str, tuple[float, float]]] = None,
    unnormalise: bool = False,
) -> torch.Tensor:
    if stats and name in stats:
        location, scale = stats[name]
    else:
        location = locations[name]
        scale = scales[name]
    if unnormalise:
        return x * scale + location
    else:
        return (x - location) / scale


def normalise_atmos_var(
    x: torch.Tensor,
    name: str,
    atmos_levels: tuple[int | float, ...],
    unnormalise: bool = False,
) -> torch.Tensor:
    level_locations: list[int | float] = []
    level_scales: list[int | float] = []
    for level in atmos_levels:
        level_locations.append(locations[f"{name}_{level}"])
        level_scales.append(scales[f"{name}_{level}"])
    location = torch.tensor(level_locations, dtype=x.dtype, device=x.device)
    scale = torch.tensor(level_scales, dtype=x.dtype, device=x.device)
    if unnormalise:
        return x * scale[..., None, None] + location[..., None, None]
    else:
        return (x - location[..., None, None]) / scale[..., None, None]


unnormalise_surf_var = partial(normalise_surf_var, unnormalise=True)
unnormalise_atmos_var = partial(normalise_atmos_var, unnormalise=True)


locations: dict[str, float] = {
    "z": -1.386496e03,
    "lsm": 0.000000e00,
    "slt": 0.000000e00,
    "2t": 2.785140e02,
    "10u": -5.135059e-02,
    "10v": 1.891580e-01,
    "msl": 1.009578e05,
    # --- PLACEHOLDERS added for the SWIO steering-flow / shear branch ---
    # These are NOT fitted to real data. Replace with the actual mean/std
    # computed over your own SWIO ERA5 training window before real training
    # (see steering_shear.py: compute these the same way you fit
    # StandardScalerNP in data_prep.py, just on the derived fields instead
    # of the raw ones).
    "steer_u": 0.0,
    "steer_v": 0.0,
    "shear_u": 0.0,
    "shear_v": 0.0,
}

scales: dict[str, float] = {
    "z": 5.884467e04,
    "lsm": 1.000000e00,
    "slt": 7.000000e00,
    "2t": 2.122036e01,
    "10u": 5.547512e00,
    "10v": 4.765339e00,
    "msl": 1.332246e03,
    # --- PLACEHOLDERS, see note above ---
    "steer_u": 8.0,
    "steer_v": 8.0,
    "shear_u": 10.0,
    "shear_v": 10.0,
}
