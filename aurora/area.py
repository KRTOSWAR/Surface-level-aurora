"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import torch

__all__ = ["area", "compute_patch_areas", "radius_earth"]

radius_earth = 6378137 / 1000


def area(polygon: torch.Tensor) -> torch.Tensor:
    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)
    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]
    rad = torch.deg2rad
    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n
            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]
            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))
    area = area * radius_earth * radius_earth / 2
    return torch.abs(area)
