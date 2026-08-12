"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Modified for the surface-only branch:

- Import paths corrected to `aurora.*` (previously `DecodersAurora.aurora.*`,
  a leftover from a different fork's package layout) and to import
  `AuroraLite` from `aurora.model.aurora_lite`, which is the class this repo
  actually defines (the original `Aurora` class from `aurora.model.aurora`
  doesn't exist here).

- `predict_vars`-aware history update. The original implementation assumed
  every key in `batch.surf_vars` is also a key in `pred.surf_vars`, i.e. that
  the decoder predicts everything it's given as input. That's true for the
  standard full-model decoder, but `SurfaceOnlyMLPDecoder` deliberately
  predicts only a subset (`predict_vars`) while the encoder is conditioned on
  a wider set (e.g. `steer_u`/`steer_v`/`shear_u`/`shear_v`). Since
  `dataclasses.replace(pred, surf_vars={...})` starts from `pred` - which only
  ever contains `predict_vars` keys - any surface variable that isn't
  predicted would silently vanish from `batch.surf_vars` after the first
  step, and the model would stop being conditioned on it for the rest of the
  roll-out. This version explicitly carries non-predicted
  ("conditioning-only") surface variables forward via persistence (repeats
  the last known value), or via a caller-supplied update function if you
  later want to feed in externally forecast steering flow / shear instead.
"""

import dataclasses
from typing import Callable, Generator, Optional

import torch

from aurora.batch import Batch
from aurora.model.aurora_lite import AuroraLite

__all__ = ["rollout"]


def rollout(
    model: AuroraLite,
    batch: Batch,
    steps: int,
    conditioning_update_fn: Optional[Callable[[str, torch.Tensor, int], Optional[torch.Tensor]]] = None,
) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions.

    Args:
        model (:class:`aurora.model.aurora_lite.AuroraLite`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
        conditioning_update_fn (callable, optional): For surface variables the decoder
            does *not* predict (e.g. `steer_u`/`shear_v` on the surface-only branch),
            this optionally computes the next step's value. Called as
            `fn(var_name, last_value, step_index)`, where `last_value` has shape
            `(B, 1, H, W)` (the most recent history slice) and `step_index` is the
            0-based roll-out step about to be produced. Must return a tensor of the
            same shape, or `None` to fall back to persistence for that call. If this
            argument is omitted entirely, every conditioning-only variable is carried
            forward unchanged (persistence) for the whole roll-out.

    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    # We will need to concatenate data, so ensure that everything is already of the right form.
    # Use an arbitrary parameter of the model to derive the data type and device.
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # Which surface variables does the decoder actually predict? Falls back to "all of
    # them" for decoders (e.g. the standard `Perceiver3DDecoderLite`) that don't define
    # `predict_vars` and predict every surface variable they're given - this keeps the
    # standard (non-surface-only) roll-out path behaving exactly as before.
    predict_vars = set(getattr(model.decoder, "predict_vars", tuple(batch.surf_vars.keys())))
    conditioning_vars = [k for k in batch.surf_vars if k not in predict_vars]

    for step in range(steps):
        pred = model.forward(batch)

        yield pred

        # Roll the history window forward for the variables the decoder predicted.
        new_surf_vars = {
            k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in pred.surf_vars.items()
        }

        # Everything else (conditioning-only surface vars, e.g. steer_u/steer_v/
        # shear_u/shear_v) never comes back from the decoder, so it has to be carried
        # forward explicitly here or it silently disappears from the batch on the next
        # iteration - see the module docstring.
        for k in conditioning_vars:
            last_value = batch.surf_vars[k][:, -1:]
            next_value = conditioning_update_fn(k, last_value, step) if conditioning_update_fn else None
            if next_value is None:
                next_value = last_value  # Persistence.
            new_surf_vars[k] = torch.cat([batch.surf_vars[k][:, 1:], next_value], dim=1)

        # Add the appropriate history so the model can be run on the prediction.
        batch = dataclasses.replace(
            pred,
            surf_vars=new_surf_vars,
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                for k, v in pred.atmos_vars.items()
            },
        )
