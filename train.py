import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aurora.batch import Batch, Metadata
from aurora.model.aurora_lite import AuroraLite
from aurora.normalisation import normalise_surf_var

import xarray as xr

SURF_VARS = ("2t", "10u", "10v", "msl", "steer_u", "steer_v", "shear_u", "shear_v")
STATIC_VARS = ("lsm", "z")
PREDICT_VARS = ("msl", "10u", "10v", "2t")
PATCH_SIZE = 4
HISTORY_STEPS = 2


# =====================================================================
# 1. DATASET (unchanged from the original - this part was already correct)
# =====================================================================
class ERA5SWIODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        static_path: str,
        history_steps: int = HISTORY_STEPS,
        surf_vars: tuple = SURF_VARS,
        static_vars: tuple = STATIC_VARS,
        predict_vars: tuple = PREDICT_VARS,
        time_slice: slice | None = None,
    ):
        super().__init__()
        self.history_steps = history_steps
        self.surf_vars = surf_vars
        self.static_vars_keys = static_vars
        self.predict_vars = predict_vars

        self.ds = xr.open_dataset(data_path)

        rename_map = {"u_steer": "steer_u", "v_steer": "steer_v", "u_shear": "shear_u", "v_shear": "shear_v"}
        rename_map = {k: v for k, v in rename_map.items() if k in self.ds}
        if rename_map:
            self.ds = self.ds.rename(rename_map)

        if self.ds.latitude.values[0] < self.ds.latitude.values[-1]:
            self.ds = self.ds.reindex(latitude=self.ds.latitude[::-1])
        longitudes = self.ds.longitude.values % 360
        self.ds = self.ds.assign_coords(longitude=longitudes)
        if not np.all(np.diff(self.ds.longitude.values) > 0):
            self.ds = self.ds.sortby("longitude")

        lon_rem = len(self.ds.longitude) % PATCH_SIZE
        if lon_rem != 0:
            self.ds = self.ds.isel(longitude=slice(0, -lon_rem))
        lat_rem = len(self.ds.latitude) % PATCH_SIZE
        if lat_rem != 0:
            self.ds = self.ds.isel(latitude=slice(0, -lat_rem))

        # Restrict to a time window (train/val split) AFTER windowing considerations -
        # done by the caller via `time_slice` on the already-cleaned coords.
        if time_slice is not None:
            self.ds = self.ds.isel(time=time_slice)

        self.lats = torch.from_numpy(self.ds.latitude.values.astype(np.float32))
        self.lons = torch.from_numpy(self.ds.longitude.values.astype(np.float32))

        self.static_ds = xr.open_dataset(static_path)
        if self.static_ds.latitude.values[0] < self.static_ds.latitude.values[-1]:
            self.static_ds = self.static_ds.reindex(latitude=self.static_ds.latitude[::-1])
        static_longitudes = self.static_ds.longitude.values % 360
        self.static_ds = self.static_ds.assign_coords(longitude=static_longitudes)
        if not np.all(np.diff(self.static_ds.longitude.values) > 0):
            self.static_ds = self.static_ds.sortby("longitude")
        self.static_ds = self.static_ds.sel(latitude=self.ds.latitude, longitude=self.ds.longitude)

        self.static_vars = {}
        for var in self.static_vars_keys:
            if var in self.static_ds:
                data = self.static_ds[var].values.squeeze()
                self.static_vars[var] = torch.from_numpy(data.astype(np.float32))

        self.times = [pd.to_datetime(t).to_pydatetime() for t in self.ds.time.values]

        # Flag (not silently ignore) any gap where consecutive steps aren't 6h apart -
        # windowing below assumes uniform spacing, and a missing day in the raw
        # download would otherwise splice non-adjacent times together unnoticed.
        deltas = np.diff(self.ds.time.values).astype("timedelta64[h]").astype(int)
        if len(deltas) and not np.all(deltas == deltas[0]):
            bad = np.where(deltas != deltas[0])[0]
            raise ValueError(
                f"Non-uniform time spacing at indices {bad.tolist()[:5]}... "
                f"(deltas: {set(deltas.tolist())}h). Check for missing daily files "
                f"before training - the windowing logic assumes uniform 6h spacing."
            )

        self.num_samples = len(self.times) - self.history_steps

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def __getitem__(self, idx: int):
        input_indices = list(range(idx, idx + self.history_steps))
        target_index = idx + self.history_steps
        input_times = [self.times[i] for i in input_indices]

        sample_surf = {v: torch.from_numpy(self.ds[v].isel(time=input_indices).values.astype(np.float32))
                       for v in self.surf_vars}
        sample_target = {v: torch.from_numpy(self.ds[v].isel(time=target_index).values.astype(np.float32))
                          for v in self.predict_vars}
        return {"surf_vars": sample_surf, "target_vars": sample_target, "times": input_times}


def aura_collate_fn(batch_list, static_vars, lats, lons):
    surf_keys = batch_list[0]["surf_vars"].keys()
    target_keys = batch_list[0]["target_vars"].keys()
    batched_surf = {k: torch.stack([item["surf_vars"][k] for item in batch_list], dim=0) for k in surf_keys}
    batched_targets = {k: torch.stack([item["target_vars"][k] for item in batch_list], dim=0) for k in target_keys}
    metadata = Metadata(lat=lats, lon=lons, time=tuple(batch_list[0]["times"]), atmos_levels=(), rollout_step=0)
    input_batch = Batch(surf_vars=batched_surf, static_vars=static_vars, atmos_vars={}, metadata=metadata)
    return input_batch, batched_targets


# =====================================================================
# 2. FIT REAL NORMALISATION STATS FROM THE TRAINING SPLIT
# =====================================================================
def fit_surf_stats(ds: xr.Dataset, var_names: tuple) -> dict:
    """Mean/std per variable over the *training* time range only - never fit
    stats using data the model will later be validated/tested on."""
    stats = {}
    for name in var_names:
        arr = ds[name].values
        stats[name] = (float(np.mean(arr)), float(np.std(arr)) + 1e-6)
    return stats


# =====================================================================
# 3. NORMALIZED, LATITUDE-WEIGHTED LOSS
# =====================================================================
class NormalizedLatitudeWeightedLoss(nn.Module):
    def __init__(self, lats: torch.Tensor, surf_stats: dict):
        super().__init__()
        weights = torch.cos(torch.deg2rad(lats))
        weights = weights / weights.mean()
        self.register_buffer("weights", weights.view(1, 1, -1, 1))
        self.surf_stats = surf_stats

    def forward(self, pred_batch: Batch, target_vars: dict) -> torch.Tensor:
        H_crop, W_crop = pred_batch.spatial_shape
        w = self.weights[:, :, :H_crop, :]
        total = 0.0
        for name, pred_tensor in pred_batch.surf_vars.items():
            target_tensor = target_vars[name][:, None, :H_crop, :W_crop].to(pred_tensor.device)
            pred_n = normalise_surf_var(pred_tensor, name, stats=self.surf_stats)
            target_n = normalise_surf_var(target_tensor, name, stats=self.surf_stats)
            total = total + ((pred_n - target_n) ** 2 * w).mean()
        return total / len(pred_batch.surf_vars)


# =====================================================================
# 4. TRAIN / VAL LOOP
# =====================================================================
@dataclass
class TrainConfig:
    data_path: str
    static_path: str
    epochs: int = 50
    batch_size: int = 5
    lr: float = 1e-4
    val_fraction: float = 0.2
    surface_embed_dim: int = 256
    patience: int = 30
    checkpoint_dir: str = "./checkpoints"


def run_training(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Time-ordered split: last `val_fraction` of the period is validation.
    # (Temporal, not random, split - adjacent 6h steps are highly autocorrelated,
    # so a random split would leak information between train and val.)
    full_ds_probe = xr.open_dataset(cfg.data_path)
    n_time = full_ds_probe.sizes["time"]
    n_val = max(cfg.batch_size + HISTORY_STEPS, int(n_time * cfg.val_fraction))
    n_train = n_time - n_val
    full_ds_probe.close()
    print(f"Time steps: {n_time} total -> {n_train} train / {n_val} val "
          f"({(n_train - HISTORY_STEPS)} train samples, {(n_val - HISTORY_STEPS)} val samples)")
    if n_train - HISTORY_STEPS < cfg.batch_size:
        raise ValueError(
            f"Only {n_train - HISTORY_STEPS} train samples for batch_size={cfg.batch_size} - "
            f"this dataset is too short. This is expected for a 10-day pilot; treat this run as "
            f"a pipeline smoke test, not a real training run."
        )

    train_dataset = ERA5SWIODataset(cfg.data_path, cfg.static_path, time_slice=slice(0, n_train))
    val_dataset = ERA5SWIODataset(cfg.data_path, cfg.static_path, time_slice=slice(n_train, n_time))

    train_stats = fit_surf_stats(train_dataset.ds, SURF_VARS)
    print("\nFitted training-split stats (name: mean, std):")
    for k, (m, s) in train_stats.items():
        print(f"  {k:8s}: {m:12.4f}, {s:10.4f}")

    collate = partial(aura_collate_fn, static_vars=train_dataset.static_vars,
                       lats=train_dataset.lats, lons=train_dataset.lons)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_collate = partial(aura_collate_fn, static_vars=val_dataset.static_vars,
                           lats=val_dataset.lats, lons=val_dataset.lons)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=val_collate)

    model = AuroraLite(
        surf_vars=SURF_VARS,
        static_vars=STATIC_VARS,
        surface_only=True,
        surface_embed_dim=cfg.surface_embed_dim,
        patch_size=PATCH_SIZE,
        max_history_size=HISTORY_STEPS,
        surface_predict_vars=PREDICT_VARS,
        surf_stats=train_stats,  # <- the fix: real stats, not the fallback placeholders/globals
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\nModel size: {n_params:,} params (~{n_params * 4 / 1e6:.1f} MB fp32)")
    print(f"  encoder:  {n_enc:,} ({100*n_enc/n_params:.1f}%)")
    print(f"  backbone: {n_bb:,} ({100*n_bb/n_params:.1f}%)")
    print(f"  decoder:  {n_dec:,} ({100*n_dec/n_params:.1f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = NormalizedLatitudeWeightedLoss(train_dataset.lats, train_stats).to(device)
    val_criterion = NormalizedLatitudeWeightedLoss(val_dataset.lats, train_stats).to(device)  # train stats, always

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0

    print("\nStarting training...")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for input_batch, target_vars in train_loader:
            optimizer.zero_grad()
            pred_batch, _ = model(input_batch)
            loss = criterion(pred_batch, target_vars)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_batch, target_vars in val_loader:
                pred_batch, _ = model(input_batch)
                val_loss += val_criterion(pred_batch, target_vars).item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch [{epoch}/{cfg.epochs}] train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "surf_stats": train_stats, "epoch": epoch,
                 "val_loss": val_loss},
                Path(cfg.checkpoint_dir) / "best.pt",
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"No val improvement for {cfg.patience} epochs, stopping early.")
                break

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/downloads/swio_10days_surface_expanded.nc")
    p.add_argument("--static_path", default="./data/downloads/static.nc")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    run_training(TrainConfig(data_path=args.data_path, static_path=args.static_path,
                              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr))
