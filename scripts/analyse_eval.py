"""Quick analysis of vnm_monthly_eval.nc — surface obvious failure modes:
- Are predicted samples in the right ballpark?
- Per-location bias / scale errors
- Are samples degenerate (constant, all zero, exploded)?
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NC = Path("/Users/knutdr/Sources/chtorch/vnm_monthly_eval.nc")
CSV = Path("/Users/knutdr/Downloads/chap_VNM_admin1_monthly.csv")


def main():
    ds = xr.open_dataset(NC)
    print("== NetCDF variables ==")
    for name, v in ds.data_vars.items():
        print(f"  {name:20s} dims={v.dims} shape={v.shape} dtype={v.dtype}")
    print("\n== Coords ==")
    for name, c in ds.coords.items():
        print(f"  {name:20s} size={c.size}")

    samples = ds["forecast"]
    truth = ds["observed"]
    # Broadcast truth across horizon for fair per-(time,horizon) comparison
    truth = truth.expand_dims({"horizon_distance": samples.sizes["horizon_distance"]}, axis=-1)
    sample_dim = "sample"
    print(f"\nsamples.shape = {samples.shape}")
    print(f"truth.shape   = {truth.shape}")

    pred_mean = samples.mean(sample_dim)
    pred_med = samples.median(sample_dim)
    pred_std = samples.std(sample_dim)

    # Flatten across split/period/location for global stats, drop NaN truth
    truth_flat = truth.values.ravel().astype(float)
    mean_flat = pred_mean.values.ravel().astype(float)
    med_flat = pred_med.values.ravel().astype(float)
    std_flat = pred_std.values.ravel().astype(float)
    mask = ~np.isnan(truth_flat)
    truth_flat = truth_flat[mask]
    mean_flat = mean_flat[mask]
    med_flat = med_flat[mask]
    std_flat = std_flat[mask]

    print("\n== Truth vs prediction (all points) ==")
    print(f"truth   :  min {truth_flat.min():.1f}  median {np.median(truth_flat):.1f}  mean {truth_flat.mean():.1f}  max {truth_flat.max():.1f}")
    print(f"pred mu :  min {mean_flat.min():.1f}  median {np.median(mean_flat):.1f}  mean {mean_flat.mean():.1f}  max {mean_flat.max():.1f}")
    print(f"pred med:  min {med_flat.min():.1f}  median {np.median(med_flat):.1f}  mean {med_flat.mean():.1f}  max {med_flat.max():.1f}")
    print(f"pred sd :  min {std_flat.min():.1f}  median {np.median(std_flat):.1f}  mean {std_flat.mean():.1f}  max {std_flat.max():.1f}")

    # Error metrics
    err = mean_flat - truth_flat
    abs_err = np.abs(err)
    print("\n== Error metrics (mean prediction vs truth) ==")
    print(f"bias (mean err): {err.mean():.2f}")
    print(f"MAE            : {abs_err.mean():.2f}")
    print(f"RMSE           : {np.sqrt((err**2).mean()):.2f}")
    nz = truth_flat > 0
    if nz.any():
        smape = 2 * np.abs(mean_flat[nz] - truth_flat[nz]) / (np.abs(mean_flat[nz]) + np.abs(truth_flat[nz]) + 1e-9)
        print(f"SMAPE (truth>0): {smape.mean():.3f}")

    # Per-location bias (across all splits)
    if 'location' in samples.dims:
        per_loc_mean_pred = pred_mean.mean([d for d in pred_mean.dims if d != 'location']).values
        per_loc_mean_truth = truth.mean([d for d in truth.dims if d != 'location']).values
        ratio = per_loc_mean_pred / np.maximum(per_loc_mean_truth, 1e-3)
        print("\n== Per-location ratio (pred_mean / truth_mean), first 10 locations ==")
        for i in range(min(10, len(ratio))):
            print(f"  loc {i:3d}: pred={per_loc_mean_pred[i]:.2f}  truth={per_loc_mean_truth[i]:.2f}  ratio={ratio[i]:.2f}")

    # Sample dispersion check — are samples actually varying?
    print("\n== Sample dispersion summary ==")
    cv = pred_std.values / np.maximum(pred_mean.values, 1e-3)
    cv_flat = cv.ravel()
    cv_flat = cv_flat[~np.isnan(cv_flat)]
    print(f"coefficient of variation (sd/mu): median {np.median(cv_flat):.3f}  p5 {np.percentile(cv_flat,5):.3f}  p95 {np.percentile(cv_flat,95):.3f}")


def dataset_stats():
    df = pd.read_csv(CSV)
    print("\n== Dataset stats ==")
    print(f"rows={len(df)}  locations={df['location'].nunique()}  periods={df['time_period'].nunique()}")
    by_loc = df.groupby('location')['disease_cases']
    means = by_loc.mean()
    zeros = (df['disease_cases'] == 0).mean()
    nans = df['disease_cases'].isna().mean()
    print(f"frac zero cases: {zeros:.3f}    frac NaN: {nans:.3f}")
    print(f"per-loc mean cases: min {means.min():.1f}  median {means.median():.1f}  mean {means.mean():.1f}  max {means.max():.1f}")
    print(f"per-loc max cases:  median {by_loc.max().median():.0f}  p95 {by_loc.max().quantile(0.95):.0f}  max {by_loc.max().max():.0f}")


if __name__ == "__main__":
    main()
    dataset_stats()
