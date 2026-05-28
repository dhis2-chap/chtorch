"""Compute log-CRPS for every archived chap eval run on the VNM dataset."""
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path("/Users/knutdr/chap-evaluations/mlruns/949198700587758246")

LABELS = {
    "3a9a5401bc0945ec8a1509b69a3218a2": "03:16  ep=30  num_workers=3  decay bug (10h run)",
    "bcf75b4026b344d0ac1dd6070591533e": "11:02  ep=30  num_workers=0  decay bug (2m)",
    "4161b4f5426e44b2a5d4ebfe85fa0f08": "11:41  ep=100 decay FIX",
    "f564e440afcd40968c51a6f543822bff": "12:06  ep=100 + sin/cos + direct_ar + past_ratio=0.5",
    "2b7c238512ff4baab3f5bfccf590343e": "12:12  + proper inverse  past_ratio=0.5",
    "196561674db048ec8876fc95aea89f8a": "13:28  + proper inverse  past_ratio=0.2 (current)",
}


def sample_crps(samples, truth):
    n = samples.shape[-1]
    truth = truth[..., None]
    term1 = np.mean(np.abs(samples - truth), axis=-1)
    sorted_s = np.sort(samples, axis=-1)
    i = np.arange(1, n + 1)
    weights = (2 * i - n - 1).astype(sorted_s.dtype)
    term2 = (sorted_s * weights).sum(axis=-1) / (n * n)
    return term1 - term2


def metrics_for_nc(nc_path):
    ds = xr.open_dataset(nc_path)
    samples = ds["forecast"].values
    truth = ds["observed"].values
    truth_b = np.broadcast_to(truth[:, :, None], samples.shape[:3])

    log_s = np.log1p(np.maximum(samples, 0))
    log_t = np.log1p(np.maximum(truth_b, 0))
    crps_log = sample_crps(log_s, log_t)
    mask = np.isfinite(truth_b)

    overall = np.nanmean(np.where(mask, crps_log, np.nan))
    per_h = [np.nanmean(np.where(mask[:, :, h], crps_log[:, :, h], np.nan)) for h in range(samples.shape[2])]
    return overall, per_h


print(f"{'label':70s}  {'mean':>7s}  {'h=0':>7s}  {'h=1':>7s}  {'h=2':>7s}")
print("-" * 105)
for run_id, label in LABELS.items():
    nc = ROOT / run_id / "artifacts" / "vnm_monthly_eval.nc"
    if not nc.exists():
        print(f"MISSING: {nc}")
        continue
    overall, per_h = metrics_for_nc(nc)
    print(f"{label:70s}  {overall:7.4f}  {per_h[0]:7.4f}  {per_h[1]:7.4f}  {per_h[2]:7.4f}")
