"""Compute log-CRPS for vnm_monthly_eval.nc.

CRPS for a sample ensemble is:
    CRPS(F, y) = (1/n) Σ |x_i - y|  -  (1/(2 n²)) Σ_{i,j} |x_i - x_j|

Using the order-statistic identity
    Σ_{i,j} |x_i - x_j| = 2 Σ_i (2i - n - 1) * x_(i)
this can be computed in O(n log n) per cell. log-CRPS applies log1p to
both samples and truth first so the metric is on a comparable scale
across high- and low-incidence locations.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr

NC = Path("/Users/knutdr/Sources/chtorch/vnm_monthly_eval.nc")


def sample_crps(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Vectorised sample CRPS along the last axis of `samples`.

    samples : (..., n)
    truth   : (...)   — broadcastable to the prefix dims of samples
    returns : (...) — CRPS per cell
    """
    n = samples.shape[-1]
    truth = truth[..., None]  # broadcast against samples
    term1 = np.mean(np.abs(samples - truth), axis=-1)

    sorted_s = np.sort(samples, axis=-1)
    # weights = (2i - n - 1) for i=1..n  →  -(n-1), -(n-3), ..., (n-1)
    i = np.arange(1, n + 1)
    weights = (2 * i - n - 1).astype(sorted_s.dtype)
    term2 = (sorted_s * weights).sum(axis=-1) / (n * n)
    return term1 - term2


def main():
    ds = xr.open_dataset(NC)
    samples = ds["forecast"].values  # (loc, time, horizon, sample)
    truth = ds["observed"].values    # (loc, time)

    # Broadcast truth across horizon
    horizon_n = samples.shape[2]
    truth_b = np.broadcast_to(truth[:, :, None], samples.shape[:3])

    # log-CRPS: transform both
    log_samples = np.log1p(np.maximum(samples, 0))
    log_truth = np.log1p(np.maximum(truth_b, 0))

    crps_log = sample_crps(log_samples, log_truth)  # (loc, time, horizon)
    crps_raw = sample_crps(samples, truth_b)

    # Mask out cells where truth is NaN
    mask = np.isfinite(truth_b)
    print(f"valid cells: {mask.sum()} of {mask.size}")
    print()
    print("== Aggregate metrics ==")
    print(f"  log-CRPS  (mean over all cells): {np.nanmean(np.where(mask, crps_log, np.nan)):.4f}")
    print(f"  raw CRPS  (mean over all cells): {np.nanmean(np.where(mask, crps_raw, np.nan)):.2f}")

    print("\n== Per horizon ==")
    for h in range(horizon_n):
        lc = np.nanmean(np.where(mask[:, :, h], crps_log[:, :, h], np.nan))
        rc = np.nanmean(np.where(mask[:, :, h], crps_raw[:, :, h], np.nan))
        print(f"  h={h}:  log-CRPS {lc:.4f}    raw CRPS {rc:.2f}")

    print("\n== Per location (top 5 and bottom 5 by historical mean) ==")
    # Reuse the historical observations to rank
    hist = ds["historical_observed"].values
    hist_mean = np.nanmean(hist, axis=1)
    order = np.argsort(-hist_mean)
    locs = ds.coords['location'].values
    picks = list(order[:5]) + list(order[-5:])
    for idx in picks:
        loc = locs[idx]
        lc = np.nanmean(np.where(mask[idx], crps_log[idx], np.nan))
        print(f"  {loc:8s}  hist_mean={hist_mean[idx]:>8.1f}  log-CRPS={lc:.4f}")


if __name__ == "__main__":
    main()
