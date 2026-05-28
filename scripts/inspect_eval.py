"""Look at top-cases locations to see if the model captures dynamic range."""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

NC = Path("/Users/knutdr/Sources/chtorch/vnm_monthly_eval.nc")
CSV = Path("/Users/knutdr/Downloads/chap_VNM_admin1_monthly.csv")


def main():
    ds = xr.open_dataset(NC)
    df = pd.read_csv(CSV)
    by_loc = df.groupby('location')['disease_cases']

    # Build location-id → name mapping
    name_by_loc = df.groupby('location')['location_name'].first().to_dict()

    locations = ds.coords['location'].values
    samples = ds['forecast']
    truth = ds['observed']

    pred_med = samples.median('sample')
    pred_q05 = samples.quantile(0.05, dim='sample')
    pred_q95 = samples.quantile(0.95, dim='sample')

    # Top-5 by historical mean and bottom-5
    hist_means = df.groupby('location')['disease_cases'].mean().sort_values(ascending=False)
    pick = list(hist_means.head(5).index) + list(hist_means.tail(5).index)

    print(f"{'location':14s} {'name':15s} {'hist_mean':>10s} {'eval_truth':>10s} {'pred_med':>10s} {'q05':>8s} {'q95':>8s}")
    for loc in pick:
        if loc not in locations:
            continue
        i = list(locations).index(loc)
        # average across time/horizon for compact view
        t_avg = truth.isel(location=i).mean().item()
        m_avg = pred_med.isel(location=i).mean().item()
        l_avg = pred_q05.isel(location=i).mean().item()
        u_avg = pred_q95.isel(location=i).mean().item()
        name = name_by_loc.get(loc, '?')[:14]
        print(f"{loc:14s} {name:15s} {hist_means[loc]:>10.1f} {t_avg:>10.1f} {m_avg:>10.1f} {l_avg:>8.1f} {u_avg:>8.1f}")

    # Per-horizon error
    print("\nMAE per horizon (forecast 1m / 2m / 3m ahead):")
    obs_b = truth.expand_dims({'horizon_distance': samples.sizes['horizon_distance']}, axis=-1)
    err = (samples.mean('sample') - obs_b)
    for h in range(samples.sizes['horizon_distance']):
        vals = err.isel(horizon_distance=h).values.ravel()
        vals = vals[~np.isnan(vals)]
        print(f"  h={h}: bias {vals.mean():>+8.2f}  MAE {np.abs(vals).mean():>8.2f}  RMSE {np.sqrt((vals**2).mean()):>8.2f}")


if __name__ == "__main__":
    main()
