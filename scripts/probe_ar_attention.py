"""Probe whether the model adapts predictions to recent cases.

For a chosen location, show per-eval-period:
  - truth observation
  - the most recent 6 months of historical cases seen by the model
  - prediction median (h=0)

If the model is paying attention to the AR signal, prediction at time t
should rise when the trailing context shows a surge.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

NC = Path("/Users/knutdr/Sources/chtorch/lao_monthly_eval.nc")
CSV = Path("/Users/knutdr/Data/CH/chap_LAO_admin1_monthly-3.csv")

# Locations to inspect — the largest outbreak underpredictions
PICKS = ["LA-VT", "LA-SV", "LA-CH", "LA-KH"]


def main():
    ds = xr.open_dataset(NC)
    df = pd.read_csv(CSV)
    df['time_period'] = pd.PeriodIndex(df['time_period'], freq='M')

    samples = ds['forecast']
    truth = ds['observed']
    pred_med = samples.median('sample').values
    pred_q05 = samples.quantile(0.05, dim='sample').values
    pred_q95 = samples.quantile(0.95, dim='sample').values

    # Time labels for the eval window (9 periods)
    eval_periods = pd.PeriodIndex(ds['time_period'].values, freq='M')
    locs = list(ds.coords['location'].values)

    for loc in PICKS:
        if loc not in locs:
            continue
        i = locs.index(loc)
        loc_df = df[df.location == loc].sort_values('time_period').set_index('time_period')

        print(f"\n==== {loc} ({loc_df['location_name'].iloc[0]}) ====")
        print(f"{'eval_period':>12s} {'trail-6 cases (recent first)':>40s} {'truth':>8s} {'pred(h=0)':>10s} {'q05':>6s} {'q95':>8s}")

        for j, p in enumerate(eval_periods):
            # Build the trailing context — 6 months before this period
            trail = []
            for k in range(6):
                pp = p - (k + 1)
                if pp in loc_df.index:
                    trail.append(loc_df.loc[pp, 'disease_cases'])
                else:
                    trail.append(float('nan'))
            trail_str = "  ".join(f"{x:>5.0f}" if not np.isnan(x) else "  nan" for x in trail)
            t = float(truth.values[i, j])
            m = float(pred_med[i, j, 0])
            lo = float(pred_q05[i, j, 0])
            hi = float(pred_q95[i, j, 0])
            print(f"   {str(p):>9s}  {trail_str:>40s}  {t:>8.0f}  {m:>10.1f}  {lo:>6.1f}  {hi:>8.1f}")


if __name__ == "__main__":
    main()
