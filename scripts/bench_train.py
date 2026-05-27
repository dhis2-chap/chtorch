"""Benchmark a single Estimator.train call on the VNM monthly dataset.

Runs with the production config (max_epochs=30) but you can override via CLI.
Times: dataset load, tensorifier, train_dataset build, model construction,
Trainer.fit (with per-epoch hook). Also dumps a cProfile.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import lightning as L
import yaml
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.estimator import Estimator


class TimingCallback(L.Callback):
    def __init__(self):
        self.epoch_times = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t0 is not None:
            self.epoch_times.append(time.perf_counter() - self._t0)
            self._t0 = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/knutdr/Downloads/chap_VNM_admin1_monthly.csv")
    parser.add_argument("--config", default="/Users/knutdr/Sources/chtorch/vnm_monthly_config.yaml")
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Override max_epochs for the benchmark")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override num_workers in DataLoader")
    parser.add_argument("--profile-out", default="/tmp/bench_train.prof")
    args = parser.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    user_opts = cfg_raw.get("user_option_values", {})
    user_opts["max_epochs"] = args.max_epochs
    additional = cfg_raw.get("additional_continuous_covariates", [])

    model_cfg = ModelConfiguration(**user_opts, additional_covariates=additional)
    prob_cfg = ProblemConfiguration(prediction_length=3)

    t0 = time.perf_counter()
    dataset = DataSet.from_csv(args.data)
    t_load = time.perf_counter() - t0

    estimator = Estimator(prob_cfg, model_cfg)
    # Optionally override num_workers by monkey-patching DataLoader default.
    if args.num_workers is not None:
        import torch.utils.data as _du
        orig = _du.DataLoader.__init__

        def patched(self, *a, **kw):
            kw["num_workers"] = args.num_workers
            return orig(self, *a, **kw)

        _du.DataLoader.__init__ = patched

    # Install a Lightning timing callback by monkey-patching L.Trainer.
    timing_cb = TimingCallback()
    orig_trainer_init = L.Trainer.__init__

    def patched_trainer(self, *a, **kw):
        cbs = list(kw.get("callbacks") or [])
        cbs.append(timing_cb)
        kw["callbacks"] = cbs
        return orig_trainer_init(self, *a, **kw)

    L.Trainer.__init__ = patched_trainer

    prof = cProfile.Profile()
    prof.enable()
    t0 = time.perf_counter()
    estimator.train(dataset)
    t_train = time.perf_counter() - t0
    prof.disable()
    prof.dump_stats(args.profile_out)

    print()
    print("==== BENCHMARK SUMMARY ====")
    print(f"data load:        {t_load:.2f}s")
    print(f"full train call:  {t_train:.2f}s")
    if timing_cb.epoch_times:
        et = timing_cb.epoch_times
        print(f"per-epoch:        min {min(et):.2f}s  median {sorted(et)[len(et)//2]:.2f}s  max {max(et):.2f}s  total {sum(et):.2f}s")
        print(f"#epochs measured: {len(et)}")
    print(f"profile dumped:   {args.profile_out}")

    # Print top hotspots inline.
    print()
    print("==== TOP 25 cumulative ====")
    stats = pstats.Stats(prof).sort_stats("cumulative")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
