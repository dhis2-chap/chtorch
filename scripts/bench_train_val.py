"""Train with validation hold-out and log per-epoch train/val loss.

Use this to answer: is the current setup undertrained at N epochs?
Holds out the last 3 months (default) as validation, trains for the
requested number of epochs, prints both loss trajectories at the end.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightning as L
import yaml
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.estimator import Estimator


class LossLogger(L.Callback):
    def __init__(self):
        self.train = []
        self.val = []
        self._t0 = None
        self.epoch_times = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._t0 is not None:
            self.epoch_times.append(time.perf_counter() - self._t0)
            self._t0 = None
        tl = trainer.callback_metrics.get('train_loss')
        if tl is not None:
            self.train.append(float(tl))

    def on_validation_epoch_end(self, trainer, pl_module):
        vl = trainer.callback_metrics.get('validation_loss')
        if vl is not None:
            self.val.append(float(vl))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/knutdr/Downloads/chap_VNM_admin1_monthly.csv")
    parser.add_argument("--config", default="/Users/knutdr/Sources/chtorch/vnm_monthly_config.yaml")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--prediction-length", type=int, default=3)
    args = parser.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    user_opts = dict(cfg_raw.get("user_option_values", {}))
    user_opts["max_epochs"] = args.max_epochs
    additional = cfg_raw.get("additional_continuous_covariates", [])
    model_cfg = ModelConfiguration(**user_opts, additional_covariates=additional)
    prob_cfg = ProblemConfiguration(prediction_length=args.prediction_length, validate=True)

    dataset = DataSet.from_csv(args.data)
    train_dataset, val_generator = train_test_generator(
        dataset, prediction_length=args.prediction_length, n_test_sets=1
    )
    validation_dataset = next(val_generator)[-1]

    estimator = Estimator(prob_cfg, model_cfg)
    estimator.add_validation(validation_dataset)

    logger_cb = LossLogger()
    orig_init = L.Trainer.__init__

    def patched(self, *a, **kw):
        cbs = list(kw.get("callbacks") or [])
        cbs.append(logger_cb)
        kw["callbacks"] = cbs
        return orig_init(self, *a, **kw)
    L.Trainer.__init__ = patched

    estimator.train(train_dataset)

    print()
    print("==== SUMMARY ====")
    print(f"epochs measured: {len(logger_cb.train)} train, {len(logger_cb.val)} val")
    if logger_cb.epoch_times:
        et = logger_cb.epoch_times
        print(f"per-epoch wall: median {sorted(et)[len(et)//2]:.2f}s  total {sum(et):.1f}s")

    print("\nepoch  train_loss  val_loss")
    n = max(len(logger_cb.train), len(logger_cb.val))
    every = max(1, n // 25)  # print ~25 rows
    for i in range(n):
        if i % every == 0 or i == n - 1:
            tl = f"{logger_cb.train[i]:.4f}" if i < len(logger_cb.train) else " -- "
            vl = f"{logger_cb.val[i]:.4f}" if i < len(logger_cb.val) else " -- "
            print(f"{i:5d}   {tl:>10s}   {vl:>10s}")


if __name__ == "__main__":
    main()
