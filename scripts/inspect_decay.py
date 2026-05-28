"""Show which parameters end up in which weight-decay group, and the
post-training norms of the location embeddings, to confirm whether
location_embeddings.0.weight is being over-regularized.
"""
import torch

from chtorch.configuration import ModelConfiguration, ProblemConfiguration
from chtorch.estimator import Estimator
from chtorch.lightning_module import DeepARLightningModule


def main():
    # Tiny synthetic to get a module quickly
    import numpy as np
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
    dataset = DataSet.from_csv('/Users/knutdr/Downloads/chap_VNM_admin1_monthly.csv')

    model_cfg = ModelConfiguration(context_length=12, embed_dim=4, n_hidden=16,
                                    state_dim=16, num_rnn_layers=1, max_dim=32,
                                    output_embedding_dim=0, dropout=0.1,
                                    weight_decay=1e-5, max_epochs=1,
                                    additional_covariates=['rainfall', 'mean_temperature', 'mean_relative_humidity'])
    prob_cfg = ProblemConfiguration(prediction_length=3, debug=True)

    estimator = Estimator(prob_cfg, model_cfg)
    predictor = estimator.train(dataset)
    module = predictor.module

    # Reconstruct decay groups
    weight_decay = model_cfg.weight_decay
    decay, embed_decay, level_2_decay, no_decay = [], [], [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('bias') or 'norm' in name.lower():
            bucket = 'no_decay'; no_decay.append((name, param))
        elif 'embed' in name:
            if '.0.' in name:
                bucket = 'level_2 (×100)'; level_2_decay.append((name, param))
            else:
                bucket = 'embed (×10)'; embed_decay.append((name, param))
        else:
            bucket = 'normal (×1)'; decay.append((name, param))
        norm = param.detach().norm().item()
        print(f"  {bucket:18s} wd={weight_decay * (100 if '×100' in bucket else 10 if '×10' in bucket else 1 if '×1' in bucket else 0):.0e}  {name:60s} shape={tuple(param.shape)}  L2={norm:.3f}")

    print("\n== Critical embeddings ==")
    for name, p in level_2_decay + embed_decay:
        if 'location_embeddings' in name:
            row_norms = p.detach().norm(dim=1)
            print(f"  {name:50s} per-row L2 mean={row_norms.mean():.4f}  min={row_norms.min():.4f}  max={row_norms.max():.4f}")


if __name__ == "__main__":
    main()
