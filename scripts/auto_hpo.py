import json

from itertools import product
from chtorch.estimator import Estimator
from chtorch.configuration import ModelConfiguration, ProblemConfiguration
import logging
logger = logging.getLogger(__name__)
from chtorch.hpo import optuna_search


def main(dataset):
    # using grid search
    param_grid = {
        "weight_decay": [4, 8, 16, 32],
        "n_hidden": [4, 8, 16, 32],
        "max_epochs": [2, 3],
        "context_length": [7, 10, 12, 15, 20],
        "embed_dim": [2, 4, 8],
        "num_rnn_layers": [4, 8, 16, 32],
        "n_layers": [4, 8, 16, 32]
    }

    param_grid = {
        "weight_decay": [4],
        "n_hidden": [4],
        "max_epochs": [2],
        "context_length": [7],
        "embed_dim": [2],
        "num_rnn_layers": [4],
        "n_layers": [4]
    }

    best_loss = float("inf")
    best_model = None

    for wd, nh, me, cl, ed, nrl, nl in product(param_grid["weight_decay"],
                                               param_grid["n_hidden"],
                                               param_grid["max_epochs"],
                                               param_grid["context_length"],
                                               param_grid["embed_dim"],
                                               param_grid["num_rnn_layers"],
                                               param_grid["n_layers"]):
        prob_config = ProblemConfiguration(replace_zeros=True)
        model_config = ModelConfiguration(weight_decay=wd,
                                          n_hidden=nh,
                                          max_epochs=me,
                                          context_length=cl,
                                          embed_dim=ed,
                                          num_rnn_layers=nrl,
                                          n_layers=nl)
        estimator = Estimator(prob_config, model_config, validate=True)

        estimator.train(dataset)
        val_loss = estimator.last_val_loss  # if val_loss is hooked to estimator.train

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model_config

        logger.info(f"Best loss so far: {best_loss}")

    with open('model_config_grid.json', 'w') as f:
        json.dump(best_model.model_dump(), f, indent=4)


# using optuna

if __name__ == "__main__":
    #path = input("Dataset path: ")
    import sys
    path = sys.argv[1]
    n_trials= int(sys.argv[2]) if len(sys.argv) > 2 else 10
    output_name = 'model_config_optuna.json'
    optuna_search(path, n_trials, output_name)
