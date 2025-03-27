import optuna 
import torch 

from itertools import product
from chtorch.estimator import Estimator 

from chap_core.data import DataSet
from chap_core.datatypes import FullData

# using grid search
param_grid = {
    "hidden_dim": [4, 8, 16, 32],
    "embed_dim": [2, 4, 8],
    "regularization": [1e-6, 1e-5, 1e-4]
}

best_loss = float("inf")
best_params = {}
for hd, ed, reg in product(param_grid["hidden_dim"],
                           param_grid["embed_dim"],
                           param_grid["regularization"]):
    estimator = Estimator(context_length=12, prediction_length=3, debug=False, validate=True)
    estimator.hidden_dim = hd
    estimator.embed_dim = ed 
    estimator.weight_decay = reg 

    path = input("Dataset path: ")
    # dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)
    dataset = DataSet.from_csv(path, FullData)
    estimator.train(dataset)
    val_loss = estimator.last_val_loss # if val_loss is hooked to estimator.train 

    if val_loss < best_loss:
        best_loss = val_loss 
        best_params = {"hidden_dim": hd, "embed_dim": ed, "regularization": reg}

    print("Best loss:", best_loss)
    print("Best params:", best_params)

# using optuna
def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8, 16, 32])
    embed_dim = trial.suggest_categorical("embed_dim", [2, 4, 8])
    weight_decay = trial.suggest_loguniform("regularization", 1e-8, 1e-3)

    estimator = Estimator(context_length=12, prediction_length=3, debug=False, validate=True)
    estimator.hidden_dim = hidden_dim
    estimator.embed_dim = embed_dim 
    estimator.weight_decay = weight_decay

    dataset = DataSet.from_csv('/home/knut/Data/ch_data/rwanda_harmonized.csv', FullData)
    predictor = estimator.train(dataset)

    val_loss = estimator.final_val_loss

    return val_loss 

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Loss value: ", best_trial.value)
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
