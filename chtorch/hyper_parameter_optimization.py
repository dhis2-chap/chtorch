import optuna as optuna

from chtorch.estimator import ModelConfiguration


def suggest_model_config(trial: optuna.Trial,
                         base_configuration: ModelConfiguration = ModelConfiguration()) -> ModelConfiguration:
    cp = base_configuration.copy()
    cp.dropout = trial.suggest_float("model.dropout", 0.0, 0.5)
    return cp


def objective(trial: optuna.Trial) -> float:
    config = suggest_model_config(trial)
    model = MyModel(config.model)

    # Train and validate using config
    val_loss = train_and_evaluate(model, config)

    return val_loss
