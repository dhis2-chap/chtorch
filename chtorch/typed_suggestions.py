from typing import Literal

import optuna


def suggest_model(model_class: type, trial: optuna.Trial):
    """
    Suggests a model configuration based on the model class and the trial

    Parameters
    ----------
    model_class : type
        The model class to suggest a configuration for
    trial : optuna.Trial
        The optuna trial to suggest the configuration for

    Returns
    -------
    model_class
        The suggested model configuration
    """
    suggestion_dict = {}
    for key, value in model_class.__annotations__.items():
        if value.__origin__ == Literal:
            suggestion_dict[key] = trial.suggest_categorical(key, value.__args__)


    return model_class(**suggestion_dict)
