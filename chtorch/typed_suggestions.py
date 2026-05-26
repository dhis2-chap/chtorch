from typing import Literal, get_args, get_origin

import optuna


def suggest_model(model_class: type, trial: optuna.Trial):
    """
    Suggests a model configuration based on the model class and the trial.

    Only fields annotated as `Literal[...]` are supported. Any other
    annotation raises NotImplementedError so the caller can extend coverage
    rather than silently get an under-specified config.
    """
    suggestion_dict = {}
    for key, annotation in model_class.__annotations__.items():
        if get_origin(annotation) is Literal:
            suggestion_dict[key] = trial.suggest_categorical(key, list(get_args(annotation)))
        else:
            raise NotImplementedError(
                f"suggest_model does not yet handle field '{key}' with annotation {annotation!r}"
            )
    return model_class(**suggestion_dict)
