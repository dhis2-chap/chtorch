from pydantic import BaseModel

from chtorch.module import RNNConfiguration


class TensorifierConfig(BaseModel):
    mask_covid: bool = True
    previous_cases: bool = False
    use_population: bool = True
    additional_covariates: list[str] = ['rainfall', 'mean_temperature']


class ModelConfiguration(RNNConfiguration, TensorifierConfig, extra='forbid'):
    """Should be composition not inheritance"""
    # Very technical hp
    weight_decay: float = 1e-6  # Regularization
    max_epochs: int | None = None  # Training
    learning_rate: float = 1e-3  # Training/Convergence
    batch_size: int = 64  # Training/Convergence
    augmentations: list[str] = []  # Regularization
    context_length: int = 12
    past_ratio: float = 0.2  # Regularization


class ProblemConfiguration(BaseModel):
    prediction_length: int | None = 3
    replace_zeros: bool = False
    replace_nans: bool = False
    predict_nans: bool = False  # This can also be a model configuration
    debug: bool = False
    validate: bool = False
    validation_splits: int = 5
    validation_index: int = 4
