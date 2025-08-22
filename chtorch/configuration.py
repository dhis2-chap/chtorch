from pydantic import BaseModel

from chtorch.module import RNNConfiguration
from yacs.config import CfgNode as CN


class TensorifierConfig(BaseModel): # data preprocessing config
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



_C = CN()

_C.NAME = CN()
_C.NAME.EXPERIMENT_NAME = "PBMOHPO"

_C.PROBLEM = CN()
_C.PROBLEM.PROBLEM_TYPE = "TuneDeepAR"
_C.PROBLEM.DATAPATH = "data/vietnam_monthly.csv"

_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER_TYPE = "BO"

_C.DECISION_MAKER = CN()
_C.DECISION_MAKER.DECISION_MAKER_TYPE = "DecisionMaker"
_C.DECISION_MAKER.SEED = 123

_C.BUDGET = CN()
_C.BUDGET.EVAL_BUDGET = 50
_C.BUDGET.DM_BUDGET = 50

_C.BATCH_SIZE = CN()
_C.BATCH_SIZE.EVAL_BATCH_SIZE = 1
_C.BATCH_SIZE.DM_BATCH_SIZE = 1

_C.REPL = CN()
_C.REPL.SEEDREPL = 0

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()