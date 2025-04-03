from typing import Literal, Annotated
from chtorch.typed_suggestions import suggest_model
import optuna
import pytest
from pydantic import BaseModel, Field


class TestModel(BaseModel):
    category: Literal['A', 'B', 'C']
    dropout: Annotated[float, Field(strict=True, gt=0, lt=1)]


@pytest.fixture
def trial():
    return optuna.create_trial(state=optuna.trial.TrialState.RUNNING)


@pytest.fixture
def study():
    return optuna.create_study(direction="maximize")


def objective(trial):
    model = suggest_model(TestModel, trial)
    assert isinstance(model, TestModel)
    return 0.0

@pytest.mark.skip(reason="Test is not implemented yet")
def test_suggest_model(study):
    # model = suggest_model(TestModel, trial)
    study.optimize(objective, n_trials=1)
