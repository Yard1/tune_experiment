from abc import ABC, abstractmethod
from typing import Callable, Dict
from ray import tune
from ray.tune.sample import Domain, Categorical
from ray.tune.utils.placement_groups import PlacementGroupFactory
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd

class DummyTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


class Problem(ABC):
    def __init__(self):
        return

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def get_preprocessor(self, data: pd.DataFrame) -> Pipeline:
        return Pipeline([("identity", DummyTransformer())])

    @property
    @abstractmethod
    def config(self) -> Dict[str, Domain]:
        return

    @property
    def config_early_stopping(self) -> Dict[str, Domain]:
        if self.early_stopping_key:
            config = self.config.copy()
            config[self.early_stopping_key] = self.early_stopping_iters
        else:
            config = self.config
        return config

    @property
    @abstractmethod
    def init_config(self) -> Dict[str, Domain]:
        return

    @property
    def has_categorical_parameters(self) -> bool:
        return any(isinstance(v, Categorical) for v in self.config.values())

    @property
    @abstractmethod
    def early_stopping_key(self) -> str:
        return

    @property
    @abstractmethod
    def early_stopping_iters(self) -> int:
        return

    @property
    @abstractmethod
    def metric_name(self) -> str:
        return

    @property
    @abstractmethod
    def metric_mode(self) -> str:
        return

    @abstractmethod
    def get_trainable(self) -> Callable:
        return

    @property
    @abstractmethod
    def resource_requirements(self) -> PlacementGroupFactory:
        return

    def trainable_with_parameters(self, X, y, num_classes: int, cv_folds: int,
                                  random_seed: int, results_path: str):
        return tune.with_parameters(self.get_trainable(),
                                    X=X,
                                    y=y,
                                    num_classes=num_classes,
                                    cv_folds=cv_folds,
                                    random_seed=random_seed,
                                    results_path=results_path)
