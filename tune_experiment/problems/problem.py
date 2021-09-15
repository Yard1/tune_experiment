from abc import ABC, abstractmethod
from typing import Callable, Dict
import pandas as pd
from ray import tune
from ray.tune.sample import Sampler, Categorical
from ray.tune.utils.placement_groups import PlacementGroupFactory

class Problem(ABC):
    def __init__(self):
        return

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    @property
    def config(self) -> Dict[str, Sampler]:
        return

    @property
    def config_early_stopping(self) -> Dict[str, Sampler]:
        config = self.config.copy()
        config[self.early_stopping_key] = self.early_stopping_iters
        return config

    @abstractmethod
    @property
    def init_config(self) -> Dict[str, Sampler]:
        return

    @property
    def has_categorical_parameters(self) -> bool:
        return any(isinstance(v, Categorical) for v in self.config.values())

    @abstractmethod
    @property
    def early_stopping_key(self) -> str:
        return

    @abstractmethod
    @property
    def early_stopping_iters(self) -> int:
        return

    @abstractmethod
    @property
    def metric_name(self) -> str:
        return

    @abstractmethod
    @property
    def metric_mode(self) -> str:
        return

    @abstractmethod
    @property
    def trainable(self) -> Callable:
        return

    @abstractmethod
    @property
    def resource_requirements(self) -> PlacementGroupFactory:
        return

    def trainable_with_parameters(
        self,
        dataset: pd.DataFrame,
        target_column: str,
        cv_folds: int,
        random_seed: int
    ):
        return tune.with_parameters(
            self.trainable,
            dataset=dataset,
            target_column=target_column,
            cv_folds=cv_folds,
            random_seed=random_seed
        )