from abc import abstractproperty
from typing import Callable, Dict, Optional
import pandas as pd
from tune_experiment.problems.problem import Problem
from ray import tune
from ray.tune.sample import Sampler
from ray.tune.utils.placement_groups import PlacementGroupFactory

import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class SklearnProblem(Problem):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    @property
    def config(self) -> Dict[str, Sampler]:
        return

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return

    @property
    def early_stopping_key(self) -> str:
        return None

    @property
    def early_stopping_iters(self) -> int:
        return None

    def _get_fit_kwargs(self):
        return {}

    def _get_estimator(self, config: dict, random_seed: int):
        return

    @property
    def trainable(self) -> Callable:
        def sklearn_trainable(config: dict,
                              dataset: pd.DataFrame,
                              target_column: str,
                              cv_folds: int,
                              random_seed: int,
                              checkpoint_dir: Optional[str] = None):
            X = dataset.drop(target_column, axis=1)
            y = dataset[target_column]

            estimator = self._get_estimator(config, random_seed)

            cv_splitter = StratifiedKFold(n_splits=cv_folds,
                                          shuffle=True,
                                          random_state=random_seed)

            is_multiclass = len(np.unique(y)) > 2
            metric = partial(roc_auc_score,
                             multi_class="ovr" if is_multiclass else "raise")

            results = []
            for i, train_test in enumerate(cv_splitter.split(X, y)):
                train, test = train_test
                estimator_fold = clone(estimator)
                estimator_fold.fit(_safe_indexing(X, train),
                                   _safe_indexing(y, train),
                                   **self._get_fit_kwargs())
                pred_proba = estimator_fold.predict_proba(
                    _safe_indexing(X, test))
                if not is_multiclass:
                    pred_proba = pred_proba[:, 1]
                results.append(metric(_safe_indexing(y, test), pred_proba))
            tune.report(**{self.metric_name: np.mean(results)})

        return sklearn_trainable

    @property
    def resource_requirements(self) -> PlacementGroupFactory:
        return PlacementGroupFactory([{"CPU": self.n_jobs}])

    @property
    def metric_name(self) -> str:
        return "roc_auc"

    @property
    def metric_mode(self) -> str:
        return "max"


class LRProblem(SklearnProblem):
    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "C": tune.loguniform(1e-2, 1e2),
            "penalty": tune.choice(["l1", "l2"]),
            "class_weight": tune.choice(["balanced", None])
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {"C": 1.0, "penalty": "l2", "class_weight": None}

    def _get_estimator(self, config: dict, random_seed: int):
        return LogisticRegression(**{
            k: v
            for k, v in config.items() if k in self.config
        },
                                  random_state=random_seed,
                                  solver="saga")


class SVMProblem(SklearnProblem):
    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "C": tune.loguniform(1.0, 1e3),
            "gamma": tune.loguniform(1e-4, 1e-3),
            "tol": tune.loguniform(1e-5, 1e-1),
            "class_weight": tune.choice(["balanced", None])
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {"C": 1.0, "gamma": 1, "tol": 1e-3, "class_weight": None}

    def _get_estimator(self, config: dict, random_seed: int):
        return SVC(
            **{k: v
               for k, v in config.items() if k in self.config},
            kernel="rbf",
            probability=True,
            random_state=random_seed,
        )


class KNNProblem(SklearnProblem):
    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "n_neighbors": tune.randint(1, 25),
            "p": tune.randint(1, 4),
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {"n_neighbors": 5, "p": 2}

    def _get_estimator(self, config: dict, random_seed: int):
        return KNeighborsClassifier(
            **{k: v
               for k, v in config.items() if k in self.config})


class MLPProblem(SklearnProblem):
    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "hidden_layer_sizes": tune.randint(50, 200),
            "alpha": tune.loguniform(1e-5, 1e1),
            "batch_size": tune.randint(10, 250),
            "learning_rate_init": tune.loguniform(1e-5, 1e-1),
            "tol": tune.loguniform(1e-5, 1e-1),
            "beta_1": tune.loguniform(1 - 0.99, 1 - 0.5),  # logit
            "beta_2": tune.loguniform(1 - (1.0 - 1e-6), 1 - 0.9),  # logit
            "epsilon": tune.loguniform(1e-9, 1e-6),
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {
            "hidden_layer_sizes": 100,
            "alpha": 0.0001,
            "batch_size": 200,
            "learning_rate_init": 0.001,
            "tol": 1e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1e-8,
        }

    def _get_estimator(self, config: dict, random_seed: int):
        config = config.copy()
        config["beta_1"] = (1 - config["beta_1"]) * -1
        config["beta_2"] = (1 - config["beta_2"]) * -1
        return MLPClassifier(
            **{k: v
               for k, v in config.items() if k in self.config},
            random_state=random_seed,
            early_stopping=True,
        )