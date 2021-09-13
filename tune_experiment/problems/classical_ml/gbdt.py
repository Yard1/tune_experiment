from tune_experiment.problems.problem import Problem
from typing import Callable, Dict, Optional
from ray import tune
from ray.tune.sample import Sampler
from ray.tune.utils.placement_groups import PlacementGroupFactory

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
import xgboost as xgb


class XGBoostProblem(Problem):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": tune.lograndint(10, 10000),
            "max_depth": tune.randint(1, 10),
            "min_child_weight": tune.loguniform(0.001, 128),
            "learning_rate": tune.loguniform(1 / 1024, 1.0),
            "subsample": tune.uniform(0.1, 1.0),
            "colsample_bylevel": tune.uniform(0.1, 1.0),
            "colsample_bytree": tune.uniform(0.1, 1.0),
            "reg_alpha": tune.loguniform(1 / 1024, 1024),
            "reg_lambda": tune.loguniform(1 / 1024, 1024),
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.3,
            "subsample": 1.0,
            "colsample_bylevel": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 1 / 1024,
            "reg_lambda": 1 / 1024,
        }

    @property
    def early_stopping_key(self) -> str:
        return "n_estimators"

    @property
    def early_stopping_iters(self) -> int:
        return 1000

    def _get_estimator(self, config: dict, random_seed: int):
        return xgb.XGBClassifier(
                **{k: v
                   for k, v in config.items() if k in self.config},
                n_jobs=1,
                random_state=random_seed,
            )

    def _get_booster(self, booster: xgb.XGBModel):
        return booster.get_booster()

    def _get_num_boosted_rounds(self, booster: xgb.Booster):
        return booster.num_boosted_rounds()

    def _set_params(self, estimator: xgb.XGBModel, n_estimators_left: int, init_model: xgb.Booster) -> None:
        estimator.set_params(**{self.early_stopping_key: n_estimators_left, "xgb_model": init_model})

    @property
    def trainable(self) -> Callable:
        def xgboost_trainable(config: dict,
                              dataset: pd.DataFrame,
                              target_column: str,
                              cv_folds: int,
                              random_seed: int,
                              checkpoint_dir: Optional[str] = None):
            X = dataset.drop(target_column, axis=1)
            y = dataset[target_column]
            if checkpoint_dir:
                with open(os.path.join(checkpoint_dir, "checkpoint"),
                          "rb") as f:
                    xgb_models = pickle.load(f)
            else:
                xgb_models = [None] * cv_folds

            xgb_estimator = self._get_estimator(config, random_seed)

            cv_splitter = StratifiedKFold(n_splits=cv_folds,
                                          shuffle=True,
                                          random_state=random_seed)
            metric = partial(
                roc_auc_score,
                multi_class="ovr" if len(np.unique(y)) > 2 else "raise")

            for tree in range(self.early_stopping_iters):
                results = []
                n_estimators_left = (
                    config[self.early_stopping_key] -
                    self._get_num_boosted_rounds(xgb_models[0])
                ) if xgb_models[0] else config[self.early_stopping_key]
                for i, train_test in enumerate(cv_splitter.split(X, y)):
                    train, test = train_test
                    xgb_estimator_fold = clone(xgb_estimator)
                    self._set_params(xgb_estimator_fold, n_estimators_left, xgb_models[i])
                    xgb_estimator_fold.fit(X[train], y[train])
                    results.append(
                        metric(y[test],
                               xgb_estimator_fold.predict_proba(X[test])))
                    xgb_models[i] = self._get_booster(xgb_estimator)
                with tune.checkpoint_dir(step=tree) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    with open(path, "w") as f:
                        pickle.dump(xgb_models, f)
                tune.report(metric=np.mean(results))
        return xgboost_trainable

    @property
    def resource_requirements(self) -> PlacementGroupFactory:
        return PlacementGroupFactory([{"CPU": self.n_jobs}])