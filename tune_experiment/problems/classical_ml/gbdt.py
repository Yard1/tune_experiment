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
from sklearn.utils import _safe_indexing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from functools import partial
import xgboost as xgb
import lightgbm as lgbm

def get_xgb_num_trees(bst: xgb.Booster) -> int:
    import json
    data = [json.loads(d) for d in bst.get_dump(dump_format="json")]
    return len(data) // 4

class XGBoostProblem(Problem):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": tune.lograndint(10, 1000),
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
        return 200

    def _get_estimator(self, config: dict, random_seed: int):
        return xgb.XGBClassifier(
                **{k: v
                   for k, v in config.items() if k in self.config},
                n_jobs=self.n_jobs,
                random_state=random_seed,
                use_label_encoder=False
            )

    def _get_booster(self, booster: xgb.XGBModel):
        return booster.get_booster()

    def _get_fit_kwargs(self, init_model: xgb.Booster):
        return {"xgb_model": init_model}

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
                    xgb_models, trees_already_boosted = pickle.load(f)
            else:
                xgb_models = [None] * cv_folds
                trees_already_boosted = 0

            xgb_estimator = self._get_estimator(config, random_seed)

            cv_splitter = StratifiedKFold(n_splits=cv_folds,
                                          shuffle=True,
                                          random_state=random_seed)

            is_multiclass = len(np.unique(y)) > 2
            metric = partial(
                roc_auc_score,
                multi_class="ovr" if is_multiclass else "raise")

            n_trees = config[self.early_stopping_key] - trees_already_boosted

            for tree in range(n_trees):
                results = []
                for i, train_test in enumerate(cv_splitter.split(X, y)):
                    train, test = train_test
                    xgb_estimator_fold = clone(xgb_estimator)
                    xgb_estimator_fold.set_params(**{self.early_stopping_key: 1})
                    xgb_estimator_fold.fit(_safe_indexing(X, train), _safe_indexing(y, train), **self._get_fit_kwargs(xgb_models[i]))
                    pred_proba = xgb_estimator_fold.predict_proba(_safe_indexing(X, test))
                    if not is_multiclass:
                        pred_proba = pred_proba[:, 1]
                    results.append(
                        metric(_safe_indexing(y, test), pred_proba))
                    xgb_models[i] = self._get_booster(xgb_estimator_fold)
                if tree % 10 == 0 or tree-1 == config[self.early_stopping_key]:
                    with tune.checkpoint_dir(step=tree) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        with open(path, "wb") as f:
                            pickle.dump((xgb_models, tree), f)
                tune.report(**{self.metric_name: np.mean(results)})
        return xgboost_trainable

    @property
    def resource_requirements(self) -> PlacementGroupFactory:
        return PlacementGroupFactory([{"CPU": self.n_jobs}])

    @property
    def metric_name(self) -> str:
        return "roc_auc"

    @property
    def metric_mode(self) -> str:
        return "max"

class LightGBMProblem(XGBoostProblem):
    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": tune.lograndint(10, 1000),
            "num_leaves": tune.lograndint(4, 1000),
            "min_child_samples": tune.lograndint(2, 1+2**7),
            "learning_rate": tune.loguniform(1 / 1024, 1.0),
            "subsample": tune.uniform(0.1, 1.0),
            "log_max_bin": tune.lograndint(3, 11),
            "colsample_bytree": tune.uniform(0.1, 1.0),
            "reg_alpha": tune.loguniform(1 / 1024, 1024),
            "reg_lambda": tune.loguniform(1 / 1024, 1024),
        }

    @property
    def init_config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": 100,
            "num_leaves": 31,
            "min_child_samples": 20,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "log_max_bin": 8,
            "colsample_bytree": 1.0,
            "reg_alpha": 1 / 1024,
            "reg_lambda": 1 / 1024,
        }

    @property
    def early_stopping_iters(self) -> int:
        return 400

    def _get_estimator(self, config: dict, random_seed: int):
        config = config.copy()
        config["max_bin"] = 1 << int(round(config.pop("log_max_bin"))) - 1
        return lgbm.LGBMClassifier(
                **{k: v
                   for k, v in config.items() if k in self.config or k == "max_bin"},
                n_jobs=1,
                random_state=random_seed,
            )

    def _get_booster(self, booster: lgbm.LGBMModel):
        return booster.booster_

    def _get_fit_kwargs(self, init_model: lgbm.Booster):
        return {"init_model": init_model}