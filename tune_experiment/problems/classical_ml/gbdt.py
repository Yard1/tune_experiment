from tune_experiment.problems.problem import Problem, DummyTransformer
from typing import Callable, Dict, Optional, Tuple
from ray import tune
from ray.tune.sample import Sampler
from ray.tune.utils.placement_groups import PlacementGroupFactory

import os
import pandas as pd
from pandas.api.types import is_categorical_dtype
import numpy as np
import pickle
from sklearn.utils import _safe_indexing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.special import softmax
from functools import partial
import xgboost as xgb
import lightgbm as lgbm

from tune_experiment.utils import set_up_s3fs


def get_xgb_num_trees(bst: xgb.Booster) -> int:
    import json
    data = [json.loads(d) for d in bst.get_dump(dump_format="json")]
    return len(data) // 4


def oridinal_transform_column(col: pd.Series):
    if is_categorical_dtype(col.dtype):
        col = col.cat.codes
    return col


def ordinal_transform_pandas(X: pd.DataFrame, y=None):
    return X.apply(oridinal_transform_column)


class XGBoostProblem(Problem):
    def __init__(self, n_jobs: int):
        self.n_jobs = n_jobs

    @property
    def config(self) -> Dict[str, Sampler]:
        return {
            "n_estimators": tune.lograndint(1, 50),
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
            "n_estimators": 10,
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
    def preprocessor(self) -> Pipeline:
        return Pipeline([("encode",
                          FunctionTransformer(ordinal_transform_pandas))])

    @property
    def early_stopping_key(self) -> str:
        return "n_estimators"

    @property
    def early_stopping_iters(self) -> int:
        return 50

    def _get_params(self, config: dict, random_seed: int,
                    num_classes: int) -> dict:
        params = {k: v for k, v in config.items() if k in self.config}
        default_params = dict(
            nthread=self.n_jobs,
            seed=random_seed,
            objective="multi:softprob"
            if num_classes > 2 else "binary:logistic",
            num_class=num_classes,
            verbosity=0,
        )
        if num_classes <= 2:
            default_params.pop("num_class")
        return {**default_params, **params}

    def _get_dataset(self, X, y):
        return xgb.DMatrix(X, y)

    def _get_eval_metric(self, metric, num_classes: int):
        def eval_metric(y_score: np.ndarray,
                        dtrain: xgb.DMatrix) -> Tuple[str, float]:
            y_true = dtrain.get_label()
            if num_classes > 2:
                y_score = softmax(y_score, axis=1)
            return self.metric_name, metric(y_true, y_score)

        return eval_metric

    def _train(self, params: dict, train: xgb.DMatrix,
               num_boosting_rounds: int, test: xgb.DMatrix,
               eval_metric: Callable, evals_result: dict, init_model):
        params.pop("n_estimators", None)
        return xgb.train(params,
                         train,
                         num_boosting_rounds,
                         evals=[(test, "test")],
                         feval=eval_metric,
                         evals_result=evals_result,
                         xgb_model=init_model,
                         verbose_eval=False)

    def get_trainable(self) -> Callable:
        def xgboost_trainable(config: dict,
                              X,
                              y,
                              num_classes: int,
                              cv_folds: int,
                              random_seed: int,
                              results_path: str,
                              checkpoint_dir: Optional[str] = None):
            set_up_s3fs(results_path)
            if checkpoint_dir:
                with open(os.path.join(checkpoint_dir, "checkpoint"),
                          "rb") as f:
                    xgb_models, trees_already_boosted = pickle.load(f)
            else:
                xgb_models = [None] * cv_folds
                trees_already_boosted = 0

            cv_splitter = StratifiedKFold(n_splits=cv_folds,
                                          shuffle=True,
                                          random_state=random_seed)

            is_multiclass = num_classes > 2
            metric = partial(roc_auc_score,
                             average="weighted",
                             multi_class="ovr" if is_multiclass else "raise")
            eval_metric = self._get_eval_metric(metric, num_classes)

            n_trees = config[self.early_stopping_key] - trees_already_boosted
            data_per_fold = []
            for i, train_test in enumerate(cv_splitter.split(X, y)):
                train, test = train_test
                data_per_fold.append(
                    (self._get_dataset(_safe_indexing(X, train),
                                       _safe_indexing(y, train)),
                     self._get_dataset(_safe_indexing(X, test),
                                       _safe_indexing(y, test))))

            params = self._get_params(config, random_seed, num_classes)

            for tree in range(n_trees):
                results = []
                for i, data in enumerate(data_per_fold):
                    train, test = data
                    evals_result = {}
                    bst = self._train(params, train, 10, test, eval_metric,
                                      evals_result, xgb_models[i])
                    results.append(evals_result["test"][self.metric_name][-1])
                    xgb_models[i] = bst
                with tune.checkpoint_dir(step=tree) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    with open(path, "wb") as f:
                        pickle.dump((xgb_models, tree), f)
                tune.report(**{self.metric_name: np.mean(results)})

        xgboost_trainable.__name__ = f"{self.__class__.__name__}_trainable"
        xgboost_trainable.__qualname__ = xgboost_trainable.__name__
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
            "n_estimators": tune.lograndint(1, 50),
            "num_leaves": tune.lograndint(4, 1000),
            "min_child_samples": tune.lograndint(2, 1 + 2**7),
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
            "n_estimators": 10,
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
        return 50

    @property
    def preprocessor(self) -> Pipeline:
        return Pipeline([("identity", DummyTransformer())])

    def _get_params(self, config: dict, random_seed: int,
                    num_classes: int) -> dict:
        config = config.copy()
        config["max_bin"] = 1 << int(round(config.pop("log_max_bin"))) - 1
        params = {
            k: v
            for k, v in config.items() if k in self.config or k == "max_bin"
        }
        default_params = dict(
            n_jobs=self.n_jobs,
            random_state=random_seed,
            objective="multiclass" if num_classes > 2 else "binary",
            num_class=num_classes,
            verbose=-1,
        )
        if num_classes <= 2:
            default_params.pop("num_class")
        return {**default_params, **params}

    def _get_dataset(self, X, y):
        return lgbm.Dataset(X, y, free_raw_data=False)

    def _get_eval_metric(self, metric, num_classes: int):
        def eval_metric(y_score: np.ndarray,
                        dtrain: lgbm.Dataset) -> Tuple[str, float, bool]:
            y_true = dtrain.get_label()
            if num_classes > 2:
                y_score = y_score.reshape(num_classes, -1).T
                y_score = softmax(y_score, axis=1)
            return self.metric_name, metric(y_true, y_score), True

        return eval_metric

    def _train(self, params: dict, train: xgb.DMatrix,
               num_boosting_rounds: int, test: xgb.DMatrix,
               eval_metric: Callable, evals_result: dict, init_model):
        params.pop("n_estimators", None)
        return lgbm.train(params,
                          train,
                          num_boosting_rounds,
                          valid_sets=[test],
                          valid_names=["test"],
                          feval=eval_metric,
                          evals_result=evals_result,
                          init_model=init_model,
                          verbose_eval=False,
                          keep_training_booster=True)
