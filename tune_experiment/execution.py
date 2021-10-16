from typing import Optional
import pickle
import os
import shutil
import gc
import numpy as np
import pandas as pd
import subprocess
from ray import tune
from ray.tune.syncer import SyncConfig
from tune_experiment.problems.problem import Problem
from tune_experiment.searchers import searcher_registry
from tune_experiment.utils import set_up_s3fs, run_on_every_ray_node
from sklearn.preprocessing import LabelEncoder


def benchmark_classical_ml(data_url: str,
                           problem: Problem,
                           time_budget_s: float,
                           max_concurrent: int = 1,
                           random_seed: int = 1,
                           cv_folds: int = 5,
                           searcher_name: Optional[str] = None,
                           force_redo: bool = False):
    gc.collect()

    print(f"Downloading dataset {data_url}")
    data = pd.read_parquet(data_url).select_dtypes(exclude=['object'])
    print("Dataset downloaded, preprocessing...")
    y = pd.Series(LabelEncoder().fit_transform(data["target"]))
    X = problem.get_preprocessor(data).fit_transform(data.drop("target", axis=1), y)
    print("Dataset preprocessed.")

    if searcher_name is None:
        searchers = searcher_registry
    else:
        searchers = {searcher_name: searcher_registry[searcher_name]}

    for name, searcher in searchers.items():
        if not (searcher.supports_categorical
                and searcher.suitable_for_classical_ml):
            if searcher_name is not None:
                raise ValueError(
                    f"{searcher_name} doesn't support categorical variables or is not suitable for classical ml."
                )
            continue

        searcher = searcher()
        scheduler = searcher.get_scheduler_instance(
            problem.early_stopping_iters)
        if problem.early_stopping_key is None and scheduler is not None:
            if searcher_name is not None:
                raise ValueError(
                    f"{searcher_name} cannot be used without a probelm that supports early stopping."
                )
            continue

        if scheduler is None:
            config = problem.config
        else:
            config = problem.config_early_stopping
        search_alg = searcher.get_searcher_instance(config,
                                                    problem.init_config,
                                                    time_budget_s, random_seed,
                                                    max_concurrent)
        name = (f"{problem.__class__.__name__}-{searcher.__class__.__name__}"
                f"-{data_url.split('/')[-1].split('.')[0]}"
                f"-{cv_folds}-{random_seed}-{time_budget_s}-{max_concurrent}")
        results_path = "~/results"
        run_on_every_ray_node(set_up_s3fs, path=results_path)
        results_path_expanded = os.path.expanduser(results_path)
        save_path = os.path.join(results_path_expanded,
                                 f"{name}_analysis.pickle")
        if not force_redo and os.path.exists(save_path):
            print(f"Skipping tune run {name}")
            continue
        print(f"Starting tune run {name}")
        analysis = tune.run(problem.trainable_with_parameters(
            X=X,
            y=y,
            num_classes=len(np.unique(y)),
            cv_folds=cv_folds,
            random_seed=random_seed,
            results_path=results_path),
                            metric=problem.metric_name,
                            mode=problem.metric_mode,
                            config=config,
                            name=name,
                            search_alg=search_alg,
                            scheduler=scheduler,
                            num_samples=-1,
                            time_budget_s=time_budget_s,
                            resources_per_trial=problem.resource_requirements,
                            max_concurrent_trials=max_concurrent,
                            verbose=1,
                            keep_checkpoints_num=1,
                            max_failures=2,
                            raise_on_failed_trial=False,
                            local_dir=results_path_expanded,
                            sync_config=SyncConfig(sync_to_cloud=False,
                                                   sync_on_checkpoint=False,
                                                   sync_to_driver=False))
        print(analysis.results_df)
        with open(save_path, "wb") as f:
            pickle.dump(analysis, f)
        shutil.rmtree(results_path_expanded)
