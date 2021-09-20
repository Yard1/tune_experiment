from typing import Optional
import pickle
import os
import pandas as pd
from ray import tune
from tune_experiment.problems.problem import Problem
from tune_experiment.searchers import searcher_registry


def benchmark_classical_ml(data_url: str,
                           problem: Problem,
                           time_budget_s: float,
                           max_concurrent: int = 1,
                           random_seed: int = 1,
                           cv_folds: int = 5,
                           searcher_name: Optional[str] = None):
    print(f"Downloading dataset {data_url}")
    data = pd.read_parquet(data_url)
    print("Dataset downloaded, preprocessing...")
    X = problem.preprocessor.fit_transform(data.drop("target", axis=1))
    y = data["target"]
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
        name = (
            f"{problem.__class__.__name__}-{search_alg.__class__.__name__}"
            f"-{scheduler.__class__.__name__ if scheduler else 'FIFOScheduler'}"
            f"-{cv_folds}-{random_seed}-{time_budget_s}")
        print(f"Starting tune run {name}")
        analysis = tune.run(problem.trainable_with_parameters(
            X=X, y=y, cv_folds=cv_folds, random_seed=random_seed),
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
                            reuse_actors=True,
                            verbose=1,
                            keep_checkpoints_num=1,
                            max_failures=2,
                            local_dir="results")
        with open(os.path.join("results", f"{name}_analysis.pickle"),
                  "wb") as f:
            pickle.dump(analysis, f)
