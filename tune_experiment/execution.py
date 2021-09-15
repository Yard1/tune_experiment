import ray
import pandas as pd
from ray import tune
from tune_experiment.problems.classical_ml.gbdt import XGBoostProblem, LightGBMProblem
import sklearn.datasets

def benchmark_classical_ml():
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    problem = LightGBMProblem(1)
    searcher = tune.create_searcher("optuna")
    scheduler = tune.create_scheduler("async_hyperband", max_t=problem.early_stopping_iters)
    random_seed = 1
    cv_folds = 5
    name = f"{problem.__class__.__name__}-{searcher.__class__.__name__}-{scheduler.__class__.__name__}-{cv_folds}-{random_seed}"
    data = sklearn.datasets.load_breast_cancer(as_frame=True)
    data = pd.concat((data["data"], data["target"]), axis=1)
    tune.run(
        problem.trainable_with_parameters(
            dataset=data, target_column="target", cv_folds=cv_folds, random_seed=random_seed
        ),
        metric=problem.metric_name,
        mode=problem.metric_mode,
        config=problem.config_early_stopping,
        name=name,
        search_alg=searcher,
        scheduler=scheduler,
        num_samples=10,
        resources_per_trial=problem.resource_requirements
    )