import ray
import pandas as pd
from ray import tune
from tune_experiment.problems.classical_ml.gbdt import XGBoostProblem, LightGBMProblem
from tune_experiment.problems.classical_ml.sklearn import LRProblem, MLPProblem, SVMProblem, KNNProblem
from tune_experiment.searchers import searcher_registry
import sklearn.datasets

def benchmark_classical_ml():
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    problem = LightGBMProblem(1)
    time_budget_s = 60
    max_concurrent = 8
    random_seed = 1
    cv_folds = 5
    data = pd.read_parquet("https://tune-experiment-datasets.s3.us-west-2.amazonaws.com/classical_ml/Australian.parquet")
    X = problem.preprocessor.fit_transform(data.drop("target", axis=1))
    y = data["target"]
    for name, searcher in searcher_registry.items():
        if not (searcher.supports_categorical and searcher.suitable_for_classical_ml):
            continue
        searcher = searcher()
        scheduler = searcher.get_scheduler_instance(problem.early_stopping_iters)
        if problem.early_stopping_key is None and scheduler is not None:
            continue
        if scheduler is None:
            config = problem.config
        else:
            config = problem.config_early_stopping
        search_alg = searcher.get_searcher_instance(config, problem.init_config, time_budget_s, random_seed, max_concurrent)
        name = f"{problem.__class__.__name__}-{search_alg.__class__.__name__}-{scheduler.__class__.__name__}-{cv_folds}-{random_seed}"
        print(name)
        tune.run(
            problem.trainable_with_parameters(
                X=X, y=y, cv_folds=cv_folds, random_seed=random_seed
            ),
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
            reuse_actors=True
        )