import argparse
import os
import ray
from itertools import product

from tune_experiment.execution import benchmark_classical_ml
from tune_experiment.problems.classical_ml.gbdt import XGBoostProblem, LightGBMProblem
from tune_experiment.problems.classical_ml.sklearn import LRProblem, MLPProblem, SVMProblem, KNNProblem
from tune_experiment.datasets.classical_ml import get_classical_ml_datasets

if __name__ == "__main__":
    datasets = get_classical_ml_datasets()
    datasets = {v.split("/")[-1].split(".")[0]: v for v in datasets}
    problems = {
        v.__name__: v
        for v in [
            XGBoostProblem, LightGBMProblem, LRProblem, MLPProblem, SVMProblem,
            KNNProblem
        ]
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument("problem", type=str, help="Problem name")

    parser.add_argument("--searcher",
                        required=False,
                        type=str,
                        default="all",
                        help="Searcher name")
    parser.add_argument("--time-budget-s",
                        required=False,
                        type=float,
                        default=60,
                        help="Time budget in seconds")
    parser.add_argument("--max-concurrent",
                        required=False,
                        type=int,
                        default=8,
                        help="Max concurrent trials")
    parser.add_argument("--random-seed",
                        required=False,
                        type=int,
                        default=1,
                        help="Random seed")
    parser.add_argument("--cv",
                        required=False,
                        type=int,
                        default=5,
                        help="Number of cv folds")
    parser.add_argument("--server-address",
                        type=str,
                        default=None,
                        required=False,
                        help="The address of server to connect to if using "
                        "Ray Client.")
    args, _ = parser.parse_known_args()

    assert args.dataset == "all" or args.dataset in datasets, f"dataset must be 'all' or one of {', '.join(datasets.keys())}"
    assert args.problem == "all" or args.problem in problems, f"problem must be 'all' or one of {', '.join(problems.keys())}"
    assert os.path.isdir("results"), "'results' folder must exist in localdir"

    ray.init(address=args.server_address)

    kwargs = dict(
        time_budget_s=args.time_budget_s,
        max_concurrent=args.max_concurrent,
        random_seed=args.random_seed,
        cv_folds=args.cv,
        searcher_name=None if args.searcher == "all" else args.searcher)

    combinations = product(
        datasets.values() if args.dataset == "all" else [args.dataset],
        problems.values() if args.problem == "all" else [args.problem])

    for dataset, problem in combinations:
        benchmark_classical_ml(dataset, problem(1), **kwargs)
