# example usage:
# python bin/benchmark_classical_ml.py all LightGBMProblem --max-concurrent 8 --server-address auto --time-budget-s 60 --biggest-first

import argparse
import os
import ray
from itertools import product

from tune_experiment.execution import benchmark_deep_learning
from tune_experiment.problems.deep_learning.cifar import CIFARProblem

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--n-jobs",
                        required=False,
                        type=int,
                        default=1,
                        help="Number of CPU threads per trial")
    parser.add_argument("--server-address",
                        type=str,
                        default=None,
                        required=False,
                        help="The address of server to connect to if using "
                        "Ray Client.")
    parser.add_argument("--force-redo",
                        action="store_true",
                        help="Start with biggest datasets.")
    args, _ = parser.parse_known_args()
    problems = {
        v.__name__: v
        for v in [
            CIFARProblem
        ]
    }
    assert args.problem == "all" or args.problem in problems, f"problem must be 'all' or one of {', '.join(problems.keys())}"
    #assert os.path.isdir(os.path.expanduser("~/results")), "'~/results' folder must exist"

    ray.init(address=args.server_address)

    kwargs = dict(
        time_budget_s=args.time_budget_s,
        max_concurrent=args.max_concurrent,
        random_seed=args.random_seed,
        searcher_name=None if args.searcher == "all" else args.searcher,
        force_redo=args.force_redo)

    combinations = problems.keys() if args.problem == "all" else [args.problem]

    for problem in combinations:
        benchmark_deep_learning(problems[problem](args.n_jobs), **kwargs)
