from tune_experiment.execution import benchmark_classical_ml
from tune_experiment.problems.classical_ml.gbdt import XGBoostProblem, LightGBMProblem
from tune_experiment.problems.classical_ml.sklearn import LRProblem, MLPProblem, SVMProblem, KNNProblem
from tune_experiment.datasets.classical_ml import get_classical_ml_datasets

datasets = get_classical_ml_datasets()

benchmark_classical_ml(datasets[0], LightGBMProblem(1), time_budget_s=60, max_concurrent=8, random_seed=1, cv_folds=5)