from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type, Union
import pandas as pd
import inspect
from ray import tune
from ray.tune.sample import Sampler, Categorical
from ray.tune.suggest import Searcher, SearchAlgorithm
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.flaml import BlendSearch, CFO
from ray.tune.suggest.hebo import HEBOSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.nevergrad import NevergradSearch
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest.zoopt import ZOOptSearch

from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import TrialScheduler, ASHAScheduler, HyperBandForBOHB, PopulationBasedTraining

from nevergrad.optimization.optimizerlib import NGOpt
from optuna.samplers import TPESampler, CmaEsSampler


class Searcher(ABC):
    @abstractmethod
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return

    def get_scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return None

    supports_categorical: bool = True
    uses_partial_results: bool = False
    designed_for_parallel: bool = True
    suitable_for_classical_ml: bool = True
    suitable_for_dl: bool = True
    suitable_for_rf: bool = True
    suitable_for_test_functions: bool = True


class RandomSearch(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return BasicVariantGenerator()


class AxSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return AxSearch()


class BayesOptSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return BayesOptSearch(random_state=random_state or 42)

    designed_for_parallel: bool = False
    supports_categorical: bool = False


class BOHBSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return TuneBOHB(max_concurrent=max_concurrent, seed=random_state)

    def get_scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return HyperBandForBOHB(max_t=max_t)

    uses_partial_results: bool = True


class DragonflyBanditSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return DragonflySearch(optimizer="bandit", domain="euclidean")


class DragonflyGeneticSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return DragonflySearch(optimizer="genetic", domain="cartesian")


class BlendSearchSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        low_cost_config = {
            k: v
            for k, v in init_config.items()
            if k in ["n_estimators", "learning_rate", "batch_size"]
        }

        class BlendSearchPatched(BlendSearch):
            def set_search_properties(self,
                                      metric: Optional[str] = None,
                                      mode: Optional[str] = None,
                                      config: Optional[Dict] = None) -> bool:
                config = config.copy()
                config["time_budget_s"] = time_budget_s
                return super().set_search_properties(metric=metric,
                                                     mode=mode,
                                                     config=config)

            def suggest(self, trial_id: str) -> Optional[Dict]:
                ret = super().suggest(trial_id)
                ret.pop("time_budget_s")
                return ret

        return BlendSearchPatched(low_cost_partial_config=low_cost_config,
                                  seed=random_state)

    uses_partial_results: bool = True


class CFOSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        low_cost_config = {
            k: v
            for k, v in init_config.items()
            if k in ["n_estimators", "learning_rate", "batch_size"]
        }

        class CFOPatched(CFO):
            def set_search_properties(self,
                                      metric: Optional[str] = None,
                                      mode: Optional[str] = None,
                                      config: Optional[Dict] = None) -> bool:
                config = config.copy()
                config["time_budget_s"] = time_budget_s
                return super().set_search_properties(metric=metric,
                                                     mode=mode,
                                                     config=config)

            def suggest(self, trial_id: str) -> Optional[Dict]:
                ret = super().suggest(trial_id)
                ret.pop("time_budget_s")
                return ret

        return CFOPatched(low_cost_partial_config=low_cost_config,
                          seed=random_state)

    uses_partial_results: bool = True


class HEBOSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return HEBOSearch(max_concurrent=max_concurrent,
                          random_state_seed=random_state)


class HyperOptSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return HyperOptSearch(random_state_seed=random_state)

    designed_for_parallel: bool = False


class NevergradSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return NevergradSearch(optimizer=NGOpt)


class OptunaTPESearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        sampler = TPESampler(seed=random_state,
                             multivariate=True,
                             constant_liar=True)
        return OptunaSearch(sampler=sampler)

    uses_partial_results: bool = True


class OptunaCMAESSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        sampler = CmaEsSampler(seed=random_state, consider_pruned_trials=True)
        return OptunaSearch(sampler=sampler)

    uses_partial_results: bool = True
    supports_categorical: bool = False
    designed_for_parallel: bool = False


class SkOptSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return SkOptSearch()


class ZOOptSearcher(Searcher):
    def get_searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return ZOOptSearch(parallel_num=max_concurrent)


class PBTSearcher(RandomSearch):
    def get_scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return PopulationBasedTraining()

    uses_partial_results: bool = True
    suitable_for_classical_ml: bool = False
    suitable_for_dl: bool = True
    suitable_for_rf: bool = True
    suitable_for_test_functions: bool = False


# dynamically generate ASHA classes
new_globals = {}
searcher_registry: Dict[str, Type[Searcher]] = {}
__all__ = ["searcher_registry"]
temp_globals = globals().copy()
for k, v in temp_globals.items():

    def get_asha_scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return ASHAScheduler(max_t=max_t)

    if v is not Searcher and inspect.isclass(v) and issubclass(
            v,
            Searcher) and v.get_scheduler_instance == Searcher.get_scheduler_instance:
        asha_name = f"ASHA{v.__name__}"
        asha_class = type(asha_name, (v, ),
                          {"get_scheduler_instance": get_asha_scheduler_instance})
        new_globals[asha_name] = asha_class
        __all__.append(k)
        __all__.append(asha_name)
        searcher_registry[k] = v
        searcher_registry[asha_name] = asha_class

globals().update(new_globals)