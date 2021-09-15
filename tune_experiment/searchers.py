from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union
import pandas as pd
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
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return

    @property
    def scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return None

    @property
    def name(self) -> str:
        return f"{self.searcher_instance.__class__.__name__}-{self.scheduler_instance.__class__.__name__}"

    supports_categorical: bool = True
    uses_partial_results: bool = False
    designed_for_parallel: bool = True
    suitable_for_classical_ml: bool = True
    suitable_for_dl: bool = True
    suitable_for_rf: bool = True
    suitable_for_test_functions: bool = True


class RandomSearch(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return BasicVariantGenerator()


class AxSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return AxSearch()


class BayesOptSearcher(Searcher):
    @property
    def searcher_instance(
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
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return TuneBOHB(max_concurrent=max_concurrent, seed=random_state)

    @property
    def scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return HyperBandForBOHB(max_t=max_t)

    uses_partial_results: bool = True


class DragonflyBanditSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return DragonflySearch(optimizer="bandit", domain="euclidean")


class DragonflyGeneticSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return DragonflySearch(optimizer="genetic", domain="cartesian")


class BlendSearchSearcher(Searcher):
    @property
    def searcher_instance(
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
    @property
    def searcher_instance(
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
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return HEBOSearch(max_concurrent=max_concurrent,
                          random_state_seed=random_state)


class HyperOptSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return HyperOptSearch(random_state_seed=random_state)

    designed_for_parallel: bool = False


class NevergradSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return NevergradSearch(optimizer=NGOpt)


class OptunaTPESearcher(Searcher):
    @property
    def searcher_instance(
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
    @property
    def searcher_instance(
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
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return SkOptSearch()


class ZOOptSearcher(Searcher):
    @property
    def searcher_instance(
            self,
            config: Dict[str, Sampler],
            init_config: Dict[str, Any],
            time_budget_s: float,
            random_state: Optional[int] = None,
            max_concurrent: int = 10) -> Union[SearchAlgorithm, Searcher]:
        return ZOOptSearch(parallel_num=max_concurrent)


class PBTSearcher(RandomSearch):
    @property
    def scheduler_instance(self, max_t: int) -> Optional[TrialScheduler]:
        return PopulationBasedTraining()

    uses_partial_results: bool = True
    suitable_for_classical_ml: bool = False
    suitable_for_dl: bool = True
    suitable_for_rf: bool = True
    suitable_for_test_functions: bool = False