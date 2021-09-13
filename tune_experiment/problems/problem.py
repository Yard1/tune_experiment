from abc import ABC, abstractproperty
from typing import Callable, Dict
from ray.tune.sample import Sampler
from ray.tune.utils.placement_groups import PlacementGroupFactory

class Problem(ABC):
    def __init__(self):
        return

    @abstractproperty
    def config(self) -> Dict[str, Sampler]:
        return

    @abstractproperty
    def init_config(self) -> Dict[str, Sampler]:
        return

    @abstractproperty
    def early_stopping_key(self) -> str:
        return

    @abstractproperty
    def early_stopping_iters(self) -> int:
        return

    @abstractproperty
    def trainable(self) -> Callable:
        return

    @abstractproperty
    def resource_requirements(self) -> PlacementGroupFactory:
        return