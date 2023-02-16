from typing import Type

from torch.optim import Optimizer, RMSprop

from medley.configs import DatasetConfig
from medley.modules.supervised.nets import AlexNet, BaseNeuralNet
from medley.modules.supervised.objectives import BaseObjective
from medley.utils.misc import get_class_name

from .base import BaseConfig
from .param import ParamConfig
from .path import PathConfig


class OptimConfig(BaseConfig):
    klass: Type[Optimizer] = RMSprop
    params: dict = dict(lr=1e-5, weight_decay=10**-5)


class NetConfig(BaseConfig):
    klass: Type[BaseNeuralNet] = AlexNet
    criterion_klass: Type[BaseObjective]
    params: ParamConfig = ParamConfig(alpha=0.2)  # learning rate

    @property
    def name(self):
        return get_class_name(self.klass).lower()

    @property
    def criterion_name(self):
        return get_class_name(self.criterion_klass).lower()


class TrainingConfig(BaseConfig):
    dataset: DatasetConfig
    net: NetConfig

    optim: OptimConfig = OptimConfig()

    bit_size: int = 32
    validate_after_epoches: int = 20
    epoches: int = 200

    @property
    def name(self):
        return '{}.{}.{}.{}'.format(
            self.bit_size, self.net.name, self.net.criterion_name, self.dataset.name
        )

    @property
    def save_dir(self):
        return PathConfig().save_dir.format(self.name)
