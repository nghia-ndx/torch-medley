from typing import Callable, Optional, Type

from medley.datasets import BaseDataset
from medley.utils.misc import get_class_name

from .base import BaseConfig
from .path import PathConfig


class DatasetConfig(BaseConfig):
    klass: Type[BaseDataset]

    top_k: Optional[int] = None

    batch_size: int = 64

    train_transform: Callable = lambda *_: _
    eval_transform: Callable = lambda *_: _

    @property
    def name(self):
        return get_class_name(self.klass).lower()

    @property
    def root_path(self):
        return PathConfig().dataset_dir.format(self.name)
