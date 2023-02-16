from typing import Type

from medley.modules.unsupervised.methods import BaseHashMethod
from medley.utils.misc import get_class_name

from .base import BaseConfig
from .embedding import EmbeddingConfig
from .path import PathConfig


class HashingConfig(BaseConfig):
    bit_size: int = 32
    embedding: EmbeddingConfig
    method_klass: Type[BaseHashMethod]

    @property
    def dataset(self):
        return self.embedding.dataset

    @property
    def name(self):
        return '{}.{}.{}.{}'.format(
            self.bit_size,
            self.embedding.model_name,
            get_class_name(self.method_klass).lower(),
            self.dataset.name,
        )

    @property
    def save_dir(self):
        return PathConfig().save_dir.format(self.name)
