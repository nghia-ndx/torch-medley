from medley.configs import DatasetConfig

from .base import BaseConfig
from .path import PathConfig


class EmbeddingConfig(BaseConfig):
    model_name: str
    dataset: DatasetConfig

    @property
    def name(self):
        return f'{self.model_name}.{self.dataset.name}'.lower()

    @property
    def save_dir(self):
        return PathConfig().embedding_dir.format(self.name)
