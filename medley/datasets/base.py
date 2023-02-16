from abc import ABC, abstractmethod

from torch_deephash_dataset.base import BaseDeepHashDataset


class BaseDataset(BaseDeepHashDataset, ABC):
    @property
    @abstractmethod
    def n_class(self) -> int:
        pass

    def __getitem__(self, index):
        return index, *super().__getitem__(index)
