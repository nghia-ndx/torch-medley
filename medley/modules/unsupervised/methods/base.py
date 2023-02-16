from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor


class BaseHashMethod(ABC):
    def __init__(self, bit_size: int):
        self.bit_size = bit_size
        self._called = False

    def encode(self, data: Tensor) -> Tensor:
        """
        Can only be invoked after `__call__` has been invoked
        """
        assert self._called
        return self._encode(data)

    def train(
        self, train_data: Tensor, test_data: Tensor, db_data: Tensor
    ) -> Tuple[Tensor, Tensor]:
        self._called = True
        return self._train(train_data, test_data, db_data)

    @abstractmethod
    def _train(
        self, train_data: Tensor, test_data: Tensor, db_data: Tensor
    ) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def _encode(self, data) -> Tensor:
        pass
