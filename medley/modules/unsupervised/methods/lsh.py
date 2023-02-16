import torch

from medley.configs import device
from medley.utils.torch import binarize_code

from .base import BaseHashMethod


class LocalitySensitiveHashing(BaseHashMethod):
    def _encode(self, data):
        return binarize_code(data.to(device) @ self.W)

    def _train(self, _, test_data, db_data):
        self.W = torch.randn(db_data.shape[1], self.bit_size).to(device)

        test_code = self._encode(test_data)
        db_code = self._encode(db_data)
        return test_code, db_code
