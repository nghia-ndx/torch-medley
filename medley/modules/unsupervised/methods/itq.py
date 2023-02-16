from typing import Tuple

import torch
from sklearn.decomposition import PCA
from torch import Tensor

from medley.configs import device
from medley.utils.torch import binarize_code, numpify

from .base import BaseHashMethod


class IterativeQuantization(BaseHashMethod):
    def __init__(self, bit_size: int, max_iter: int = 3):
        super().__init__(bit_size)
        self.max_iter = max_iter

    def _encode(self, data: Tensor) -> Tensor:
        code = torch.from_numpy(self.pca.transform(numpify(data))).to(device) @ self.R
        return binarize_code(code)

    def _train(
        self, train_data: Tensor, test_data: Tensor, db_data: Tensor
    ) -> Tuple[Tensor, Tensor]:
        R = torch.randn(self.bit_size, self.bit_size).to(device)
        [U, _, _] = torch.svd(R)
        R = U[:, : self.bit_size]

        # PCA
        pca = PCA(n_components=self.bit_size)
        V = torch.from_numpy(pca.fit_transform(numpify(train_data))).to(device)

        # Training
        for _ in range(self.max_iter):
            V_tilde = V @ R
            B = V_tilde.sign()
            [U, _, VT] = torch.svd(B.t() @ V)
            R = VT.t() @ U.t()

        self.pca, self.R = pca, R
        test_code = self._encode(test_data)
        db_code = self._encode(db_data)

        return test_code, db_code
