from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from medley.configs import device
from medley.utils.torch import binarize_code, numpify

from .base import BaseHashMethod


class SpectralHashing(BaseHashMethod):
    def _encode(self, data):
        data = self.pca.transform(numpify(data)) - self.mn.reshape(1, -1)
        omega0 = np.pi / self.R
        omegas = self.modes * omega0.reshape(1, -1)
        U = np.zeros((len(data), self.bit_size))
        for i in range(self.bit_size):
            omegai = omegas[i, :]
            ys = np.sin(data * omegai + np.pi / 2)
            yi = np.prod(ys, 1)
            U[:, i] = yi

        return binarize_code(torch.from_numpy(U))

    def _train(
        self, train_data: Tensor, test_data: Tensor, db_data: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # PCA
        pca = PCA(n_components=self.bit_size)
        X = pca.fit_transform(numpify(train_data))

        # Fit uniform distribution
        eps = np.finfo(float).eps
        mn = X.min(0) - eps
        mx = X.max(0) + eps

        # Enumerate eigenfunctions
        R = mx - mn
        max_mode = np.ceil((self.bit_size + 1) * R / R.max()).astype(np.intc)
        n_modes = max_mode.sum() - len(max_mode) + 1
        modes = np.ones([n_modes, self.bit_size])
        m = 0
        for i in range(self.bit_size):
            modes[m + 1 : m + max_mode[i], i] = np.arange(1, max_mode[i]) + 1
            m = m + max_mode[i] - 1

        modes -= 1
        omega0 = np.pi / R
        omegas = modes * omega0.reshape(1, -1).repeat(n_modes, 0)
        eig_val = -(omegas**2).sum(1)
        ii = (-eig_val).argsort()
        modes = modes[ii[1 : self.bit_size + 1], :]

        self.pca, self.R, self.mn, self.modes = pca, R, mn, modes
        test_code = self._encode(test_data).to(device)
        db_code = self._encode(db_data).to(device)

        return test_code, db_code
