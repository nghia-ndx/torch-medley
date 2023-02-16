import torch
from torch.utils.data import DataLoader

from medley.configs import ParamConfig, device
from medley.datasets import BaseDataset

from ..nets import BaseNeuralNet
from .base import BaseObjective


class DeepSupervisedHashingLoss(BaseObjective):
    def __init__(
        self,
        bit_size: int,
        params: ParamConfig,
        net: BaseNeuralNet,
        loader: DataLoader[BaseDataset],
    ):
        super().__init__(bit_size, params, net, loader)

        self.alpha = self.params.alpha

        self.m = 2 * self.bit_size
        train_size = len(self.loader.dataset)  # type: ignore
        self.U = torch.zeros(train_size, self.bit_size).float().to(device)
        self.Y = (
            torch.zeros(train_size, self.loader.dataset.n_class)  # type: ignore
            .float()
            .to(device)
        )

    def forward(self, u, y, ind):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = self.alpha * (1 - u.abs()).abs().mean()

        return loss1 + loss2
