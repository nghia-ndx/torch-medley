import torch
from torch.utils.data import DataLoader

from medley.configs import ParamConfig
from medley.datasets import BaseDataset

from ..nets import BaseNeuralNet
from .base import BaseObjective


class HashNetLoss(BaseObjective):
    def __init__(
        self,
        bit_size: int,
        params: ParamConfig,
        net: BaseNeuralNet,
        loader: DataLoader[BaseDataset],
    ):
        super().__init__(bit_size, params, net, loader)
        self.scale = 1

    def forward(self, u, y, ind):
        u = torch.tanh(self.scale * u)
        S = (y @ y.t() > 0).float()
        sigmoid_alpha = self.params.alpha
        dot_product = sigmoid_alpha * u @ u.t()
        mask_positive = S > 0
        mask_negative = (1 - S).bool()

        neg_log_probe = (
            dot_product + torch.log(1 + torch.exp(-dot_product)) - S * dot_product
        )
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1

        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        loss = torch.sum(neg_log_probe) / S
        return loss
