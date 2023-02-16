import torch
from torch.utils.data import DataLoader

from medley.configs import ParamConfig, device
from medley.datasets import BaseDataset

from ..nets import BaseNeuralNet
from .base import BaseObjective


class DeepCauchyHashingLoss(BaseObjective):
    def __init__(
        self,
        bit_size: int,
        params: ParamConfig,
        net: BaseNeuralNet,
        loader: DataLoader[BaseDataset],
    ):
        super().__init__(bit_size, params, net, loader)

        self.gamma = self.params.gamma
        self.lambda_ = self.params.lambda_
        self.K = self.bit_size
        self.one = torch.ones(self.loader.batch_size or 1, self.bit_size).to(device)

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = (
            hi.pow(2).sum(dim=1, keepdim=True).pow(0.5)
            @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        )
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (
            s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj)
        )
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.lambda_ * quantization_loss.mean()

        return loss
