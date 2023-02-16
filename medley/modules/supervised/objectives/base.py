import torch.nn as nn
from torch.utils.data import DataLoader

from medley.configs import ParamConfig
from medley.datasets import BaseDataset
from medley.modules.supervised.nets import BaseNeuralNet


class BaseObjective(nn.Module):
    def __init__(
        self,
        bit_size: int,
        params: ParamConfig,
        net: BaseNeuralNet,
        loader: DataLoader[BaseDataset],
    ):
        super().__init__()
        self.bit_size = bit_size
        self.params = params
        self.net = net
        self.loader = loader
