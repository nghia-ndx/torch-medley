import torch.nn as nn


class BaseNeuralNet(nn.Module):
    def __init__(self, bit_size: int):
        super().__init__()
        self.bit_size = bit_size
