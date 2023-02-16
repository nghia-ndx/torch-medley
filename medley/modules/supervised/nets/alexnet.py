import torch.nn as nn
from torchvision.models.alexnet import AlexNet_Weights, alexnet

from .base import BaseNeuralNet


class AlexNet(BaseNeuralNet):
    def __init__(self, bit_size: int):
        super().__init__(bit_size)

        model_alexnet = alexnet(weights=AlexNet_Weights)

        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight  # type: ignore
        cl1.bias = model_alexnet.classifier[1].bias  # type: ignore

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight  # type: ignore
        cl2.bias = model_alexnet.classifier[4].bias  # type: ignore

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.bit_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x
