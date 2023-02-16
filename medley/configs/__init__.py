import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .base import BaseConfig
from .dataset import DatasetConfig
from .embedding import EmbeddingConfig
from .hashing import HashingConfig
from .param import ParamConfig
from .training import NetConfig, OptimConfig, TrainingConfig
