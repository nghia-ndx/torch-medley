from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from medley.utils.torch import binarize_code

from .configs import DatasetConfig, device
from .datasets import BaseDataset

_Loaders = namedtuple('Loaders', ['train', 'test', 'db'])


def create_loaders(config: DatasetConfig, num_workers=4):
    train_dataset, test_dataset, db_dataset = [
        config.klass(config.root_path, split=split, transform=transform)
        for split, transform in [
            ('train', config.train_transform),
            ('test', config.eval_transform),
            ('db', config.eval_transform),
        ]
    ]

    train_loader, test_loader, db_loader = [
        DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        for dataset, shuffle in [
            (train_dataset, True),
            (test_dataset, False),
            (db_dataset, False),
        ]
    ]

    return _Loaders(train=train_loader, test=test_loader, db=db_loader)


def feed_forward(net: nn.Module, loader: DataLoader[BaseDataset], binarize=True):
    results, labels = [], []
    for _, image, label in tqdm(loader, desc='Calculating'):
        results.append(net(image.to(device)).cpu().detach())
        labels.append(label)

    results, labels = torch.cat(results), torch.cat(labels)

    if binarize:
        return binarize_code(results), binarize_code(labels)
    return results, labels
