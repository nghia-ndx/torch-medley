import os
from functools import lru_cache

import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

from medley.utils.logging import logger

from .base import BaseDataset


class Cifar(BaseDataset):
    train_size = 5000
    test_size = 1000

    @property
    def n_class(self) -> int:
        return 10

    @property
    def cifar10_data_list(self):
        return CIFAR10.train_list + CIFAR10.test_list

    @property
    def cifar10_base_folder(self):
        return self.get_full_path(CIFAR10.base_folder)

    def load_dataset(self):
        self.imgs = []
        self.labels = []

        for img, label in self.get_split_iterator(self.split):
            self.imgs.append(img)
            self.labels.append(label)

    def download_dataset(self):
        logger.info('Using torch\'s CIFAR10 dataset')
        CIFAR10(self.root, download=True)

    def is_dataset_existed(self):
        for batch_file, _ in self.cifar10_data_list:
            batch_file_path = os.path.join(self.cifar10_base_folder, batch_file)
            if not os.path.exists(batch_file_path):
                return False
        return True

    @lru_cache(maxsize=3)
    def _get_split_data_and_labels(self, split):
        all_data = []
        all_labels = []

        cifar_train = CIFAR10(self.root, train=True)
        cifar_test = CIFAR10(self.root, train=False)

        all_data = np.concatenate([cifar_train.data, cifar_test.data], axis=0)
        all_labels = np.concatenate([cifar_train.targets, cifar_test.targets], axis=0)
        all_labels = np.array(
            [np.eye(10, dtype=np.int8)[label] for label in all_labels]
        )

        if split == 'train':
            all_data = all_data[: self.train_size]
            all_labels = all_labels[: self.train_size]
        elif split == 'test':
            all_data = all_data[self.train_size : self.train_size + self.test_size]
            all_labels = all_labels[self.train_size : self.train_size + self.test_size]
        else:
            all_data = all_data[: self.train_size + self.test_size]
            all_labels = all_labels[: self.train_size + self.test_size]

        return all_data, all_labels

    def get_split_iterator(self, split: str):
        for data, label in zip(*self._get_split_data_and_labels(split)):
            yield data, label

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index])
        img = self.transform(img) if self.transform else img
        label = self.labels[index]
        label = self.target_transform(label) if self.target_transform else label
        return index, img, label

    def __len__(self):
        return len(self.imgs)
