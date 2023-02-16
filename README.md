# Torch Medley

**[Work in progress]**

A collection of supervised DeepHashing methods and unsupervised similarity preserving hashing methods. 

# Installation
Install via `pip`
```
pip install git+https://github.com/nghia-ndx/torch-medley
```

# Usages
### Dataset configs
```python
from medley.configs import DatasetConfig
from medley.datasets import Cifar, Coco, NusWide

import torchvision.transforms as tf
from typing import Callable

# Datasets
class CifarConfig(DatasetConfig):
    crop_size: int = 244

    train_transform: Callable = tf.Compose(
        [
            tf.Resize(crop_size),
            tf.ToTensor(),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

    eval_transform: Callable = train_transform

class NusWideConfig(DatasetConfig):
    batch_size: int = 128

    crop_size: int = 244
    resize_size: int = 256

    top_k: int = 5000
    train_transform: Callable = tf.Compose(
        [
            tf.Resize(resize_size),
            tf.RandomHorizontalFlip(),
            tf.RandomCrop(crop_size),
            tf.ToTensor(),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    eval_transform: Callable = tf.Compose(
        [
            tf.Resize(resize_size),
            tf.CenterCrop(crop_size),
            tf.ToTensor(),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

class CocoConfig(DatasetConfig):
    batch_size: int = 128

    crop_size: int = 244
    resize_size: int = 256

    top_k: int = 5000
    train_transform = tf.Compose(
        [
            tf.Resize(resize_size),
            tf.RandomHorizontalFlip(),
            tf.RandomCrop(crop_size),
            tf.ToTensor(),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    eval_transform = tf.Compose(
        [
            tf.Resize(resize_size),
            tf.CenterCrop(crop_size),
            tf.ToTensor(),
            tf.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

cifar = CifarConfig(klass=Cifar)
nuswide = NusWideConfig(klass=NusWide)
coco = CocoConfig(klass=Coco)
```

### Supervised training
```python
from medley.configs import TrainingConfig, NetConfig, ParamConfig
from medley.modules.supervised.objectives import DeepSupervisedHashingLoss, DeepCauchyHashingLoss, HashNetLoss
from medley.modules.supervised.trainer import Trainer

dsh = NetConfig(
    criterion_klass=DeepSupervisedHashingLoss
)
dch = NetConfig(
    criterion_klass=DeepCauchyHashingLoss,
    params=ParamConfig(alpha=0.2, gamma = 20.0, lambda_=0.1)
)
hashnet = NetConfig(
    criterion_klass=HashNetLoss,
)

Trainer(
    TrainingConfig(
        dataset=cifar,
        net=dsh
    )
).run()
```

### Unsupervised hashing
```python
from medley.configs import EmbeddingConfig, HashingConfig
from medley.modules.unsupervised.embedder import Embedder
from medley.modules.unsupervised.hasher import Hasher
from medley.modules.unsupervised.methods import IterativeQuantization, LocalitySensitiveHashing, SpectralHashing

coatnet_cifar = EmbeddingConfig(
    'coatnext_nano_rw_224',
    dataset=cifar
)
vit_cifar = EmbeddingConfig(
    'vit_tiny_patch16_224',
    dataset=cifar
)

Hasher(
    embedding=coatnet_cifar,
    method_klass=IterativeQuantization
).run()
```
