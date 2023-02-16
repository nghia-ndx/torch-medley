import os

from timm import create_model
from timm.data import create_transform, resolve_data_config

from medley.configs import EmbeddingConfig, device
from medley.helpers import create_loaders, feed_forward
from medley.utils.logging import logger
from medley.utils.torch import save_torch_objects


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

        self.loaders = create_loaders(self.config.dataset)

        self.model = create_model(
            config.model_name,
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )

        self.data_config = resolve_data_config(
            self.model.pretrained_cfg, model=self.model
        )

        self.transform = create_transform(**self.data_config, is_training=False)

        self.model.eval()
        self.model.to(device)

    def run(self):
        for split, loader in self.loaders._asdict().items():
            # Overide dataset transform with model transform
            loader.dataset.transform = self.transform  # type: ignore
            logger.info(f'Extracting in {split}')
            embeddings, labels = feed_forward(self.model, loader)  # type: ignore

            save_dir = os.path.join(self.config.save_dir, split)
            os.makedirs(save_dir, exist_ok=True)

            save_torch_objects(
                [
                    {'name': 'embeddings', 'val': embeddings},
                    {'name': 'labels', 'val': labels},
                ],
                save_dir=save_dir,
            )
