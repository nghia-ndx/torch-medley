import os
from functools import lru_cache

from medley.configs import HashingConfig
from medley.metrics import mean_average_precision, pr_curve
from medley.utils.logging import logger
from medley.utils.torch import load_torch_objects, save_torch_objects

from .embedder import Embedder


class Hasher:
    def __init__(self, config: HashingConfig):
        self.config = config

        try:
            self.load_embeddings_and_labels()
        except Exception:
            Embedder(self.config.embedding).run()
        finally:
            (
                self.train_embeddings,
                _,
                self.test_embeddings,
                self.test_labels,
                self.db_embeddings,
                self.db_labels,
            ) = self.load_embeddings_and_labels()

    @lru_cache(maxsize=1)
    def load_embeddings_and_labels(self):
        save_dirs = [
            os.path.join(self.config.embedding.save_dir, split)
            for split in ['train', 'test', 'db']
        ]
        torch_object_paths = [
            os.path.join(save_dir, file_name)
            for save_dir in save_dirs
            for file_name in ['embeddings.pt', 'labels.pt']
        ]
        return load_torch_objects(torch_object_paths)

    def run(self):
        test_hash_codes, db_hash_codes = self.config.method_klass(
            self.config.bit_size
        ).train(self.train_embeddings, self.test_embeddings, self.db_embeddings)

        mAP = mean_average_precision(
            test_hash_codes,
            db_hash_codes,
            self.test_labels,
            self.db_labels,
            top_k=self.config.dataset.top_k,
        )

        P, R = pr_curve(
            test_hash_codes,
            db_hash_codes,
            self.test_labels,
            self.db_labels,
        )

        logger.info(f'Mean Average Precision: {mAP:#.3f}')
        save_dir = os.path.join(self.config.save_dir, f'{mAP:#.3f}')
        os.makedirs(save_dir, exist_ok=True)
        save_torch_objects(
            [
                {'name': 'P', 'val': P},
                {'name': 'R', 'val': R},
                {'name': 'test_hash_codes', 'val': test_hash_codes},
                {'name': 'test_labels', 'val': self.test_labels},
                {'name': 'db_hash_codes', 'val': db_hash_codes},
                {'name': 'db_labels', 'val': self.db_labels},
            ],
            save_dir=save_dir,
        )
        logger.info(f'Saved in {save_dir}')
