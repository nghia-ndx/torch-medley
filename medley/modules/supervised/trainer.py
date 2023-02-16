import os

from tqdm.auto import tqdm

from medley.configs import TrainingConfig, device
from medley.helpers import create_loaders, feed_forward
from medley.metrics import mean_average_precision, pr_curve
from medley.utils.logging import logger
from medley.utils.torch import save_torch_objects


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self.loaders = create_loaders(self.config.dataset)

        net_class = self.config.net.klass
        optim_class = self.config.optim.klass

        self.net = net_class(self.config.bit_size).to(device)
        self.optim = optim_class(self.net.parameters(), **self.config.optim.params)
        self.criterion = self.config.net.criterion_klass(
            self.config.bit_size, self.config.net.params, self.net, self.loaders.train
        )

        self.train_loss_map = {}
        self.mAP_map = {}
        self.current_epoch = 0

    def _train_epoch(self, index, image, label):
        image = image.to(device)
        label = label.to(device)

        self.optim.zero_grad()
        u = self.net(image)

        loss = self.criterion(u, label.float(), index)
        loss_value = loss.item()

        loss.backward()
        self.optim.step()

        return loss_value

    def run(self):
        logger.info(
            'Training: {} -- Bit: {} -- Dataset: {}'.format(
                self.config.net.criterion_name,
                self.config.bit_size,
                self.config.dataset.name,
            )
        )
        best_mAP = 0

        for epoch in range(1, self.config.epoches + 1):
            self.current_epoch = epoch
            self.net.train()

            loader = self.loaders.train
            n_batches = len(loader)

            epoch_str = f'[{epoch:#3d}/{self.config.epoches:#3d}]'
            epoch_iter = tqdm(loader, desc=epoch_str)

            epoch_total_loss = 0.0
            epoch_loss = 0.0

            for i, (index, image, label) in enumerate(epoch_iter, start=1):
                epoch_total_loss += self._train_epoch(index, image, label)

                # Check for last iteration to calculate loss and update the pbar prefix
                if i == n_batches:
                    epoch_loss = epoch_total_loss / n_batches
                    epoch_iter.set_description_str(f'{epoch_str} Loss: {epoch_loss}')

            self.train_loss_map[epoch] = epoch_loss

            if epoch % self.config.validate_after_epoches == 0:
                logger.info(f'Validating and save epoch {epoch}')
                mAP = self.validate_and_save()
                self.mAP_map[epoch] = mAP
                if mAP > best_mAP:
                    best_mAP = mAP

        self.finalize_and_save()
        logger.info(f'Training completed! Best mAP: {best_mAP}')

    def validate_and_save(self):
        logger.info('Binarizing test set')
        test_hash_codes, test_labels = feed_forward(self.net, self.loaders.test)
        logger.info('Binarizing db set')
        db_hash_codes, db_labels = feed_forward(self.net, self.loaders.db)

        mAP = mean_average_precision(
            test_hash_codes,
            db_hash_codes,
            test_labels,
            db_labels,
            top_k=self.config.dataset.top_k,
        )

        P, R = pr_curve(
            test_hash_codes,
            db_hash_codes,
            test_labels,
            db_labels,
        )

        logger.info(
            f'Mean Average Precision for epoch {self.current_epoch}: {mAP:#.3f}'
        )
        save_dir = os.path.join(
            self.config.save_dir, f'{self.current_epoch}_{mAP:#.3f}'
        )
        os.makedirs(save_dir, exist_ok=True)

        save_torch_objects(
            [
                {'name': 'P', 'val': P},
                {'name': 'R', 'val': R},
                {'name': 'test_hash_codes', 'val': test_hash_codes},
                {'name': 'test_labels', 'val': test_labels},
                {'name': 'db_hash_codes', 'val': db_hash_codes},
                {'name': 'db_labels', 'val': db_labels},
                {'name': 'model_state_dict', 'val': self.net.state_dict()},
            ],
            save_dir=save_dir,
        )
        logger.info(f'Saved training progress in {save_dir}')

        return mAP

    def finalize_and_save(self):
        save_torch_objects(
            [
                {'name': 'training_loss_dict', 'val': self.train_loss_map},
                {'name': 'mAP_dict', 'val': self.mAP_map},
            ],
            save_dir=self.config.save_dir,
        )
