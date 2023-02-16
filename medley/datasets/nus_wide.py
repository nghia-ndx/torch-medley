from torch_deephash_dataset.nus_wide import NUSWIDEDataset

from .base import BaseDataset


class NusWide(BaseDataset, NUSWIDEDataset):
    @property
    def n_class(self) -> int:
        return 81
