from torch_deephash_dataset.coco import COCODataset

from .base import BaseDataset


class Coco(BaseDataset, COCODataset):
    @property
    def n_class(self) -> int:
        return 91
