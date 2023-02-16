import os
from typing import List, TypedDict, Union

import torch
from numpy.typing import ArrayLike
from torch import Tensor

from medley.configs import device

_NamedTorchObject = TypedDict('_NamedTorchObject', name=str, val=Union[Tensor, dict])


def save_torch_objects(named_torch_objects: List[_NamedTorchObject], save_dir: str):
    for torch_object in named_torch_objects:
        if isinstance(torch_object['val'], Tensor):
            torch_object['val'] = torch_object['val'].cpu().detach()

        torch.save(
            torch_object['val'], os.path.join(save_dir, f'{torch_object["name"]}.pt')
        )


def load_torch_objects(load_paths):
    return (torch.load(path) for path in load_paths)


def numpify(tensor: Tensor) -> ArrayLike:
    return tensor.cpu().detach().numpy()


def binarize_code(tensor: Tensor):
    return torch.where(
        tensor.to(device) > 0, torch.ones(1).to(device), torch.zeros(1).to(device)
    )
