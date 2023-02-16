from .base import BaseConfig


class PathConfig(BaseConfig):
    base_dir: str = 'project'
    save_dir: str = base_dir + '/saves/{}'
    dataset_dir: str = base_dir + '/datasets/{}'
    embedding_dir: str = base_dir + '/embeddings/{}'
