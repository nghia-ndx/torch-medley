from typing import Any, Type, Union


def get_class_name(obj: Union[Any, Type]) -> str:
    if isinstance(obj, Type):
        return obj.__name__
    return type(obj).__name__
