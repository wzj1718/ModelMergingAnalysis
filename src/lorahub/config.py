from dataclasses import dataclass
from src.base.base_config import BaseConfig
from src.base.base_method import CombineMethod
from enum import Enum


@dataclass
class LoraHubConfig(BaseConfig):
    N: int = 10
    max_iter: int = 50
    do_search: bool = True