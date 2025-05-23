from dataclasses import dataclass
from src.base.base_config import BaseConfig
from src.base.base_method import CombineMethod
from enum import Enum


@dataclass
class PackLLMConfig(BaseConfig):
    N: int = 10
    method: str = "simple"
    tao: float = 0.1
    topK: int = 5