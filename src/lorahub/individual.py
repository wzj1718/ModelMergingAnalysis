import json
import os
import random
import uuid
from typing import Dict, List

import torch
from loguru import logger
from src.utils import save_lora_weight
from src.base.base_individual import BaseIndividual

class Individual(BaseIndividual):
    def __init__(self, 
                 id: str, x: Dict, weight_path: str, model_name_or_path: str, lora_config_path: str,
                 seed: int = 42,
    ):
        super().__init__(
            id=id, x=x, weight_path=weight_path,
            model_name_or_path=model_name_or_path,
            lora_config_path=lora_config_path, seed=seed
        )
        self.p = x
    
    def save_individual(self, save_path: str):
        save_lora_weight(
            lora_weight=self.x,
            lora_path=save_path,
            tokenizer=self.tokenizer,
            config=self.config_path
        )
        # update save path
        self.weight_path = save_path