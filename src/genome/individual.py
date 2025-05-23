import os
import random
import uuid
from typing import Dict, List

import torch
from loguru import logger
from src.utils import save_lora_weight
from src.base.base_individual import BaseIndividual

class Individual(BaseIndividual):
    def __init__(
        self, id: str, x: Dict, parent: List[str], 
        weight_path: str, model_name_or_path: str, lora_config_path: str,
        seed: int = 42, from_mutation: str = None,
    ):
        super().__init__(
            id=id, x=x, weight_path=weight_path,
            model_name_or_path=model_name_or_path,
            lora_config_path=lora_config_path, seed=seed
        )
        self.parent = parent
        self.from_mutation = from_mutation

    def mutation(self, individual_mutation_rate: float, gene_mutation_rate: float, sigma: float=0.01, best_fitness_score: float=-100):
        # if self.fitness_score >= best_fitness_score:
        #     logger.info(f"Individual {self.id} (fitness score: {self.fitness_score:.4f}) is the best individual, no mutation")
        #     # do nothing
        #     return
        if random.random() > individual_mutation_rate:
            # donot mutate, do nothing
            return None
        else:
            mutated_weights = {}
            for key, tensor in self.x.items():
                device = tensor.device
                dtype = tensor.dtype

                mutation_mask = torch.rand(tensor.shape, device=device) < gene_mutation_rate
                noise = torch.randn(tensor.shape, device=device, dtype=dtype) * sigma

                tensor_mutated = tensor + noise * mutation_mask.to(dtype)
                mutated_weights[key] = tensor_mutated
            
            # reset individual state
            # self.x = mutated_weights
            old_id = self.id
            new_id = uuid.uuid4().hex

            logger.info(f"Individual {old_id} (fitness score: {self.fitness_score:.4f}) mutated into {new_id}")
            
            new_weight_path = os.path.join("/".join(self.weight_path.split("/")[:-1]), f"individual_{new_id}")
            save_lora_weight(
                lora_weight=mutated_weights,
                lora_path=new_weight_path,
                tokenizer=self.tokenizer,
                config=self.config_path
            )
            new_individual = Individual(
                id=new_id, x = mutated_weights, from_mutation=self.weight_path, parent=[], weight_path=new_weight_path,
                model_name_or_path=self.model_name_or_path, lora_config_path=self.config_path, 
            )
            new_individual.evaluated = {}
            
            return new_individual