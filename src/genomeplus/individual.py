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
    """individual for hybrid (PSO-GA) optimizer method."""
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
        self.v = None
        self.p = x
        
    def save_individual(self, save_path: str):
        save_lora_weight(
            lora_weight=self.x,
            lora_path=save_path,
            tokenizer = self.tokenizer,
            config=self.config_path
        )
        # update save path
        self.weight_path = save_path
            
    def init_velocity(self, lora: dict) -> None:
        assert set(lora.keys()) == set(self.x.keys()), "The architecture of the two LORAs must be the same."

        self.v = dict()
        for key in lora.keys():
            self.v[key] = lora[key] - self.x[key]
    
    def update_velocity(
        self, global_max_weight: Dict, global_min_weight: Dict,
        r: List[float] | None, phi: List[float], C: float
    ):
        for key in self.x.keys():
            try:
                self.v[key] = r[0] * phi[0] * self.v[key] \
                    + r[1] * phi[1] * (self.p[key] - self.x[key]) \
                        + r[2] * phi[2] * (global_max_weight[key] - self.x[key]) \
                            - r[3] * phi[3]* (global_min_weight[key] - self.x[key])
                self.v[key] = 1/C * self.v[key]
            except Exception as e:
                print(f"Error processing key {key}")
                print(f"Shapes: v={self.v[key].shape}, p={self.p[key].shape}, x={self.x[key].shape}")
                print(f"max={global_max_weight[key].shape}, min={global_min_weight[key].shape}")
                raise e
    
    def update_weight(self, _lambda: float):
        for key in self.x.keys():
            self.x[key] = self.x[key] + _lambda * self.v[key]
    

    def update_position(self, fitness_score: float, path: str):
        """dropout method"""
        if self.fitness_score > self.best_fitness_score:
            self.best_fitness_score = fitness_score
            self.p = self.x
        # TODO: reset individual position
    
    def mutation(self, individual_mutation_rate: float, gene_mutation_rate: float, sigma: float):
        if random.random() > individual_mutation_rate:
            # do not mutate, do nothing
            return None

        else:
            mutated_weights = dict()
            for key, tensor in self.x.items():
                device = tensor.device
                dtype = tensor.dtype  
                
                mutation_mask = torch.rand(tensor.shape, device=device) < gene_mutation_rate
                noise = torch.randn(tensor.shape, device=device, dtype=dtype) * sigma
                
                tensor_mutated = tensor + noise * mutation_mask.to(dtype)
                mutated_weights[key] = tensor_mutated
            
            old_id = self.id
            new_individual_id = uuid.uuid4().hex
            
            logger.info(
                f"Individual {old_id} (fitness score: {self.fitness_score:.4f}) is mutated to Individual {new_individual_id}"
            )
            
            new_weight_path = os.path.join("/".join(self.weight_path.split("/")[:-1]), f"individual_{new_individual_id}")
            save_lora_weight(
                lora_weight=mutated_weights,
                lora_path=new_weight_path,
                tokenizer=self.tokenizer,
                config=self.config_path
            )
            new_individual = Individual(
                id=new_individual_id, x = mutated_weights, from_mutation=self.weight_path, parent=[], weight_path=new_weight_path,
                model_name_or_path=self.model_name_or_path, lora_config_path=self.config_path, 
            )
            new_individual.evaluated = {}
            
            return new_individual