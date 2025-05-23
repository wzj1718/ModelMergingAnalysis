import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
import shutil
import numpy as np
from loguru import logger
from openai import OpenAI
from src.lorahub.config import LoraHubConfig
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod
from src.lorahub.individual import Individual
import nevergrad as ng

class LoraHub(BaseMethod):
    def __init__(self, config: LoraHubConfig):
        super().__init__(config)
        self.individuals = []
        self.do_search = config.do_search
    
    def _evaluate(self):
        pass
    
    def initialize(self):
        logger.info(f"Initializing Lora Model ...")
        for i in range(len(self.pools)):
            individual_id = uuid.uuid4().hex
            individual = Individual(
                id=individual_id,
                x=load_lora_weight(self.pools[i]),
                weight_path=self.pools[i],
                model_name_or_path=self.model_name_or_path,
                lora_config_path=self.pools[i],
            )
            self.individuals.append(individual)
            
    def default_l1_regularization(self, weights):
        """
        Get the L1 regularization term for the weights
        """
        sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
        return 0.05 * sum_of_squares
    
    def get_init_weights(self) -> List[float]:
        # calculate the fitness score on valid set
        self.initialize()
        weighted_scores = self.evaluate(
            individuals=self.individuals, 
            split="valid"
        )
        weight = [
            value['weighted_score']
            for _, value in weighted_scores.items()
        ]
        # normalize the weights
        weight = [x / sum(weight) for x in weight]
        logger.info(f"Initial score: {weight}")
        return weight
    
    def get_score(self, weights, split="valid"):
        merged_lora_weight = self.merge_lora_weights(
            lora_state_dicts=[load_lora_weight(self.pools[i]) for i in range(len(self.pools))],
            weights=weights,
            method=self.combine_method,
            density=0.7,
            majority_sign_method="total"
        )
        individual_id = uuid.uuid4().hex
        individual = Individual(
            id=individual_id,
            x=merged_lora_weight,
            weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
            model_name_or_path=self.model_name_or_path,
            lora_config_path=self.pools[0],
        )
        individual.save_individual(save_path=individual.weight_path)
        weighted_scores = self.evaluate(individuals=[individual], split=split)
        score = weighted_scores[individual_id]['weighted_score']
        regular_score = score + self.default_l1_regularization(weights)
        logger.info(f"- Score: {score}- Regular score: {regular_score}")
        
        if split == "valid":
            # for minimization
            return -1 * regular_score
        elif split == "test":
            # for maximization
            return score
        else:
            raise ValueError(f"Invalid split: {split}")
        
    def search(self):
        start_time = time.time()
        logger.info(f"Start LoraHub searching ...")        
        # self.initialize()
        # self.get_init_score(individuals=self.individuals)
        init_weights = self.get_init_weights()
        if self.do_search:
            instrum = ng.p.Array(
                init=init_weights,
                upper=[1.5]*self.config.N,
                lower=[-1.5]*self.config.N,
            )
            optimizer = ng.optimizers.NGOpt(
                parametrization=instrum,
                budget=self.config.max_iter
            )
            logger.info("> Begin to perform gradient-free optimization ...")
            recommendations = optimizer.minimize(self.get_score, verbosity=1)
            final_weights = recommendations.value
        else:
            logger.warning(f"Search switch is off, the algorithm is equal to EXPERT FUSION.")
            final_weights = init_weights
            
        for pool, weight in zip(self.pools, final_weights):
            logger.info(f"-> Pool: {pool.split('/')[-1]}, Weight: {weight:.4f}")
        
        if self.test_tasks != self.tasks:
            logger.warning(f"Test tasks and tasks are not consistent, got {self.test_tasks} and {self.tasks}.")
            self.tasks = list(set(self.tasks + self.test_tasks))
            self.task_weights = [1.0 / len(self.tasks) for _ in range(len(self.tasks))]
            
        final_score = self.get_score(final_weights, split="test")
        logger.info(f"Final score: {final_score}")
        end_time = time.time()
        logger.info(f"LoraHub searching completed, time: {end_time - start_time}s")
        
        return final_weights