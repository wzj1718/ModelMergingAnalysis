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
from src.packllm.config import PackLLMConfig
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod
from src.lorahub.individual import Individual


class PackLLM(BaseMethod):
    def __init__(self, config: PackLLMConfig):
        super().__init__(config)
        self.individuals = []
        self.method = config.method
        self.tao = config.tao
        self.topK = config.topK
        
    def initialize(self):
        logger.info(f"Initializing Pack of LLMs Model ...")
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
    
    def get_ppl(self, split="valid"):
        self.initialize()
        # logger.warning(f"Pack of LLMs only support single task now! Current task: {self.tasks[0]}")
        lambda_ppl_group_by_task = dict()
        id_to_task_ppl = {ind.id: [] for ind in self.individuals}  # 每个 id 的 ppl 列表
        for task in self.tasks:
            logger.info(f"Evaluating task: {task}")
            ppl_results = self.evaluate_single_task_ppl(
                individuals=self.individuals,
                task=task,
                split=split
            )
            for result in ppl_results.values():
                id_ = result['id']
                perplexity = result['perplexity']
                id_to_task_ppl[id_].append(perplexity)

        avg_ppl_list = []  # 存储每个 id 的平均 ppl
        for id_, ppls in id_to_task_ppl.items():
            avg_ppl = np.mean(ppls)  # 计算均值
            avg_ppl_list.append(avg_ppl)
            logger.info(f"ID: {id_}, Average PPL across tasks: {avg_ppl:.4f}")  # 打印每个id的均值
        
        lambda_ppl = -np.log(avg_ppl_list) / self.tao
        lambda_ppl = lambda_ppl - np.max(lambda_ppl)
        lambda_ppl = np.exp(lambda_ppl)
        lambda_ppl = lambda_ppl / np.sum(lambda_ppl)

        for i in range(len(self.pools)):
            logger.info(f"ID: {self.pools[i].split('/')[-1]}, Final Lambda PPL: {lambda_ppl[i]:.4f}")

        return lambda_ppl
    
    def merge_lora_by_lambda(self, lambda_ppl, topK):
        sorted_indices = np.argsort(lambda_ppl)[::-1]  # Sort in descending order
        topK_indices = sorted_indices[:topK]
        topK_individuals = [self.individuals[i] for i in topK_indices]
        for idx, individual in enumerate(topK_individuals):
            individual.fitness_score = lambda_ppl[topK_indices[idx]]
        logger.info(f"Merge {topK} individuals ...")
        
        merged_lora_weight = self.merge_lora_weights(
            lora_state_dicts=[individual.x for individual in topK_individuals],
            weights=[individual.fitness_score for individual in topK_individuals],
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
        
        return individual
    
    def search(self):
        logger.info(f"Searching Pack of LLMs Model ...")
        logger.info(f"Method: {self.config.method}")
        logger.info(f"Tasks: {self.tasks}")
        logger.info(f"Tao: {self.tao}")
        logger.info(f"TopK: {self.topK}")
        
        start_time = time.time()
        ppl = self.get_ppl()
        logger.info(f"PPL: {ppl}")
        merged_individual = self.merge_lora_by_lambda(ppl, topK=self.topK)
        # get test performace
        if self.test_tasks != self.tasks:
            logger.warning(f"Test tasks and tasks are not consistent, got {self.test_tasks} and {self.tasks}.")
            self.tasks = list(set(self.tasks + self.test_tasks))
            self.task_weights = [1.0 / len(self.tasks) for _ in range(len(self.tasks))]
            
        test_results = self.evaluate(individuals=[merged_individual], split="test")
        logger.info(f"Test Results: {test_results}")
        end_time = time.time()
        logger.info(f"Time: {end_time - start_time:.2f}s")
        return test_results
