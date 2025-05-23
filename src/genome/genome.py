import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import numpy as np
from loguru import logger
from src.genome.config import GenomeConfig
from src.genome.individual import Individual
from src.utils import load_lora_weight
from src.base.base_method import BaseMethod

class Genome(BaseMethod):
    def __init__(self, config: GenomeConfig):
        """
        Initialize the Genome.
        """
        
        # order is important.
        self.cross_method = config.cross_method
        super().__init__(config)
        config.validate()
        
        self.N = config.N
        self.epochs = config.max_iter

        # hyper params
        self.cross_rate = config.cross_rate
        self.individual_mutation_rate = config.individual_mutation_rate
        self.gene_mutation_rate = config.gene_mutation_rate
        self.sigma = config.sigma
        self.elite_percent = config.elite_percent
        self.elite_number = int(self.elite_percent * self.N)

        self.method = config.method

    def initialize(self) -> None:
        logger.info(f"Initializing population...")
        start_time = time.time()

        self.individuals = []

        expert_pairs = self.generate_pair_sequences(pools=self.pools, n_samples=self.N, seed=self.seed)
        print(expert_pairs)
        for expert_pair in expert_pairs:
            print(expert_pair)
            print(type(expert_pair))
            individual_id = uuid.uuid4().hex
            individual_weight = self.merge_lora_weights(
                lora_state_dicts=[load_lora_weight(expert_pair[0]), load_lora_weight(expert_pair[1])],
                weights=[], 
                method=self.combine_method,
                density=0.7,
                majority_sign_method="total"
            )
            ### 权重分布直方图
            # individual_weight = self.merge_lora_weights_for_weight_fenbu(
            #     lora_state_dicts=[load_lora_weight(expert_pair[0]), load_lora_weight(expert_pair[1])],
            #     expert_pair=expert_pair, #####
            #     weights=[], 
            #     method=self.combine_method,
            #     density=0.7,
            #     majority_sign_method="total"
            # )
            individual = Individual(
                id=individual_id,
                x=individual_weight,
                weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                parent=expert_pair,
                lora_config_path=self.pools[0],
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(save_path=individual.weight_path)
            self.individuals.append(individual)
        
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        end_time = time.time()
        logger.info(f"Init time: {(end_time - start_time):.2f} seconds.")

        self.update_optim_state(step = 0, time=end_time-start_time, weighted_scores=weighted_scores)
        self.save_optim_state(self.state)
        self.report_state(step=0)

    def selection(self, method: str) -> List[Individual]:
        # check if the population is full
        if len(self.individuals) == self.N:
            logger.info(f"Population size is equal to N (N={self.N}).")
            return self.individuals
        
        logger.info(f"Start selection, method: {method}")
        all_individual_paths = {ind.weight_path for ind in self.individuals}

        # reserve elite individuals
        sorted_individuals = sorted(
            self.individuals, key=lambda x: x.fitness_score, reverse=True
        )
        elites = sorted_individuals[:self.elite_number]
        remaining_individuals = sorted_individuals[self.elite_number:]
        remaining_size = self.N - self.elite_number

        # method: roulette, tournament, rank, random, elite
        # 选择种群
        if method == "roulette":
            fitness_scores = [i.fitness_score for i in remaining_individuals]
            # compute the probability of each individual
            prob = [score/sum(fitness_scores) for score in fitness_scores]
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                p=prob,
                replace=False
            )
            selected_individuals = np.concatenate(
                [elites, selected_remaining]
            )
        elif method == "tournament":
            tournament_size = 3
            selected_remaining = []
            available_individuals = remaining_individuals.copy()
            while len(selected_remaining) < remaining_size and len(available_individuals) > 0:
                current_tournament_size = min(tournament_size, len(available_individuals))
                tournament_candidates = np.random.choice(
                    available_individuals,
                    size=current_tournament_size,
                    replace=False
                )
                winner = max(tournament_candidates, key=lambda x: x.fitness_score)
                selected_remaining.append(winner)
                available_individuals.remove(winner)
        elif method == "rank":
            ranks = range(1, len(remaining_individuals)+1)
            prob = [rank/sum(ranks) for rank in ranks]
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                replace=False,
                p=prob
            )
        elif method == "elite":
            selected_remaining = sorted_individuals[self.elite_number:self.N]
        elif method == "random":
            selected_remaining = np.random.choice(
                remaining_individuals,
                size=remaining_size,
                replace=False
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        selected_individuals = np.concatenate(
            [elites, selected_remaining]
        )
        selected_paths = {ind.weight_path for ind in selected_individuals}
        paths_to_remove = all_individual_paths - selected_paths
        for path in paths_to_remove:
            if os.path.exists(path):
                try:
                    import shutil
                    shutil.rmtree(path)
                    logger.info(f"Removed unselected individual: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {str(e)}")

        return selected_individuals.tolist()

    def crossover(self, step: int, method: str):
        logger.info(f"Start Crossover, method: {method}")
        num_pairs = len(self.individuals) // 2
        if method == "roulette":
            fitness_scores = [i.fitness_score for i in self.individuals]
            total_fitness = sum(fitness_scores)
            prob = [i / total_fitness for i in fitness_scores]

            pairs = []
            for _ in range(num_pairs):
                if random.random() <= self.cross_rate:
                    parents = np.random.choice(
                        self.individuals, size=2, p=prob, 
                        replace=False,
                    )
                    pairs.append((parents[0], parents[1]))
        elif method == "random":
            pairs = []
            for _ in range(num_pairs):
                parents = np.random.choice(
                    self.individuals, size=2, replace=False,
                )
                pairs.append((parents[0], parents[1]))
        else:
            raise ValueError(f"Invalid method: {method}")

        for pair in pairs:
            individual_id = uuid.uuid4().hex
            individual_weight = self.merge_lora_weights(
                lora_state_dicts=[pair[0].x, pair[1].x],
                weights=[pair[0].fitness_score, pair[1].fitness_score],
                method=self.cross_method,
                density=0.7,
                majority_sign_method="total",
                alpha=1.2,
            )
            individual = Individual(
                id=individual_id,
                x=individual_weight,
                weight_path=os.path.join(self.workspace, f"individual_{individual_id}"),
                parent=pair,
                lora_config_path=self.pools[0],
                model_name_or_path=self.model_name_or_path,
            )
            individual.save_individual(save_path=individual.weight_path)
            self.individuals.append(individual)

        logger.info(f"Crossover completed, current population size: {len(self.individuals)}")

    def mutation(self):
        logger.info(f"Start mutation...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    individual.mutation,
                    individual_mutation_rate=self.individual_mutation_rate,
                    gene_mutation_rate=self.gene_mutation_rate,
                    sigma=self.sigma,
                    best_fitness_score=self.global_max_fitness_score,
                ) for individual in self.individuals
            ]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    self.individuals.append(result)

        logger.info(f"Mutation completed, current population size: {len(self.individuals)}")

    def _step(self, step: int, method: str):
        start_time = time.time()
        # 1. crossover
        self.crossover(step=step, method=method)
        # 2. mutation, mutation create new individual
        self.mutation()
        # 3. evaluate
        ## 3.1 get weighted scores
        weighted_scores = self.evaluate(individuals=self.individuals, split="valid")
        ## 3.2 update global
        for individual_id, result in weighted_scores.items():
            self.update_global(
                id=individual_id,
                fitness_score=result['weighted_score'],
                path=result['path'],
                task_scores=result['task_scores']
            )
        # 4. selection
        self.individuals = self.selection(method="tournament")
        end_time = time.time()
        logger.info(f"Step {step} takes {(end_time - start_time):.2f} seconds.")
        # 5. save state
        self.update_optim_state(step=step, time=end_time - start_time, weighted_scores=weighted_scores)
        self.save_optim_state(self.state)
        self.report_state(step=step)
    
    def print_config(self):
        logger.info(f"GA config: ")
        logger.info(
            f"Tasks = {self.tasks}, Task weights = {self.task_weights}, "
            f"Seed = {self.seed}, N = {self.N}, Epochs = {self.epochs}, "
            f"Combine method = {self.combine_method}, "
            f"Cross method = {self.cross_method}, "
            f"cross rate = {self.cross_rate}, mutation rate = {self.individual_mutation_rate}, "
            f"gene mutation rate = {self.gene_mutation_rate}, sigma = {self.sigma}\n"
            f"Early Stop = {self.early_stop}, Early Stop Iter = {self.early_stop_iter}\n"
        )
    
    def search(self):
        start_time = time.time()
        logger.info("Start GA search.")
        self.print_config()

        method = self.method
        self.initialize()
        for i in range(1, self.epochs+1):
            self._step(step=i, method=method)

            if self.patience_flag == True:
                self.global_patience_counter += 1
                if self.global_patience_counter > self.early_stop_iter and self.early_stop:
                    logger.info("Early stop triggered.")
                    break
            else:
                self.global_patience_counter = 0
                self.patience_flag = True
            
        # test performance
        self.ensemble_test(individuals=self.individuals, split="test")
        end_time = time.time()  
        try:
            self.save_final_state(individuals=self.individuals, time=end_time-start_time)
        except Exception as e:
            self.save_optim_state(self.state)
            logger.error(f"Error saving final state: {e}")
            
        if self.plot_enabled:
            try:
                self.generate_plots()
            except Exception as e:
                logger.error(f"Error generating plots: {e}")