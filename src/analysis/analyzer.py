import os
import json
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


class OptimizerAnalyzer:
    def __init__(self, base_dir: str, method: str, task_name: str, population_size: int, combine_method: str, model: str):
        """
        Initialize the optimizer analyzer.
        
        Args:
            base_dir: Base directory where workspace is located
            method: Optimization method name (e.g., 'ga', 'pso')
            task_name: Name of the task(s)
            population_size: Population size used in optimization
            combine_method: Method used for combining weights
            model: Model name
        """
        self.method = method.lower()
        self.base_path = Path(base_dir) / f"{self.method}_workspace/{task_name}/N{population_size}_{combine_method}/{model}"
        self.states: List[Dict] = []
        self.tasks = task_name.split('_')
        
    def load_states(self) -> None:
        """Load all state.json files from different runs."""
        method_prefix = self.method.upper()
        valid_states = []
        
        for exp_dir in self.base_path.glob(f"{method_prefix}-*"):
            state_path = exp_dir / "state.json"
            if state_path.exists():
                try:
                    with open(state_path) as f:
                        state = json.load(f)
                    # Check if state contains necessary fields
                    if all(field in state for field in ["test", "final", "ensemble_test"]):
                        valid_states.append(state)
                    else:
                        logger.warning(f"Incomplete state file found in {exp_dir}, skipping...")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON file found in {exp_dir}, skipping...")
                except Exception as e:
                    logger.warning(f"Error loading state file from {exp_dir}: {str(e)}")
        
        if not valid_states:
            raise ValueError(f"No valid state files found in {self.base_path}")
        
        self.states = valid_states
        logger.info(f"Loaded {len(self.states)} valid state files")
            
    def _extract_convergence_data(self) -> tuple:
        """Extract convergence data from all runs."""
        # Find the minimum number of steps across all runs
        min_steps = min(len([k for k in state.keys() if k.startswith("step_")]) for state in self.states)
        
        # Initialize arrays for weighted scores
        scores = np.zeros((len(self.states), min_steps))
        
        # Collect data from each run
        for i, state in enumerate(self.states):
            for step in range(min_steps):
                scores[i, step] = state[f"step_{step}"]["weighted_scores"]["max"]
                
        # Calculate mean and std
        mean_scores = np.mean(scores, axis=0)
        std_scores = np.std(scores, axis=0)
        
        return mean_scores, std_scores, range(min_steps)
    
    def plot_convergence(self, save_path: Optional[str] = None) -> None:
        """Plot convergence curve with variance."""
        mean_scores, std_scores, steps = self._extract_convergence_data()
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, mean_scores, 'b-', label='Mean Performance')
        plt.fill_between(steps, 
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        alpha=0.2)
        plt.xlabel('Step')
        plt.ylabel('Weighted Score')
        plt.title(f'{self.method.upper()} Convergence Analysis')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_final_performance(self) -> Dict:
        """Analyze the final performance on test set across all runs."""
        final_stats = {task: {"scores": []} for task in self.tasks}
        final_stats["weighted"] = {"scores": []}
        
        # Collect test scores from each run
        for state in self.states:
            try:
                # Get the best individual's test score
                final_score = state["final"]["test_score"]
                final_stats["weighted"]["scores"].append(final_score)
                
                # Get task-specific scores
                for task in self.tasks:
                    best_id = state["final"]["test_id"]
                    task_score = state["test"][task][best_id]
                    final_stats[task]["scores"].append(task_score)
            except KeyError as e:
                logger.warning(f"Missing key in state file: {str(e)}, skipping this run...")
                continue
        
        if not final_stats["weighted"]["scores"]:
            raise ValueError("No valid test results found in any state file")
        
        # Calculate statistics
        for metric in final_stats:
            scores = np.array(final_stats[metric]["scores"])
            final_stats[metric].update({
                "mean": np.mean(scores),
                "std": np.std(scores),
                "max": np.max(scores),
                "min": np.min(scores)
            })
        
        return final_stats
    
    def analyze_ensemble_performance(self) -> Dict:
        """Analyze ensemble test performance across all runs."""
        ensemble_stats = {task: {"top_k_scores": {}} for task in self.tasks}
        ensemble_stats["weighted"] = {"top_k_scores": {}}
        
        # Collect ensemble scores from each run
        for state in self.states:
            try:
                ensemble_results = state["ensemble_test"]
                
                # Process task-specific results
                for task in self.tasks:
                    task_results = ensemble_results["task_results"][task]
                    for k in range(2, len(task_results) + 1):
                        top_k = f"top_{k}"
                        if top_k not in ensemble_stats[task]["top_k_scores"]:
                            ensemble_stats[task]["top_k_scores"][top_k] = []
                        ensemble_stats[task]["top_k_scores"][top_k].append(task_results[top_k]["score"])
                
                # Process weighted results
                weighted_results = ensemble_results["weighted_results"]
                for k in range(2, len(weighted_results) + 1):
                    top_k = f"top_{k}"
                    if top_k not in ensemble_stats["weighted"]["top_k_scores"]:
                        ensemble_stats["weighted"]["top_k_scores"][top_k] = []
                    ensemble_stats["weighted"]["top_k_scores"][top_k].append(weighted_results[top_k]["score"])
            except KeyError as e:
                logger.warning(f"Missing key in ensemble results: {str(e)}, skipping this run...")
                continue
        
        if not ensemble_stats["weighted"]["top_k_scores"]:
            raise ValueError("No valid ensemble results found in any state file")
        
        # Calculate statistics for each top-k
        for metric in ensemble_stats:
            stats = {}
            best_k = None
            best_mean = float('-inf')
            
            for top_k, scores in ensemble_stats[metric]["top_k_scores"].items():
                scores_array = np.array(scores)
                current_mean = np.mean(scores_array)
                stats[top_k] = {
                    "mean": np.mean(scores_array),
                    "std": np.std(scores_array),
                    "max": np.max(scores_array),
                    "min": np.min(scores_array)
                }
                
                if current_mean > best_mean:
                    best_mean = current_mean
                    best_k = top_k
            
            ensemble_stats[metric]["stats"] = stats
            ensemble_stats[metric]["best_k"] = {
                "k": best_k,
                "mean": best_mean
            }
            
        return ensemble_stats
    
    def plot_ensemble_performance(self, save_path: Optional[str] = None) -> None:
        """Plot ensemble performance for different top-k."""
        ensemble_stats = self.analyze_ensemble_performance()
        
        plt.figure(figsize=(12, 6))
        
        # Plot for each task and weighted score
        for metric in ensemble_stats:
            stats = ensemble_stats[metric]["stats"]
            k_values = range(2, len(stats) + 1)
            means = [stats[f"top_{k}"]["mean"] for k in k_values]
            stds = [stats[f"top_{k}"]["std"] for k in k_values]
            
            plt.errorbar(k_values, means, yerr=stds, label=metric, marker='o', capsize=5)
        
        plt.xlabel('Top-k Models')
        plt.ylabel('Score')
        plt.title(f'{self.method.upper()} Ensemble Performance')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def analyze_task_correlation(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Analyze correlation between different tasks if multiple tasks exist."""
        if len(self.tasks) <= 1:
            return None
            
        # Collect final performance for each task
        task_scores = []
        for task in self.tasks:
            scores = []
            for state in self.states:
                try:
                    best_id = state["final"]["test_id"]
                    scores.append(state["test"][task][best_id])
                except KeyError as e:
                    logger.warning(f"Missing key in state file: {str(e)}, skipping this run...")
                    continue
            task_scores.append(scores)
            
        if not all(task_scores):
            raise ValueError("No valid test results found for correlation analysis")
            
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(task_scores)
        
        # Plot correlation heatmap
        if save_path:
            plt.figure(figsize=(8, 6))
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            
            # Add text annotations
            for i in range(len(self.tasks)):
                for j in range(len(self.tasks)):
                    plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha='center', va='center')
            
            plt.xticks(range(len(self.tasks)), self.tasks)
            plt.yticks(range(len(self.tasks)), self.tasks)
            plt.title('Task Correlation Matrix')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        return corr_matrix