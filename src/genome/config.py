import json
from dataclasses import asdict, dataclass
from typing import List, Optional
from src.base.base_config import BaseConfig
import numpy as np
from src.base.base_method import CombineMethod

@dataclass
class GenomeConfig(BaseConfig):    
    N: int
    max_iter: int

    cross_rate: float
    cross_method: str
    individual_mutation_rate: float
    gene_mutation_rate: float
    sigma: float
    elite_percent: float
    method: str = "roulette"

    def __post_init__(self):
        super().__post_init__()
        self.cross_method = CombineMethod(self.cross_method)
        
    def validate(self):
        """Validate GA-specific configuration parameters."""
        # First validate base configuration
        super().validate()
        
        # Validate population parameters
        if self.N < 1:
            raise ValueError("Population size (N) must be greater than 0")
        if self.max_iter < 1:
            raise ValueError("Maximum iterations must be greater than 0")
            
        # Validate GA hyperparameters
        if not (0 <= self.cross_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if not (0 <= self.individual_mutation_rate <= 1):
            raise ValueError("Individual mutation rate must be between 0 and 1")
        if not (0 <= self.gene_mutation_rate <= 1):
            raise ValueError("Gene mutation rate must be between 0 and 1")
        if not (0 <= self.sigma <= 1):
            raise ValueError("Sigma must be between 0 and 1")
        if not (0 <= self.elite_percent <= 1):
            raise ValueError("Elite percentage must be between 0 and 1")
        if self.method not in ["roulette", "tournament"]:
            raise ValueError("Selection method must be either 'roulette' or 'tournament'")
    
    def save(self, path):
        if isinstance(self.cross_method, CombineMethod):
            self.cross_method = self.cross_method.value
        
        super().save(path)