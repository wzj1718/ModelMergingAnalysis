from dataclasses import dataclass
from src.base.base_config import BaseConfig
from src.base.base_method import CombineMethod
from enum import Enum

# todo: add more selection methods
class SelectionMethod(Enum):
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"
    RANDOM = "random"

@dataclass
class GenomePlusConfig(BaseConfig):
    # hyperparameters
    N_init: int = 10
    N_max: int = 20
    max_iter: int = 20
    do_init: bool = True
    lambda_step: float = 0.5
    phi_lambda: float = 0.95
    phi_inertia: float = 0.2
    phi_cognitive: float = 0.2
    phi_social: float = 0.2
    phi_repel: float = 0.1
    unable_random: bool = False
    
    ## Genome & GenomePlus
    cross_method: str = "ties"
    cross_rate: float = 0.8
    individual_mutation_rate: float=0.3
    gene_mutation_rate: float=0.1
    sigma: float=0.05
    elite_percent:float=0.1
    method: str = "roulette"
    selection_method: str = "roulette"

    def __post_init__(self):
        super().__post_init__()
        self.N = self.N_init
        self.cross_method = CombineMethod(self.cross_method)
    
    def save(self, path):
        if isinstance(self.cross_method, CombineMethod):
            self.cross_method = self.cross_method.value
        
        super().save(path)
    
    def validate(self):
        """validate config"""
        super().validate()
        if self.N_init < 1:
            raise ValueError("N_init must be greater than 0")
        if self.N_max < 1:
            raise ValueError("N_max must be greater than 0")
        if self.N_init > self.N_max:
            raise ValueError("N_init must be less than or equal to N_max")
        if self.max_iter < 1:
            raise ValueError("max_iter must be greater than 0")
        if not (0 <= self.lambda_step <= 1):
            raise ValueError("lambda_step must be in [0, 1]")
        if not (0 <= self.phi_lambda <= 1):
            raise ValueError("phi_lambda must be in [0, 1]")
        if not (0 <= self.phi_inertia <= 1):
            raise ValueError("phi_inertia must be in [0, 1]")
        if not (0 <= self.phi_cognitive <= 1):
            raise ValueError("phi_cognitive must be in [0, 1]")
        if not (0 <= self.phi_social <= 1):
            raise ValueError("phi_social must be in [0, 1]")
        if not (0 <= self.phi_repel <= 1):
            raise ValueError("phi_repel must be in [0, 1]")
        if not (0 <= self.cross_rate <= 1):
            raise ValueError("cross_rate must be in [0, 1]")
        if not (0 <= self.individual_mutation_rate <= 1):
            raise ValueError("individual_mutation_rate must be in [0, 1]")
        if not (0 <= self.gene_mutation_rate <= 1):
            raise ValueError("gene_mutation_rate must be in [0, 1]")
        if not (0 <= self.sigma <= 1):
            raise ValueError("sigma must be in [0, 1]")
        if not (0 <= self.elite_percent <= 1):
            raise ValueError("elite_percent must be in [0, 1]")