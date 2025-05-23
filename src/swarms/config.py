from dataclasses import dataclass
from src.base.base_config import BaseConfig

@dataclass
class PSOConfig(BaseConfig):
    # hyperparameters
    N: int = 10
    max_iter: int = 50
    lambda_step: float=0.5
    phi_lambda: float=0.95
    phi_inertia: float=0.2
    phi_cognitive: float=0.2
    phi_social: float=0.2
    phi_repel: float=0.1
    
    # control parameters
    particle_reset_iter: int = 3
    unable_random: bool = False
    max_workers: int = 1
    
    def validate(self):
        super().validate()
        if self.N < 1:
            raise ValueError("N must be greater than 0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be greater than 0")
        if not (0 <= self.lambda_step <= 1):
            raise ValueError("lambda_step must be in range [0, 1]")
        if not (0 <= self.phi_lambda <= 1):
            raise ValueError("phi_lambda must be in range [0, 1]")
        if not all(0 <= phi <= 1 for phi in [self.phi_cognitive, self.phi_inertia, self.phi_repel, self.phi_social]):
            raise ValueError("phi values must be in range [0, 1]")
        if self.max_workers < 1:
            raise ValueError("max_workers must be greater than 0")
        if len(self.llm_base_url) < self.max_workers:
            raise ValueError("Not enough llm_base_url for workers")