"""Fig. 1 configuration."""

from dataclasses import field
import numpy as np

from src.config import BaseConfig


class ValidationConfig(BaseConfig):
    """Configuration for Fig. 1 validation."""
    
    a: float = 0.5  # Coupling strength delta = a * beta
    
    beta_range: np.ndarray = field(
        default_factory=lambda: np.logspace(-6, -2, 10)
    )
    mc_trials: int = 1_000_000
    max_steps: int = 1_000_000_000
    n_jobs: int = -1
    
    filename: str = "fig_validation.csv"
