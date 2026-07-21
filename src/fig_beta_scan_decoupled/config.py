"""Fig. 3 configuration."""

from dataclasses import field
import numpy as np

from src.config import BaseConfig


class DecoupledConfig(BaseConfig):
    """Configuration for decoupled rationality model."""
    
    delta: float = 0.005  # Independent rewiring rationality
    c: float = 2.0  # Note: c=2 for this figure
    
    k_cd_star: float = 2.0
    k_dd_star: float = 4.0
    
    beta_range: np.ndarray = field(
        default_factory=lambda: np.logspace(-3, -1, 15)
    )
    mc_trials: int = 1_000_000
    max_steps: int = 1_000_000_000
    n_jobs: int = -1
    
    filename: str = "fig_beta_scan_decoupled.csv"
