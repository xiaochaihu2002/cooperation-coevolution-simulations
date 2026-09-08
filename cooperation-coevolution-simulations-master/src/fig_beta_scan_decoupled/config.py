"""Configuration for the decoupled beta scan (Fig. 3)."""

from dataclasses import dataclass, field

import numpy as np

from src.config import BaseConfig


@dataclass
class DecoupledConfig(BaseConfig):
    """Configuration of the model with decoupled rationalities.

    The rewiring rationality delta is held fixed while the strategy-updating
    intensity beta is swept.  Parameters follow Fig. 3 of the manuscript:
    N = 100, b = 4, c = 1, L = 4, k0 = 0.5, delta = 0.005,
    k_CC* = -3, k_CD* = -2 and k_DD* = -10.
    """

    delta: float = 0.005

    N: int = 100
    b: float = 4.0
    c: float = 1.0
    L: int = 4
    k0: float = 0.5
    k_cc_star: float = -3.0
    k_cd_star: float = -2.0
    k_dd_star: float = -10.0

    beta_range: np.ndarray = field(
        default_factory=lambda: np.logspace(-4, -2, 10)
    )
    mc_trials: int = 1_000_000
    max_steps: int = 1_000_000_000
    n_jobs: int = -1

    filename: str = "fig_beta_scan_decoupled.csv"
