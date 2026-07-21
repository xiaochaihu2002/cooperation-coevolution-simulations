"""
Shared configuration base class.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class BaseConfig:
    """
    Base configuration for all simulation models.
    
    Attributes:
        N: Population size.
        L: Network average degree.
        b: Benefit of cooperation.
        c: Cost of cooperation.
        w: Time scale separation.
        k0: Baseline link breaking rate.
        k_cc_star, k_cd_star, k_dd_star: Link breaking perturbations.
        output_dir: Directory for output files.
    """
    N: int = 100
    L: int = 4
    b: float = 4.0
    c: float = 1.0
    w: float = 0.01
    k0: float = 0.5
    
    k_cc_star: float = -3.0
    k_cd_star: float = -1.0
    k_dd_star: float = 3.0
    
    output_dir: str = "./data"
