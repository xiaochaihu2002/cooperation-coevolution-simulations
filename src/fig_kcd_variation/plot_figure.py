"""
Plotter for Fig. 2: Non-monotonic dependence of critical selection intensity 
on CD link breaking probability (large-N limit).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.plot_utils import setup_academic_style


def compute_beta_star_large_n(
    k_cd_star: np.ndarray,
    k_cc_star: float,
    k_dd_star: float,
    n: int = 100,
    b: float = 4.0,
    c: float = 1.0,
    a: float = 0.5,
    L: int = 4,
    k0: float = 0.5
) -> np.ndarray:
    """
    Compute beta^*(k_CD^*) in the large-N limit.
    
    Uses simplified expression: beta^* = -B'(k_CD^*) / A'(k_CD^*)
    """
    struct_term = ((n + 1) / 3.0) * (k_cc_star - 2.0 * k_cd_star + k_dd_star)
    struct_term += n * (k_cd_star - k_dd_star)
    B_prime = -(c * L) / (2.0 * n) - (a / (2.0 * n**2 * k0)) * struct_term
    
    A1_prime = (c**2 * L**2 * (n - 2)) / (12.0 * n)
    
    A2_prime = (a * L * (n + 1) * (
        4.0 * b * (k_cd_star - k_cc_star)
        + c * (
            4.0 * k_dd_star
            + 2.0 * k_cd_star * (n - 2)
            + k_cc_star * n
            - 3.0 * k_dd_star * n
        )
    )) / (24.0 * k0 * n**2)
    
    common = k_cc_star - 2.0 * k_cd_star + k_dd_star
    common += (k_cc_star + k_cd_star - 2.0 * k_dd_star) * n
    
    A3_term = (
        2.0 * (n - 1) * common**2
        - 3.0 * n * (
            4.0 * k_cd_star**2
            - 2.0 * k_cd_star * k_dd_star
            - k_dd_star**2
            - 2.0 * k_cd_star**2 * n
            - 2.0 * k_cd_star * k_dd_star * n
            + 5.0 * k_dd_star**2 * n
            - 3.0 * k_cc_star**2 * (1.0 + n)
            + 2.0 * k_cc_star * k_cd_star * (1.0 + n)
            + 2.0 * k0 * common
        )
    )
    A3_prime = (a**2 / (72.0 * k0**2 * n**3)) * A3_term
    
    A_prime = A1_prime + A2_prime + A3_prime
    
    return -B_prime / A_prime


def main():
    """Generate Fig. 2."""
    setup_academic_style()
    
    k_cd_star = np.linspace(-50, 50, 2000)
    
    param_sets = [
        {'k_cc_star': -5, 'color': '#B22222', 'linestyle': '-'},
        {'k_cc_star': -10, 'color': '#000080', 'linestyle': '--'},
        {'k_cc_star': -20, 'color': '#006400', 'linestyle': '-.'},
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.axhline(0, color='black', linestyle='-', linewidth=2, zorder=2)
    
    max_beta = 0.0
    
    for p in param_sets:
        beta_star = compute_beta_star_large_n(
            k_cd_star, p['k_cc_star'], 5.0
        )
        
        max_beta = max(max_beta, np.max(beta_star))
        
        ax.plot(k_cd_star, beta_star, color=p['color'], linewidth=3,
                linestyle=p['linestyle'],
                label=rf'$k_{{CC}}^* = {p["k_cc_star"]}$', zorder=4)
        
        gradient = np.gradient(beta_star)
        ascending_mask = (beta_star >= 0) & (gradient > 0)
        ax.fill_between(k_cd_star, beta_star, 0, where=ascending_mask,
                        color=p['color'], alpha=0.3, edgecolor='none', zorder=3)
    
    y_bottom = -0.001
    ax.fill_between(k_cd_star, y_bottom, 0, color='gray', alpha=0.1,
                    hatch='//////', edgecolor='gray', zorder=1)
    
    ax.set_xlabel(r'Breaking probability between cooperators and defectors, $k_{CD}^*$',
                  fontsize=18, labelpad=8)
    ax.set_ylabel(r'Critical selection intensity, $\beta^*$',
                  fontsize=18, labelpad=8)
    ax.set_xlim(-50, 50)
    ax.set_ylim(y_bottom, max_beta * 1.05)
    ax.tick_params(axis='both', which='major', length=7, labelsize=18)
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    
    out = Path("./figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_kcd_variation.jpg", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
