"""
Plotter for Fig. 4: Exact beta* vs. k_CD in decoupled rationality model.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.plot_utils import setup_academic_style


def compute_beta_star_exact(
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
    Compute exact critical selection intensity from full second-order expansion.
    """
    k_diff = k_cc_star - 2.0 * k_cd_star + k_dd_star
    k_sum = k_cc_star + k_cd_star - 2.0 * k_dd_star
    
    denom = 12.0 * c**2 * k0**2 * L**2 * (n - 2) * n**2
    
    t1 = 36.0 * c * k0**2 * L * n**2
    t2 = 6.0 * a * k0 * n * (k_diff + k_sum * n)
    t3 = -3.0 * k0 * L * n * (1.0 + n) * (
        4.0 * b * (k_cc_star - k_cd_star)
        + c * (
            4.0 * k_dd_star
            + 2.0 * k_cd_star * (n - 2)
            + k_cc_star * n
            - 3.0 * k_dd_star * n
        )
    )
    
    inner1 = k0**2 * n**2 * (
        -8.0 * c**2 * L**2 * (n - 2) * (
            -2.0 * k_diff**2
            + (7.0 * k_cc_star**2
               - 10.0 * k_cc_star * k_cd_star
               + 4.0 * k_cd_star**2
               + 8.0 * k_cc_star * k_dd_star
               - 22.0 * k_cd_star * k_dd_star
               + 13.0 * k_dd_star**2
               - 24.0 * k0 * k_diff) * n
            + (11.0 * k_cc_star**2
               - 14.0 * k_cc_star * k_cd_star
               - 4.0 * k_cd_star**2
               - 24.0 * k0 * k_sum
               + 4.0 * k_cc_star * k_dd_star
               + 34.0 * k_cd_star * k_dd_star
               - 31.0 * k_dd_star**2) * n**2
            + 2.0 * k_sum**2 * n**3
        )
    )
    
    inner2 = 3.0 * (
        4.0 * b * (k_cc_star - k_cd_star) * L * (1.0 + n)
        + 2.0 * a * (k_diff + k_sum * n)
        - c * L * (
            2.0 * k_cd_star * (n - 2) * (1.0 + n)
            + n * (-12.0 * k0 + k_cc_star + k_cc_star * n)
            + k_dd_star * (4.0 + n - 3.0 * n**2)
        )
    )**2
    
    val_inside_sqrt = inner1 + inner2
    sqrt_term = np.where(
        val_inside_sqrt < 0,
        np.nan,
        np.sqrt(3.0) * np.sqrt(np.maximum(val_inside_sqrt, 0.0))
    )
    
    return (t1 + t2 + t3 + sqrt_term) / denom


def main():
    """Generate Fig. 4."""
    setup_academic_style()
    
    k_cd_star = np.linspace(0.51, 0.53, 500)
    
    param_sets = [
        {
            'k_cc_star': 0.490,
            'k_dd_star': 0.505152,
            'color': '#000080',
            'marker': 'o',
            'label': r'$k_{CC}^* = 0.490$'
        },
        {
            'k_cc_star': 0.493,
            'k_dd_star': 0.505152,
            'color': '#006400',
            'marker': 's',
            'label': r'$k_{CC}^* = 0.493$'
        },
        {
            'k_cc_star': 0.497,
            'k_dd_star': 0.505152,
            'color': '#B22222',
            'marker': '^',
            'label': r'$k_{CC}^* = 0.497$'
        },
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    y_max_global = -np.inf
    y_min_global = np.inf
    
    for p in param_sets:
        beta_star = compute_beta_star_exact(
            k_cd_star, p['k_cc_star'], p['k_dd_star']
        )
        
        valid_mask = ~np.isnan(beta_star)
        x_valid = k_cd_star[valid_mask]
        y_valid = beta_star[valid_mask]
        
        if len(x_valid) == 0:
            continue
        
        y_max_global = max(y_max_global, np.max(y_valid))
        y_min_global = min(y_min_global, np.min(y_valid))
        
        ax.plot(x_valid, y_valid, color=p['color'], linewidth=2.5,
                alpha=0.9, zorder=3)
        
        ax.plot(x_valid, y_valid, color=p['color'], marker=p['marker'],
                markersize=7, markevery=20, linestyle='None',
                markerfacecolor='white', markeredgewidth=2,
                label=p['label'], zorder=4)
    
    margin = (y_max_global - y_min_global) * 0.15
    ax.set_xlim(0.51, 0.53)
    ax.set_ylim(y_min_global - margin, y_max_global + margin * 1.5)
    
    ax.set_xlabel(r'Breaking probability between cooperators and defectors, $k_{CD}^*$',
                  fontsize=18, labelpad=8)
    ax.set_ylabel(r'Critical selection intensity, $\beta^*$',
                  fontsize=18, labelpad=8)
    ax.tick_params(axis='both', which='major', length=7, labelsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    
    out = Path("./figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_decoupled_validation.jpg", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
