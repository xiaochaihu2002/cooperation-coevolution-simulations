"""Plot Fig. 1."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.theory import generate_smooth_theory
from src.plot_utils import setup_academic_style


def main():
    """Generate Fig. 1."""
    csv_path = Path("./data/fig_validation.csv")
    df = pd.read_csv(csv_path)
    
    beta_dense, psi_1st, psi_2nd, A, B, beta_cross = generate_smooth_theory(
        beta_min=df['beta'].min(),
        beta_max=df['beta'].max(),
        n_points=2000,
        N=100, L=4, b=4.0, c=1.0, k0=0.5, a=0.5,
        k_cc_star=-3.0, k_cd_star=-1.0, k_dd_star=3.0
    )
    
    setup_academic_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xscale('log')
    
    rho0 = 1.0 / 100
    
    ax.plot(beta_dense, psi_2nd, color='#B22222', linewidth=3,
            label=r'Theoretical ($2^{\mathrm{nd}}$ Order)', zorder=2)
    ax.plot(beta_dense, psi_1st, color='#000080', linewidth=3,
            linestyle='--', label=r'Theoretical ($1^{\mathrm{st}}$ Order)', zorder=2)
    
    ax.errorbar(df['beta'], df['sim_prob'], yerr=df['sim_error'],
                fmt='o', color='black', ecolor='black', elinewidth=2,
                capsize=4, markersize=7, markerfacecolor='white',
                markeredgewidth=2, label='Simulation', zorder=3)
    
    ax.axhline(rho0, color='gray', linestyle=':', linewidth=2,
               label=rf'Neutral ($1/N = {rho0:.4f}$)', zorder=1)
    
    ax.set_xlabel(r'Selection Intensity, $\beta$', fontsize=18, labelpad=8)
    ax.set_ylabel(r'Fixation Probability, $\psi_C$', fontsize=18, labelpad=8)
    ax.tick_params(axis='both', which='major', length=7, labelsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.tight_layout()
    
    out = Path("./figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "fig_validation.svg", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
