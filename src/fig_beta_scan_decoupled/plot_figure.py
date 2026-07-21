"""Plot Fig. 3."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plot_utils import setup_academic_style
from .theory import calculate_coefficients_beta_expansion, theoretical_prediction


def main():
    """Generate Fig. 3."""
    csv_path = Path("./data/fig_beta_scan_decoupled.csv")
    df = pd.read_csv(csv_path)
    
    delta = df['delta'].iloc[0]
    
    coeffs = calculate_coefficients_beta_expansion(
        N=100, L=4, b=4.0, c=2.0, k0=0.5,
        k_cc_star=-3.0, k_cd_star=2.0, k_dd_star=4.0
    )
    
    beta_dense = np.logspace(
        np.log10(df['beta'].min()),
        np.log10(df['beta'].max()),
        2000
    )
    
    rho_1st_dense = []
    rho_2nd_dense = []
    for beta in beta_dense:
        r1, r2 = theoretical_prediction(beta, delta, coeffs)
        rho_1st_dense.append(r1)
        rho_2nd_dense.append(r2)
    
    setup_academic_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.set_xscale('log')
    
    rho0 = 1.0 / 100
    
    ax.plot(beta_dense, rho_2nd_dense, color='#B22222', linewidth=3,
            label=r'Theoretical ($2^{\mathrm{nd}}$ Order)', zorder=2)
    
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
    plt.savefig(out / "fig_beta_scan_decoupled.svg", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
