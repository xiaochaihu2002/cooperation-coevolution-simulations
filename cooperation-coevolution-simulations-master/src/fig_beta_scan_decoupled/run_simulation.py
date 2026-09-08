"""Monte Carlo simulation of the decoupled-rationality model (Fig. 3).

The rewiring rationality delta is held fixed while the strategy-updating
intensity beta is swept; the fixation probability is estimated for each
beta and stored together with the second-order theoretical prediction.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.simulation_core import (
    initialize_network,
    get_payoff,
    fermi_probability,
    rewire,
)
from .config import DecoupledConfig
from .theory import (
    calculate_coefficients_beta_expansion,
    theoretical_prediction,
    critical_intensity,
)


def single_trial(beta: float, cfg: DecoupledConfig) -> int:
    """Run one Monte Carlo trial of the decoupled model."""
    np.random.seed()

    G = initialize_network(cfg.N, cfg.L)
    adj = [set(G.neighbors(i)) for i in range(cfg.N)]

    strategies = np.zeros(cfg.N, dtype=int)
    strategies[np.random.randint(0, cfg.N)] = 1
    n_coop = 1

    k_struct = {
        "CC": cfg.k_cc_star,
        "CD": cfg.k_cd_star,
        "DD": cfg.k_dd_star,
    }

    for _ in range(cfg.max_steps):
        if np.random.random() < cfg.w:
            # Strategy update (uses beta).
            u = np.random.randint(0, cfg.N)
            neighbors = list(adj[u])
            if not neighbors:
                continue

            v = random.choice(neighbors)
            diff = get_payoff(v, strategies, adj, cfg.b, cfg.c)
            diff -= get_payoff(u, strategies, adj, cfg.b, cfg.c)

            if np.random.random() < fermi_probability(diff, beta):
                if strategies[u] != strategies[v]:
                    strategies[u] = strategies[v]
                    n_coop += 1 if strategies[u] == 1 else -1
        else:
            # Structure update (uses the fixed delta, independent of beta).
            u = np.random.randint(0, cfg.N)
            neighbors = list(adj[u])
            if not neighbors:
                continue

            v = random.choice(neighbors)
            su, sv = strategies[u], strategies[v]

            if su * sv == 1:
                k_star = k_struct["CC"]
            elif su + sv == 0:
                k_star = k_struct["DD"]
            else:
                k_star = k_struct["CD"]

            p_break = np.clip(cfg.k0 + cfg.delta * k_star, 0.0, 1.0)
            if np.random.random() < p_break:
                rewire(u, v, adj, cfg.N)

        if n_coop == 0:
            return 0
        if n_coop == cfg.N:
            return 1

    return None


def main() -> None:
    """Run the beta sweep and save the simulation table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None,
                        help="Monte Carlo trials per data point")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="number of parallel workers")
    args = parser.parse_args()

    cfg = DecoupledConfig()
    mc_trials = args.trials or cfg.mc_trials
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs

    coeffs = calculate_coefficients_beta_expansion(
        cfg.N, cfg.L, cfg.b, cfg.c, cfg.k0,
        cfg.k_cc_star, cfg.k_cd_star, cfg.k_dd_star,
    )
    beta_star = critical_intensity(cfg.delta, coeffs)
    print(f"Decoupled model | delta = {cfg.delta}")
    print(f"  A10 = {coeffs['A10']:+.6f}")
    print(f"  A01 = {coeffs['A01']:+.6f}")
    print(f"  B1  = {coeffs['B1']:+.6f}")
    print(f"  B01 = {coeffs['B01']:+.6f}")
    print(f"  B02 = {coeffs['B02']:+.6f}")
    print(f"  beta* = {beta_star:.6e}")

    rows = []
    for beta in tqdm(cfg.beta_range, desc="beta scan"):
        rho_1st, rho_2nd = theoretical_prediction(beta, cfg.delta, coeffs)

        trials = Parallel(n_jobs=cfg.n_jobs)(
            delayed(single_trial)(beta, cfg) for _ in range(mc_trials)
        )
        valid = [r for r in trials if r is not None]

        mean = np.mean(valid) if valid else 0.0
        se = np.sqrt(mean * (1 - mean) / len(valid)) if valid else 0.0

        rows.append({
            "beta": beta,
            "delta": cfg.delta,
            "sim_prob": mean,
            "sim_error": se,
            "theo_1st": rho_1st,
            "theo_2nd": rho_2nd,
        })

    df = pd.DataFrame(rows)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / cfg.filename, index=False)
    print(f"Saved to {out / cfg.filename}")


if __name__ == "__main__":
    main()
