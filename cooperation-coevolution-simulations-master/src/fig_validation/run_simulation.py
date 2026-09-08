"""Monte Carlo simulation for the validation figure (Fig. 2).

For each k_CD* in {6, -2} the fixation probability of a single cooperator
is estimated over a log-spaced beta grid in the coupled model
(delta = a*beta) and stored together with the first- and second-order
theoretical predictions.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from src.theory import compute_coefficients
from src.simulation_core import (
    initialize_network,
    get_payoff,
    fermi_probability,
    rewire,
)
from .config import ValidationConfig


def single_trial(beta: float, cfg: ValidationConfig) -> int:
    """Run one Monte Carlo trial of the coupled model."""
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
            # Strategy update.
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
            # Structure update with delta = a * beta.
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

            p_break = np.clip(cfg.k0 + cfg.a * beta * k_star, 0.0, 1.0)
            if np.random.random() < p_break:
                rewire(u, v, adj, cfg.N)

        if n_coop == 0:
            return 0
        if n_coop == cfg.N:
            return 1

    return None


def run_case(cfg: ValidationConfig, k_cd_star: float,
             mc_trials: int) -> list:
    """Simulate one k_CD* case and return the data rows."""
    cfg.k_cd_star = k_cd_star

    A, B = compute_coefficients(
        cfg.N, cfg.L, cfg.b, cfg.c, cfg.k0, cfg.a,
        cfg.k_cc_star, cfg.k_cd_star, cfg.k_dd_star,
    )
    print(f"k_CD* = {k_cd_star}: A = {A:.6f}, B = {B:.6f}, "
          f"beta* = {-A / B:.6f}")

    rows = []
    for beta in tqdm(cfg.beta_range, desc=f"k_CD* = {k_cd_star}"):
        rho_1st = 1.0 / cfg.N + A * beta
        rho_2nd = 1.0 / cfg.N + A * beta + B * beta**2

        trials = Parallel(n_jobs=cfg.n_jobs)(
            delayed(single_trial)(beta, cfg) for _ in range(mc_trials)
        )
        valid = [r for r in trials if r is not None]

        mean = np.mean(valid) if valid else 0.0
        se = np.sqrt(mean * (1 - mean) / len(valid)) if valid else 0.0

        rows.append({
            "beta": beta,
            "k_cd_star": k_cd_star,
            "sim_prob": mean,
            "sim_error": se,
            "theo_1st": rho_1st,
            "theo_2nd": rho_2nd,
        })
    return rows


def main() -> None:
    """Run the two simulation cases and save the combined table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=None,
                        help="Monte Carlo trials per data point")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="number of parallel workers")
    args = parser.parse_args()

    cfg = ValidationConfig()
    mc_trials = args.trials or cfg.mc_trials
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs

    all_rows = []
    for k_cd_star in cfg.k_cd_star_values:
        all_rows.extend(run_case(cfg, float(k_cd_star), mc_trials))

    df = pd.DataFrame(all_rows)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / cfg.filename, index=False)
    print(f"Saved to {out / cfg.filename}")


if __name__ == "__main__":
    main()
