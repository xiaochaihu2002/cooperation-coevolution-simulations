"""Run simulation for Fig. 1."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from src.theory import compute_coefficients
from src.simulation_core import initialize_network, get_payoff, fermi_probability, rewire
from .config import ValidationConfig


def single_trial(beta: float, cfg: ValidationConfig) -> int:
    """Run one MC trial for coupled model."""
    np.random.seed()
    
    G = initialize_network(cfg.N, cfg.L)
    adj = [set(G.neighbors(i)) for i in range(cfg.N)]
    
    strategies = np.zeros(cfg.N, dtype=int)
    initial_coop = np.random.randint(0, cfg.N)
    strategies[initial_coop] = 1
    n_coop = 1
    
    k_struct = {
        'CC': cfg.k_cc_star,
        'CD': cfg.k_cd_star,
        'DD': cfg.k_dd_star
    }
    
    for step in range(cfg.max_steps):
        if np.random.random() < cfg.w:
            # Strategy update
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
            # Structure update
            u = np.random.randint(0, cfg.N)
            neighbors = list(adj[u])
            if not neighbors:
                continue
            
            v = random.choice(neighbors)
            su, sv = strategies[u], strategies[v]
            
            if su * sv == 1:
                k_star = k_struct['CC']
            elif su + sv == 0:
                k_star = k_struct['DD']
            else:
                k_star = k_struct['CD']
            
            p_break = cfg.k0 + cfg.a * beta * k_star
            p_break = np.clip(p_break, 0.0, 1.0)
            
            if np.random.random() < p_break:
                rewire(u, v, adj, cfg.N)
        
        if n_coop == 0:
            return 0
        if n_coop == cfg.N:
            return 1
    
    return None


def main():
    """Run Fig. 1 simulation."""
    cfg = ValidationConfig()
    cfg.k_cc_star = -3.0
    cfg.k_cd_star = -1.0
    cfg.k_dd_star = 3.0
    
    A, B = compute_coefficients(
        cfg.N, cfg.L, cfg.b, cfg.c, cfg.k0, cfg.a,
        cfg.k_cc_star, cfg.k_cd_star, cfg.k_dd_star
    )
    print(f"A = {A:.6f}, B = {B:.6f}")
    
    results = []
    for beta in tqdm(cfg.beta_range):
        rho_1st = 1.0 / cfg.N + A * beta
        rho_2nd = 1.0 / cfg.N + A * beta + B * beta**2
        
        sim_results = Parallel(n_jobs=cfg.n_jobs)(
            delayed(single_trial)(beta, cfg) for _ in range(cfg.mc_trials)
        )
        valid = [r for r in sim_results if r is not None]
        
        mean = np.mean(valid) if valid else 0.0
        se = np.sqrt(mean * (1 - mean) / len(valid)) if valid else 0.0
        
        results.append({
            'beta': beta,
            'sim_prob': mean,
            'sim_error': se,
            'theo_1st': rho_1st,
            'theo_2nd': rho_2nd,
        })
    
    df = pd.DataFrame(results)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / cfg.filename, index=False)
    print(f"Saved to {out / cfg.filename}")


if __name__ == "__main__":
    main()
