"""
R18 emergency resubmission: blend simulator predictions for low-coverage seeds.

Seeds 1-4 have only 1 viewport each (~10% coverage). The model prior is good
but a spatially-aware simulator can add value for the unobserved 90%.

Strategy:
- For each seed, compute observation coverage
- For seeds with < 30% coverage: blend simulator predictions at unobserved cells
- Resubmit all 5 seeds
"""

import numpy as np
from pathlib import Path
from astar.client import get_round_detail, submit, get_budget, _request
from astar.model import (
    build_prediction, prediction_to_list, apply_floor,
    observation_calibrated_transitions, debias_transitions,
    HISTORICAL_TRANSITIONS, _cell_observation_frequencies,
)
from astar.replay import load_simulations, TERRAIN_TO_CLASS, NUM_CLASSES
from simulator import (
    params_from_transition_matrix, simulate_monte_carlo_vectorized,
    grid_to_numpy,
)

ROUND_ID = "b0f9d1bf-4b71-4e6e-816c-19c718d29056"
SIM_ALPHA_LOW_COV = 0.20  # blend weight for sim at UNOBSERVED cells only
N_SIMS = 500


def coverage_fraction(round_id: str, seed_index: int, map_w: int, map_h: int) -> tuple[float, np.ndarray]:
    """Returns (fraction of cells with observations, boolean mask of observed cells)."""
    obs_freq, obs_count = _cell_observation_frequencies(round_id, None, seed_index, map_w, map_h)
    observed = obs_count > 0
    frac = observed.sum() / (map_w * map_h)
    return frac, observed


def build_sim_prediction(detail: dict, seed_index: int, trans: np.ndarray,
                         map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """Run Monte Carlo simulator to get probability tensor."""
    params = params_from_transition_matrix(trans)
    init_grid = grid_to_numpy(detail["initial_states"][seed_index]["grid"], H=map_h, W=map_w)
    sim_pred = simulate_monte_carlo_vectorized(init_grid, params, n_sims=N_SIMS,
                                                H=map_h, W=map_w, seed=42)
    # Normalize
    sim_pred = np.maximum(sim_pred, 1e-10)
    sim_pred = sim_pred / sim_pred.sum(axis=-1, keepdims=True)
    return sim_pred


def main():
    print("=== R18 Emergency Resubmission ===")
    
    # Check budget
    budget = get_budget()
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']}")
    
    # Load round detail
    detail = get_round_detail(ROUND_ID)
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    print(f"Round: {map_w}x{map_h}, {n_seeds} seeds")
    
    # Get transition matrix for simulator
    cal_trans = observation_calibrated_transitions(ROUND_ID, detail, map_w, map_h)
    trans = debias_transitions(cal_trans) if cal_trans is not None else HISTORICAL_TRANSITIONS
    
    for seed_idx in range(n_seeds):
        # Check coverage
        cov_frac, observed_mask = coverage_fraction(ROUND_ID, seed_idx, map_w, map_h)
        print(f"\nSeed {seed_idx}: {cov_frac:.1%} coverage ({observed_mask.sum()}/{map_w*map_h} cells)")
        
        # Standard model prediction
        pred = build_prediction(ROUND_ID, detail, seed_idx, map_w, map_h)
        
        if cov_frac < 0.30:
            # Low coverage: blend simulator at unobserved cells
            print(f"  Low coverage → blending simulator (alpha={SIM_ALPHA_LOW_COV}) at unobserved cells")
            sim_pred = build_sim_prediction(detail, seed_idx, trans, map_w, map_h)
            
            # Only blend at UNOBSERVED cells (observed cells already have Bayesian overlay)
            unobserved = ~observed_mask
            pred[unobserved] = (
                (1 - SIM_ALPHA_LOW_COV) * pred[unobserved] +
                SIM_ALPHA_LOW_COV * sim_pred[unobserved]
            )
            # Re-normalize
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = apply_floor(pred)
        
        # Submit
        resp = submit(ROUND_ID, seed_idx, prediction_to_list(pred))
        status = resp.get("status", "?")
        score = resp.get("score", "?")
        print(f"  Submitted: {status} score={score}")
    
    # Final check
    my = _request("GET", "/my-rounds")
    for r in my:
        if r.get("round_id") == ROUND_ID or "b0f9d1bf" in str(r.get("round_id", "")):
            print(f"\nRound score: {r.get('round_score')}")
            print(f"Seed scores: {r.get('seed_scores')}")
            break
    
    print("\nDone!")


if __name__ == "__main__":
    main()
