"""
Stochastic cellular automaton simulator for Astar Island.

Models the Norse civilization as a spatial Markov process where each cell's
transition probability depends on its initial class AND its neighborhood.
Run N Monte Carlo simulations, average results → probability tensor.

Parameters per round (5 core):
  - expansion_rate: prob of empty/forest → settlement (scaled by proximity)
  - death_rate: prob of settlement → empty/forest  
  - forest_regrowth: prob of empty → forest
  - ruin_rate: prob of settlement → ruin
  - port_decay: prob of port → empty/forest

All spatially modulated by distance to nearest settlement.
"""

import json
import numpy as np
from pathlib import Path
from typing import NamedTuple

NUM_CLASSES = 6  # Empty, Settlement, Port, Ruin, Forest, Mountain
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
}
DATA_DIR = Path("data")


class SimParams(NamedTuple):
    """Parameters controlling one round of simulation."""
    # Settlement expansion
    expansion_base: float    # base prob of empty→settlement (at dist=1)
    expansion_decay: float   # spatial decay rate (higher = faster falloff)
    # Settlement death
    death_rate: float        # prob of settlement→empty
    forest_reclaim: float    # prob of settlement→forest (given it dies)
    # Forest dynamics
    forest_regrowth: float   # prob of empty→forest (near existing forest)
    forest_loss: float       # prob of forest→empty (not via settlement)
    # Ruin/Port
    ruin_rate: float         # prob of settlement→ruin (fraction of deaths)
    port_survival: float     # prob of port staying port


def compute_distance_map(init_grid: np.ndarray, target_class: int,
                          H: int, W: int) -> np.ndarray:
    """Compute minimum Euclidean distance from each cell to nearest target_class cell."""
    positions = np.argwhere(init_grid == target_class)  # (N, 2)
    if len(positions) == 0:
        return np.full((H, W), 999.0)
    
    yy, xx = np.mgrid[0:H, 0:W]
    # Use broadcasting: (H, W, 1) vs (1, 1, N)
    dy = yy[:, :, None] - positions[:, 0][None, None, :]  # (H, W, N)
    dx = xx[:, :, None] - positions[:, 1][None, None, :]
    dists = np.sqrt(dy**2 + dx**2)
    return dists.min(axis=2)  # (H, W)


def simulate_once(init_grid: np.ndarray, params: SimParams,
                   H: int = 40, W: int = 40,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Run one stochastic simulation. Returns (H, W) grid of final classes."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Precompute distance maps from initial state
    dist_to_sett = compute_distance_map(init_grid, 1, H, W)
    dist_to_forest = compute_distance_map(init_grid, 4, H, W)
    
    result = np.copy(init_grid)
    
    for y in range(H):
        for x in range(W):
            cls = init_grid[y, x]
            r = rng.random()
            
            if cls == 5:  # Mountain — immutable
                continue
            
            elif cls == 0:  # Empty
                # Can become: Settlement (expansion), Forest (regrowth), or stay Empty
                # Settlement expansion: depends on distance to nearest settlement
                d_sett = dist_to_sett[y, x]
                p_sett = params.expansion_base * np.exp(-params.expansion_decay * max(0, d_sett - 1))
                
                # Forest regrowth: depends on distance to nearest forest
                d_forest = dist_to_forest[y, x]
                p_forest = params.forest_regrowth * np.exp(-0.5 * max(0, d_forest - 1))
                
                # Port formation: very rare, near settlements and at edge
                edge_dist = min(y, x, H - 1 - y, W - 1 - x)
                p_port = 0.005 * p_sett if edge_dist <= 2 and d_sett <= 3 else 0.001 * p_sett
                
                # Ruin: tiny chance
                p_ruin = 0.01 * params.expansion_base
                
                # Normalize if overshoot
                p_stay = max(0, 1.0 - p_sett - p_forest - p_port - p_ruin)
                total = p_stay + p_sett + p_forest + p_port + p_ruin
                
                if r < p_stay / total:
                    result[y, x] = 0
                elif r < (p_stay + p_sett) / total:
                    result[y, x] = 1
                elif r < (p_stay + p_sett + p_port) / total:
                    result[y, x] = 2
                elif r < (p_stay + p_sett + p_port + p_ruin) / total:
                    result[y, x] = 3
                else:
                    result[y, x] = 4
            
            elif cls == 1:  # Settlement
                # Can die (→Empty), be reclaimed by forest, become ruin, or survive
                p_die = params.death_rate
                p_forest = params.forest_reclaim * p_die  # fraction of deaths → forest
                p_ruin = params.ruin_rate * p_die          # fraction of deaths → ruin
                p_empty = p_die - p_forest - p_ruin        # remaining deaths → empty
                p_stay = 1.0 - p_die
                
                if r < p_stay:
                    result[y, x] = 1
                elif r < p_stay + p_empty:
                    result[y, x] = 0
                elif r < p_stay + p_empty + p_forest:
                    result[y, x] = 4
                elif r < p_stay + p_empty + p_forest + p_ruin:
                    result[y, x] = 3
                else:
                    result[y, x] = 0  # fallback
            
            elif cls == 2:  # Port
                # Simplified: survives with port_survival, otherwise like settlement death
                if r < params.port_survival:
                    result[y, x] = 2
                else:
                    # Port dies — distribute like settlement death
                    p_remaining = 1.0 - params.port_survival
                    p_empty_frac = 0.5
                    p_forest_frac = 0.3
                    p_sett_frac = 0.1
                    p_ruin_frac = 0.1
                    r2 = (r - params.port_survival) / p_remaining
                    if r2 < p_empty_frac:
                        result[y, x] = 0
                    elif r2 < p_empty_frac + p_forest_frac:
                        result[y, x] = 4
                    elif r2 < p_empty_frac + p_forest_frac + p_sett_frac:
                        result[y, x] = 1
                    else:
                        result[y, x] = 3
            
            elif cls == 3:  # Ruin
                # Ruins are somewhat stable
                if r < 0.5:
                    result[y, x] = 3
                else:
                    result[y, x] = 0
            
            elif cls == 4:  # Forest
                # Can be colonized by settlements, cleared to empty, or stay
                d_sett = dist_to_sett[y, x]
                p_sett = params.expansion_base * 0.8 * np.exp(-params.expansion_decay * max(0, d_sett - 1))
                p_empty = params.forest_loss
                p_stay = max(0, 1.0 - p_sett - p_empty)
                total = p_stay + p_sett + p_empty
                
                if r < p_stay / total:
                    result[y, x] = 4
                elif r < (p_stay + p_sett) / total:
                    result[y, x] = 1
                else:
                    result[y, x] = 0
    
    return result


def simulate_monte_carlo(init_grid: np.ndarray, params: SimParams,
                          n_sims: int = 500, H: int = 40, W: int = 40,
                          seed: int | None = None) -> np.ndarray:
    """Run N simulations and average to get probability tensor.
    
    Returns: (H, W, NUM_CLASSES) probability tensor.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
    
    for i in range(n_sims):
        result = simulate_once(init_grid, params, H, W, rng)
        for c in range(NUM_CLASSES):
            counts[:, :, c] += (result == c)
    
    probs = counts / n_sims
    return probs


def simulate_monte_carlo_vectorized(init_grid: np.ndarray, params: SimParams,
                                     n_sims: int = 500, H: int = 40, W: int = 40,
                                     seed: int | None = None) -> np.ndarray:
    """Vectorized Monte Carlo — all cells processed in parallel per simulation.
    
    Returns: (H, W, NUM_CLASSES) probability tensor.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
    
    # Precompute distance maps (same for all sims — based on initial state)
    dist_to_sett = compute_distance_map(init_grid, 1, H, W)
    dist_to_forest = compute_distance_map(init_grid, 4, H, W)
    
    # Precompute per-cell transition probabilities (H, W, NUM_CLASSES)
    trans_probs = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
    
    for y in range(H):
        for x in range(W):
            cls = init_grid[y, x]
            
            if cls == 5:  # Mountain
                trans_probs[y, x, 5] = 1.0
                continue
            
            if cls == 0:  # Empty
                d_s = dist_to_sett[y, x]
                d_f = dist_to_forest[y, x]
                p_sett = params.expansion_base * np.exp(-params.expansion_decay * max(0, d_s - 1))
                p_forest = params.forest_regrowth * np.exp(-0.5 * max(0, d_f - 1))
                edge_dist = min(y, x, H - 1 - y, W - 1 - x)
                p_port = 0.005 * p_sett if edge_dist <= 2 and d_s <= 3 else 0.001 * p_sett
                p_ruin = 0.01 * params.expansion_base
                p_stay = max(0, 1.0 - p_sett - p_forest - p_port - p_ruin)
                trans_probs[y, x] = [p_stay, p_sett, p_port, p_ruin, p_forest, 0]
            
            elif cls == 1:  # Settlement
                p_die = params.death_rate
                p_forest = params.forest_reclaim * p_die
                p_ruin = params.ruin_rate * p_die
                p_empty = p_die - p_forest - p_ruin
                p_stay = 1.0 - p_die
                trans_probs[y, x] = [max(0, p_empty), max(0, p_stay), 0, max(0, p_ruin), max(0, p_forest), 0]
            
            elif cls == 2:  # Port
                ps = params.port_survival
                pr = 1.0 - ps
                trans_probs[y, x] = [pr * 0.5, pr * 0.1, ps, pr * 0.1, pr * 0.3, 0]
            
            elif cls == 3:  # Ruin
                trans_probs[y, x] = [0.5, 0.0, 0.0, 0.5, 0.0, 0]
            
            elif cls == 4:  # Forest
                d_s = dist_to_sett[y, x]
                p_sett = params.expansion_base * 0.8 * np.exp(-params.expansion_decay * max(0, d_s - 1))
                p_empty = params.forest_loss
                p_stay = max(0, 1.0 - p_sett - p_empty)
                trans_probs[y, x] = [p_empty, p_sett, 0, 0, p_stay, 0]
            
            # Normalize
            total = trans_probs[y, x].sum()
            if total > 0:
                trans_probs[y, x] /= total
    
    # Now run N simulations using precomputed probabilities
    # Each sim: draw random numbers, convert to class via cumulative probs
    cum_probs = np.cumsum(trans_probs, axis=-1)  # (H, W, 6)
    
    for _ in range(n_sims):
        r = rng.random((H, W))  # uniform [0,1) per cell
        # Determine which class each cell transitions to
        # class = first index where cum_probs > r
        classes = np.zeros((H, W), dtype=np.int32)
        for c in range(NUM_CLASSES):
            mask = (r > cum_probs[:, :, c]) if c > 0 else np.zeros((H, W), dtype=bool)
            classes[mask & (r <= cum_probs[:, :, c])] = c
        
        # Simpler: use searchsorted-like logic
        classes = (r[:, :, None] < cum_probs).argmax(axis=-1)  # (H, W)
        
        for c in range(NUM_CLASSES):
            counts[:, :, c] += (classes == c)
    
    probs = counts / n_sims
    return probs


def load_round_gt(round_id: str) -> tuple[dict, list[np.ndarray]]:
    """Load round detail and all GT arrays."""
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return {}, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt_key = "ground_truth" if "ground_truth" in gt_data else "prediction"
        gts.append(np.array(gt_data[gt_key], dtype=np.float64))
    
    return detail, gts


def grid_to_numpy(grid: list[list[int]], H: int = 40, W: int = 40) -> np.ndarray:
    """Convert API grid (list of lists of terrain IDs) to numpy class grid."""
    arr = np.zeros((H, W), dtype=np.int32)
    for y in range(H):
        for x in range(W):
            arr[y, x] = TERRAIN_TO_CLASS.get(grid[y][x], 0)
    return arr


def score_prediction(pred: np.ndarray, gt: np.ndarray) -> float:
    """Score a prediction against GT using entropy-weighted KL."""
    pred = np.clip(pred, 1e-10, 1.0)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    gt = np.clip(gt, 1e-10, 1.0)
    
    # Per-cell entropy of GT
    entropy = -np.sum(gt * np.log(gt), axis=-1)
    
    # Per-cell KL divergence
    kl = np.sum(gt * np.log(gt / pred), axis=-1)
    
    # Only dynamic cells (entropy > 0.01)
    dynamic = entropy > 0.01
    if dynamic.sum() == 0:
        return 100.0
    
    weights = entropy[dynamic] / entropy[dynamic].sum()
    weighted_kl = np.sum(weights * kl[dynamic])
    
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score


def fit_params_to_gt(detail: dict, gts: list[np.ndarray],
                      n_sims: int = 200, n_trials: int = 50,
                      seed: int = 42) -> tuple[SimParams, float]:
    """Fit simulator parameters to ground truth via random search.
    
    Returns: (best_params, best_score)
    """
    rng = np.random.default_rng(seed)
    H = detail.get("map_height", 40)
    W = detail.get("map_width", 40)
    
    best_params = None
    best_score = -1
    
    for trial in range(n_trials):
        # Random parameter sample
        params = SimParams(
            expansion_base=rng.uniform(0.01, 0.35),
            expansion_decay=rng.uniform(0.1, 0.8),
            death_rate=rng.uniform(0.3, 0.95),
            forest_reclaim=rng.uniform(0.2, 0.6),
            forest_regrowth=rng.uniform(0.01, 0.15),
            forest_loss=rng.uniform(0.01, 0.15),
            ruin_rate=rng.uniform(0.01, 0.15),
            port_survival=rng.uniform(0.05, 0.50),
        )
        
        scores = []
        for s_idx, gt in enumerate(gts[:2]):  # Use first 2 seeds for speed
            init_grid = grid_to_numpy(detail["initial_states"][s_idx]["grid"], H, W)
            pred = simulate_monte_carlo_vectorized(
                init_grid, params, n_sims=n_sims, H=H, W=W, seed=seed + s_idx
            )
            pred = np.maximum(pred, 0.0001)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            scores.append(score_prediction(pred, gt))
        
        avg = np.mean(scores)
        if avg > best_score:
            best_score = avg
            best_params = params
            if trial % 10 == 0:
                print(f"  Trial {trial}: score={avg:.2f} params={params}")
    
    return best_params, best_score


def params_from_transition_matrix(T: np.ndarray) -> SimParams:
    """Derive simulator parameters from a (6,6) transition matrix.
    
    Works with any transition source: GT-derived, observation-calibrated, or historical.
    """
    e_to_s = T[0, 1]
    s_to_s = T[1, 1]
    death_rate = 1.0 - s_to_s
    s_to_f = T[1, 4]
    s_to_r = T[1, 3]
    forest_reclaim = s_to_f / max(death_rate, 0.01)
    ruin_rate = s_to_r / max(death_rate, 0.01)
    f_to_e = T[4, 0]
    e_to_f = T[0, 4]
    p_to_p = T[2, 2]

    # expansion_base ~ E→S * 1.5 (distance avg correction)
    expansion_base = e_to_s * 1.5

    return SimParams(
        expansion_base=min(0.4, expansion_base),
        expansion_decay=0.1,   # calibrated: softer decay works best
        death_rate=max(0.05, min(0.98, death_rate)),
        forest_reclaim=max(0.05, min(0.8, forest_reclaim)),
        forest_regrowth=max(0.005, min(0.2, e_to_f)),
        forest_loss=max(0.005, min(0.2, f_to_e)),
        ruin_rate=max(0.005, min(0.2, ruin_rate)),
        port_survival=max(0.05, min(0.6, p_to_p)),
    )


def fit_params_from_transitions(detail: dict, gts: list[np.ndarray]) -> SimParams:
    """Derive simulator parameters analytically from GT transition matrix.
    
    This is much faster than Monte Carlo fitting — use as initial estimate.
    """
    H = detail.get("map_height", 40)
    W = detail.get("map_width", 40)
    
    transition_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s_idx, gt in enumerate(gts):
        init_grid = detail["initial_states"][s_idx]["grid"]
        for y in range(H):
            for x in range(W):
                cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                transition_counts[cls] += gt[y, x]
    
    row_sums = np.maximum(transition_counts.sum(axis=1, keepdims=True), 1.0)
    T = transition_counts / row_sums
    return params_from_transition_matrix(T)


def evaluate_simulator(n_sims: int = 300, use_analytical: bool = True,
                        verbose: bool = True) -> dict[int, float]:
    """Evaluate simulator on all rounds with GT. Returns {round_num: score}."""
    results = {}
    
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_gt(rid)
        if not gts:
            if verbose:
                print(f"R{rnum}: no GT data")
            continue
        
        H = detail.get("map_height", 40)
        W = detail.get("map_width", 40)
        
        if use_analytical:
            params = fit_params_from_transitions(detail, gts)
        else:
            params, _ = fit_params_to_gt(detail, gts, n_sims=n_sims // 2)
        
        if verbose:
            print(f"R{rnum}: params = exp_base={params.expansion_base:.3f}, "
                  f"exp_decay={params.expansion_decay:.3f}, "
                  f"death={params.death_rate:.3f}, "
                  f"forest_reclaim={params.forest_reclaim:.3f}, "
                  f"regrowth={params.forest_regrowth:.3f}")
        
        scores = []
        for s_idx, gt in enumerate(gts):
            init_grid = grid_to_numpy(detail["initial_states"][s_idx]["grid"], H, W)
            pred = simulate_monte_carlo_vectorized(
                init_grid, params, n_sims=n_sims, H=H, W=W, seed=42 + s_idx
            )
            pred = np.maximum(pred, 0.0001)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            sc = score_prediction(pred, gt)
            scores.append(sc)
        
        avg = np.mean(scores)
        results[rnum] = avg
        if verbose:
            print(f"  Score: {avg:.2f}  seeds: {[f'{s:.1f}' for s in scores]}")
    
    if verbose and results:
        print(f"\nAverage: {np.mean(list(results.values())):.2f}")
    
    return results


if __name__ == "__main__":
    print("=== Stochastic Simulator Evaluation ===\n")
    print("--- Analytical parameter fitting (from GT transitions) ---")
    results = evaluate_simulator(n_sims=300, use_analytical=True, verbose=True)
