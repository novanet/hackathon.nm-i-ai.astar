"""Per-class alpha optimization: maybe spatial model is good for some classes but bad for others."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, apply_floor,
    HISTORICAL_TRANSITIONS,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = "data/round_ae78003a"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Build calibrated transitions
def load_all_observations(seed_idx):
    obs_list = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.startswith(f"sim_s{seed_idx}_"):
            with open(os.path.join(DATA_DIR, fn)) as f:
                obs_list.append(json.load(f))
    return obs_list

obs_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                init_cls = TERRAIN_TO_CLASS.get(init_grid[iy][ix], 0)
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                obs_counts[init_cls, final_cls] += 1

prior_counts = HISTORICAL_TRANSITIONS * 0.5
blended = prior_counts + obs_counts
row_sums = np.maximum(blended.sum(axis=1, keepdims=True), 1e-10)
cal_trans = blended / row_sums

# Precompute spatial priors
print("Precomputing spatial priors...")
sp_cache = {}
for seed_idx in range(n_seeds):
    sp_cache[seed_idx] = spatial_prior(detail, seed_idx, map_w, map_h)

# Load observation data for evaluation
obs_eval = {}  # seed -> list of (vy, vx, vh, vw, classes)
for seed_idx in range(n_seeds):
    obs_list = []
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        classes = np.zeros((vh, vw), dtype=int)
        for dy in range(vh):
            for dx in range(vw):
                classes[dy, dx] = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
        obs_list.append((vy, vx, vh, vw, classes))
    obs_eval[seed_idx] = obs_list

# Evaluate per-initial-class: for each initial class, what alpha is best?
NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
print("\n=== PER-INITIAL-CLASS ALPHA OPTIMIZATION ===")

for init_cls in range(6):
    best_alpha = 0.0
    best_ll = -999
    
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        total_ll = 0.0
        total_cells = 0
        
        for seed_idx in range(n_seeds):
            sp = sp_cache[seed_idx]
            tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
            init_grid = detail["initial_states"][seed_idx]["grid"]
            
            if sp is not None and alpha > 0:
                pred = alpha * sp + (1.0 - alpha) * tp
                pred = pred / pred.sum(axis=-1, keepdims=True)
            else:
                pred = tp.copy()
            pred = apply_floor(pred, 0.001)
            
            for vy, vx, vh, vw, classes in obs_eval[seed_idx]:
                for dy in range(vh):
                    for dx in range(vw):
                        iy, ix = vy + dy, vx + dx
                        cell_init = TERRAIN_TO_CLASS.get(init_grid[iy][ix], 0)
                        if cell_init != init_cls:
                            continue
                        cls = classes[dy, dx]
                        prob = pred[iy, ix, cls]
                        total_ll += np.log(max(prob, 1e-10))
                        total_cells += 1
        
        if total_cells > 0:
            avg_ll = total_ll / total_cells
            if avg_ll > best_ll:
                best_ll = avg_ll
                best_alpha = alpha
    
    print(f"  {NAMES[init_cls]:>12}: best_alpha={best_alpha:.1f}, avg_ll={best_ll:.4f}")

print("\nDone!")
