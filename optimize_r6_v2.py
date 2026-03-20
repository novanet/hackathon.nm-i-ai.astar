"""Quick R6 alpha optimization — precompute spatial prior once, then sweep alpha."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, observation_calibrated_transitions,
    apply_floor, HISTORICAL_TRANSITIONS, PROB_FLOOR,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Get calibrated transitions
cal_trans = observation_calibrated_transitions(ROUND_ID, detail, map_w, map_h)
print("Calibrated transitions for R6:")
names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
for r in range(6):
    row = cal_trans[r]
    top = sorted(range(6), key=lambda c: row[c], reverse=True)[:3]
    parts = [f"{names[c]}={row[c]:.1%}" for c in top]
    print(f"  {names[r]:>12} -> {', '.join(parts)}")

# Precompute spatial priors and transition priors for each seed
print("\nPrecomputing spatial priors (slow)...")
sp_cache = {}
tp_cache = {}
for seed_idx in range(n_seeds):
    print(f"  Seed {seed_idx}...")
    sp_cache[seed_idx] = spatial_prior(detail, seed_idx, map_w, map_h)
    tp_cache[seed_idx] = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)

# Load observations once
obs_cache = {}  # seed_idx -> list of (vy, vx, vh, vw, grid_classes)
obs_dir = "data/round_ae78003a"
for seed_idx in range(n_seeds):
    obs_list = []
    for fname in sorted(os.listdir(obs_dir)):
        if not fname.startswith(f"sim_s{seed_idx}_"):
            continue
        with open(os.path.join(obs_dir, fname)) as f:
            obs = json.load(f)
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        # Precompute class array
        classes = np.zeros((vh, vw), dtype=int)
        for dy in range(vh):
            for dx in range(vw):
                classes[dy, dx] = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
        obs_list.append((vy, vx, vh, vw, classes))
    obs_cache[seed_idx] = obs_list

# Alpha sweep
print("\n=== ALPHA SWEEP ===")
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

for alpha in alphas:
    total_ll = 0.0
    total_cells = 0
    
    for seed_idx in range(n_seeds):
        sp = sp_cache[seed_idx]
        tp = tp_cache[seed_idx]
        
        if sp is not None and alpha > 0:
            pred = alpha * sp + (1.0 - alpha) * tp
            pred = pred / pred.sum(axis=-1, keepdims=True)
        else:
            pred = tp.copy()
        pred = apply_floor(pred)
        
        for vy, vx, vh, vw, classes in obs_cache[seed_idx]:
            for dy in range(vh):
                for dx in range(vw):
                    cls = classes[dy, dx]
                    prob = pred[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    
    avg_ll = total_ll / max(1, total_cells)
    print(f"  alpha={alpha:.2f}: avg_ll={avg_ll:.4f}")

# Historical vs calibrated comparison
print("\n=== HISTORICAL vs CALIBRATED ===")
for label, trans_mat in [("Historical", HISTORICAL_TRANSITIONS), ("Calibrated", cal_trans)]:
    total_ll = 0.0
    total_cells = 0
    for seed_idx in range(n_seeds):
        tp = _apply_transition_matrix(detail, seed_idx, trans_mat, map_w, map_h)
        tp = apply_floor(tp)
        for vy, vx, vh, vw, classes in obs_cache[seed_idx]:
            for dy in range(vh):
                for dx in range(vw):
                    cls = classes[dy, dx]
                    prob = tp[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    avg_ll = total_ll / max(1, total_cells)
    print(f"  {label:>12}: avg_ll={avg_ll:.4f}")

print("\nDone!")
