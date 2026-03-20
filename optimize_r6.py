"""Quick R6 alpha optimization — use observations as proxy for ground truth to find best blend."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, observation_calibrated_transitions,
    apply_floor, HISTORICAL_TRANSITIONS, PROB_FLOOR, load_spatial_model,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Get calibrated transitions for this round
cal_trans = observation_calibrated_transitions(ROUND_ID, detail, map_w, map_h)
print("Calibrated transitions for R6:")
names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
for r in range(6):
    row = cal_trans[r]
    top = sorted(range(6), key=lambda c: row[c], reverse=True)[:3]
    parts = [f"{names[c]}={row[c]:.1%}" for c in top]
    print(f"  {names[r]:>12} -> {', '.join(parts)}")

# For each alpha, compute log-likelihood against observations
# (proxy for how well the model would match ground truth)
print("\n=== ALPHA SWEEP (observations as proxy) ===")

alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.0]
alpha_scores = {}

for alpha in alphas:
    total_ll = 0.0
    total_cells = 0
    
    for seed_idx in range(n_seeds):
        # Build predictions at this alpha
        sp = spatial_prior(detail, seed_idx, map_w, map_h)
        if cal_trans is not None:
            tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
        else:
            tp = _apply_transition_matrix(detail, seed_idx, HISTORICAL_TRANSITIONS, map_w, map_h)
        
        if sp is not None:
            pred = alpha * sp + (1.0 - alpha) * tp
            pred = pred / pred.sum(axis=-1, keepdims=True)
        else:
            pred = tp
        pred = apply_floor(pred)
        
        # Score against observations
        obs_dir = f"data/round_ae78003a"
        for fname in os.listdir(obs_dir):
            if not fname.startswith(f"sim_s{seed_idx}_"):
                continue
            with open(os.path.join(obs_dir, fname)) as f:
                obs = json.load(f)
            
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            
            for dy in range(vh):
                for dx in range(vw):
                    terrain = obs["grid"][dy][dx]
                    cls = TERRAIN_TO_CLASS.get(terrain, 0)
                    prob = pred[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    
    avg_ll = total_ll / max(1, total_cells)
    alpha_scores[alpha] = avg_ll
    print(f"  alpha={alpha:.2f}: avg log-lik = {avg_ll:.4f} ({total_cells} cells)")

best_alpha = max(alpha_scores, key=alpha_scores.get)
print(f"\n  BEST alpha = {best_alpha} (avg_ll = {alpha_scores[best_alpha]:.4f})")

# Also check: pure transitions (alpha=0) vs historical vs calibrated
print("\n=== TRANSITION COMPARISON (not blended) ===")
for label, trans_mat in [("Historical", HISTORICAL_TRANSITIONS), ("Calibrated", cal_trans)]:
    total_ll = 0.0
    total_cells = 0
    for seed_idx in range(n_seeds):
        tp = _apply_transition_matrix(detail, seed_idx, trans_mat, map_w, map_h)
        tp = apply_floor(tp)
        obs_dir = f"data/round_ae78003a"
        for fname in os.listdir(obs_dir):
            if not fname.startswith(f"sim_s{seed_idx}_"):
                continue
            with open(os.path.join(obs_dir, fname)) as f:
                obs = json.load(f)
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            for dy in range(vh):
                for dx in range(vw):
                    terrain = obs["grid"][dy][dx]
                    cls = TERRAIN_TO_CLASS.get(terrain, 0)
                    prob = tp[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    avg_ll = total_ll / max(1, total_cells)
    print(f"  {label:>12}: avg log-lik = {avg_ll:.4f}")
