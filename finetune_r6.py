"""Fine-tune R6: smoothing, floor, and cell-level observation blending."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, apply_floor,
    HISTORICAL_TRANSITIONS, load_spatial_model,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail, load_simulations

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Precompute spatial priors
print("Precomputing spatial priors...")
sp_cache = {}
for seed_idx in range(n_seeds):
    sp_cache[seed_idx] = spatial_prior(detail, seed_idx, map_w, map_h)
    print(f"  Seed {seed_idx} done")

# Load observations
obs_cache = {}
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
        classes = np.zeros((vh, vw), dtype=int)
        for dy in range(vh):
            for dx in range(vw):
                classes[dy, dx] = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
        obs_list.append((vy, vx, vh, vw, classes))
    obs_cache[seed_idx] = obs_list

def calibrate_transitions(smoothing: float):
    """Get calibrated transition matrix with given smoothing."""
    obs_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for seed_idx in range(n_seeds):
        sims = load_simulations(ROUND_ID, seed_idx)
        if not sims:
            continue
        init_grid = detail["initial_states"][seed_idx]["grid"]
        for sim in sims:
            req = sim["request"]
            resp = sim["response"]
            vx, vy = req["viewport_x"], req["viewport_y"]
            for dy, row in enumerate(resp["grid"]):
                for dx, terrain_code in enumerate(row):
                    iy, ix = vy + dy, vx + dx
                    if iy >= map_h or ix >= map_w:
                        continue
                    init_cls = TERRAIN_TO_CLASS.get(init_grid[iy][ix], 0)
                    final_cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                    obs_counts[init_cls, final_cls] += 1
    prior_counts = HISTORICAL_TRANSITIONS * smoothing
    blended = prior_counts + obs_counts
    row_sums = np.maximum(blended.sum(axis=1, keepdims=True), 1e-10)
    return blended / row_sums

def eval_config(alpha, smoothing, floor):
    """Evaluate a configuration against observations."""
    cal_trans = calibrate_transitions(smoothing)
    total_ll = 0.0
    total_cells = 0
    
    for seed_idx in range(n_seeds):
        sp = sp_cache[seed_idx]
        tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
        
        if sp is not None and alpha > 0:
            pred = alpha * sp + (1.0 - alpha) * tp
            pred = pred / pred.sum(axis=-1, keepdims=True)
        else:
            pred = tp.copy()
        pred = apply_floor(pred, floor)
        
        for vy, vx, vh, vw, classes in obs_cache[seed_idx]:
            for dy in range(vh):
                for dx in range(vw):
                    cls = classes[dy, dx]
                    prob = pred[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    
    return total_ll / max(1, total_cells)

# Finer alpha sweep around 0.2
print("\n=== FINE ALPHA SWEEP (smoothing=5, floor=0.001) ===")
for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    ll = eval_config(alpha, 5.0, 0.001)
    print(f"  alpha={alpha:.2f}: {ll:.5f}")

# Smoothing sweep
print("\n=== SMOOTHING SWEEP (alpha=0.2, floor=0.001) ===")
for smoothing in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
    ll = eval_config(0.2, smoothing, 0.001)
    print(f"  smoothing={smoothing:.1f}: {ll:.5f}")

# Floor sweep
print("\n=== FLOOR SWEEP (alpha=0.2, smoothing=5) ===")
for floor in [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05]:
    ll = eval_config(0.2, 5.0, floor)
    print(f"  floor={floor:.4f}: {ll:.5f}")

# Now try cell-level observation blending on top
print("\n=== CELL-LEVEL OBS BLENDING ===")
# For observed cells, blend directly with the observation (one-hot)
# For unobserved cells, use the alpha-blend approach
for obs_weight in [0.0, 0.1, 0.2, 0.3, 0.5]:
    cal_trans = calibrate_transitions(5.0)
    total_ll = 0.0
    total_cells = 0
    
    for seed_idx in range(n_seeds):
        sp = sp_cache[seed_idx]
        tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
        
        alpha = 0.2
        if sp is not None:
            pred = alpha * sp + (1.0 - alpha) * tp
            pred = pred / pred.sum(axis=-1, keepdims=True)
        else:
            pred = tp.copy()
        
        # Build observation frequency map for this seed
        obs_freq = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        obs_count = np.zeros((map_h, map_w), dtype=np.float64)
        
        sims = load_simulations(ROUND_ID, seed_idx)
        if sims:
            init_grid = detail["initial_states"][seed_idx]["grid"]
            for sim in sims:
                req = sim["request"]
                resp = sim["response"]
                vx, vy = req["viewport_x"], req["viewport_y"]
                for dy, row in enumerate(resp["grid"]):
                    for dx, terrain_code in enumerate(row):
                        iy, ix = vy + dy, vx + dx
                        if iy >= map_h or ix >= map_w:
                            continue
                        final_cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                        obs_freq[iy, ix, final_cls] += 1.0
                        obs_count[iy, ix] += 1.0
        
        # Blend: for observed cells, mix obs frequency with model prediction
        if obs_weight > 0:
            for y in range(map_h):
                for x in range(map_w):
                    if obs_count[y, x] > 0:
                        obs_dist = obs_freq[y, x] / obs_count[y, x]
                        pred[y, x] = (1 - obs_weight) * pred[y, x] + obs_weight * obs_dist
        
        pred = apply_floor(pred, 0.001)
        
        for vy, vx, vh, vw, classes in obs_cache[seed_idx]:
            for dy in range(vh):
                for dx in range(vw):
                    cls = classes[dy, dx]
                    prob = pred[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    
    avg_ll = total_ll / max(1, total_cells)
    print(f"  obs_weight={obs_weight:.1f}: avg_ll={avg_ll:.5f}")

print("\nDone!")
