"""Analyze repeat observations and resubmit R6 Pass 5 with all data."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, apply_floor,
    HISTORICAL_TRANSITIONS,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail
from astar.client import submit

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = "data/round_ae78003a"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Analyze repeat observations
print("=== REPEAT vs FIRST OBSERVATION (center viewport 13,13) ===")
for seed_idx in range(n_seeds):
    first_path = os.path.join(DATA_DIR, f"sim_s{seed_idx}_x13_y13.json")
    repeat_path = os.path.join(DATA_DIR, f"sim_s{seed_idx}_x13_y13_repeat.json")
    
    if not os.path.exists(first_path) or not os.path.exists(repeat_path):
        print(f"  Seed {seed_idx}: missing files")
        continue
    
    with open(first_path) as f:
        first = json.load(f)
    with open(repeat_path) as f:
        repeat = json.load(f)
    
    grid1 = np.array(first["grid"])
    grid2 = np.array(repeat["grid"])
    
    matches = (grid1 == grid2).sum()
    total = grid1.size
    diffs = total - matches
    print(f"  Seed {seed_idx}: {matches}/{total} match ({matches/total*100:.1f}%), {diffs} cells differ")

# Now build predictions with ALL 50 observations (45 original + 5 repeats)
# Strategy: 
# 1. Build observation frequency map (cells with 2 obs get empirical distribution)
# 2. Use calibrated transitions from all 50 obs
# 3. Blend: alpha * spatial + (1-alpha) * calibrated_transitions
# 4. For cells with observations: gently blend in empirical data

print("\n=== BUILDING IMPROVED PREDICTIONS WITH ALL 50 OBS ===")

ALPHA = 0.25
SMOOTHING = 0.5
FLOOR = 0.001
OBS_WEIGHT = 0.10

# Build calibrated transitions from ALL observations (load_simulations should pick up repeat files too)
# But load_simulations might not find the repeat files... Let me build transitions manually.

def load_all_observations(seed_idx: int):
    """Load all observation files for a seed, including repeats."""
    obs_list = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.startswith(f"sim_s{seed_idx}_"):
            continue
        if fn == "round_detail.json":
            continue
        fpath = os.path.join(DATA_DIR, fn)
        with open(fpath) as f:
            obs = json.load(f)
        obs_list.append(obs)
    return obs_list

# Build round-level transition matrix from ALL obs
obs_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
total_obs_cells = 0
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
                total_obs_cells += 1

prior_counts = HISTORICAL_TRANSITIONS * SMOOTHING
blended = prior_counts + obs_counts
row_sums = np.maximum(blended.sum(axis=1, keepdims=True), 1e-10)
cal_trans = blended / row_sums

print(f"Total observation cells: {total_obs_cells} (was 9680 with 45 queries)")
print("\nCalibrated transitions:")
names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
for r in range(6):
    row = cal_trans[r]
    top = sorted(range(6), key=lambda c: row[c], reverse=True)[:3]
    parts = [f"{names[c]}={row[c]:.1%}" for c in top]
    print(f"  {names[r]:>12} -> {', '.join(parts)}")

# Submit
print(f"\n=== SUBMITTING R6 PASS 5 (alpha={ALPHA}, obs_weight={OBS_WEIGHT}, 50 obs) ===")

for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    
    if sp is not None:
        pred = ALPHA * sp + (1.0 - ALPHA) * tp
        pred = pred / pred.sum(axis=-1, keepdims=True)
    else:
        pred = tp.copy()
    
    # Cell-level observation blending
    obs_freq = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    obs_count = np.zeros((map_h, map_w), dtype=np.float64)
    
    init_grid = detail["initial_states"][seed_idx]["grid"]
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                obs_freq[iy, ix, final_cls] += 1.0
                obs_count[iy, ix] += 1.0
    
    # Blend observations into prediction
    for y in range(map_h):
        for x in range(map_w):
            if obs_count[y, x] > 0:
                obs_dist = obs_freq[y, x] / obs_count[y, x]
                pred[y, x] = (1 - OBS_WEIGHT) * pred[y, x] + OBS_WEIGHT * obs_dist
    
    pred = apply_floor(pred, FLOOR)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
