"""Submit R6 Pass 6: per-class alpha optimization."""
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
FLOOR = 0.001
OBS_WEIGHT = 0.10

# Per-initial-class optimal alphas (from per-class sweep)
CLASS_ALPHAS = {
    0: 0.3,   # Empty: spatial model has some value for position-dependent expansion
    1: 0.1,   # Settlement
    2: 0.0,   # Port: pure transitions
    3: 0.0,   # Ruin
    4: 0.1,   # Forest
    5: 0.0,   # Mountain: never changes, transitions = 100%
}

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

def load_all_observations(seed_idx):
    obs_list = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.startswith(f"sim_s{seed_idx}_"):
            with open(os.path.join(DATA_DIR, fn)) as f:
                obs_list.append(json.load(f))
    return obs_list

# Build calibrated transitions
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

print(f"Submitting R6 Pass 6 with per-class alphas: {CLASS_ALPHAS}")

for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    
    # Per-cell blend using class-specific alpha
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1.0 - alpha) * tp[y, x]
            else:
                pred[y, x] = tp[y, x]
    
    # Normalize
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    
    # Cell-level observation blending
    obs_freq = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    obs_count = np.zeros((map_h, map_w), dtype=np.float64)
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
    
    for y in range(map_h):
        for x in range(map_w):
            if obs_count[y, x] > 0:
                obs_dist = obs_freq[y, x] / obs_count[y, x]
                pred[y, x] = (1 - OBS_WEIGHT) * pred[y, x] + OBS_WEIGHT * obs_dist
    
    pred = apply_floor(pred, FLOOR)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
