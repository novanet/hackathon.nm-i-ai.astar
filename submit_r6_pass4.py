"""Submit R6 Pass 4: optimized alpha=0.25, lower smoothing, gentle cell-level obs blending."""
import numpy as np
import json
import os
from astar.model import (
    spatial_prior, _apply_transition_matrix, apply_floor,
    HISTORICAL_TRANSITIONS,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail, load_simulations
from astar.client import submit

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
ALPHA = 0.25      # Optimized for this round
SMOOTHING = 0.5   # Low smoothing (data dominates)
FLOOR = 0.001     # Conservative floor
OBS_WEIGHT = 0.10 # Gentle cell-level observation blending

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Compute calibrated transitions with low smoothing
def calibrate_transitions(smoothing: float):
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

cal_trans = calibrate_transitions(SMOOTHING)
print(f"Submitting R6 with alpha={ALPHA}, smoothing={SMOOTHING}, floor={FLOOR}, obs_weight={OBS_WEIGHT}")

for seed_idx in range(n_seeds):
    # Spatial model prior
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    
    # Transition prior from calibrated matrix
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    
    # Blend spatial + transitions
    if sp is not None:
        pred = ALPHA * sp + (1.0 - ALPHA) * tp
        pred = pred / pred.sum(axis=-1, keepdims=True)
    else:
        pred = tp.copy()
    
    # Cell-level observation blending
    if OBS_WEIGHT > 0:
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
        
        # For observed cells, blend obs frequency with model prediction
        for y in range(map_h):
            for x in range(map_w):
                if obs_count[y, x] > 0:
                    obs_dist = obs_freq[y, x] / obs_count[y, x]
                    pred[y, x] = (1 - OBS_WEIGHT) * pred[y, x] + OBS_WEIGHT * obs_dist
    
    pred = apply_floor(pred, FLOOR)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
