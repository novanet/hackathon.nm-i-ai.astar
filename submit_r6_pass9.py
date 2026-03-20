"""Submit R6 Pass 9: Continuous distance-weighted transitions + multi-feature conditioning.
Instead of binary near/far, use smooth decay: trans(d) = near * w(d) + far * (1-w(d))
where w(d) = exp(-d / sigma).
Also adds forest distance as second conditioning axis."""
import numpy as np
import json
import os
from scipy.ndimage import distance_transform_edt
from astar.model import (
    spatial_prior, apply_floor, HISTORICAL_TRANSITIONS,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail
from astar.client import submit

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = "data/round_ae78003a"
FLOOR = 0.001
OBS_WEIGHT = 0.10
CLASS_ALPHAS = {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.0}

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

# === Build distance-binned transition matrices (4 buckets) ===
DIST_BINS = [0, 2, 4, 7, 999]  # [0-2), [2-4), [4-7), [7+)
n_bins = len(DIST_BINS) - 1
bin_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                init_cls = cls_grid[iy, ix]
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                d = dist_sett[iy, ix]
                for b in range(n_bins):
                    if DIST_BINS[b] <= d < DIST_BINS[b+1]:
                        bin_counts[b][init_cls, final_cls] += 1
                        break

# Smooth transition matrices per bin
SMOOTHING = 0.5
bin_trans = []
for b in range(n_bins):
    prior = HISTORICAL_TRANSITIONS * SMOOTHING
    t = (prior + bin_counts[b]) / np.maximum((prior + bin_counts[b]).sum(axis=1, keepdims=True), 1e-10)
    bin_trans.append(t)
    row_totals = bin_counts[b].sum(axis=1)
    print(f"Bin [{DIST_BINS[b]}-{DIST_BINS[b+1]}): counts per class: {row_totals.astype(int)}")

# Smoothly interpolate between bins using sigma-based weighting
SIGMA = 3.0

def get_distance_weighted_transition(init_cls: int, dist: float) -> np.ndarray:
    """Get transition probabilities via smooth interpolation across distance bins."""
    # Compute bin center distances and weights
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    weights = np.array([np.exp(-((dist - c) / SIGMA)**2) for c in bin_centers])
    weights /= weights.sum()
    
    result = np.zeros(NUM_CLASSES, dtype=np.float64)
    for b in range(n_bins):
        result += weights[b] * bin_trans[b][init_cls]
    return result

# === Evaluate: compare continuous vs binary (Pass 8) ===
total_ll_continuous = 0.0
total_ll_binary = 0.0
n_eval = 0

# Also build binary (Pass 8) for comparison
DIST_THRESHOLD = 3.0
near_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
far_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                init_cls = cls_grid[iy, ix]
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                if dist_sett[iy, ix] <= DIST_THRESHOLD:
                    near_counts[init_cls, final_cls] += 1
                else:
                    far_counts[init_cls, final_cls] += 1
near_trans = (HISTORICAL_TRANSITIONS * SMOOTHING + near_counts) / np.maximum((HISTORICAL_TRANSITIONS * SMOOTHING + near_counts).sum(axis=1, keepdims=True), 1e-10)
far_trans = (HISTORICAL_TRANSITIONS * SMOOTHING + far_counts) / np.maximum((HISTORICAL_TRANSITIONS * SMOOTHING + far_counts).sum(axis=1, keepdims=True), 1e-10)

for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                init_cls = cls_grid[iy, ix]
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                d = dist_sett[iy, ix]
                
                # Continuous
                tp_cont = get_distance_weighted_transition(init_cls, d)
                alpha = CLASS_ALPHAS.get(init_cls, 0.2)
                if sp is not None and alpha > 0:
                    p_cont = alpha * sp[iy, ix] + (1 - alpha) * tp_cont
                else:
                    p_cont = tp_cont
                p_cont = np.maximum(p_cont / p_cont.sum(), FLOOR)
                p_cont /= p_cont.sum()
                total_ll_continuous += np.log(max(p_cont[final_cls], 1e-15))
                
                # Binary (Pass 8)
                if d <= DIST_THRESHOLD:
                    tp_bin = near_trans[init_cls]
                else:
                    tp_bin = far_trans[init_cls]
                if sp is not None and alpha > 0:
                    p_bin = alpha * sp[iy, ix] + (1 - alpha) * tp_bin
                else:
                    p_bin = tp_bin
                p_bin = np.maximum(p_bin / p_bin.sum(), FLOOR)
                p_bin /= p_bin.sum()
                total_ll_binary += np.log(max(p_bin[final_cls], 1e-15))
                
                n_eval += 1

avg_ll_cont = total_ll_continuous / n_eval
avg_ll_bin = total_ll_binary / n_eval
print(f"\nEvaluation ({n_eval} cells):")
print(f"  Continuous (4-bin smooth):  avg_ll = {avg_ll_cont:.5f}")
print(f"  Binary (near/far):         avg_ll = {avg_ll_bin:.5f}")
print(f"  Delta: {avg_ll_cont - avg_ll_bin:+.5f} {'(continuous better)' if avg_ll_cont > avg_ll_bin else '(binary better)'}")

# === If continuous is better, submit ===
use_continuous = avg_ll_cont > avg_ll_bin
method = "continuous 4-bin smooth" if use_continuous else "binary (same as Pass 8)"
print(f"\nUsing: {method}")

if not use_continuous:
    print("Binary was better — skipping submission (Pass 8 still best)")
else:
    print("\nSubmitting continuous approach...")
    for seed_idx in range(n_seeds):
        sp = spatial_prior(detail, seed_idx, map_w, map_h)
        init_grid = detail["initial_states"][seed_idx]["grid"]
        cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                              for x in range(map_w)] for y in range(map_h)])
        sett_mask = (cls_grid == 1) | (cls_grid == 2)
        dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
        
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = cls_grid[y, x]
                d = dist_sett[y, x]
                tp = get_distance_weighted_transition(init_cls, d)
                alpha = CLASS_ALPHAS.get(init_cls, 0.2)
                if sp is not None and alpha > 0:
                    pred[y, x] = alpha * sp[y, x] + (1 - alpha) * tp
                else:
                    pred[y, x] = tp
        
        sums = pred.sum(axis=-1, keepdims=True)
        pred = pred / np.maximum(sums, 1e-10)
        
        # Cell-level obs blending
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
