"""R6 Pass 11 (FINAL): Best structural improvements with conservative obs_weight.
- sigma=1.5 (sharp distance weighting - nearby matters most)  
- Forest distance conditioning (marginal but principled)
- obs_weight=0.10 (conservative - proxy metric is biased toward higher values)
- floor=0.001 (safer than 0.0005 for KL scoring which punishes confident errors)
- Per-class spatial alpha
"""
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

# Conservative parameters
FLOOR = 0.001
OBS_WEIGHT = 0.10
SIGMA = 1.5
FOREST_THRESH = 3.0
CLASS_ALPHAS = {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.0}

DIST_BINS = [0, 2, 4, 7, 999]
n_bins = len(DIST_BINS) - 1

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

# Build distance × forest conditioned transition matrices
SMOOTHING = 0.5
near_forest_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
far_forest_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = (cls_grid == 4)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), 999.0)
    
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
                df = dist_forest[iy, ix]
                for b in range(n_bins):
                    if DIST_BINS[b] <= d < DIST_BINS[b+1]:
                        if df <= FOREST_THRESH:
                            near_forest_counts[b][init_cls, final_cls] += 1
                        else:
                            far_forest_counts[b][init_cls, final_cls] += 1
                        break

nf_trans = []
ff_trans = []
for b in range(n_bins):
    prior = HISTORICAL_TRANSITIONS * SMOOTHING
    nft = (prior + near_forest_counts[b]) / np.maximum((prior + near_forest_counts[b]).sum(axis=1, keepdims=True), 1e-10)
    fft = (prior + far_forest_counts[b]) / np.maximum((prior + far_forest_counts[b]).sum(axis=1, keepdims=True), 1e-10)
    nf_trans.append(nft)
    ff_trans.append(fft)

bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]

print(f"R6 Pass 11 (FINAL): sigma={SIGMA}, obs_weight={OBS_WEIGHT}, floor={FLOOR}")
print(f"  Forest conditioning: YES, per-class alpha: {CLASS_ALPHAS}")

# Pre-build obs data
obs_data = {}
for seed_idx in range(n_seeds):
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
    obs_data[seed_idx] = (obs_freq, obs_count)

for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = (cls_grid == 4)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), 999.0)
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = cls_grid[y, x]
            d = dist_sett[y, x]
            df = dist_forest[y, x]
            
            # Distance-weighted transition (smooth interpolation across bins)
            weights = np.array([np.exp(-((d - c) / SIGMA)**2) for c in bin_centers])
            weights /= weights.sum()
            
            # Forest-conditioned
            if df <= FOREST_THRESH:
                tp = sum(w * nf_trans[b][init_cls] for b, w in enumerate(weights))
            else:
                tp = sum(w * ff_trans[b][init_cls] for b, w in enumerate(weights))
            
            # Per-class spatial blend
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1 - alpha) * tp
            else:
                pred[y, x] = tp
    
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    
    # Conservative cell-level obs blending
    obs_freq, obs_count = obs_data[seed_idx]
    mask = obs_count > 0
    obs_dist = np.zeros_like(obs_freq)
    obs_dist[mask] = obs_freq[mask] / obs_count[mask, None]
    pred[mask] = (1 - OBS_WEIGHT) * pred[mask] + OBS_WEIGHT * obs_dist[mask]
    
    pred = apply_floor(pred, FLOOR)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
