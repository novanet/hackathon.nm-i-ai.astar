"""Evaluate all R6 approaches against observations."""
import numpy as np
import json
import os
from scipy.ndimage import distance_transform_edt
from astar.model import (
    spatial_prior, _apply_transition_matrix, apply_floor,
    HISTORICAL_TRANSITIONS,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = "data/round_ae78003a"
FLOOR = 0.001
OBS_WEIGHT = 0.10

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

# Build global calibrated transitions
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

# Build proximity-conditioned transitions
DIST_THRESHOLD = 3.0
near_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
far_counts_arr = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
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
                    far_counts_arr[init_cls, final_cls] += 1

near_prior = HISTORICAL_TRANSITIONS * 0.5
near_trans = (near_prior + near_counts) / np.maximum((near_prior + near_counts).sum(axis=1, keepdims=True), 1e-10)
far_prior = HISTORICAL_TRANSITIONS * 0.5
far_trans = (far_prior + far_counts_arr) / np.maximum((far_prior + far_counts_arr).sum(axis=1, keepdims=True), 1e-10)

# Precompute spatial priors
print("Precomputing spatial priors...")
sp_cache = {}
for seed_idx in range(n_seeds):
    sp_cache[seed_idx] = spatial_prior(detail, seed_idx, map_w, map_h)

# Load obs for eval
obs_eval = {}
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

CLASS_ALPHAS = {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.0}

def eval_approach(name, build_pred_fn):
    """Evaluate a prediction approach against observations."""
    total_ll = 0.0
    total_cells = 0
    for seed_idx in range(n_seeds):
        pred = build_pred_fn(seed_idx)
        
        # Cell-level obs blending
        obs_freq = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        obs_count_map = np.zeros((map_h, map_w), dtype=np.float64)
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
                    obs_count_map[iy, ix] += 1.0
        
        for y in range(map_h):
            for x in range(map_w):
                if obs_count_map[y, x] > 0:
                    obs_dist = obs_freq[y, x] / obs_count_map[y, x]
                    pred[y, x] = (1 - OBS_WEIGHT) * pred[y, x] + OBS_WEIGHT * obs_dist
        
        pred = apply_floor(pred, FLOOR)
        
        for vy, vx, vh, vw, classes in obs_eval[seed_idx]:
            for dy in range(vh):
                for dx in range(vw):
                    cls = classes[dy, dx]
                    prob = pred[vy + dy, vx + dx, cls]
                    total_ll += np.log(max(prob, 1e-10))
                    total_cells += 1
    
    avg_ll = total_ll / max(1, total_cells)
    print(f"  {name:>40}: avg_ll={avg_ll:.5f}")
    return avg_ll

# Approach 1: Global alpha=0.25 + cal transitions
def build_global_alpha(seed_idx):
    sp = sp_cache[seed_idx]
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    pred = 0.25 * sp + 0.75 * tp
    return pred / pred.sum(axis=-1, keepdims=True)

# Approach 2: Per-class alpha + cal transitions
def build_perclass_alpha(seed_idx):
    sp = sp_cache[seed_idx]
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1.0 - alpha) * tp[y, x]
            else:
                pred[y, x] = tp[y, x]
    sums = pred.sum(axis=-1, keepdims=True)
    return pred / np.maximum(sums, 1e-10)

# Approach 3: Proximity-conditioned transitions only
def build_proximity(seed_idx):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = cls_grid[y, x]
            if dist_sett[y, x] <= DIST_THRESHOLD:
                pred[y, x] = near_trans[init_cls]
            else:
                pred[y, x] = far_trans[init_cls]
    sums = pred.sum(axis=-1, keepdims=True)
    return pred / np.maximum(sums, 1e-10)

# Approach 4: Per-class alpha + proximity-conditioned for Empty/Forest
def build_hybrid(seed_idx):
    sp = sp_cache[seed_idx]
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = cls_grid[y, x]
            # Use proximity transitions as base
            if dist_sett[y, x] <= DIST_THRESHOLD:
                tp = near_trans[init_cls]
            else:
                tp = far_trans[init_cls]
            
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1.0 - alpha) * tp
            else:
                pred[y, x] = tp
    sums = pred.sum(axis=-1, keepdims=True)
    return pred / np.maximum(sums, 1e-10)

# Approach 5: Pure calibrated transitions (no spatial)
def build_pure_trans(seed_idx):
    tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    return tp

print("\n=== APPROACH COMPARISON (all with obs_weight=0.10) ===")
results = {}
results["Global alpha=0.25"] = eval_approach("Global alpha=0.25", build_global_alpha)
results["Per-class alpha"] = eval_approach("Per-class alpha", build_perclass_alpha)
results["Proximity-conditioned"] = eval_approach("Proximity-conditioned", build_proximity)
results["Hybrid (prox + spatial)"] = eval_approach("Hybrid (prox + spatial)", build_hybrid)
results["Pure calibrated transitions"] = eval_approach("Pure calibrated transitions", build_pure_trans)

best = max(results, key=results.get)
print(f"\n  BEST: {best} ({results[best]:.5f})")
