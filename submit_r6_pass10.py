"""R6 Pass 10: Multi-distance conditioning (settlement dist + forest dist) + optimize sigma/bins.
Also try ensembling multiple approaches."""
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

# === Try multiple sigma values for distance weighting ===
DIST_BINS = [0, 2, 4, 7, 999]
n_bins = len(DIST_BINS) - 1

# Collect all observations with metadata for quick evaluation
print("Loading observations...")
eval_data = []  # (seed, init_cls, final_cls, dist_sett, dist_forest, sp_vec)
for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
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
                sp_vec = sp[iy, ix] if sp is not None else None
                eval_data.append((seed_idx, init_cls, final_cls, dist_sett[iy, ix], dist_forest[iy, ix], sp_vec))

print(f"Total eval cells: {len(eval_data)}")

# Build bin counts
bin_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
for _, init_cls, final_cls, ds, _, _ in eval_data:
    for b in range(n_bins):
        if DIST_BINS[b] <= ds < DIST_BINS[b+1]:
            bin_counts[b][init_cls, final_cls] += 1
            break

SMOOTHING = 0.5
bin_trans = []
for b in range(n_bins):
    prior = HISTORICAL_TRANSITIONS * SMOOTHING
    t = (prior + bin_counts[b]) / np.maximum((prior + bin_counts[b]).sum(axis=1, keepdims=True), 1e-10)
    bin_trans.append(t)

# === Sweep sigma ===
print("\n=== Sigma sweep ===")
best_sigma = None
best_ll = -np.inf

for sigma in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    
    total_ll = 0.0
    for _, init_cls, final_cls, ds, df, sp_vec in eval_data:
        weights = np.array([np.exp(-((ds - c) / sigma)**2) for c in bin_centers])
        weights /= weights.sum()
        tp = sum(w * bin_trans[b][init_cls] for b, w in enumerate(weights))
        alpha = CLASS_ALPHAS.get(init_cls, 0.2)
        if sp_vec is not None and alpha > 0:
            p = alpha * sp_vec + (1 - alpha) * tp
        else:
            p = tp
        p = np.maximum(p / p.sum(), FLOOR)
        p /= p.sum()
        total_ll += np.log(max(p[final_cls], 1e-15))
    
    avg_ll = total_ll / len(eval_data)
    print(f"  sigma={sigma:.1f}: avg_ll={avg_ll:.5f}")
    if avg_ll > best_ll:
        best_ll = avg_ll
        best_sigma = sigma

print(f"\nBest sigma: {best_sigma} (avg_ll={best_ll:.5f})")

# === Try forest distance as additional conditioning ===
print("\n=== Forest distance conditioning ===")
# Split into 2 forest bins: near-forest (<3) vs far-forest (>=3)
FOREST_THRESH = 3.0
near_forest_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
far_forest_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]

for _, init_cls, final_cls, ds, df, _ in eval_data:
    for b in range(n_bins):
        if DIST_BINS[b] <= ds < DIST_BINS[b+1]:
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

# Evaluate with forest conditioning
total_ll_forest = 0.0
for _, init_cls, final_cls, ds, df, sp_vec in eval_data:
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    weights = np.array([np.exp(-((ds - c) / best_sigma)**2) for c in bin_centers])
    weights /= weights.sum()
    
    if df <= FOREST_THRESH:
        tp = sum(w * nf_trans[b][init_cls] for b, w in enumerate(weights))
    else:
        tp = sum(w * ff_trans[b][init_cls] for b, w in enumerate(weights))
    
    alpha = CLASS_ALPHAS.get(init_cls, 0.2)
    if sp_vec is not None and alpha > 0:
        p = alpha * sp_vec + (1 - alpha) * tp
    else:
        p = tp
    p = np.maximum(p / p.sum(), FLOOR)
    p /= p.sum()
    total_ll_forest += np.log(max(p[final_cls], 1e-15))

avg_ll_forest = total_ll_forest / len(eval_data)
print(f"  With forest conditioning: avg_ll={avg_ll_forest:.5f}")
print(f"  Without forest:           avg_ll={best_ll:.5f}")
print(f"  Delta: {avg_ll_forest - best_ll:+.5f}")

# === Try different obs_weight values ===
print("\n=== Obs weight sweep ===")
# For this we need to rebuild per-cell predictions with obs blending
# Quick eval: just use the transition-level predictions + obs blend

best_ow = None
best_ll_ow = -np.inf

# Precompute obs frequencies per cell per seed
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

# Precompute base predictions (no obs blending)
base_preds = {}
for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    for y in range(map_h):
        for x in range(map_w):
            init_cls = cls_grid[y, x]
            d = dist_sett[y, x]
            weights = np.array([np.exp(-((d - c) / best_sigma)**2) for c in bin_centers])
            weights /= weights.sum()
            tp = sum(w * bin_trans[b][init_cls] for b, w in enumerate(weights))
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1 - alpha) * tp
            else:
                pred[y, x] = tp
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    base_preds[seed_idx] = pred

for ow in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    total_ll = 0.0
    n = 0
    for seed_idx, init_cls, final_cls, ds, df, sp_vec in eval_data:
        # Find cell coords (approximate from ds + seed info)
        # Actually we need the actual cell coords for obs blending
        pass
    
    # Simpler: evaluate on observed cells only
    total_ll = 0.0
    n = 0
    for seed_idx in range(n_seeds):
        pred = base_preds[seed_idx].copy()
        obs_freq, obs_count = obs_data[seed_idx]
        if ow > 0:
            mask = obs_count > 0
            obs_dist = np.zeros_like(obs_freq)
            obs_dist[mask] = obs_freq[mask] / obs_count[mask, None]
            pred[mask] = (1 - ow) * pred[mask] + ow * obs_dist[mask]
        
        pred = np.maximum(pred, FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        # Evaluate on observed cells
        for obs in load_all_observations(seed_idx):
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            for dy in range(vh):
                for dx in range(vw):
                    iy, ix = vy + dy, vx + dx
                    if iy >= map_h or ix >= map_w:
                        continue
                    final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                    total_ll += np.log(max(pred[iy, ix, final_cls], 1e-15))
                    n += 1
    
    avg_ll = total_ll / n
    print(f"  obs_weight={ow:.2f}: avg_ll={avg_ll:.5f}")
    if avg_ll > best_ll_ow:
        best_ll_ow = avg_ll
        best_ow = ow

print(f"\nBest obs_weight: {best_ow} (avg_ll={best_ll_ow:.5f})")

# === Try different floor values ===
print("\n=== Floor sweep ===")
best_floor = None
best_ll_floor = -np.inf

for floor in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
    total_ll = 0.0
    n = 0
    for seed_idx in range(n_seeds):
        pred = base_preds[seed_idx].copy()
        obs_freq, obs_count = obs_data[seed_idx]
        if best_ow > 0:
            mask = obs_count > 0
            obs_dist = np.zeros_like(obs_freq)
            obs_dist[mask] = obs_freq[mask] / obs_count[mask, None]
            pred[mask] = (1 - best_ow) * pred[mask] + best_ow * obs_dist[mask]
        pred = np.maximum(pred, floor)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        for obs in load_all_observations(seed_idx):
            vp = obs["viewport"]
            vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
            for dy in range(vh):
                for dx in range(vw):
                    iy, ix = vy + dy, vx + dx
                    if iy >= map_h or ix >= map_w:
                        continue
                    final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                    total_ll += np.log(max(pred[iy, ix, final_cls], 1e-15))
                    n += 1
    avg_ll = total_ll / n
    print(f"  floor={floor}: avg_ll={avg_ll:.5f}")
    if avg_ll > best_ll_floor:
        best_ll_floor = avg_ll
        best_floor = floor

print(f"\nBest floor: {best_floor} (avg_ll={best_ll_floor:.5f})")

# === Submit with best parameters ===
print(f"\n=== FINAL SUBMISSION ===")
print(f"sigma={best_sigma}, obs_weight={best_ow}, floor={best_floor}")
use_forest = avg_ll_forest > best_ll
print(f"Forest conditioning: {'YES' if use_forest else 'NO'}")

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
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    
    for y in range(map_h):
        for x in range(map_w):
            init_cls = cls_grid[y, x]
            d = dist_sett[y, x]
            df = dist_forest[y, x]
            weights = np.array([np.exp(-((d - c) / best_sigma)**2) for c in bin_centers])
            weights /= weights.sum()
            
            if use_forest and df <= FOREST_THRESH:
                tp = sum(w * nf_trans[b][init_cls] for b, w in enumerate(weights))
            elif use_forest:
                tp = sum(w * ff_trans[b][init_cls] for b, w in enumerate(weights))
            else:
                tp = sum(w * bin_trans[b][init_cls] for b, w in enumerate(weights))
            
            alpha = CLASS_ALPHAS.get(init_cls, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1 - alpha) * tp
            else:
                pred[y, x] = tp
    
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    
    # Obs blending
    if best_ow > 0:
        obs_freq, obs_count = obs_data[seed_idx]
        mask = obs_count > 0
        obs_dist = np.zeros_like(obs_freq)
        obs_dist[mask] = obs_freq[mask] / obs_count[mask, None]
        pred[mask] = (1 - best_ow) * pred[mask] + best_ow * obs_dist[mask]
    
    pred = apply_floor(pred, best_floor)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
