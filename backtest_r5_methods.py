"""Backtest the R6-style approach (continuous distance weighting, forest conditioning, 
per-class alpha) on R5 ground truth to validate improvements vs our original submission."""
import numpy as np
import json
import os
import math
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from astar.model import (
    spatial_prior, apply_floor, build_prediction, 
    HISTORICAL_TRANSITIONS, observation_calibrated_transitions,
)
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

R5_ID = "fd3c92ff-3178-4dc9-8d9b-acf389b3982b"
DATA_DIR = Path("data/round_fd3c92ff-3178-4dc9-8d9b-acf389b3982b")

detail = load_round_detail(R5_ID)
map_w = detail["map_width"]
map_h = detail["map_height"]
n_seeds = len(detail["initial_states"])

def load_ground_truth(seed_idx):
    """Load ground truth from analysis API. Returns (H,W) array of class indices."""
    gt_file = DATA_DIR / f"analysis_s{seed_idx}"
    # Find the latest analysis file
    candidates = list(DATA_DIR.glob(f"analysis_s{seed_idx}_*.json"))
    if candidates:
        with open(candidates[-1]) as f:
            data = json.load(f)
        gt = data["ground_truth"]  # H x W x 6 one-hot
        # Convert to class indices
        return np.argmax(np.array(gt), axis=-1)
    # Fallback: query API
    from astar.client import get_analysis
    data = get_analysis(R5_ID, seed_idx)
    gt = data["ground_truth"]
    return np.argmax(np.array(gt), axis=-1)

def load_all_observations(seed_idx):
    obs_list = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.startswith(f"sim_s{seed_idx}_"):
            with open(os.path.join(DATA_DIR, fn)) as f:
                data = json.load(f)
            # Handle both formats: direct (R6-style) and wrapped (R5-style with request/response)
            if "response" in data and "viewport" in data["response"]:
                obs_list.append(data["response"])
            elif "viewport" in data:
                obs_list.append(data)
    return obs_list

def score_prediction(pred, gt_cls, init_grid):
    """Compute the actual competition score for a prediction vs ground truth.
    gt_cls: (H,W) array of class indices."""
    
    # Identify dynamic cells (cells that could change)
    total_kl = 0.0
    total_entropy_weight = 0.0
    
    for y in range(map_h):
        for x in range(map_w):
            p = pred[y, x]
            true_cls = gt_cls[y, x]
            
            # Entropy weight: -sum(p * log(p))
            entropy = 0.0
            for c in range(NUM_CLASSES):
                if p[c] > 1e-15:
                    entropy -= p[c] * math.log(p[c])
            
            # KL contribution: -log(p[true_cls])
            kl = -math.log(max(p[true_cls], 1e-15))
            
            total_kl += entropy * kl if entropy > 0 else 0
            total_entropy_weight += entropy
    
    if total_entropy_weight > 0:
        weighted_kl = total_kl / total_entropy_weight
    else:
        weighted_kl = 0.0
    
    score = max(0, min(100, 100 * math.exp(-3 * weighted_kl)))
    return score, weighted_kl

# ===================== METHOD 1: Original submission (alpha=0.85) =====================
print("=" * 60)
print("METHOD 1: Original submission (alpha=0.85, global transitions)")
for seed_idx in range(n_seeds):
    pred = build_prediction(R5_ID, detail, seed_idx, map_w, map_h)
    gt_cls = load_ground_truth(seed_idx)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    score, wkl = score_prediction(pred, gt_cls, init_grid)
    print("  Seed %d: score=%.2f (wkl=%.5f)" % (seed_idx, score, wkl))

# ===================== METHOD 2: R6-style (adaptive alpha) =====================
print()
print("=" * 60)
print("METHOD 2: R6-style (per-class alpha, proximity-conditioned, sigma=1.5)")

DIST_BINS = [0, 2, 4, 7, 999]
n_bins = len(DIST_BINS) - 1
SIGMA = 1.5
FOREST_THRESH = 3.0
# Use lower alphas since we build transitions from observations
CLASS_ALPHAS_LOW = {0: 0.3, 1: 0.1, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.0}

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = (cls_grid == 4)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), 999.0)
    
    # Build transition matrices from THIS seed's observations
    nf_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    ff_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                ic = cls_grid[iy, ix]
                fc = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                d = dist_sett[iy, ix]
                df = dist_forest[iy, ix]
                for b in range(n_bins):
                    if DIST_BINS[b] <= d < DIST_BINS[b+1]:
                        if df <= FOREST_THRESH:
                            nf_counts[b][ic, fc] += 1
                        else:
                            ff_counts[b][ic, fc] += 1
                        break
    
    SMOOTHING = 0.5
    nf_trans = []
    ff_trans = []
    for b in range(n_bins):
        prior = HISTORICAL_TRANSITIONS * SMOOTHING
        nft = (prior + nf_counts[b]) / np.maximum((prior + nf_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        fft = (prior + ff_counts[b]) / np.maximum((prior + ff_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        nf_trans.append(nft)
        ff_trans.append(fft)
    
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            ic = cls_grid[y, x]
            d = dist_sett[y, x]
            df = dist_forest[y, x]
            weights = np.array([np.exp(-((d - c) / SIGMA)**2) for c in bin_centers])
            weights /= weights.sum()
            
            if df <= FOREST_THRESH:
                tp = sum(w * nf_trans[b][ic] for b, w in enumerate(weights))
            else:
                tp = sum(w * ff_trans[b][ic] for b, w in enumerate(weights))
            
            alpha = CLASS_ALPHAS_LOW.get(ic, 0.2)
            if sp is not None and alpha > 0:
                pred[y, x] = alpha * sp[y, x] + (1 - alpha) * tp
            else:
                pred[y, x] = tp
    
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    
    # Obs blending
    OBS_WEIGHT = 0.10
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
                fc = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                obs_freq[iy, ix, fc] += 1.0
                obs_count[iy, ix] += 1.0
    
    mask = obs_count > 0
    obs_dist = np.zeros_like(obs_freq)
    obs_dist[mask] = obs_freq[mask] / obs_count[mask, None]
    pred[mask] = (1 - OBS_WEIGHT) * pred[mask] + OBS_WEIGHT * obs_dist[mask]
    
    pred = apply_floor(pred, 0.001)
    
    gt_cls = load_ground_truth(seed_idx)
    score, wkl = score_prediction(pred, gt_cls, init_grid)
    print("  Seed %d: score=%.2f (wkl=%.5f)" % (seed_idx, score, wkl))

# ===================== METHOD 3: High alpha (keep spatial model dominant) =====================
print()
print("=" * 60)
print("METHOD 3: High alpha (0.85) + continuous distance + forest conditioning")

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = (cls_grid == 4)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), 999.0)
    
    nf_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    ff_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                ic = cls_grid[iy, ix]
                fc = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                d = dist_sett[iy, ix]
                df = dist_forest[iy, ix]
                for b in range(n_bins):
                    if DIST_BINS[b] <= d < DIST_BINS[b+1]:
                        if df <= FOREST_THRESH:
                            nf_counts[b][ic, fc] += 1
                        else:
                            ff_counts[b][ic, fc] += 1
                        break
    
    SMOOTHING = 0.5
    nf_trans = []
    ff_trans = []
    for b in range(n_bins):
        prior = HISTORICAL_TRANSITIONS * SMOOTHING
        nft = (prior + nf_counts[b]) / np.maximum((prior + nf_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        fft = (prior + ff_counts[b]) / np.maximum((prior + ff_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        nf_trans.append(nft)
        ff_trans.append(fft)
    
    bin_centers = [(DIST_BINS[b] + min(DIST_BINS[b+1], 20)) / 2.0 for b in range(n_bins)]
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    ALPHA = 0.85
    for y in range(map_h):
        for x in range(map_w):
            ic = cls_grid[y, x]
            d = dist_sett[y, x]
            df = dist_forest[y, x]
            weights = np.array([np.exp(-((d - c) / SIGMA)**2) for c in bin_centers])
            weights /= weights.sum()
            
            if df <= FOREST_THRESH:
                tp = sum(w * nf_trans[b][ic] for b, w in enumerate(weights))
            else:
                tp = sum(w * ff_trans[b][ic] for b, w in enumerate(weights))
            
            if sp is not None:
                pred[y, x] = ALPHA * sp[y, x] + (1 - ALPHA) * tp
            else:
                pred[y, x] = tp
    
    sums = pred.sum(axis=-1, keepdims=True)
    pred = pred / np.maximum(sums, 1e-10)
    pred = apply_floor(pred, 0.001)
    
    gt_cls = load_ground_truth(seed_idx)
    score, wkl = score_prediction(pred, gt_cls, init_grid)
    print("  Seed %d: score=%.2f (wkl=%.5f)" % (seed_idx, score, wkl))
