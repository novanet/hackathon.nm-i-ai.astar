"""Submit R6 Pass 7: Proximity-conditioned transitions learned from THIS round's observations."""
import numpy as np
import json
import os
from scipy.ndimage import distance_transform_edt
from astar.model import apply_floor, HISTORICAL_TRANSITIONS
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail
from astar.client import submit

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

# Build proximity-conditioned transition matrices
# Two buckets: near settlements (dist <= 3) and far (dist > 3)
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

# Normalize with smoothing
SMOOTHING = 0.5
for label, counts in [("Near", near_counts), ("Far", far_counts)]:
    prior = HISTORICAL_TRANSITIONS * SMOOTHING
    blended = prior + counts
    row_sums = np.maximum(blended.sum(axis=1, keepdims=True), 1e-10)
    trans = blended / row_sums
    
    if label == "Near":
        near_trans = trans
    else:
        far_trans = trans
    
    names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    print(f"\n{label} settlements (dist {'<=' if label == 'Near' else '>'} {DIST_THRESHOLD}):")
    for r in range(6):
        row = trans[r]
        top = sorted(range(6), key=lambda c: row[c], reverse=True)[:3]
        parts = [f"{names[c]}={row[c]:.1%}" for c in top]
        n = int(counts[r].sum())
        print(f"  {names[r]:>12} -> {', '.join(parts)}  (n={n})")

# Build predictions using proximity-conditioned transitions
print(f"\n=== SUBMITTING R6 PASS 7 (proximity-conditioned transitions) ===")

for seed_idx in range(n_seeds):
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
