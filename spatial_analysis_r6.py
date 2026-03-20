"""Explore position-dependent transition patterns to find local structure."""
import numpy as np
import json
import os
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, load_round_detail

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = "data/round_ae78003a"

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

# For empty cells specifically: do cells near settlements have different outcomes
# than isolated empty cells? This would tell us if the spatial model adds value.

print("=== EMPTY CELL ANALYSIS: PROXIMITY TO SETTLEMENTS ===")
# Bucket: near-settlement (within 3) vs far (>3)
near_trans = np.zeros((NUM_CLASSES,), dtype=np.float64)
far_trans = np.zeros((NUM_CLASSES,), dtype=np.float64)

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    
    # Build distance to nearest settlement
    from scipy.ndimage import distance_transform_edt
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999)
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                if cls_grid[iy, ix] != 0:  # only empty cells
                    continue
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                if dist_sett[iy, ix] <= 3:
                    near_trans[final_cls] += 1
                else:
                    far_trans[final_cls] += 1

names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
print("\nNear settlements (dist <= 3):")
near_total = near_trans.sum()
for c in range(6):
    pct = near_trans[c] / max(1, near_total) * 100
    print(f"  {names[c]:>12}: {pct:.1f}% (n={int(near_trans[c])})")

print("\nFar from settlements (dist > 3):")
far_total = far_trans.sum()
for c in range(6):
    pct = far_trans[c] / max(1, far_total) * 100
    print(f"  {names[c]:>12}: {pct:.1f}% (n={int(far_trans[c])})")

# Forest cells: near settlement vs far
print("\n=== FOREST CELL ANALYSIS: PROXIMITY TO SETTLEMENTS ===")
near_f = np.zeros((NUM_CLASSES,), dtype=np.float64)
far_f = np.zeros((NUM_CLASSES,), dtype=np.float64)

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999)
    
    for obs in load_all_observations(seed_idx):
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        for dy in range(vh):
            for dx in range(vw):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                if cls_grid[iy, ix] != 4:  # only forest cells
                    continue
                final_cls = TERRAIN_TO_CLASS.get(obs["grid"][dy][dx], 0)
                if dist_sett[iy, ix] <= 3:
                    near_f[final_cls] += 1
                else:
                    far_f[final_cls] += 1

print("\nForest near settlements (dist <= 3):")
near_total = near_f.sum()
for c in range(6):
    pct = near_f[c] / max(1, near_total) * 100
    print(f"  {names[c]:>12}: {pct:.1f}% (n={int(near_f[c])})")

print("\nForest far from settlements (dist > 3):")
far_total = far_f.sum()
for c in range(6):
    pct = far_f[c] / max(1, far_total) * 100
    print(f"  {names[c]:>12}: {pct:.1f}% (n={int(far_f[c])})")
