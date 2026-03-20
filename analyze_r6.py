"""Analyze R6 observations to understand round characteristics and find improvement opportunities."""
import json
import os
import numpy as np
from collections import Counter

ROUND_DIR = "data/round_ae78003a"
ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"

# Load round detail
with open(os.path.join(ROUND_DIR, "round_detail.json")) as f:
    detail = json.load(f)

# Analyze initial states
print("=== INITIAL STATE ANALYSIS ===")
for seed_idx, state in enumerate(detail["initial_states"]):
    grid = np.array(state["grid"])
    setts = [s for s in state["settlements"] if s["alive"]]
    ports = [s for s in state["settlements"] if s.get("has_port")]
    
    # Count terrain types
    # Map internal codes to classes: 10,11,0->0, 1->1, 2->2, 3->3, 4->4, 5->5
    class_grid = np.zeros_like(grid)
    class_grid[grid == 0] = 0   # Empty
    class_grid[grid == 10] = 0  # Ocean -> Empty
    class_grid[grid == 11] = 0  # Plains -> Empty
    class_grid[grid == 1] = 1   # Settlement
    class_grid[grid == 2] = 2   # Port
    class_grid[grid == 3] = 3   # Ruin
    class_grid[grid == 4] = 4   # Forest
    class_grid[grid == 5] = 5   # Mountain
    
    counts = Counter(class_grid.flatten())
    total = class_grid.size
    print(f"\nSeed {seed_idx}: {len(setts)} settlements, {len(ports)} ports")
    for cls in range(6):
        name = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"][cls]
        pct = counts.get(cls, 0) / total * 100
        print(f"  {name}: {counts.get(cls, 0)} ({pct:.1f}%)")

# Analyze observations
print("\n\n=== OBSERVATION ANALYSIS ===")
for seed_idx in range(5):
    obs_files = sorted([f for f in os.listdir(ROUND_DIR) if f.startswith(f"sim_s{seed_idx}_")])
    if not obs_files:
        print(f"Seed {seed_idx}: no observations")
        continue
    
    all_terrain = Counter()
    for obs_file in obs_files:
        with open(os.path.join(ROUND_DIR, obs_file)) as f:
            obs = json.load(f)
        grid_obs = np.array(obs["grid"])
        # Map to classes
        class_obs = np.zeros_like(grid_obs)
        class_obs[grid_obs == 0] = 0
        class_obs[grid_obs == 10] = 0
        class_obs[grid_obs == 11] = 0
        class_obs[grid_obs == 1] = 1
        class_obs[grid_obs == 2] = 2
        class_obs[grid_obs == 3] = 3
        class_obs[grid_obs == 4] = 4
        class_obs[grid_obs == 5] = 5
        
        for c in class_obs.flatten():
            all_terrain[c] += 1
    
    total_obs = sum(all_terrain.values())
    print(f"\nSeed {seed_idx} ({len(obs_files)} viewports, {total_obs} cells observed):")
    for cls in range(6):
        name = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"][cls]
        pct = all_terrain.get(cls, 0) / total_obs * 100
        print(f"  {name}: {all_terrain.get(cls, 0)} ({pct:.1f}%)")

# Compute per-seed transition stats (initial -> observed)
print("\n\n=== TRANSITION ANALYSIS (initial -> observed) ===")
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Need to load initial grids properly
transitions = np.zeros((6, 6), dtype=int)

for seed_idx in range(5):
    state = detail["initial_states"][seed_idx]
    init_grid = np.array(state["grid"])
    
    # Map initial grid to classes
    init_class = np.zeros_like(init_grid)
    init_class[(init_grid == 0) | (init_grid == 10) | (init_grid == 11)] = 0
    init_class[init_grid == 1] = 1
    init_class[init_grid == 2] = 2
    init_class[init_grid == 3] = 3
    init_class[init_grid == 4] = 4
    init_class[init_grid == 5] = 5
    
    obs_files = sorted([f for f in os.listdir(ROUND_DIR) if f.startswith(f"sim_s{seed_idx}_")])
    for obs_file in obs_files:
        with open(os.path.join(ROUND_DIR, obs_file)) as f:
            obs = json.load(f)
        
        vp = obs["viewport"]
        vx, vy, vw, vh = vp["x"], vp["y"], vp["w"], vp["h"]
        obs_grid = np.array(obs["grid"])
        
        # Map obs to classes
        obs_class = np.zeros_like(obs_grid)
        obs_class[(obs_grid == 0) | (obs_grid == 10) | (obs_grid == 11)] = 0
        obs_class[obs_grid == 1] = 1
        obs_class[obs_grid == 2] = 2
        obs_class[obs_grid == 3] = 3
        obs_class[obs_grid == 4] = 4
        obs_class[obs_grid == 5] = 5
        
        for dy in range(vh):
            for dx in range(vw):
                init_c = init_class[vy + dy, vx + dx]
                obs_c = obs_class[dy, dx]
                transitions[init_c, obs_c] += 1

print("\nTransition matrix (rows=initial, cols=observed):")
print(f"{'':>12}", end="")
for c in range(6):
    print(f"{CLASS_NAMES[c]:>12}", end="")
print()

for r in range(6):
    row_total = transitions[r].sum()
    print(f"{CLASS_NAMES[r]:>12}", end="")
    for c in range(6):
        if row_total > 0:
            pct = transitions[r, c] / row_total * 100
            print(f"{pct:>11.1f}%", end="")
        else:
            print(f"{'N/A':>12}", end="")
    print(f"  (n={row_total})")

print("\n\nKey stats:")
print(f"  Empty->Empty: {transitions[0,0]/max(1,transitions[0].sum())*100:.1f}%")
print(f"  Settlement->Settlement: {transitions[1,1]/max(1,transitions[1].sum())*100:.1f}%")
print(f"  Forest->Forest: {transitions[4,4]/max(1,transitions[4].sum())*100:.1f}%")
print(f"  Port->Port: {transitions[2,2]/max(1,transitions[2].sum())*100:.1f}%")

# Compare to historical averages
print("\n\n=== COMPARISON TO HISTORICAL ===")
hist = {
    "Empty->Empty": 86.5,
    "Settlement->Settlement": 28.2,
    "Forest->Forest": 79.4,
    "Port->Port": 18.4,
}
obs_vals = {
    "Empty->Empty": transitions[0,0]/max(1,transitions[0].sum())*100,
    "Settlement->Settlement": transitions[1,1]/max(1,transitions[1].sum())*100,
    "Forest->Forest": transitions[4,4]/max(1,transitions[4].sum())*100,
    "Port->Port": transitions[2,2]/max(1,transitions[2].sum())*100,
}
for key in hist:
    diff = obs_vals[key] - hist[key]
    print(f"  {key}: observed={obs_vals[key]:.1f}% vs historical={hist[key]:.1f}% (diff={diff:+.1f}%)")
