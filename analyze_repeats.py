"""Analyze repeat observations and resubmit with all 50 queries of data."""
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
DATA_DIR = "data/round_ae78003a"

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

# Check repeat observation format
repeat_file = os.path.join(DATA_DIR, "sim_s0_x13_y13_repeat.json")
with open(repeat_file) as f:
    repeat_data = json.load(f)
print("Repeat observation keys:", list(repeat_data.keys()))

# Find matching first obs
first_files = {}
for seed_idx in range(n_seeds):
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.startswith(f"sim_s{seed_idx}_x13_y13_") and "repeat" not in fn:
            first_files[seed_idx] = fn
            break

print("\n=== REPEAT vs FIRST OBSERVATION ===")
for seed_idx in range(n_seeds):
    repeat_fn = f"sim_s{seed_idx}_x13_y13_repeat.json"
    repeat_path = os.path.join(DATA_DIR, repeat_fn)
    if not os.path.exists(repeat_path):
        continue
    
    with open(repeat_path) as f:
        repeat = json.load(f)
    
    # Get the grid from repeat - check format
    if "grid" in repeat:
        grid2 = np.array(repeat["grid"])
    elif "response" in repeat and "grid" in repeat["response"]:
        grid2 = np.array(repeat["response"]["grid"])
    else:
        print(f"Seed {seed_idx}: unknown repeat format, keys={list(repeat.keys())}")
        continue
    
    if seed_idx in first_files:
        first_path = os.path.join(DATA_DIR, first_files[seed_idx])
        with open(first_path) as f:
            first = json.load(f)
        
        if "grid" in first:
            grid1 = np.array(first["grid"])
        elif "response" in first and "grid" in first["response"]:
            grid1 = np.array(first["response"]["grid"])
        else:
            print(f"Seed {seed_idx}: unknown first format")
            continue
        
        matches = (grid1 == grid2).sum()
        total = grid1.size
        print(f"  Seed {seed_idx}: {matches}/{total} match ({matches/total*100:.1f}%)")
        
        # Per-class disagreement
        for cls_code in [0, 10, 11, 1, 2, 3, 4, 5]:
            in1 = (grid1 == cls_code).sum()
            in2 = (grid2 == cls_code).sum()
            if in1 > 0 or in2 > 0:
                name = {0: "Empty", 10: "Ocean", 11: "Plains", 1: "Sett", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}.get(cls_code, str(cls_code))
                print(f"    {name}: first={in1}, repeat={in2}, diff={in2-in1:+d}")
    else:
        print(f"  Seed {seed_idx}: no first observation found")

print("\nDone!")
