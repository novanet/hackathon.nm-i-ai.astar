"""Use remaining 5 queries for repeat observations on high-value viewports."""
import json
import os
import numpy as np
from astar.client import simulate, get_budget
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
DATA_DIR = f"data/round_ae78003a"

# Check budget
budget = get_budget()
print(f"Budget: {budget}")

remaining = budget.get("queries_max", 50) - budget.get("queries_used", 50)
if remaining <= 0:
    print("No queries remaining!")
    exit()

print(f"\n{remaining} queries available")

# Strategy: repeat-observe the most "interesting" viewport for multiple seeds
# The center viewport (13,13) covers the most settlement-dense area
# Use 5 queries: one repeat per seed on viewport (13,13) 
# This gives us 2 observations per cell in center, allowing empirical distribution

queries = []
for seed_idx in range(min(5, remaining)):
    queries.append((seed_idx, 13, 13))

print(f"\nPlanned queries: {len(queries)}")
for seed_idx, vx, vy in queries:
    print(f"  Seed {seed_idx}: viewport ({vx}, {vy})")

input_str = input("\nProceed? (y/n): ") if False else "y"

for seed_idx, vx, vy in queries:
    print(f"\nQuerying seed {seed_idx} at ({vx},{vy})...")
    result = simulate(ROUND_ID, seed_idx, vx, vy)
    
    # Save
    fname = f"sim_s{seed_idx}_x{vx}_y{vy}_repeat.json"
    with open(os.path.join(DATA_DIR, fname), "w") as f:
        json.dump(result, f)
    
    # Quick analysis: compare with first observation
    first_fname = None
    for fn in os.listdir(DATA_DIR):
        if fn.startswith(f"sim_s{seed_idx}_x{vx}_y{vy}_") and "repeat" not in fn:
            first_fname = fn
            break
    
    if first_fname:
        with open(os.path.join(DATA_DIR, first_fname)) as f:
            first_obs = json.load(f)
        
        # Count matches
        grid1 = np.array(first_obs["grid"])
        grid2 = np.array(result["grid"])
        matches = (grid1 == grid2).sum()
        total = grid1.size
        print(f"  Match with first obs: {matches}/{total} ({matches/total*100:.1f}%)")
        
        # Count differences by class
        diffs = (grid1 != grid2)
        n_diffs = diffs.sum()
        print(f"  {n_diffs} cells differ between observations")
    
    print(f"  Status: saved to {fname}")

print(f"\nDone! Used {len(queries)} queries.")
