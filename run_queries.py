"""Run grid queries for active round and save observations."""
import json
import os
import time

from astar.client import simulate, get_budget

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
GRID = [(0,0),(13,0),(26,0),(0,13),(13,13),(26,13),(0,26),(13,26),(26,26)]

os.makedirs("data/round_ae78003a", exist_ok=True)

for seed in range(5):
    print(f"Seed {seed}:")
    for vx, vy in GRID:
        w = min(15, 40 - vx)
        h = min(15, 40 - vy)
        try:
            result = simulate(ROUND_ID, seed, vx, vy, w, h)
            fname = f"data/round_ae78003a/sim_s{seed}_x{vx}_y{vy}.json"
            with open(fname, "w") as f:
                json.dump(result, f)
            used = result.get("queries_used", "?")
            print(f"  ({vx},{vy}) ok, used={used}")
            time.sleep(0.25)
        except Exception as e:
            print(f"  ({vx},{vy}) FAILED: {e}")
            break

b = get_budget()
print(f"\nBudget: {b['queries_used']}/{b['queries_max']}")
