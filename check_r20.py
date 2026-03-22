"""Quick check R20 status and details."""
import os, json, sys
from astar.client import get_rounds, get_round_detail

rounds = get_rounds()
r20 = [r for r in rounds if r['round_number'] == 20][0]
print("R20 info:")
print(f"  ID: {r20['id']}")
print(f"  Status: {r20['status']}")
print(f"  Closes at: {r20['closes_at']}")
print(f"  Weight: {r20['round_weight']}")

det = get_round_detail(r20['id'])
ns = len(det.get('initial_states', []))
w = det['map_width']
h = det['map_height']
print(f"  Grid: {w}x{h}")
print(f"  Seeds: {ns}")
print(f"  Top keys: {sorted(det.keys())}")

# Show initial state structure
if ns > 0:
    s0 = det['initial_states'][0]
    grid = s0.get('grid', s0.get('initial_grid', []))
    print(f"  Seed0 keys: {sorted(s0.keys())}")
    if grid:
        import numpy as np
        g = np.array(grid)
        print(f"  Grid shape: {g.shape}")
        unique, counts = np.unique(g, return_counts=True)
        for u, c in zip(unique, counts):
            pct = 100 * c / g.size
            print(f"    Class {u}: {c} ({pct:.1f}%)")

# Check budget
from astar.client import _request
try:
    budget = _request("GET", "/budget")
    print(f"\nBudget: {json.dumps(budget, indent=2)}")
except Exception as e:
    print(f"Budget error: {e}")

# Check R19 score details
print("\n=== R19 final score ===")
r19 = [r for r in rounds if r['round_number'] == 19][0]
print(f"  R19 ID: {r19['id']}")
print(f"  R19 status: {r19['status']}")
