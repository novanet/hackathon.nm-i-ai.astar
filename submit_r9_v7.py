"""Resubmit R9 with V7 + settlement stats fix."""
import importlib
import numpy as np

# Force fresh model reload
import astar.model as mod
mod._spatial_model = None
importlib.reload(mod)

from astar.model import build_prediction, prediction_to_list
from astar.client import get_round_detail, submit, _request

ROUND_ID = "2a341ace-0f57-4309-9b89-e59fe0f09179"

print("=== LOADING R9 ===")
detail = get_round_detail(ROUND_ID)
n_seeds = len(detail.get("initial_states", []))
map_w = detail.get("map_width", 40)
map_h = detail.get("map_height", 40)
print(f"Map: {map_w}x{map_h}, {n_seeds} seeds")

print("\n=== SUBMITTING R9 (V7 + stats fix) ===")
for seed_idx in range(n_seeds):
    try:
        pred = build_prediction(ROUND_ID, detail, seed_idx, map_w, map_h)
        # Sanity check
        assert pred.shape == (map_h, map_w, 6), f"Bad shape: {pred.shape}"
        assert np.allclose(pred.sum(axis=-1), 1.0, atol=1e-3), "Not normalized"
        assert pred.min() >= 0, "Negative probs"
        
        resp = submit(ROUND_ID, seed_idx, prediction_to_list(pred))
        status = resp.get("status", "?")
        score = resp.get("score", "?")
        print(f"  Seed {seed_idx}: {status} score={score}")
    except Exception as e:
        print(f"  Seed {seed_idx}: ERROR - {e}")

# Check final score
print("\n=== CHECKING SCORE ===")
my = _request("GET", "/my-rounds")
for r in my:
    if r.get("round_number") == 9:
        rs = r.get("round_score")
        ss = r.get("seed_scores")
        print(f"R9: score={rs}")
        if ss:
            print(f"Seeds: {ss}")
        if rs:
            weight = 1.05 ** 9
            print(f"Weighted: {rs * weight:.2f} (weight={weight:.4f})")
        break
