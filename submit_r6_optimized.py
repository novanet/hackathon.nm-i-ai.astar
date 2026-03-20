"""Submit R6 with optimized alpha=0.2 (observation-calibrated transitions get higher weight)."""
import os
import numpy as np

os.environ.setdefault("ASTAR_TOKEN", os.environ.get("ASTAR_TOKEN", ""))

from astar.model import (
    spatial_prior, _apply_transition_matrix, observation_calibrated_transitions,
    apply_floor, HISTORICAL_TRANSITIONS,
)
from astar.replay import load_round_detail, NUM_CLASSES
from astar.client import submit

ROUND_ID = "ae78003a-4efe-425a-881a-d16a39bca0ad"
ALPHA = 0.2  # Optimized for this round (was 0.85)

detail = load_round_detail(ROUND_ID)
n_seeds = len(detail["initial_states"])
map_w = detail["map_width"]
map_h = detail["map_height"]

cal_trans = observation_calibrated_transitions(ROUND_ID, detail, map_w, map_h)

print(f"Submitting R6 with alpha={ALPHA}")
for seed_idx in range(n_seeds):
    sp = spatial_prior(detail, seed_idx, map_w, map_h)
    if cal_trans is not None:
        tp = _apply_transition_matrix(detail, seed_idx, cal_trans, map_w, map_h)
    else:
        tp = _apply_transition_matrix(detail, seed_idx, HISTORICAL_TRANSITIONS, map_w, map_h)
    
    if sp is not None:
        pred = ALPHA * sp + (1.0 - ALPHA) * tp
        pred = pred / pred.sum(axis=-1, keepdims=True)
    else:
        pred = tp
    
    pred = apply_floor(pred)
    
    result = submit(ROUND_ID, seed_idx, pred.tolist())
    print(f"  Seed {seed_idx}: {result}")

print("Done!")
