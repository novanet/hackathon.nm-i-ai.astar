"""Full pipeline backtest: build_prediction() on each round with GT, measuring actual scores."""
import importlib
import json
import numpy as np
from pathlib import Path

# Force reload model with fresh weights
import astar.model as mod
mod._spatial_model = None
importlib.reload(mod)

from astar.model import build_prediction, apply_floor
from astar.submit import score_prediction
from train_mlp import load_round_data, ROUND_IDS

print("=== FULL PIPELINE BACKTEST (build_prediction) ===\n")

scores_all = {}
for rnum in sorted(ROUND_IDS.keys()):
    rid = ROUND_IDS[rnum]
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    
    seed_scores = []
    for s, gt in enumerate(gts):
        pred = build_prediction(rid, detail, s, 40, 40)
        sc = score_prediction(pred, gt)
        seed_scores.append(sc)
    
    avg = np.mean(seed_scores)
    scores_all[rnum] = avg
    seeds_str = ", ".join(f"{s:.1f}" for s in seed_scores)
    print(f"  R{rnum}: avg={avg:.2f}  seeds=[{seeds_str}]")

print(f"\n  Overall avg: {np.mean(list(scores_all.values())):.2f}")

# What weighted leaderboard score would we get from each round?
print("\n=== WEIGHTED LEADERBOARD PROJECTIONS ===")
best_weighted = 0
best_round = 0
for rnum, raw in sorted(scores_all.items()):
    w = 1.05 ** rnum
    weighted = raw * w
    marker = " *** BEST ***" if weighted > best_weighted else ""
    if weighted > best_weighted:
        best_weighted = weighted
        best_round = rnum
    print(f"  R{rnum}: raw={raw:.2f} × {w:.4f} = {weighted:.1f}{marker}")

print(f"\n  Best: R{best_round} = {best_weighted:.1f}")
