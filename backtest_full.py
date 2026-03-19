"""Backtest: sweep Bayesian prior_strength and test with R2 observations."""
import json, numpy as np
from pathlib import Path
from astar.model import (
    compute_cell_features, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, spatial_prior, bayesian_update, PRIOR_STRENGTH,
    cross_seed_transition_prior, build_prediction
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, build_observation_grid
import astar.model as m

R1_ID = "71451d74-be9f-471f-aacd-a41f3b68a9cd"
DATA_DIR = Path("data") / f"round_{R1_ID}"

detail = json.loads(sorted(DATA_DIR.glob("round_detail_*.json"))[-1].read_text(encoding="utf-8"))
ground_truths = []
for s in range(5):
    gt = json.loads((DATA_DIR / f"ground_truth_s{s}.json").read_text(encoding="utf-8"))
    ground_truths.append(np.array(gt["ground_truth"], dtype=np.float64))

map_w, map_h = 40, 40

# Summary of all strategies
print("=== STRATEGY COMPARISON (R1 ground truth, no observations) ===\n")

# 1. Historical transitions only
scores = []
for s in range(5):
    pred = _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
    pred = apply_floor(pred)
    scores.append(score_prediction(pred, ground_truths[s]))
print(f"  Historical transitions:     {np.mean(scores):.2f}  {[f'{s:.1f}' for s in scores]}")

# 2. Spatial model only
m._spatial_model = None
scores = []
for s in range(5):
    sp = spatial_prior(detail, s, map_w, map_h)
    pred = apply_floor(sp)
    scores.append(score_prediction(pred, ground_truths[s]))
print(f"  Spatial model only:         {np.mean(scores):.2f}  {[f'{s:.1f}' for s in scores]}")

# 3. Spatial + historical blend  
for alpha in [0.3, 0.5, 0.7]:
    scores = []
    for s in range(5):
        sp = spatial_prior(detail, s, map_w, map_h)
        trans = _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
        blended = alpha * sp + (1 - alpha) * trans
        blended = blended / blended.sum(axis=-1, keepdims=True)
        pred = apply_floor(blended)
        scores.append(score_prediction(pred, ground_truths[s]))
    print(f"  Spatial({alpha:.0%})+Trans({1-alpha:.0%}):    {np.mean(scores):.2f}  {[f'{s:.1f}' for s in scores]}")

# 4. Full pipeline (spatial + transition + floor, no obs since R1 had 0 queries)
scores = []
for s in range(5):
    pred = build_prediction(R1_ID, detail, s, map_w, map_h)
    scores.append(score_prediction(pred, ground_truths[s]))
print(f"  Full pipeline (no obs):     {np.mean(scores):.2f}  {[f'{s:.1f}' for s in scores]}")

# === Simulate the Bayesian update with synthetic observations ===
print("\n=== BAYESIAN UPDATE SIMULATION ===")
print("(Simulating having N observations per cell drawn from ground truth)\n")

for n_obs_per_cell in [1, 2, 5, 10]:
    scores_bayes = []
    scores_naive = []
    for s in range(5):
        gt = ground_truths[s]
        sp = spatial_prior(detail, s, map_w, map_h)
        prior = sp if sp is not None else _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
        
        # Simulate observations: sample from ground truth
        rng = np.random.default_rng(42 + s)
        pred_bayes = prior.copy()
        pred_naive = prior.copy()
        
        for y in range(map_h):
            for x in range(map_w):
                # Sample n observations from ground truth distribution
                obs_classes = rng.choice(NUM_CLASSES, size=n_obs_per_cell, p=gt[y, x])
                
                # Bayesian update
                alpha = prior[y, x] * PRIOR_STRENGTH
                for cls in obs_classes:
                    alpha[cls] += 1.0
                pred_bayes[y, x] = alpha / alpha.sum()
                
                # Naive frequency replacement
                counts = np.zeros(NUM_CLASSES)
                for cls in obs_classes:
                    counts[cls] += 1.0
                pred_naive[y, x] = counts / counts.sum()
        
        pred_bayes = apply_floor(pred_bayes)
        pred_naive = apply_floor(pred_naive)
        scores_bayes.append(score_prediction(pred_bayes, gt))
        scores_naive.append(score_prediction(pred_naive, gt))
    
    print(f"  {n_obs_per_cell} obs/cell: Bayesian={np.mean(scores_bayes):.2f}, Naive={np.mean(scores_naive):.2f}, Δ={np.mean(scores_bayes)-np.mean(scores_naive):+.2f}")

# Sweep prior_strength
print("\n=== PRIOR STRENGTH SWEEP (1 obs per cell) ===")
for strength in [1, 3, 5, 8, 10, 15, 20, 30, 50]:
    scores = []
    for s in range(5):
        gt = ground_truths[s]
        sp = spatial_prior(detail, s, map_w, map_h)
        prior = sp if sp is not None else _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
        
        rng = np.random.default_rng(42 + s)
        pred = prior.copy()
        for y in range(map_h):
            for x in range(map_w):
                obs_cls = rng.choice(NUM_CLASSES, p=gt[y, x])
                alpha = prior[y, x] * strength
                alpha[obs_cls] += 1.0
                pred[y, x] = alpha / alpha.sum()
        pred = apply_floor(pred)
        scores.append(score_prediction(pred, gt))
    print(f"  strength={strength:3d}: avg={np.mean(scores):.2f}")
