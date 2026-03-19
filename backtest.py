"""
Download Round 1 ground truth and backtest our model against it.

Since we didn't query Round 1, this tests the static prior + cross-seed model
against actual ground truth — the best possible offline validation.
"""
import json
import numpy as np
from pathlib import Path

from astar.client import _request, get_round_detail
from astar.model import initial_prior, apply_floor, cross_seed_transition_prior
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES

R1_ID = "71451d74-be9f-471f-aacd-a41f3b68a9cd"
DATA_DIR = Path("data") / f"round_{R1_ID}"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download ground truth for all 5 seeds
print("=== Downloading Round 1 ground truth ===")
detail = get_round_detail(R1_ID)
n_seeds = len(detail.get("initial_states", []))
print(f"Map: {detail['map_width']}x{detail['map_height']}, {n_seeds} seeds")

ground_truths = []
for seed_idx in range(n_seeds):
    analysis = _request("GET", f"/analysis/{R1_ID}/{seed_idx}")
    gt = np.array(analysis["ground_truth"], dtype=np.float64)
    ground_truths.append(gt)
    
    # Save to disk
    gt_path = DATA_DIR / f"ground_truth_s{seed_idx}.json"
    gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"  Seed {seed_idx}: shape={gt.shape}, saved to {gt_path}")

# === Backtest different model strategies ===
print("\n=== BACKTESTING MODEL STRATEGIES ===\n")

map_w, map_h = detail["map_width"], detail["map_height"]

# Strategy 1: Uniform baseline (worst case)
print("Strategy 1: Uniform (1/6 each class)")
for seed_idx in range(n_seeds):
    pred = np.full((map_h, map_w, NUM_CLASSES), 1.0 / NUM_CLASSES)
    pred = apply_floor(pred)
    score = score_prediction(pred, ground_truths[seed_idx])
    print(f"  Seed {seed_idx}: {score:.2f}")

# Strategy 2: Initial state prior only
print("\nStrategy 2: Initial state prior (no queries)")
scores_prior = []
for seed_idx in range(n_seeds):
    pred = initial_prior(detail, seed_idx, map_w, map_h)
    pred = apply_floor(pred)
    score = score_prediction(pred, ground_truths[seed_idx])
    scores_prior.append(score)
    print(f"  Seed {seed_idx}: {score:.2f}")
print(f"  Average: {np.mean(scores_prior):.2f}")

# Strategy 3: Cross-seed transition model (using Round 2 transitions applied to Round 1 initial states)
# This tests if Round 2's transition patterns generalize
print("\nStrategy 3: Hand-tuned prior from Round 2 transition knowledge")
# Use the transition probabilities we observed in Round 2
r2_transitions = np.array([
    [0.768, 0.164, 0.011, 0.018, 0.040, 0.000],  # Empty→
    [0.388, 0.381, 0.004, 0.054, 0.173, 0.000],  # Settlement→
    [0.417, 0.000, 0.583, 0.000, 0.000, 0.000],  # Port→
    [0.500, 0.000, 0.000, 0.500, 0.000, 0.000],  # Ruin→ (extrapolated, tiny sample)
    [0.110, 0.191, 0.015, 0.020, 0.665, 0.000],  # Forest→
    [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # Mountain→
])
scores_transition = []
for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            pred[y, x] = r2_transitions[init_cls]
    pred = apply_floor(pred)
    score = score_prediction(pred, ground_truths[seed_idx])
    scores_transition.append(score)
    print(f"  Seed {seed_idx}: {score:.2f}")
print(f"  Average: {np.mean(scores_transition):.2f}")

# Strategy 4: Blend (40% prior + 60% transition)
print("\nStrategy 4: Blend 60% transition + 40% prior")
scores_blend = []
for seed_idx in range(n_seeds):
    prior = initial_prior(detail, seed_idx, map_w, map_h)
    init_grid = detail["initial_states"][seed_idx]["grid"]
    trans_pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            trans_pred[y, x] = r2_transitions[init_cls]
    blended = 0.6 * trans_pred + 0.4 * prior
    blended = blended / blended.sum(axis=-1, keepdims=True)
    blended = apply_floor(blended)
    score = score_prediction(blended, ground_truths[seed_idx])
    scores_blend.append(score)
    print(f"  Seed {seed_idx}: {score:.2f}")
print(f"  Average: {np.mean(scores_blend):.2f}")

# Strategy 5: Try different blend ratios
print("\n=== BLEND RATIO SWEEP ===")
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    scores = []
    for seed_idx in range(n_seeds):
        prior = initial_prior(detail, seed_idx, map_w, map_h)
        init_grid = detail["initial_states"][seed_idx]["grid"]
        trans_pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                trans_pred[y, x] = r2_transitions[init_cls]
        blended = alpha * trans_pred + (1 - alpha) * prior
        blended = blended / blended.sum(axis=-1, keepdims=True)
        blended = apply_floor(blended)
        scores.append(score_prediction(blended, ground_truths[seed_idx]))
    print(f"  alpha={alpha:.1f} (transition weight): avg={np.mean(scores):.2f}  per-seed: {[f'{s:.1f}' for s in scores]}")

# Strategy 6: Try different probability floors
print("\n=== PROBABILITY FLOOR SWEEP ===")
best_floor_score = 0
best_floor = 0.01
for floor in [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
    scores = []
    for seed_idx in range(n_seeds):
        init_grid = detail["initial_states"][seed_idx]["grid"]
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = r2_transitions[init_cls]
        pred = apply_floor(pred, floor=floor)
        scores.append(score_prediction(pred, ground_truths[seed_idx]))
    avg = np.mean(scores)
    if avg > best_floor_score:
        best_floor_score = avg
        best_floor = floor
    print(f"  floor={floor:.3f}: avg={avg:.2f}")
print(f"  Best floor: {best_floor} (score={best_floor_score:.2f})")

print("\n=== GROUND TRUTH ANALYSIS (Round 1) ===")
for seed_idx in range(n_seeds):
    gt = ground_truths[seed_idx]
    gt_argmax = np.argmax(gt, axis=-1)
    eps = 1e-10
    p = np.clip(gt, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    print(f"\nSeed {seed_idx}:")
    for c in range(6):
        pct = (gt_argmax == c).sum() / gt_argmax.size * 100
        print(f"  {CLASS_NAMES[c]}: {pct:.1f}%")
    print(f"  Mean entropy: {entropy.mean():.3f}, Dynamic cells: {(entropy > 0.01).sum()}/{entropy.size}")
