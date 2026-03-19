"""
Learn transition matrices from Round 1 ground truth and compare to Round 2 observations.
Also try fitting more sophisticated models.
"""
import json
import numpy as np
from pathlib import Path

from astar.client import _request, get_round_detail
from astar.model import apply_floor
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES

R1_ID = "71451d74-be9f-471f-aacd-a41f3b68a9cd"
DATA_DIR = Path("data") / f"round_{R1_ID}"

# Load Round 1 detail
detail = get_round_detail(R1_ID)
n_seeds = len(detail["initial_states"])
map_w, map_h = detail["map_width"], detail["map_height"]

# Load cached ground truths
ground_truths = []
for seed_idx in range(n_seeds):
    gt_path = DATA_DIR / f"ground_truth_s{seed_idx}.json"
    analysis = json.loads(gt_path.read_text(encoding="utf-8"))
    ground_truths.append(np.array(analysis["ground_truth"], dtype=np.float64))

# ======= LEARN TRANSITION MATRIX FROM ROUND 1 =======
print("=== ROUND 1 ORACLE TRANSITION MATRIX ===")
print("(If we could perfectly learn transitions from data)\n")

# Count transitions: initial_class -> ground_truth distribution
transition_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    gt = ground_truths[seed_idx]
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            transition_counts[init_cls] += gt[y, x]

# Normalize
r1_transitions = transition_counts / transition_counts.sum(axis=1, keepdims=True)

print("Round 1 oracle transitions (initial → final):")
print(f"{'':>12}", end="")
for c in range(NUM_CLASSES):
    print(f" {CLASS_NAMES[c]:>10}", end="")
print()
for from_cls in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[from_cls]:>12}", end="")
    for to_cls in range(NUM_CLASSES):
        print(f" {r1_transitions[from_cls, to_cls]:>10.3f}", end="")
    print()

# Compare Round 1 vs Round 2 transition matrices
print("\n=== ROUND 2 TRANSITION MATRIX (from observations) ===")
r2_transitions = np.array([
    [0.768, 0.164, 0.011, 0.018, 0.040, 0.000],  # Empty→
    [0.388, 0.381, 0.004, 0.054, 0.173, 0.000],  # Settlement→
    [0.417, 0.000, 0.583, 0.000, 0.000, 0.000],  # Port→
    [0.500, 0.000, 0.000, 0.500, 0.000, 0.000],  # Ruin→
    [0.110, 0.191, 0.015, 0.020, 0.665, 0.000],  # Forest→
    [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # Mountain→
])

print(f"{'':>12}", end="")
for c in range(NUM_CLASSES):
    print(f" {CLASS_NAMES[c]:>10}", end="")
print()
for from_cls in range(NUM_CLASSES):
    print(f"{CLASS_NAMES[from_cls]:>12}", end="")
    for to_cls in range(NUM_CLASSES):
        print(f" {r2_transitions[from_cls, to_cls]:>10.3f}", end="")
    print()

# Score with the oracle matrix
print("\n=== SCORING: ORACLE vs R2 TRANSITIONS ===")

for label, trans_mat in [("R1 oracle", r1_transitions), ("R2 observed", r2_transitions)]:
    scores = []
    for seed_idx in range(n_seeds):
        init_grid = detail["initial_states"][seed_idx]["grid"]
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = trans_mat[init_cls]
        pred = apply_floor(pred, floor=0.001)
        scores.append(score_prediction(pred, ground_truths[seed_idx]))
    print(f"  {label}: avg={np.mean(scores):.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

# ======= How different are the transition matrices? =======
print("\n=== DIFFERENCE: R1 oracle - R2 observed ===")
diff = r1_transitions - r2_transitions
for from_cls in range(NUM_CLASSES):
    max_diff = np.abs(diff[from_cls]).max()
    if max_diff > 0.01:
        print(f"  {CLASS_NAMES[from_cls]}: max diff = {max_diff:.3f}")
        for to_cls in range(NUM_CLASSES):
            if abs(diff[from_cls, to_cls]) > 0.01:
                print(f"    → {CLASS_NAMES[to_cls]}: R1={r1_transitions[from_cls,to_cls]:.3f} R2={r2_transitions[from_cls,to_cls]:.3f} (Δ={diff[from_cls,to_cls]:+.3f})")

# ======= AVERAGE TRANSITION MATRIX =======
print("\n=== AVERAGED (R1+R2)/2 TRANSITION MATRIX ===")
avg_transitions = (r1_transitions + r2_transitions) / 2
scores = []
for seed_idx in range(n_seeds):
    init_grid = detail["initial_states"][seed_idx]["grid"]
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            pred[y, x] = avg_transitions[init_cls]
    pred = apply_floor(pred, floor=0.001)
    scores.append(score_prediction(pred, ground_truths[seed_idx]))
print(f"  Averaged: avg={np.mean(scores):.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

# ======= CROSS-VALIDATION: Per-seed transition matrices =======
print("\n=== LEAVE-ONE-OUT CROSS-VALIDATION ===")
print("(Train on 4 seeds, test on 1)")
loo_scores = []
for test_seed in range(n_seeds):
    # Learn transitions from other 4 seeds
    train_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s in range(n_seeds):
        if s == test_seed:
            continue
        init_grid = detail["initial_states"][s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                train_counts[init_cls] += ground_truths[s][y, x]
    # Smooth with small uniform prior
    train_counts += 0.1
    loo_trans = train_counts / train_counts.sum(axis=1, keepdims=True)
    
    # Test on held-out seed
    init_grid = detail["initial_states"][test_seed]["grid"]
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            pred[y, x] = loo_trans[init_cls]
    pred = apply_floor(pred, floor=0.001)
    score = score_prediction(pred, ground_truths[test_seed])
    loo_scores.append(score)
    print(f"  Test seed {test_seed}: {score:.2f}")
print(f"  LOO average: {np.mean(loo_scores):.2f}")

# ======= SPATIAL-AWARE MODEL =======
print("\n=== SPATIAL-AWARE MODEL ===")
print("(Use neighbor initial classes as features)")

from scipy.ndimage import gaussian_filter

for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
    scores = []
    for seed_idx in range(n_seeds):
        init_grid = detail["initial_states"][seed_idx]["grid"]
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = r1_transitions[init_cls]
        # Apply spatial smoothing
        for c in range(NUM_CLASSES):
            pred[:, :, c] = gaussian_filter(pred[:, :, c], sigma=sigma)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        pred = apply_floor(pred, floor=0.001)
        scores.append(score_prediction(pred, ground_truths[seed_idx]))
    print(f"  sigma={sigma:.1f}: avg={np.mean(scores):.2f}")

# ======= NEIGHBOR DENSITY FEATURES =======
print("\n=== NEIGHBOR DENSITY FEATURES ===")
print("(Adjust transition based on how many neighbors are settlements/forest)")

def count_neighbors(grid, target_class, radius=1):
    """Count cells of target_class in neighborhood"""
    h, w = len(grid), len(grid[0])
    counts = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            n = 0
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if TERRAIN_TO_CLASS.get(grid[ny][nx], 0) == target_class:
                            n += 1
            counts[y, x] = n
    return counts

# Simple: boost settlement probability near other settlements
for boost_factor in [1.2, 1.5, 2.0, 3.0]:
    scores = []
    for seed_idx in range(n_seeds):
        init_grid = detail["initial_states"][seed_idx]["grid"]
        settlement_density = count_neighbors(init_grid, 1, radius=2)  # cls 1 = Settlement
        
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = r1_transitions[init_cls].copy()
                # Boost settlement probability proportional to neighbor density
                if settlement_density[y, x] > 0:
                    pred[y, x, 1] *= (1 + (boost_factor - 1) * settlement_density[y, x] / 8)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        pred = apply_floor(pred, floor=0.001)
        scores.append(score_prediction(pred, ground_truths[seed_idx]))
    print(f"  boost={boost_factor:.1f}: avg={np.mean(scores):.2f}")
