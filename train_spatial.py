"""
Train the spatial conditional model on ALL completed round ground truths.

Features per cell: initial class one-hot (6) + 3×3 neighbor fractions (6) 
  + 5×5 outer ring fractions (6) + distance/count features (4) 
  + round-level features (5) = 27
Target: ground truth probability distribution (6 dims)

Uses LightGBM with MultiOutputRegressor for per-class probability prediction.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import compute_cell_features, apply_floor, _apply_transition_matrix, HISTORICAL_TRANSITIONS
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

DATA_DIR = Path("data")
MODEL_PATH = DATA_DIR / "spatial_model.pkl"

# All completed rounds with ground truth
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
}


def load_round_data(round_id: str):
    """Load round detail and ground truths for a completed round."""
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    ground_truths = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            break
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        ground_truths.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, ground_truths


def compute_gt_round_features(detail: dict, gts: list[np.ndarray]) -> np.ndarray:
    """Compute round-level features from GT transition matrix.
    Returns 8 features matching compute_round_features() in model.py:
    E→E, S→S, F→F, E→S, settlement_density, mean_food, mean_wealth, mean_defense.
    For GT training, settlement stats are derived as proxies from GT distributions.
    """
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    # Compute GT transition matrix
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = 0
    total = 0
    for s in range(len(gts)):
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                gt_counts[init_cls] += gts[s][y, x]
                if init_cls == 1:
                    n_sett += 1
                total += 1
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / row_sums
    sett_density = n_sett / max(total, 1)

    # Derive settlement stat proxies from GT:
    # S→S correlates with food/defense; high S→S = good food + defense
    ss_rate = gt_trans[1, 1]
    # Proxy: food ~ S→S persistence (high survival = good food)
    mean_food = 0.3 + 0.7 * ss_rate  # range ~[0.3, 1.0]
    # Proxy: wealth ~ E→S expansion rate (more expansion = more trade/wealth)
    mean_wealth = gt_trans[0, 1] * 0.3  # range ~[0, 0.04]
    # Proxy: defense ~ 1 - S→E collapse rate
    mean_defense = 1.0 - gt_trans[1, 0]  # range ~[0.4, 0.7]

    return np.array([
        gt_trans[0, 0],  # E→E
        gt_trans[1, 1],  # S→S
        gt_trans[4, 4],  # F→F
        gt_trans[0, 1],  # E→S
        sett_density,
        mean_food,
        mean_wealth,
        mean_defense,
    ], dtype=np.float64)


ENTROPY_WEIGHT_POWER = 0.25  # entropy^power weighting for training samples

def build_training_data_multi(round_ids: dict[int, str], compute_weights: bool = False):
    """Build feature matrix X and target matrix Y from multiple rounds.
    If compute_weights=True, also return entropy-based sample weights.
    """
    X_parts = []
    Y_parts = []
    W_parts = []
    round_labels = []  # track which round each sample came from

    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            print(f"  R{rnum}: no ground truth, skipping")
            continue
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)

        # Compute round-level features from GT
        round_feats = compute_gt_round_features(detail, gts)

        for seed_idx, gt in enumerate(gts):
            init_grid = detail["initial_states"][seed_idx]["grid"]
            features = compute_cell_features(init_grid, map_w, map_h,
                                             round_features=round_feats)
            X_parts.append(features.reshape(-1, features.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            if compute_weights:
                # Entropy of GT distribution per cell — high-entropy cells matter more for scoring
                p = np.clip(gt, 1e-10, 1.0)
                entropy = -np.sum(p * np.log(p), axis=-1).flatten()
                W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))
            round_labels.extend([(rnum, seed_idx)] * (map_w * map_h))

    X = np.vstack(X_parts)
    Y = np.vstack(Y_parts)
    W = np.concatenate(W_parts) if compute_weights else None
    return X, Y, round_labels, W


def train_and_evaluate():
    # ── Load all data ──
    print("=== LOADING ALL ROUND DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds, map {detail['map_width']}x{detail['map_height']}")
        else:
            print(f"  R{rnum}: no data")

    total_seeds = sum(len(gts) for _, _, gts in all_data.values())
    total_cells = total_seeds * 1600
    print(f"  Total: {total_seeds} seeds, {total_cells} cells")

    # ── Leave-One-Round-Out cross-validation ──
    print("\n=== LEAVE-ONE-ROUND-OUT CROSS-VALIDATION ===")
    loro_results = {}
    for test_rnum in sorted(all_data.keys()):
        # Train on all other rounds
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, _, W_train = build_training_data_multi(train_ids, compute_weights=True)

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                num_leaves=15, min_child_samples=50, subsample=0.7,
                colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
                verbosity=-1,
            ),
            n_jobs=1,
        )
        model.fit(X_train, Y_train, sample_weight=W_train)

        # Test on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)
        scores = []
        baseline_scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))

            # Baseline
            bpred = _apply_transition_matrix(test_detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
            bpred = apply_floor(bpred)
            baseline_scores.append(score_prediction(bpred, gt))

        avg = np.mean(scores)
        bavg = np.mean(baseline_scores)
        loro_results[test_rnum] = (avg, bavg)
        print(f"  Test R{test_rnum}: spatial={avg:.2f}, baseline={bavg:.2f}, Δ={avg-bavg:+.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

    spatial_avg = np.mean([v[0] for v in loro_results.values()])
    base_avg = np.mean([v[1] for v in loro_results.values()])
    print(f"\n  LORO average:  spatial={spatial_avg:.2f}, baseline={base_avg:.2f}, Δ={spatial_avg-base_avg:+.2f}")

    # ── Update historical transitions from all data ──
    print("\n=== UPDATED TRANSITION MATRIX (all rounds) ===")
    transition_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for rnum, (rid, detail, gts) in all_data.items():
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            for y in range(map_h):
                for x in range(map_w):
                    init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                    transition_counts[init_cls] += gt[y, x]
    # Normalize
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    new_transitions = transition_counts / row_sums
    from astar.replay import CLASS_NAMES
    print(f"{'':>12}", end="")
    for c in range(NUM_CLASSES):
        print(f" {CLASS_NAMES[c]:>10}", end="")
    print()
    for from_cls in range(NUM_CLASSES):
        print(f"{CLASS_NAMES[from_cls]:>12}", end="")
        for to_cls in range(NUM_CLASSES):
            print(f" {new_transitions[from_cls, to_cls]:>10.4f}", end="")
        print()
    
    # Score the updated transition matrix
    print("\n  Updated transitions score on each round:")
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        scores = []
        for s, gt in enumerate(gts):
            pred = _apply_transition_matrix(detail, s, new_transitions, map_w, map_h)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))
        print(f"    R{rnum}: avg={np.mean(scores):.2f}")

    # ── Train final model on ALL rounds ──
    print("\n=== TRAINING FINAL MODEL (all rounds, entropy-weighted) ===")
    X, Y, _, W = build_training_data_multi(ROUND_IDS, compute_weights=True)
    print(f"  Training on {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Entropy weight power: {ENTROPY_WEIGHT_POWER}")

    final_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
            verbosity=-1,
        ),
        n_jobs=1,
    )
    final_model.fit(X, Y, sample_weight=W)

    MODEL_PATH.write_bytes(pickle.dumps(final_model))
    print(f"  Model saved to {MODEL_PATH}")

    # Training set scores (sanity check)
    print("\n=== TRAINING SET SCORES ===")
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        scores = []
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=round_feats)
            flat_pred = final_model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))
        print(f"  R{rnum}: avg={np.mean(scores):.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

    # Print the new transition matrix in code-ready format
    print("\n=== CODE-READY TRANSITION MATRIX ===")
    print("HISTORICAL_TRANSITIONS = np.array([")
    for from_cls in range(NUM_CLASSES):
        row = ", ".join(f"{new_transitions[from_cls, to_cls]:.4f}" for to_cls in range(NUM_CLASSES))
        comment = CLASS_NAMES[from_cls]
        print(f"    [{row}],  # {comment} →")
    print("])")


if __name__ == "__main__":
    train_and_evaluate()
