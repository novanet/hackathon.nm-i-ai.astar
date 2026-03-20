"""
Train the spatial conditional model on ALL completed round ground truths.

Features per cell: initial class one-hot (6) + 3×3 neighbor fractions (6) + 5×5 outer ring fractions (6) = 18
Target: ground truth probability distribution (6 dims)

Uses MultiOutputRegressor with GradientBoosting for per-class probability prediction.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

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


def build_training_data_multi(round_ids: dict[int, str]):
    """Build feature matrix X and target matrix Y from multiple rounds."""
    X_parts = []
    Y_parts = []
    round_labels = []  # track which round each sample came from

    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            print(f"  R{rnum}: no ground truth, skipping")
            continue
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        for seed_idx, gt in enumerate(gts):
            init_grid = detail["initial_states"][seed_idx]["grid"]
            features = compute_cell_features(init_grid, map_w, map_h)
            X_parts.append(features.reshape(-1, features.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            round_labels.extend([(rnum, seed_idx)] * (map_w * map_h))

    X = np.vstack(X_parts)
    Y = np.vstack(Y_parts)
    return X, Y, round_labels


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
        X_train, Y_train, _ = build_training_data_multi(train_ids)

        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, subsample=0.8,
            ),
            n_jobs=-1,
        )
        model.fit(X_train, Y_train)

        # Test on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        scores = []
        baseline_scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h)
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
    print("\n=== TRAINING FINAL MODEL (all rounds) ===")
    X, Y, _ = build_training_data_multi(ROUND_IDS)
    print(f"  Training on {X.shape[0]} samples, {X.shape[1]} features")

    final_model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8,
        ),
        n_jobs=-1,
    )
    final_model.fit(X, Y)

    MODEL_PATH.write_bytes(pickle.dumps(final_model))
    print(f"  Model saved to {MODEL_PATH}")

    # Training set scores (sanity check)
    print("\n=== TRAINING SET SCORES ===")
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        scores = []
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h)
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
