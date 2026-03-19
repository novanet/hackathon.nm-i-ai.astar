"""
Train the spatial conditional model on Round 1 ground truth.

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

R1_ID = "71451d74-be9f-471f-aacd-a41f3b68a9cd"
DATA_DIR = Path("data") / f"round_{R1_ID}"
MODEL_PATH = Path("data") / "spatial_model.pkl"


def load_r1_data():
    """Load Round 1 detail and ground truths."""
    detail = json.loads(sorted(DATA_DIR.glob("round_detail_*.json"))[-1].read_text(encoding="utf-8"))
    ground_truths = []
    for s in range(5):
        gt = json.loads((DATA_DIR / f"ground_truth_s{s}.json").read_text(encoding="utf-8"))
        ground_truths.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, ground_truths


def build_training_data(detail, ground_truths):
    """Build feature matrix X and target matrix Y from all seeds."""
    X_parts = []
    Y_parts = []

    for seed_idx in range(len(ground_truths)):
        init_grid = detail["initial_states"][seed_idx]["grid"]
        features = compute_cell_features(init_grid, 40, 40)  # (40, 40, 18)
        gt = ground_truths[seed_idx]  # (40, 40, 6)

        X_parts.append(features.reshape(-1, features.shape[-1]))
        Y_parts.append(gt.reshape(-1, gt.shape[-1]))

    X = np.vstack(X_parts)  # (8000, 18)
    Y = np.vstack(Y_parts)  # (8000, 6)
    return X, Y


def train_and_evaluate():
    print("Loading Round 1 data...")
    detail, ground_truths = load_r1_data()
    map_w, map_h = detail["map_width"], detail["map_height"]
    n_seeds = len(ground_truths)

    # LOO cross-validation first
    print("\n=== LEAVE-ONE-OUT CROSS-VALIDATION ===")
    loo_scores = []
    for test_seed in range(n_seeds):
        # Train on 4 seeds
        X_train, Y_train = [], []
        for s in range(n_seeds):
            if s == test_seed:
                continue
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h)
            X_train.append(feat.reshape(-1, feat.shape[-1]))
            Y_train.append(ground_truths[s].reshape(-1, NUM_CLASSES))
        X_train = np.vstack(X_train)
        Y_train = np.vstack(Y_train)

        # Train model
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=20, subsample=0.8,
            ),
            n_jobs=-1,
        )
        model.fit(X_train, Y_train)

        # Predict on test seed
        init_grid = detail["initial_states"][test_seed]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h)
        flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        pred = apply_floor(pred)

        score = score_prediction(pred, ground_truths[test_seed])
        loo_scores.append(score)
        print(f"  Test seed {test_seed}: {score:.2f}")
    print(f"  LOO average: {np.mean(loo_scores):.2f}")

    # Compare with transition-only baseline
    print("\n=== BASELINE COMPARISON ===")
    baseline_scores = []
    for s in range(n_seeds):
        pred = _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
        pred = apply_floor(pred)
        baseline_scores.append(score_prediction(pred, ground_truths[s]))
    print(f"  Transition-only baseline: {np.mean(baseline_scores):.2f}")
    print(f"  Spatial model LOO:        {np.mean(loo_scores):.2f}")
    print(f"  Improvement:              {np.mean(loo_scores) - np.mean(baseline_scores):+.2f}")

    # Train final model on ALL 5 seeds
    print("\n=== TRAINING FINAL MODEL (all 5 seeds) ===")
    X, Y = build_training_data(detail, ground_truths)
    print(f"  Training on {X.shape[0]} samples, {X.shape[1]} features")

    final_model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_leaf=20, subsample=0.8,
        ),
        n_jobs=-1,
    )
    final_model.fit(X, Y)

    # Save model
    MODEL_PATH.write_bytes(pickle.dumps(final_model))
    print(f"  Model saved to {MODEL_PATH}")

    # Score on training data (sanity check)
    print("\n=== TRAINING SET SCORES (overfitting check) ===")
    for s in range(n_seeds):
        init_grid = detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h)
        flat_pred = final_model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        pred = apply_floor(pred)
        score = score_prediction(pred, ground_truths[s])
        print(f"  Seed {s}: {score:.2f}")

    # Test the full build_prediction pipeline (with spatial model loaded)
    print("\n=== FULL PIPELINE TEST (spatial + transition + floor) ===")
    # Force reload
    import astar.model as m
    m._spatial_model = None
    from astar.model import build_prediction
    for s in range(n_seeds):
        pred = build_prediction(R1_ID, detail, s, map_w, map_h)
        score = score_prediction(pred, ground_truths[s])
        print(f"  Seed {s}: {score:.2f}")


if __name__ == "__main__":
    train_and_evaluate()
