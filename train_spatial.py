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
import warnings
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

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
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
    17: "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
    18: "b0f9d1bf-4b71-4e6e-816c-19c718d29056",
    19: "597e60cf-d1a1-4627-ac4d-2a61da68b6df",
    20: "fd82f643-15e2-40e7-9866-8d8f5157081c",
    21: "b3a0be6b-b48b-419d-916a-b7a77fa58c4d",
    22: "a8be24e1-bd48-49bb-aa46-c5593da79f6f",
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
USE_AUGMENTATION = False  # D4 augmentation HURTS GBM: LORO 82.46→82.19, R11-12 massive regression


def _augment_grid_gt(init_grid_np: np.ndarray, gt: np.ndarray):
    """Generate D4 augmentations (4 rotations × 2 flips = 8 variants).
    init_grid_np: (H, W) int array of terrain codes
    gt: (H, W, 6) float array of probabilities
    Yields (augmented_grid_np, augmented_gt) tuples.
    """
    for k in range(4):
        g = np.rot90(init_grid_np, k=k)
        t = np.rot90(gt, k=k, axes=(0, 1))
        yield g, t
        # Horizontal flip
        yield np.fliplr(g), np.flip(t, axis=1)


def _np_grid_to_list(grid_np: np.ndarray) -> list[list[int]]:
    """Convert numpy grid back to list[list[int]] for compute_cell_features."""
    return grid_np.tolist()


def build_training_data_multi(round_ids: dict[int, str], compute_weights: bool = False,
                              augment: bool = USE_AUGMENTATION):
    """Build feature matrix X and target matrix Y from multiple rounds.
    If compute_weights=True, also return entropy-based sample weights.
    If augment=True, apply D4 augmentation (8× data).
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
            init_grid_np = np.array(init_grid)

            if augment:
                variants = list(_augment_grid_gt(init_grid_np, gt))
            else:
                variants = [(init_grid_np, gt)]

            for aug_grid_np, aug_gt in variants:
                aug_h, aug_w = aug_grid_np.shape
                aug_grid_list = _np_grid_to_list(aug_grid_np)
                features = compute_cell_features(aug_grid_list, aug_w, aug_h,
                                                 round_features=round_feats)
                X_parts.append(features.reshape(-1, features.shape[-1]))
                Y_parts.append(aug_gt.reshape(-1, aug_gt.shape[-1]))
                if compute_weights:
                    p = np.clip(aug_gt, 1e-10, 1.0)
                    entropy = -np.sum(p * np.log(p), axis=-1).flatten()
                    W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))
                round_labels.extend([(rnum, seed_idx)] * (aug_w * aug_h))

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
        scores_sim = []
        baseline_scores = []

        # Sim params from GT transitions for this round
        from simulator import params_from_transition_matrix, simulate_monte_carlo_vectorized, grid_to_numpy as sim_grid_to_numpy
        gt_trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        for s, gt in enumerate(test_gts):
            ig = test_detail["initial_states"][s]["grid"]
            for y in range(map_h):
                for x in range(map_w):
                    gt_trans[TERRAIN_TO_CLASS.get(ig[y][x], 0)] += gt[y, x]
        rs = np.maximum(gt_trans.sum(axis=1, keepdims=True), 1.0)
        gt_T = gt_trans / rs
        sim_params = params_from_transition_matrix(gt_T)
        sim_params = sim_params._replace(expansion_base=sim_params.expansion_base * 1.3)

        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            scores.append(score_prediction(apply_floor(pred), gt))

            # ML + sim blend
            init_np = sim_grid_to_numpy(init_grid, map_h, map_w)
            sim_pred = simulate_monte_carlo_vectorized(init_np, sim_params, n_sims=200, seed=42+s)
            blend = 0.85 * pred + 0.15 * sim_pred
            blend = np.maximum(blend, 1e-10)
            blend = blend / blend.sum(axis=-1, keepdims=True)
            scores_sim.append(score_prediction(apply_floor(blend), gt))

            # Baseline
            bpred = _apply_transition_matrix(test_detail, s, HISTORICAL_TRANSITIONS, map_w, map_h)
            bpred = apply_floor(bpred)
            baseline_scores.append(score_prediction(bpred, gt))

        avg = np.mean(scores)
        avg_sim = np.mean(scores_sim)
        bavg = np.mean(baseline_scores)
        loro_results[test_rnum] = (avg, bavg, avg_sim)
        print(f"  Test R{test_rnum}: spatial={avg:.2f}, +sim={avg_sim:.2f}(d={avg_sim-avg:+.2f}), baseline={bavg:.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

    spatial_avg = np.mean([v[0] for v in loro_results.values()])
    sim_avg = np.mean([v[2] for v in loro_results.values()])
    base_avg = np.mean([v[1] for v in loro_results.values()])
    print(f"\n  LORO average:  spatial={spatial_avg:.2f}, +sim={sim_avg:.2f}(d={sim_avg-spatial_avg:+.2f}), baseline={base_avg:.2f}")

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

    # ── Train final ensemble model on ALL rounds ──
    print("\n=== TRAINING FINAL ENSEMBLE (LGB 70% + XGB 30%, entropy-weighted) ===")
    X, Y, _, W = build_training_data_multi(ROUND_IDS, compute_weights=True)
    aug_label = "8× D4 augmented" if USE_AUGMENTATION else "no augmentation"
    print(f"  Training on {X.shape[0]} samples, {X.shape[1]} features ({aug_label})")
    print(f"  Entropy weight power: {ENTROPY_WEIGHT_POWER}")

    lgb_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
            verbosity=-1,
        ),
        n_jobs=1,
    )
    lgb_model.fit(X, Y, sample_weight=W)
    print("  LGB model trained")

    xgb_model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=1.0, verbosity=0,
        ),
        n_jobs=1,
    )
    xgb_model.fit(X, Y, sample_weight=W)
    print("  XGB model trained")

    # Save as ensemble dict: {lgb, xgb, lgb_weight}
    ensemble = {"lgb": lgb_model, "xgb": xgb_model, "lgb_weight": 0.7}
    MODEL_PATH.write_bytes(pickle.dumps(ensemble))
    print(f"  Ensemble saved to {MODEL_PATH}")

    # For scoring below, use a wrapper that blends
    class EnsemblePredictor:
        def __init__(self, lgb_m, xgb_m, w=0.7):
            self.lgb = lgb_m
            self.xgb = xgb_m
            self.w = w
        def predict(self, X):
            return self.w * self.lgb.predict(X) + (1 - self.w) * self.xgb.predict(X)
    final_model = EnsemblePredictor(lgb_model, xgb_model, 0.7)

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
        print(f"    [{row}],  # {comment} ->")
    print("])")

    # ── Train MLP for triple-blend ──
    print("\n=== TRAINING MLP (KL-loss, for triple-blend) ===")
    try:
        import torch
        from train_mlp import train_mlp as train_mlp_fn, KLDivMLP
        mlp = train_mlp_fn(X, Y, W, n_epochs=300, verbose=True)
        mlp_path = Path("data") / "mlp_model.pt"
        torch.save({
            "state_dict": mlp.state_dict(),
            "n_features": X.shape[1],
            "n_classes": NUM_CLASSES,
            "hidden": [256, 128, 64],
        }, mlp_path)
        print(f"  MLP saved to {mlp_path}")
    except Exception as e:
        print(f"  MLP training skipped: {e}")

    # ── Train learned debiaser: maps observation-derived features → GT features ──
    print("\n=== TRAINING LEARNED DEBIASER ===")
    try:
        from sklearn.linear_model import Ridge
        from astar.model import (observation_calibrated_transitions, debias_transitions,
                                 compute_round_features, _extract_settlement_stats)

        obs_feats_list = []
        gt_feats_list = []
        debiaser_rounds = []

        for rnum, (rid, detail, gts) in sorted(all_data.items()):
            # Check if this round has simulation data
            rdir = Path("data") / f"round_{rid}"
            sims = list(rdir.glob("sim_*.json"))
            if len(sims) < 5:
                print(f"  R{rnum}: no sim data ({len(sims)} files), skipping")
                continue

            cal_trans = observation_calibrated_transitions(rid, detail)
            if cal_trans is None:
                print(f"  R{rnum}: no calibrated transitions, skipping")
                continue

            debiased = debias_transitions(cal_trans)
            sett_stats = _extract_settlement_stats(rid, detail)
            pipeline_feats = compute_round_features(debiased, detail,
                                                    settlement_stats=sett_stats)

            gt_round_feats = compute_gt_round_features(detail, gts)

            obs_feats_list.append(pipeline_feats)
            gt_feats_list.append(gt_round_feats)
            debiaser_rounds.append(rnum)

        if len(obs_feats_list) >= 5:
            X_obs = np.array(obs_feats_list)
            Y_gt = np.array(gt_feats_list)
            n_debiaser = len(debiaser_rounds)
            print(f"  Training on {n_debiaser} rounds: {debiaser_rounds}")

            # LOO evaluation
            loo_mse_before = []
            loo_mse_after = []
            for i in range(n_debiaser):
                X_tr = np.delete(X_obs, i, axis=0)
                Y_tr = np.delete(Y_gt, i, axis=0)
                X_te = X_obs[i:i+1]
                Y_te = Y_gt[i:i+1]
                ridge = Ridge(alpha=10.0)
                ridge.fit(X_tr, Y_tr)
                pred = ridge.predict(X_te)
                mse_before = np.mean((X_te[0, :4] - Y_te[0, :4]) ** 2)
                mse_after = np.mean((pred[0, :4] - Y_te[0, :4]) ** 2)
                loo_mse_before.append(mse_before)
                loo_mse_after.append(mse_after)
                print(f"    LOO R{debiaser_rounds[i]}: MSE before={mse_before:.6f}, after={mse_after:.6f}, "
                      f"{'BETTER' if mse_after < mse_before else 'WORSE'}")

            avg_before = np.mean(loo_mse_before)
            avg_after = np.mean(loo_mse_after)
            print(f"  LOO avg MSE: before={avg_before:.6f}, after={avg_after:.6f}, "
                  f"change={avg_after-avg_before:+.6f}")

            if avg_after < avg_before:
                # Fit final model on all data
                final_debiaser = Ridge(alpha=10.0)
                final_debiaser.fit(X_obs, Y_gt)
                debiaser_path = Path("data") / "learned_debiaser.pkl"
                debiaser_path.write_bytes(pickle.dumps(final_debiaser))
                print(f"  Learned debiaser saved to {debiaser_path}")
            else:
                print("  Learned debiaser did NOT improve — NOT saving")
                # Remove old model if it exists
                debiaser_path = Path("data") / "learned_debiaser.pkl"
                if debiaser_path.exists():
                    debiaser_path.unlink()
        else:
            print(f"  Only {len(obs_feats_list)} rounds with sim data — need at least 5, skipping")
    except Exception as e:
        import traceback
        print(f"  Debiaser training failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    train_and_evaluate()
