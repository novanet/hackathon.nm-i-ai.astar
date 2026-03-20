"""
Proper LORO ensemble test: MLP + GBM blend with multiple weight options.
Trains both models per fold and evaluates blends.
"""
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    compute_cell_features, apply_floor,
    HISTORICAL_TRANSITIONS, CALIBRATION_FACTORS, PER_CLASS_TEMPS,
    per_class_temperature_scale,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from train_mlp import (
    load_round_data, compute_gt_round_features, build_training_data,
    train_mlp, predict_with_mlp, KLDivMLP, ROUND_IDS,
    ENTROPY_WEIGHT_POWER,
)


def build_gbm_training_data(round_ids: dict[int, str]):
    """Build X, Y, W for GBM (reuses MLP data pipeline)."""
    return build_training_data(round_ids)


def apply_postprocessing(pred: np.ndarray, round_feats: np.ndarray,
                         detail: dict, seed_idx: int) -> np.ndarray:
    """Apply calibration + temps + floor."""
    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)
    ss_rate = round_feats[1]
    if ss_rate < 0.15:
        temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
    else:
        temps = PER_CLASS_TEMPS
    pred = per_class_temperature_scale(pred, detail, seed_idx, temps=temps)
    return apply_floor(pred)


def main():
    print("=== LORO ENSEMBLE TEST: GBM vs MLP vs BLENDS ===")
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")

    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    blend_weights = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    # blend_w = weight on GBM, (1-blend_w) on MLP
    # 0.0 = pure MLP, 1.0 = pure GBM

    results = {w: {} for w in blend_weights}  # {weight: {rnum: avg_score}}

    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, W_train = build_training_data(train_ids)
        n_train = X_train.shape[0]

        print(f"\n--- R{test_rnum} (train on {n_train} samples) ---")

        # Train GBM
        gbm = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                num_leaves=15, min_child_samples=50, subsample=0.7,
                colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
                verbosity=-1,
            ),
            n_jobs=1,
        )
        gbm.fit(X_train, Y_train, sample_weight=W_train)
        print(f"  GBM trained")

        # Train MLP
        mlp = train_mlp(X_train, Y_train, W_train, verbose=False)
        print(f"  MLP trained")

        # Free GPU memory
        torch.cuda.empty_cache()

        # Evaluate
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        blend_seed_scores = {w: [] for w in blend_weights}

        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)

            # GBM prediction (raw)
            flat_feat = feat.reshape(-1, feat.shape[-1])
            gbm_raw = gbm.predict(flat_feat).reshape(map_h, map_w, NUM_CLASSES)
            gbm_raw = np.maximum(gbm_raw, 1e-10)
            gbm_raw = gbm_raw / gbm_raw.sum(axis=-1, keepdims=True)

            # MLP prediction (raw)
            mlp_raw = predict_with_mlp(mlp, feat, map_h, map_w)

            for w in blend_weights:
                if w == 0.0:
                    blended = mlp_raw.copy()
                elif w == 1.0:
                    blended = gbm_raw.copy()
                else:
                    blended = w * gbm_raw + (1 - w) * mlp_raw
                    blended = blended / blended.sum(axis=-1, keepdims=True)

                pp = apply_postprocessing(blended, test_round_feats, test_detail, s)
                sc = score_prediction(pp, gt)
                blend_seed_scores[w].append(sc)

        for w in blend_weights:
            avg = np.mean(blend_seed_scores[w])
            results[w][test_rnum] = avg

        # Print round summary
        gbm_only = results[1.0][test_rnum]
        mlp_only = results[0.0][test_rnum]
        best_w = max(blend_weights, key=lambda w: results[w][test_rnum])
        best_sc = results[best_w][test_rnum]
        print(f"  GBM={gbm_only:.2f}, MLP={mlp_only:.2f}, best={best_sc:.2f} (w={best_w})")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("LORO SUMMARY: avg score per blend weight")
    print("=" * 70)
    print(f"{'Weight':>8} | " + " | ".join(f"R{r:d}" for r in sorted(all_data.keys())) + " |  AVG")
    print("-" * 70)
    best_avg = 0
    best_w_overall = 0
    for w in blend_weights:
        label = f"{w:.1f} GBM" if w > 0 else "MLP only"
        if w == 1.0:
            label = "GBM only"
        elif w == 0.0:
            label = "MLP only"
        else:
            label = f"{w:.0%}G+{1-w:.0%}M"
        scores = [results[w].get(r, 0) for r in sorted(all_data.keys())]
        avg = np.mean(scores)
        print(f"{label:>8} | " + " | ".join(f"{s:5.1f}" for s in scores) + f" | {avg:5.2f}")
        if avg > best_avg:
            best_avg = avg
            best_w_overall = w

    print(f"\nBest blend: {best_w_overall:.1f} GBM + {1-best_w_overall:.1f} MLP → LORO avg = {best_avg:.2f}")
    print(f"(Compare: previous GBM-only LORO was ~80.66 with pp)")


if __name__ == "__main__":
    main()
