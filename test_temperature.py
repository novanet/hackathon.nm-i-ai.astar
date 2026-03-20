"""Backtest temperature scaling via LORO cross-validation.
Sweep T values and find optimal temperature for KL-divergence scoring."""

import json, pickle, numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    compute_cell_features, apply_floor, temperature_scale,
    _apply_transition_matrix, HISTORICAL_TRANSITIONS,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from train_spatial import (
    ROUND_IDS, load_round_data, compute_gt_round_features,
    build_training_data_multi,
)

DATA_DIR = Path("data")

# Temperature values to sweep
TEMPS = [0.90, 0.95, 1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20, 1.30, 1.50]


def main():
    # Load all data
    print("=== LOADING DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    # LORO CV with temperature sweep
    print(f"\n=== LORO × TEMPERATURE SWEEP ===")
    print(f"Temperatures: {TEMPS}")

    # Store results: {T: {rnum: avg_score}}
    results = {T: {} for T in TEMPS}

    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, _ = build_training_data_multi(train_ids)

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                verbosity=-1,
            ),
            n_jobs=1,
        )
        model.fit(X_train, Y_train)

        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        # Get raw predictions for all seeds (before temperature/floor)
        raw_preds = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            raw_preds.append(pred)

        # Score each temperature
        for T in TEMPS:
            scores = []
            for s, gt in enumerate(test_gts):
                scaled = temperature_scale(raw_preds[s], T=T)
                scaled = apply_floor(scaled)
                scores.append(score_prediction(scaled, gt))
            avg = np.mean(scores)
            results[T][test_rnum] = avg

        # Print progress
        row = f"  R{test_rnum}:"
        for T in TEMPS:
            row += f"  T={T:.2f}={results[T][test_rnum]:.2f}"
        print(row)

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"{'T':>6}", end="")
    for rnum in sorted(all_data.keys()):
        print(f"  {'R'+str(rnum):>6}", end="")
    print(f"  {'AVG':>6}  {'W-AVG':>6}")

    best_T = None
    best_avg = -1
    for T in TEMPS:
        print(f"{T:6.2f}", end="")
        vals = []
        wvals = []
        for rnum in sorted(all_data.keys()):
            sc = results[T][rnum]
            print(f"  {sc:6.2f}", end="")
            vals.append(sc)
            wvals.append(sc * 1.05 ** rnum)
        avg = np.mean(vals)
        wavg = np.mean(wvals)
        print(f"  {avg:6.2f}  {wavg:6.2f}", end="")
        if avg > best_avg:
            best_avg = avg
            best_T = T
            print(" *", end="")
        print()

    print(f"\nBest temperature: T={best_T:.2f} (LORO avg={best_avg:.2f})")
    delta = best_avg - np.mean([results[1.0][r] for r in all_data])
    print(f"Improvement over T=1.0: {delta:+.2f}")


if __name__ == "__main__":
    main()
