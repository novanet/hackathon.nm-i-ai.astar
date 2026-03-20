"""
Comprehensive LORO hyperparameter optimization.

Sweeps:
  1. LightGBM regularization configs (n_estimators, max_depth, reg_alpha/lambda)
  2. Per-class temperature scaling (6 temperatures, one per initial class)
  3. Probability floor values
  4. Meta-model: use settlement stats to correct round feature estimates

Evaluates on Leave-One-Round-Out cross-validation.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    compute_cell_features, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, SHRINKAGE_MATRIX, NUM_CLASSES,
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, _extract_settlement_stats,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, CLASS_NAMES

DATA_DIR = Path("data")

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
}
# R8 excluded — no GT yet


def load_round_data(round_id: str):
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            break
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts


def compute_gt_round_features(detail: dict, gts: list[np.ndarray]) -> np.ndarray:
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
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
    return np.array([gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4],
                     gt_trans[0, 1], sett_density], dtype=np.float64)


def build_training_data(round_ids: dict[int, str], all_data: dict):
    X_parts, Y_parts = [], []
    for rnum in sorted(round_ids.keys()):
        if rnum not in all_data:
            continue
        rid, detail, gts = all_data[rnum]
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for seed_idx, gt in enumerate(gts):
            init_grid = detail["initial_states"][seed_idx]["grid"]
            features = compute_cell_features(init_grid, map_w, map_h,
                                             round_features=round_feats)
            X_parts.append(features.reshape(-1, features.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
    return np.vstack(X_parts), np.vstack(Y_parts)


def temperature_scale(pred: np.ndarray, T: float) -> np.ndarray:
    if T == 1.0:
        return pred
    scaled = np.power(np.maximum(pred, 1e-30), 1.0 / T)
    return scaled / scaled.sum(axis=-1, keepdims=True)


def per_class_temperature_scale(pred: np.ndarray, init_grid: list[list[int]],
                                temps: np.ndarray,
                                map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """Apply different temperature per initial class of each cell."""
    result = pred.copy()
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            T = temps[init_cls]
            if T != 1.0:
                scaled = np.power(np.maximum(result[y, x], 1e-30), 1.0 / T)
                result[y, x] = scaled / scaled.sum()
    return result


def evaluate_config(all_data: dict, lgb_params: dict,
                    temps: np.ndarray | float = 1.15,
                    floor: float = 0.001,
                    per_class_temp: bool = False) -> dict:
    """
    Run full LORO evaluation with given hyperparameters.
    Returns dict with per-round scores and average.
    """
    results = {}
    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train = build_training_data(train_ids, all_data)

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(**lgb_params, verbosity=-1),
            n_jobs=1,
        )
        model.fit(X_train, Y_train)

        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)

            # Temperature scaling
            if per_class_temp and isinstance(temps, np.ndarray):
                pred = per_class_temperature_scale(pred, init_grid, temps, map_w, map_h)
            elif isinstance(temps, (int, float)) and temps != 1.0:
                pred = temperature_scale(pred, float(temps))

            # Floor
            pred = np.maximum(pred, floor)
            pred = pred / pred.sum(axis=-1, keepdims=True)

            scores.append(score_prediction(pred, gt))

        results[test_rnum] = np.mean(scores)

    results["avg"] = np.mean(list(results.values()))
    return results


def main():
    # ── Load all round data ──
    print("=== LOADING DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")
        else:
            print(f"  R{rnum}: no data")

    # ═══════════════════════════════════════════════════
    # PHASE 1: LightGBM regularization sweep
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 1: LightGBM REGULARIZATION SWEEP")
    print("=" * 60)

    lgb_configs = {
        "current": dict(n_estimators=1000, max_depth=6, learning_rate=0.05,
                        num_leaves=31, min_child_samples=20, subsample=0.8,
                        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
        "heavy_reg": dict(n_estimators=500, max_depth=4, learning_rate=0.05,
                          num_leaves=15, min_child_samples=50, subsample=0.7,
                          colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0),
        "med_reg": dict(n_estimators=700, max_depth=5, learning_rate=0.05,
                        num_leaves=20, min_child_samples=30, subsample=0.75,
                        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5),
        "shallow": dict(n_estimators=300, max_depth=3, learning_rate=0.08,
                        num_leaves=8, min_child_samples=80, subsample=0.7,
                        colsample_bytree=0.6, reg_alpha=2.0, reg_lambda=2.0),
        "wide_shallow": dict(n_estimators=500, max_depth=4, learning_rate=0.05,
                             num_leaves=12, min_child_samples=40, subsample=0.8,
                             colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0),
        "fewer_trees": dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                            num_leaves=31, min_child_samples=20, subsample=0.8,
                            colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5),
    }

    # Use current temperature and floor for this phase
    best_lgb_name = "current"
    best_lgb_avg = 0.0
    lgb_results = {}

    for name, params in lgb_configs.items():
        res = evaluate_config(all_data, params, temps=1.15, floor=0.001)
        lgb_results[name] = res
        avg = res["avg"]
        per_round = "  ".join(f"R{r}={res[r]:.1f}" for r in sorted(all_data.keys()))
        print(f"  {name:20s}: avg={avg:.2f}  {per_round}")
        if avg > best_lgb_avg:
            best_lgb_avg = avg
            best_lgb_name = name

    print(f"\n  ** Best LGB config: {best_lgb_name} (avg={best_lgb_avg:.2f}) **")
    best_lgb = lgb_configs[best_lgb_name]

    # ═══════════════════════════════════════════════════
    # PHASE 2: Global temperature sweep (with best LGB)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 2: GLOBAL TEMPERATURE SWEEP")
    print("=" * 60)

    temp_values = [0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5]
    best_global_t = 1.15
    best_global_t_avg = 0.0

    for t in temp_values:
        res = evaluate_config(all_data, best_lgb, temps=t, floor=0.001)
        avg = res["avg"]
        per_round = "  ".join(f"R{r}={res[r]:.1f}" for r in sorted(all_data.keys()))
        print(f"  T={t:.2f}: avg={avg:.2f}  {per_round}")
        if avg > best_global_t_avg:
            best_global_t_avg = avg
            best_global_t = t

    print(f"\n  ** Best global T: {best_global_t} (avg={best_global_t_avg:.2f}) **")

    # ═══════════════════════════════════════════════════
    # PHASE 3: Per-class temperature sweep
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: PER-CLASS TEMPERATURE SWEEP")
    print("=" * 60)

    # Start from best global T as default, then vary per class
    # Classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
    # Mountain is always certain → T doesn't matter, fix at 1.0
    # Port/Ruin are rare → keep at global T
    # Focus sweep on Empty, Settlement, Forest (the big 3)

    base_t = best_global_t
    empty_ts = [base_t - 0.15, base_t - 0.1, base_t, base_t + 0.1, base_t + 0.2, base_t + 0.3]
    sett_ts = [base_t, base_t + 0.1, base_t + 0.2, base_t + 0.3, base_t + 0.4, base_t + 0.5]
    forest_ts = [base_t - 0.1, base_t, base_t + 0.1, base_t + 0.2, base_t + 0.3]

    best_pct_avg = best_global_t_avg  # beat the global-T baseline
    best_pct_temps = np.array([base_t, base_t, base_t, base_t, base_t, 1.0])
    use_per_class = False

    print(f"  Sweeping Empty × Settlement × Forest temps (base={base_t})")
    print(f"  Empty range: {[f'{t:.2f}' for t in empty_ts]}")
    print(f"  Settlement range: {[f'{t:.2f}' for t in sett_ts]}")
    print(f"  Forest range: {[f'{t:.2f}' for t in forest_ts]}")
    print(f"  Total combos: {len(empty_ts) * len(sett_ts) * len(forest_ts)}")

    n_combos = 0
    for te, ts, tf in product(empty_ts, sett_ts, forest_ts):
        temps = np.array([te, ts, base_t, base_t, tf, 1.0])
        res = evaluate_config(all_data, best_lgb, temps=temps, floor=0.001,
                              per_class_temp=True)
        avg = res["avg"]
        n_combos += 1
        if avg > best_pct_avg:
            best_pct_avg = avg
            best_pct_temps = temps.copy()
            use_per_class = True
            per_round = "  ".join(f"R{r}={res[r]:.1f}" for r in sorted(all_data.keys()))
            print(f"  NEW BEST: E={te:.2f} S={ts:.2f} F={tf:.2f} → avg={avg:.2f}  {per_round}")

    if use_per_class:
        print(f"\n  ** Best per-class temps: E={best_pct_temps[0]:.2f} S={best_pct_temps[1]:.2f} "
              f"P={best_pct_temps[2]:.2f} R={best_pct_temps[3]:.2f} F={best_pct_temps[4]:.2f} "
              f"M={best_pct_temps[5]:.2f} (avg={best_pct_avg:.2f}) **")
    else:
        print(f"\n  ** Per-class temp did NOT beat global T={best_global_t} (avg={best_global_t_avg:.2f}) **")

    # ═══════════════════════════════════════════════════
    # PHASE 4: Floor sweep (with best LGB + best temp)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 4: PROBABILITY FLOOR SWEEP")
    print("=" * 60)

    floor_values = [0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
    best_floor = 0.001
    best_floor_avg = 0.0

    if use_per_class:
        final_temps = best_pct_temps
        temp_is_perclass = True
    else:
        final_temps = best_global_t
        temp_is_perclass = False

    for f in floor_values:
        res = evaluate_config(all_data, best_lgb, temps=final_temps, floor=f,
                              per_class_temp=temp_is_perclass)
        avg = res["avg"]
        per_round = "  ".join(f"R{r}={res[r]:.1f}" for r in sorted(all_data.keys()))
        print(f"  floor={f:.4f}: avg={avg:.2f}  {per_round}")
        if avg > best_floor_avg:
            best_floor_avg = avg
            best_floor = f

    print(f"\n  ** Best floor: {best_floor} (avg={best_floor_avg:.2f}) **")

    # ═══════════════════════════════════════════════════
    # PHASE 5: Meta-model for round features
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 5: ROUND FEATURE ESTIMATION ANALYSIS")
    print("=" * 60)

    # For rounds with observations, compare debiased-obs features vs GT features
    # and check if settlement stats can predict the residual
    print("  (Evaluating whether a meta-model helps...)")

    # Compute GT features and debiased-obs features for each round
    obs_rounds = {}
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        gt_feats = compute_gt_round_features(detail, gts)
        # Try to compute obs-based features
        try:
            cal_trans = observation_calibrated_transitions(rid, detail)
            if cal_trans is not None:
                deb_trans = debias_transitions(cal_trans)
                obs_feats = compute_round_features(deb_trans, detail)
                stats = _extract_settlement_stats(rid, detail)
                obs_rounds[rnum] = {
                    "gt_feats": gt_feats, "obs_feats": obs_feats,
                    "error": gt_feats - obs_feats, "stats": stats,
                }
            else:
                print(f"  R{rnum}: no observations")
        except Exception as e:
            print(f"  R{rnum}: error computing obs features: {e}")

    if obs_rounds:
        print(f"\n  Rounds with observations: {sorted(obs_rounds.keys())}")
        feat_names = ["E→E", "S→S", "F→F", "E→S", "sett_dens"]
        print(f"\n  {'Round':>6}", end="")
        for fn in feat_names:
            print(f"  {fn:>8}", end="")
        print("   (errors = GT - obs)")

        errors = []
        for rnum in sorted(obs_rounds.keys()):
            err = obs_rounds[rnum]["error"]
            errors.append(err)
            print(f"  R{rnum:>4}", end="")
            for e in err:
                print(f"  {e:>+8.4f}", end="")
            print()

        errors = np.array(errors)
        mean_err = errors.mean(axis=0)
        std_err = errors.std(axis=0)
        print(f"\n  {'Mean':>6}", end="")
        for e in mean_err:
            print(f"  {e:>+8.4f}", end="")
        print(f"\n  {'Std':>6}", end="")
        for e in std_err:
            print(f"  {e:>8.4f}", end="")
        print()

        # Test if simple bias correction helps
        # Apply mean error as correction to obs features, re-evaluate
        print("\n  Testing bias-corrected round features in LORO...")
        # We need a special LORO that uses corrected obs features at test time
        # instead of GT features. This simulates real inference.
        for rnum in sorted(obs_rounds.keys()):
            gt_f = obs_rounds[rnum]["gt_feats"]
            obs_f = obs_rounds[rnum]["obs_feats"]
            # LOO bias: mean error excluding this round
            other_errors = [obs_rounds[r]["error"] for r in obs_rounds if r != rnum]
            if other_errors:
                loo_bias = np.mean(other_errors, axis=0)
                corrected = obs_f + loo_bias
                print(f"  R{rnum}: GT={gt_f[:4].round(4)}  obs={obs_f[:4].round(4)}  "
                      f"corrected={corrected[:4].round(4)}  "
                      f"err_before={np.abs(gt_f[:4] - obs_f[:4]).mean():.4f}  "
                      f"err_after={np.abs(gt_f[:4] - corrected[:4]).mean():.4f}")

        print("\n  Note: With only", len(obs_rounds), "rounds of paired data,")
        print("  a meta-model would likely overfit. Simple bias correction shown above.")
        print("  If err_after < err_before consistently, we can apply the mean bias correction.")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  Best LGB config:  {best_lgb_name}")
    print(f"    {best_lgb}")
    if use_per_class:
        print(f"  Temperature:      per-class E={best_pct_temps[0]:.2f} S={best_pct_temps[1]:.2f} "
              f"F={best_pct_temps[4]:.2f} M={best_pct_temps[5]:.2f}")
    else:
        print(f"  Temperature:      global T={best_global_t}")
    print(f"  Floor:            {best_floor}")
    print(f"  LORO avg score:   {best_floor_avg:.2f}")
    print(f"  Previous LORO:    ~73.96 (T=1.15) / ~75.50 (T=1.15 + current LGB)")

    # Print code snippets to apply
    print("\n  === CODE CHANGES TO APPLY ===")
    print(f"  # In train_spatial.py — LightGBM params:")
    for k, v in best_lgb.items():
        print(f"  #   {k}={v}")
    if use_per_class:
        print(f"\n  # In model.py — Per-class temperature:")
        print(f"  PER_CLASS_TEMPS = np.array({best_pct_temps.tolist()})")
    else:
        print(f"\n  # In model.py — Global temperature:")
        print(f"  TEMPERATURE = {best_global_t}")
    print(f"\n  # In model.py — Probability floor:")
    print(f"  PROB_FLOOR = {best_floor}")


if __name__ == "__main__":
    main()
