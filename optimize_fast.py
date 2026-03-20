"""
Fast per-class temperature + floor sweep using cached LORO predictions.

Phase 1+2 results (from optimize_hyperparams.py):
  Best LGB: heavy_reg (n_est=500, depth=4, leaves=15, child=50, sub=0.7, col=0.6, alpha=1.0, lambda=1.0)
  Best global T: 1.10 (tied 1.15 at 76.47, but 1.10 better on R3)

This script:
  1. Trains 7 LORO models (heavy_reg)
  2. Caches raw predictions (before temp/floor) for all test seeds
  3. Sweeps per-class temperatures and floors on cached predictions (pure numpy, instant)
  4. Also evaluates meta-model for round features (bias correction)
"""

import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    compute_cell_features, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, SHRINKAGE_MATRIX, NUM_CLASSES,
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, _extract_settlement_stats,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, CLASS_NAMES

warnings.filterwarnings("ignore")

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

BEST_LGB = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
                num_leaves=15, min_child_samples=50, subsample=0.7,
                colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0)


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


def compute_gt_round_features(detail, gts):
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = total = 0
    for s in range(len(gts)):
        ig = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                ic = TERRAIN_TO_CLASS.get(ig[y][x], 0)
                gt_counts[ic] += gts[s][y, x]
                if ic == 1:
                    n_sett += 1
                total += 1
    rs = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / rs
    return np.array([gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4],
                     gt_trans[0, 1], n_sett / max(total, 1)], dtype=np.float64)


def build_training_data(round_ids, all_data):
    X_parts, Y_parts = [], []
    for rnum in sorted(round_ids.keys()):
        if rnum not in all_data:
            continue
        _, detail, gts = all_data[rnum]
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        rf = compute_gt_round_features(detail, gts)
        for si, gt in enumerate(gts):
            ig = detail["initial_states"][si]["grid"]
            feat = compute_cell_features(ig, map_w, map_h, round_features=rf)
            X_parts.append(feat.reshape(-1, feat.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
    return np.vstack(X_parts), np.vstack(Y_parts)


def apply_per_class_temp_and_floor(raw_pred, init_cls_grid, temps, floor):
    """Apply per-class temperature and floor to raw predictions. All numpy, fast."""
    pred = raw_pred.copy()
    h, w = pred.shape[:2]
    for c in range(NUM_CLASSES):
        mask = (init_cls_grid == c)
        if not mask.any() or temps[c] == 1.0:
            continue
        cells = pred[mask]  # (N, 6)
        scaled = np.power(np.maximum(cells, 1e-30), 1.0 / temps[c])
        pred[mask] = scaled / scaled.sum(axis=-1, keepdims=True)
    # Floor
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def apply_global_temp_and_floor(raw_pred, T, floor):
    pred = raw_pred.copy()
    if T != 1.0:
        pred = np.power(np.maximum(pred, 1e-30), 1.0 / T)
        pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def main():
    print("=== LOADING DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    # ═══════════════════════════════════════════════════
    # STEP 1: Train LORO models and cache raw predictions
    # ═══════════════════════════════════════════════════
    print("\n=== TRAINING LORO MODELS (heavy_reg) ===")
    # cached[rnum] = list of (raw_pred, gt, init_cls_grid) per seed
    cached = {}
    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train = build_training_data(train_ids, all_data)

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(**BEST_LGB, verbosity=-1), n_jobs=1)
        model.fit(X_train, Y_train)

        _, detail, gts = all_data[test_rnum]
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        rf = compute_gt_round_features(detail, gts)

        seed_data = []
        for s, gt in enumerate(gts):
            ig = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(ig, map_w, map_h, round_features=rf)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            raw_pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            raw_pred = np.maximum(raw_pred, 1e-10)
            raw_pred = raw_pred / raw_pred.sum(axis=-1, keepdims=True)

            init_cls_grid = np.array([[TERRAIN_TO_CLASS.get(ig[y][x], 0)
                                       for x in range(map_w)] for y in range(map_h)])
            seed_data.append((raw_pred, gt, init_cls_grid))

        cached[test_rnum] = seed_data
        # Quick check with T=1.15, floor=0.001
        scores = []
        for raw_pred, gt, icg in seed_data:
            p = apply_global_temp_and_floor(raw_pred, 1.15, 0.001)
            scores.append(score_prediction(p, gt))
        print(f"  R{test_rnum}: LORO avg={np.mean(scores):.2f}  (T=1.15, floor=0.001)")

    # ═══════════════════════════════════════════════════
    # STEP 2: Global temperature sweep (verify Phase 2)
    # ═══════════════════════════════════════════════════
    print("\n=== GLOBAL TEMPERATURE SWEEP ===")
    for T in [0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        round_avgs = {}
        for rnum, seeds in cached.items():
            ss = [score_prediction(apply_global_temp_and_floor(rp, T, 0.001), gt)
                  for rp, gt, _ in seeds]
            round_avgs[rnum] = np.mean(ss)
        avg = np.mean(list(round_avgs.values()))
        per_round = "  ".join(f"R{r}={round_avgs[r]:.1f}" for r in sorted(round_avgs))
        print(f"  T={T:.2f}: avg={avg:.2f}  {per_round}")

    # ═══════════════════════════════════════════════════
    # STEP 3: Per-class temperature sweep (fast!)
    # ═══════════════════════════════════════════════════
    print("\n=== PER-CLASS TEMPERATURE SWEEP ===")
    print("  Classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain")

    # First: sweep each class independently to find promising ranges
    base_temps = np.array([1.10, 1.10, 1.10, 1.10, 1.10, 1.0])  # start from global best
    t_range = np.arange(0.8, 1.8, 0.05)

    print("\n  --- Independent per-class sweeps ---")
    best_per_class = base_temps.copy()
    for cls in range(5):  # skip Mountain
        cls_name = CLASS_NAMES[cls]
        best_t = base_temps[cls]
        best_avg = 0
        for t in t_range:
            temps = base_temps.copy()
            temps[cls] = t
            round_avgs = {}
            for rnum, seeds in cached.items():
                ss = [score_prediction(apply_per_class_temp_and_floor(rp, icg, temps, 0.001), gt)
                      for rp, gt, icg in seeds]
                round_avgs[rnum] = np.mean(ss)
            avg = np.mean(list(round_avgs.values()))
            if avg > best_avg:
                best_avg = avg
                best_t = t
        best_per_class[cls] = best_t
        print(f"  {cls_name:>12}: best T={best_t:.2f} (avg={best_avg:.2f})")

    # Now do a focused grid search around the independent optima
    print(f"\n  --- Focused grid search around independent optima ---")
    print(f"  Independent optima: E={best_per_class[0]:.2f} S={best_per_class[1]:.2f} "
          f"P={best_per_class[2]:.2f} R={best_per_class[3]:.2f} F={best_per_class[4]:.2f}")

    e_range = [best_per_class[0] + d for d in [-0.1, -0.05, 0, 0.05, 0.1]]
    s_range = [best_per_class[1] + d for d in [-0.1, -0.05, 0, 0.05, 0.1]]
    f_range = [best_per_class[4] + d for d in [-0.1, -0.05, 0, 0.05, 0.1]]
    # Port and Ruin: use independent optima (too few cells to matter)

    best_combo_avg = 0
    best_combo_temps = best_per_class.copy()
    n_combos = len(e_range) * len(s_range) * len(f_range)
    print(f"  Grid: {n_combos} combos (E×S×F)")

    for te in e_range:
        for ts in s_range:
            for tf in f_range:
                temps = np.array([te, ts, best_per_class[2], best_per_class[3], tf, 1.0])
                round_avgs = {}
                for rnum, seeds in cached.items():
                    ss = [score_prediction(
                        apply_per_class_temp_and_floor(rp, icg, temps, 0.001), gt)
                        for rp, gt, icg in seeds]
                    round_avgs[rnum] = np.mean(ss)
                avg = np.mean(list(round_avgs.values()))
                if avg > best_combo_avg:
                    best_combo_avg = avg
                    best_combo_temps = temps.copy()

    print(f"  Best combo: E={best_combo_temps[0]:.2f} S={best_combo_temps[1]:.2f} "
          f"P={best_combo_temps[2]:.2f} R={best_combo_temps[3]:.2f} "
          f"F={best_combo_temps[4]:.2f} M={best_combo_temps[5]:.2f} → avg={best_combo_avg:.2f}")

    # Compare with global T
    global_avgs = {}
    for rnum, seeds in cached.items():
        ss = [score_prediction(apply_global_temp_and_floor(rp, 1.10, 0.001), gt)
              for rp, gt, _ in seeds]
        global_avgs[rnum] = np.mean(ss)
    global_avg = np.mean(list(global_avgs.values()))
    print(f"  Global T=1.10 baseline: avg={global_avg:.2f}")
    print(f"  Per-class improvement: {best_combo_avg - global_avg:+.2f}")

    # Per-round breakdown of best combo
    use_per_class = best_combo_avg > global_avg + 0.05  # only if meaningfully better
    final_temps = best_combo_temps if use_per_class else None

    combo_round_avgs = {}
    for rnum, seeds in cached.items():
        if use_per_class:
            ss = [score_prediction(
                apply_per_class_temp_and_floor(rp, icg, best_combo_temps, 0.001), gt)
                for rp, gt, icg in seeds]
        else:
            ss = [score_prediction(apply_global_temp_and_floor(rp, 1.10, 0.001), gt)
                  for rp, gt, _ in seeds]
        combo_round_avgs[rnum] = np.mean(ss)
    per_round = "  ".join(f"R{r}={combo_round_avgs[r]:.1f}" for r in sorted(combo_round_avgs))
    print(f"  Final temps per-round: {per_round}")

    # ═══════════════════════════════════════════════════
    # STEP 4: Floor sweep
    # ═══════════════════════════════════════════════════
    print("\n=== PROBABILITY FLOOR SWEEP ===")
    floor_values = [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]
    best_floor = 0.001
    best_floor_avg = 0

    for floor in floor_values:
        round_avgs = {}
        for rnum, seeds in cached.items():
            if use_per_class:
                ss = [score_prediction(
                    apply_per_class_temp_and_floor(rp, icg, best_combo_temps, floor), gt)
                    for rp, gt, icg in seeds]
            else:
                ss = [score_prediction(apply_global_temp_and_floor(rp, 1.10, floor), gt)
                      for rp, gt, _ in seeds]
            round_avgs[rnum] = np.mean(ss)
        avg = np.mean(list(round_avgs.values()))
        per_round = "  ".join(f"R{r}={round_avgs[r]:.1f}" for r in sorted(round_avgs))
        print(f"  floor={floor:.5f}: avg={avg:.2f}  {per_round}")
        if avg > best_floor_avg:
            best_floor_avg = avg
            best_floor = floor

    print(f"\n  ** Best floor: {best_floor} (avg={best_floor_avg:.2f}) **")

    # ═══════════════════════════════════════════════════
    # STEP 5: Meta-model analysis
    # ═══════════════════════════════════════════════════
    print("\n=== ROUND FEATURE BIAS ANALYSIS ===")
    feat_names = ["E→E", "S→S", "F→F", "E→S", "sett_d"]
    obs_data = {}
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        gt_feats = compute_gt_round_features(detail, gts)
        try:
            cal = observation_calibrated_transitions(rid, detail)
            if cal is not None:
                deb = debias_transitions(cal)
                obs_f = compute_round_features(deb, detail)
                stats = _extract_settlement_stats(rid, detail)
                obs_data[rnum] = {"gt": gt_feats, "obs": obs_f, "err": gt_feats - obs_f,
                                  "stats": stats}
        except Exception:
            pass

    if obs_data:
        print(f"  {'Round':>6}", end="")
        for fn in feat_names:
            print(f" {fn:>8}", end="")
        print("  (GT - obs)")

        for rnum in sorted(obs_data.keys()):
            err = obs_data[rnum]["err"]
            print(f"  R{rnum:>4}", end="")
            for e in err:
                print(f" {e:>+8.4f}", end="")
            print()

        errors = np.array([obs_data[r]["err"] for r in sorted(obs_data.keys())])
        print(f"  {'Mean':>6}", end="")
        for e in errors.mean(axis=0):
            print(f" {e:>+8.4f}", end="")
        print(f"\n  {'Std':>6}", end="")
        for e in errors.std(axis=0):
            print(f" {e:>8.4f}", end="")
        print()

        # LOO bias correction test
        print("\n  LOO bias correction:")
        for rnum in sorted(obs_data.keys()):
            gt_f = obs_data[rnum]["gt"]
            obs_f = obs_data[rnum]["obs"]
            others = [obs_data[r]["err"] for r in obs_data if r != rnum]
            if others:
                bias = np.mean(others, axis=0)
                corrected = obs_f + bias
                err_before = np.abs(gt_f[:4] - obs_f[:4]).mean()
                err_after = np.abs(gt_f[:4] - corrected[:4]).mean()
                print(f"    R{rnum}: err_before={err_before:.4f}  err_after={err_after:.4f}  "
                      f"{'BETTER' if err_after < err_before else 'WORSE'}")

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"  LGB config:  heavy_reg (depth=4, est=500, alpha=1.0, lambda=1.0)")
    if use_per_class:
        print(f"  Temperature:  per-class E={best_combo_temps[0]:.2f} S={best_combo_temps[1]:.2f} "
              f"P={best_combo_temps[2]:.2f} R={best_combo_temps[3]:.2f} "
              f"F={best_combo_temps[4]:.2f} M={best_combo_temps[5]:.2f}")
    else:
        print(f"  Temperature:  global T=1.10")
    print(f"  Floor:        {best_floor}")
    print(f"  LORO avg:     {best_floor_avg:.2f}")
    print(f"  Previous:     ~75.50 (current model, T=1.15, floor=0.001)")
    print(f"  Gain:         {best_floor_avg - 75.50:+.2f}")


if __name__ == "__main__":
    main()
