"""
Per-Entropy-Bucket Calibration via Isotonic Regression.

Idea: High-entropy cells (settlement/forest boundaries) dominate 88-99% of KL loss.
Current global CALIBRATION_FACTORS applies the same correction to all cells.
Instead, bucket cells by predicted entropy and fit per-bucket, per-class isotonic
regression to map predicted probabilities closer to GT.

This script:
1. Runs LORO CV to collect out-of-sample predictions + GT for every cell
2. Computes predicted entropy per cell
3. Buckets cells and fits isotonic regression per (bucket, class)
4. Evaluates calibrated predictions vs uncalibrated in LORO

Does NOT modify model.py — produces a calibration artifact that can be integrated later.
"""

import json
import pickle
import warnings
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from astar.model import compute_cell_features, apply_floor, PROB_FLOOR
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES

DATA_DIR = Path("data")

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
}

# Entropy bucket boundaries — splitting dynamic vs static cells
ENTROPY_BUCKETS = [
    (0.0, 0.1, "static"),       # near-deterministic cells (mountain, deep forest)
    (0.1, 0.4, "low"),          # mostly decided, slight uncertainty
    (0.4, 0.8, "medium"),       # moderate uncertainty
    (0.8, 2.0, "high"),         # high uncertainty (boundaries, volatile cells)
]

ENTROPY_WEIGHT_POWER = 0.5


def load_round_data(round_id: str):
    """Load round detail and ground truths (matches train_spatial.py)."""
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gtp = rdir / f"ground_truth_s{s}.json"
        if not gtp.exists():
            break
        gt_data = json.loads(gtp.read_text(encoding="utf-8"))
        gt = np.array(gt_data["ground_truth"], dtype=np.float64)
        gts.append(gt)
    return detail, gts


def compute_gt_round_features(detail: dict, gts: list[np.ndarray]) -> np.ndarray:
    """Compute round features from GT (matches train_spatial.py exactly)."""
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
    ss_rate = gt_trans[1, 1]
    return np.array([
        gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4], gt_trans[0, 1],
        sett_density,
        0.3 + 0.7 * ss_rate,
        gt_trans[0, 1] * 0.3,
        1.0 - gt_trans[1, 0],
    ], dtype=np.float64)


def predicted_entropy(pred: np.ndarray) -> np.ndarray:
    """Compute per-cell entropy from predictions. Shape: (H, W) or (N,)."""
    p = np.clip(pred, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def collect_loro_predictions() -> tuple[np.ndarray, np.ndarray, list]:
    """Run LORO CV, collect all out-of-sample predictions and GT.

    Returns:
        all_preds: (N, 6) raw model predictions (before calibration/floor)
        all_gts:   (N, 6) ground truth
        all_meta:  list of (round_num, seed_idx) per sample
    """
    print("=== COLLECTING LORO PREDICTIONS ===")

    # Load all data
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    all_preds = []
    all_gts = []
    all_meta = []

    for test_rnum in sorted(all_data.keys()):
        # Train on all other rounds
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_parts, Y_parts, W_parts = [], [], []

        for rnum, rid in sorted(train_ids.items()):
            detail, gts = load_round_data(rid)
            if not gts:
                continue
            map_w = detail.get("map_width", 40)
            map_h = detail.get("map_height", 40)
            round_feats = compute_gt_round_features(detail, gts)
            for seed_idx, gt in enumerate(gts):
                init_grid = detail["initial_states"][seed_idx]["grid"]
                features = compute_cell_features(init_grid, map_w, map_h,
                                                 round_features=round_feats)
                X_parts.append(features.reshape(-1, features.shape[-1]))
                Y_parts.append(gt.reshape(-1, gt.shape[-1]))
                p = np.clip(gt, 1e-10, 1.0)
                entropy = -np.sum(p * np.log(p), axis=-1).flatten()
                W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))

        X_train = np.vstack(X_parts)
        Y_train = np.vstack(Y_parts)
        W_train = np.concatenate(W_parts)

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

        # Predict on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)

            all_preds.append(pred.reshape(-1, NUM_CLASSES))
            all_gts.append(gt.reshape(-1, NUM_CLASSES))
            all_meta.extend([(test_rnum, s)] * (map_w * map_h))

        print(f"  R{test_rnum}: done ({len(test_gts)} seeds)")

    return np.vstack(all_preds), np.vstack(all_gts), all_meta


def evaluate_from_cached(preds: np.ndarray, gts: np.ndarray, meta: list,
                         calibrators: dict, bucket_bounds: list[tuple],
                         map_w: int = 40, map_h: int = 40):
    """Evaluate calibration using cached LORO predictions (no re-training).

    Groups predictions back into per-round per-seed maps and scores them.
    """
    print("\n=== EVALUATION: BASELINE vs CALIBRATED (from cache) ===")
    nc = preds.shape[1]
    cells_per_map = map_w * map_h

    # Group by (round, seed)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, (rnum, sidx) in enumerate(meta):
        groups[(rnum, sidx)].append(i)

    # Score per round
    round_scores_base = defaultdict(list)
    round_scores_cal = defaultdict(list)

    for (rnum, sidx), indices in sorted(groups.items()):
        p = preds[indices]
        g = gts[indices]

        # Reshape to map
        pred_map = p.reshape(map_h, map_w, nc)
        gt_map = g.reshape(map_h, map_w, nc)

        # Baseline: just floor
        base_pred = apply_floor(pred_map.copy())
        score_base = score_prediction(base_pred, gt_map)
        round_scores_base[rnum].append(score_base)

        # Calibrated
        cal_pred = apply_entropy_calibration(pred_map, calibrators, bucket_bounds)
        cal_pred = apply_floor(cal_pred)
        score_cal = score_prediction(cal_pred, gt_map)
        round_scores_cal[rnum].append(score_cal)

    results = {}
    for rnum in sorted(round_scores_base.keys()):
        avg_base = np.mean(round_scores_base[rnum])
        avg_cal = np.mean(round_scores_cal[rnum])
        delta = avg_cal - avg_base
        results[rnum] = (avg_base, avg_cal, delta)
        marker = "+" if delta > 0.05 else ("-" if delta < -0.05 else "=")
        print(f"  R{rnum:2d}: base={avg_base:.2f}  cal={avg_cal:.2f}  d={delta:+.2f} {marker}")

    base_avg = np.mean([v[0] for v in results.values()])
    cal_avg = np.mean([v[1] for v in results.values()])
    delta_avg = cal_avg - base_avg
    print(f"\n  LORO avg: base={base_avg:.2f}  cal={cal_avg:.2f}  delta={delta_avg:+.2f}")

    improved = sum(1 for v in results.values() if v[2] > 0.05)
    regressed = sum(1 for v in results.values() if v[2] < -0.05)
    print(f"  Improved: {improved}/{len(results)} rounds, Regressed: {regressed}/{len(results)} rounds")

    return results


def fit_entropy_calibration(preds: np.ndarray, gts: np.ndarray,
                            bucket_bounds: list[tuple] | None = None,
                            ) -> dict:
    """Fit isotonic regression per (entropy_bucket, class).

    Args:
        preds: (N, 6) raw predictions
        gts: (N, 6) ground truth
        bucket_bounds: list of (lo, hi, name) tuples

    Returns:
        calibrators: dict mapping (bucket_name, class_idx) → IsotonicRegression
    """
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    # Compute predicted entropy
    ent = predicted_entropy(preds)

    calibrators = {}
    print("\n=== FITTING ISOTONIC CALIBRATION ===")

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        n_cells = mask.sum()
        print(f"  Bucket '{name}' [{lo:.1f}, {hi:.1f}): {n_cells} cells ({100*n_cells/len(ent):.1f}%)")

        if n_cells < 200:
            print(f"    Too few cells, skipping (min 200)")
            continue

        for c in range(NUM_CLASSES):
            p = preds[mask, c]
            g = gts[mask, c]

            # Only fit if there's variance in both pred and GT
            if p.std() < 1e-6 or g.std() < 1e-6:
                continue

            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            ir.fit(p, g)

            # Check if calibration actually improves MSE
            calibrated = ir.predict(p)
            mse_before = np.mean((p - g) ** 2)
            mse_after = np.mean((calibrated - g) ** 2)

            if mse_after < mse_before:
                calibrators[(name, c)] = ir
                print(f"    Class {c} ({CLASS_NAMES[c]:>10}): MSE {mse_before:.6f} -> {mse_after:.6f} ({100*(mse_after/mse_before-1):+.1f}%)")
            else:
                print(f"    Class {c} ({CLASS_NAMES[c]:>10}): no improvement (MSE {mse_before:.6f} -> {mse_after:.6f}), skipped")

    print(f"\n  Total calibrators fitted: {len(calibrators)}")
    return calibrators


def apply_entropy_calibration(pred: np.ndarray, calibrators: dict,
                              bucket_bounds: list[tuple] | None = None) -> np.ndarray:
    """Apply per-entropy-bucket isotonic calibration to prediction.

    Args:
        pred: (H, W, 6) prediction array
        calibrators: dict from fit_entropy_calibration
        bucket_bounds: list of (lo, hi, name) tuples

    Returns:
        calibrated: (H, W, 6) calibrated prediction
    """
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    h, w, nc = pred.shape
    flat = pred.reshape(-1, nc).copy()
    ent = predicted_entropy(flat)

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        if not mask.any():
            continue
        for c in range(nc):
            key = (name, c)
            if key in calibrators:
                flat[mask, c] = calibrators[key].predict(flat[mask, c])

    # Renormalize
    flat = np.maximum(flat, 1e-10)
    flat = flat / flat.sum(axis=1, keepdims=True)
    return flat.reshape(h, w, nc)


def evaluate_calibration(calibrators: dict, bucket_bounds: list[tuple] | None = None):
    """Run full LORO evaluation with and without per-entropy calibration.

    Uses the same LORO loop as collect_loro_predictions but applies calibration
    before scoring each seed.
    """
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    print("\n=== LORO EVALUATION: BASELINE vs CALIBRATED ===")

    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)

    results = {}

    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_parts, Y_parts, W_parts = [], [], []

        for rnum, rid in sorted(train_ids.items()):
            detail, gts = load_round_data(rid)
            if not gts:
                continue
            map_w = detail.get("map_width", 40)
            map_h = detail.get("map_height", 40)
            round_feats = compute_gt_round_features(detail, gts)
            for seed_idx, gt in enumerate(gts):
                init_grid = detail["initial_states"][seed_idx]["grid"]
                features = compute_cell_features(init_grid, map_w, map_h,
                                                 round_features=round_feats)
                X_parts.append(features.reshape(-1, features.shape[-1]))
                Y_parts.append(gt.reshape(-1, gt.shape[-1]))
                p = np.clip(gt, 1e-10, 1.0)
                entropy = -np.sum(p * np.log(p), axis=-1).flatten()
                W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))

        X_train = np.vstack(X_parts)
        Y_train = np.vstack(Y_parts)
        W_train = np.concatenate(W_parts)

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

        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        scores_base = []
        scores_cal = []

        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)

            # Score baseline (just floor)
            scores_base.append(score_prediction(apply_floor(pred), gt))

            # Score calibrated
            cal_pred = apply_entropy_calibration(pred, calibrators, bucket_bounds)
            scores_cal.append(score_prediction(apply_floor(cal_pred), gt))

        avg_base = np.mean(scores_base)
        avg_cal = np.mean(scores_cal)
        delta = avg_cal - avg_base
        results[test_rnum] = (avg_base, avg_cal, delta)
        marker = "+" if delta > 0.05 else ("-" if delta < -0.05 else "=")
        print(f"  R{test_rnum:2d}: base={avg_base:.2f}  cal={avg_cal:.2f}  d={delta:+.2f} {marker}  seeds_base={[f'{s:.1f}' for s in scores_base]}  seeds_cal={[f'{s:.1f}' for s in scores_cal]}")

    base_avg = np.mean([v[0] for v in results.values()])
    cal_avg = np.mean([v[1] for v in results.values()])
    delta_avg = cal_avg - base_avg
    print(f"\n  LORO avg: base={base_avg:.2f}  cal={cal_avg:.2f}  delta={delta_avg:+.2f}")

    improved = sum(1 for v in results.values() if v[2] > 0.05)
    regressed = sum(1 for v in results.values() if v[2] < -0.05)
    print(f"  Improved: {improved}/{len(results)} rounds, Regressed: {regressed}/{len(results)} rounds")

    worst_reg = min(results.values(), key=lambda v: v[2])
    worst_rnum = [k for k, v in results.items() if v == worst_reg][0]
    print(f"  Worst regression: R{worst_rnum} ({worst_reg[2]:+.2f})")

    return results


def sweep_bucket_boundaries():
    """Try different entropy bucket configurations."""
    configs = {
        "4-bucket (default)": [
            (0.0, 0.1, "static"),
            (0.1, 0.4, "low"),
            (0.4, 0.8, "medium"),
            (0.8, 2.0, "high"),
        ],
        "3-bucket (simple)": [
            (0.0, 0.2, "low"),
            (0.2, 0.6, "medium"),
            (0.6, 2.0, "high"),
        ],
        "2-bucket (binary)": [
            (0.0, 0.3, "low"),
            (0.3, 2.0, "high"),
        ],
        "5-bucket (fine)": [
            (0.0, 0.05, "static"),
            (0.05, 0.2, "low"),
            (0.2, 0.5, "medium"),
            (0.5, 1.0, "high"),
            (1.0, 2.0, "very_high"),
        ],
    }

    # Collect predictions once
    preds, gts, meta = collect_loro_predictions()

    print("\n" + "=" * 60)
    print("=== SWEEPING BUCKET CONFIGURATIONS ===")
    print("=" * 60)

    best_name = None
    best_avg = -1
    best_calibrators = None
    best_bounds = None

    for name, bounds in configs.items():
        print(f"\n--- Config: {name} ---")
        calibrators = fit_entropy_calibration(preds, gts, bounds)
        results = evaluate_calibration(calibrators, bounds)

        cal_avg = np.mean([v[1] for v in results.values()])
        if cal_avg > best_avg:
            best_avg = cal_avg
            best_name = name
            best_calibrators = calibrators
            best_bounds = bounds

    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_name} (LORO avg = {best_avg:.2f})")
    print(f"{'='*60}")

    return best_calibrators, best_bounds


def main():
    import sys
    import time

    t0 = time.time()

    cache_path = DATA_DIR / "loro_preds_cache.npz"
    meta_cache = DATA_DIR / "loro_meta_cache.pkl"

    if "--sweep" in sys.argv:
        calibrators, bounds = sweep_bucket_boundaries()
    else:
        # Load cached predictions or collect fresh
        if cache_path.exists() and meta_cache.exists() and "--nocache" not in sys.argv:
            print("Loading cached LORO predictions...")
            cached = np.load(cache_path)
            preds, gts = cached["preds"], cached["gts"]
            with open(meta_cache, "rb") as f:
                meta = pickle.load(f)
            print(f"  Loaded {len(preds)} cells from cache")
        else:
            preds, gts, meta = collect_loro_predictions()
            np.savez_compressed(cache_path, preds=preds, gts=gts)
            with open(meta_cache, "wb") as f:
                pickle.dump(meta, f)
            print(f"  Cached predictions to {cache_path}")

        # Analyze entropy distribution
        ent = predicted_entropy(preds)
        print(f"\n  Entropy stats: mean={ent.mean():.3f}, median={np.median(ent):.3f}, "
              f"max={ent.max():.3f}, >0.1: {(ent>0.1).sum()}, >0.5: {(ent>0.5).sum()}")

        # Distribution per bucket
        for lo, hi, name in ENTROPY_BUCKETS:
            mask = (ent >= lo) & (ent < hi)
            n = mask.sum()
            if n > 0:
                gt_ent = predicted_entropy(gts[mask])
                print(f"  Bucket '{name}' [{lo:.1f},{hi:.1f}): {n} cells, "
                      f"mean_pred_ent={ent[mask].mean():.3f}, mean_gt_ent={gt_ent.mean():.3f}")

        bounds = ENTROPY_BUCKETS
        calibrators = fit_entropy_calibration(preds, gts, bounds)

        # Fast evaluation from cached predictions
        results = evaluate_from_cached(preds, gts, meta, calibrators, bounds)

        # Save calibrators for later integration
        save_path = DATA_DIR / "entropy_calibrators.pkl"
        pickle.dump({
            "calibrators": calibrators,
            "bucket_bounds": bounds,
        }, open(save_path, "wb"))
        print(f"\n  Saved calibrators to {save_path}")

    elapsed = time.time() - t0
    print(f"\n  Wall-clock: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
