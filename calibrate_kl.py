"""
Per-Entropy-Bucket Temperature Scaling — optimized directly for KL divergence.

The isotonic regression approach (calibrate_entropy.py) reduced MSE but *hurt*
KL-divergence scoring (-0.53 LORO avg). This suggests MSE-optimal calibration
doesn't align with KL-optimal calibration.

This script:
1. Loads cached LORO predictions (from calibrate_entropy.py)
2. Buckets cells by predicted entropy
3. Fits per-class per-bucket temperature parameters that minimize KL(GT || pred)
4. Also tries: per-bucket multiplicative calibration factors
5. Evaluates via LORO scoring
"""

import json
import pickle
import warnings
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from collections import defaultdict

warnings.filterwarnings("ignore")

from astar.model import apply_floor, PROB_FLOOR
from astar.submit import score_prediction
from astar.replay import NUM_CLASSES, CLASS_NAMES

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

ENTROPY_BUCKETS = [
    (0.0, 0.1, "static"),
    (0.1, 0.4, "low"),
    (0.4, 0.8, "medium"),
    (0.8, 2.0, "high"),
]


def predicted_entropy(pred: np.ndarray) -> np.ndarray:
    p = np.clip(pred, 1e-10, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def kl_divergence(gt: np.ndarray, pred: np.ndarray) -> float:
    """Average per-cell KL(gt || pred), weighted by gt entropy."""
    p = np.clip(pred, 1e-10, 1.0)
    g = np.clip(gt, 1e-10, 1.0)
    kl_per_cell = np.sum(g * np.log(g / p), axis=-1)
    # Weight by entropy of gt (matches scoring)
    ent = -np.sum(g * np.log(g), axis=-1)
    ent_total = ent.sum()
    if ent_total < 1e-10:
        return kl_per_cell.mean()
    return (kl_per_cell * ent).sum() / ent_total


def temperature_scale(pred: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """Apply per-class temperature scaling to logit space.

    temps: (6,) array of temperatures. >1 = soften, <1 = sharpen.
    """
    p = np.clip(pred, 1e-10, 1.0)
    logits = np.log(p)
    scaled = logits / np.maximum(temps, 0.01)
    # Softmax
    exp_scaled = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    return exp_scaled / exp_scaled.sum(axis=-1, keepdims=True)


def multiplicative_calibrate(pred: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Multiply predictions by per-class factors then renormalize."""
    cal = pred * factors
    cal = np.maximum(cal, 1e-10)
    return cal / cal.sum(axis=-1, keepdims=True)


def load_cached_predictions():
    """Load cached LORO predictions from calibrate_entropy.py."""
    cache_path = DATA_DIR / "loro_preds_cache.npz"
    meta_cache = DATA_DIR / "loro_meta_cache.pkl"

    if not cache_path.exists() or not meta_cache.exists():
        raise FileNotFoundError(
            "Run calibrate_entropy.py first to generate cached predictions")

    cached = np.load(cache_path)
    with open(meta_cache, "rb") as f:
        meta = pickle.load(f)
    return cached["preds"], cached["gts"], meta


def group_by_round_seed(preds, gts, meta, map_w=40, map_h=40):
    """Group flat predictions back into per-round per-seed maps."""
    groups = defaultdict(list)
    for i, (rnum, sidx) in enumerate(meta):
        groups[(rnum, sidx)].append(i)

    result = {}
    for (rnum, sidx), indices in sorted(groups.items()):
        p = preds[indices].reshape(map_h, map_w, NUM_CLASSES)
        g = gts[indices].reshape(map_h, map_w, NUM_CLASSES)
        result[(rnum, sidx)] = (p, g)
    return result


# =================== METHOD 1: Per-bucket temperature scaling ===================

def optimize_bucket_temperatures(preds, gts, bucket_bounds=None):
    """Find optimal per-bucket, per-class temperatures that minimize total weighted KL."""
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    ent = predicted_entropy(preds)
    best_temps = {}

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        n = mask.sum()
        if n < 200:
            best_temps[name] = np.ones(NUM_CLASSES)
            continue

        bucket_preds = preds[mask]
        bucket_gts = gts[mask]

        # Optimize per-class temperatures jointly
        def objective(log_temps):
            temps = np.exp(log_temps)
            cal = temperature_scale(bucket_preds, temps)
            return kl_divergence(bucket_gts, cal)

        # Start from 1.0 (log(1)=0)
        x0 = np.zeros(NUM_CLASSES)
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.001, 'fatol': 1e-8})

        optimal_temps = np.exp(result.x)
        best_temps[name] = optimal_temps

        # Report improvement
        kl_before = kl_divergence(bucket_gts, bucket_preds)
        cal_preds = temperature_scale(bucket_preds, optimal_temps)
        kl_after = kl_divergence(bucket_gts, cal_preds)

        print(f"  Bucket '{name}' [{lo:.1f},{hi:.1f}): {n} cells")
        print(f"    KL: {kl_before:.6f} -> {kl_after:.6f} ({100*(kl_after/kl_before-1):+.1f}%)")
        print(f"    Temps: {', '.join(f'{t:.3f}' for t in optimal_temps)}")

    return best_temps


def apply_bucket_temperatures(pred, bucket_temps, bucket_bounds=None):
    """Apply per-bucket temperature scaling to a prediction map."""
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    h, w, nc = pred.shape
    flat = pred.reshape(-1, nc).copy()
    ent = predicted_entropy(flat)

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        if not mask.any() or name not in bucket_temps:
            continue
        flat[mask] = temperature_scale(flat[mask], bucket_temps[name])

    return flat.reshape(h, w, nc)


# =================== METHOD 2: Per-bucket multiplicative factors ===================

def optimize_bucket_factors(preds, gts, bucket_bounds=None):
    """Find optimal per-bucket multiplicative calibration factors."""
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    ent = predicted_entropy(preds)
    best_factors = {}

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        n = mask.sum()
        if n < 200:
            best_factors[name] = np.ones(NUM_CLASSES)
            continue

        bucket_preds = preds[mask]
        bucket_gts = gts[mask]

        def objective(log_factors):
            factors = np.exp(log_factors)
            cal = multiplicative_calibrate(bucket_preds, factors)
            return kl_divergence(bucket_gts, cal)

        x0 = np.zeros(NUM_CLASSES)
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.001, 'fatol': 1e-8})

        optimal_factors = np.exp(result.x)
        best_factors[name] = optimal_factors

        kl_before = kl_divergence(bucket_gts, bucket_preds)
        cal_preds = multiplicative_calibrate(bucket_preds, optimal_factors)
        kl_after = kl_divergence(bucket_gts, cal_preds)

        print(f"  Bucket '{name}' [{lo:.1f},{hi:.1f}): {n} cells")
        print(f"    KL: {kl_before:.6f} -> {kl_after:.6f} ({100*(kl_after/kl_before-1):+.1f}%)")
        print(f"    Factors: {', '.join(f'{f:.3f}' for f in optimal_factors)}")

    return best_factors


def apply_bucket_factors(pred, bucket_factors, bucket_bounds=None):
    """Apply per-bucket multiplicative factors to a prediction map."""
    if bucket_bounds is None:
        bucket_bounds = ENTROPY_BUCKETS

    h, w, nc = pred.shape
    flat = pred.reshape(-1, nc).copy()
    ent = predicted_entropy(flat)

    for lo, hi, name in bucket_bounds:
        mask = (ent >= lo) & (ent < hi)
        if not mask.any() or name not in bucket_factors:
            continue
        flat[mask] = multiplicative_calibrate(flat[mask], bucket_factors[name])

    return flat.reshape(h, w, nc)


# =================== METHOD 3: Global (no bucketing) for comparison ===================

def optimize_global_temperature(preds, gts):
    """Optimize a single global per-class temperature vector."""
    def objective(log_temps):
        temps = np.exp(log_temps)
        cal = temperature_scale(preds, temps)
        return kl_divergence(gts, cal)

    x0 = np.zeros(NUM_CLASSES)
    result = minimize(objective, x0, method='Nelder-Mead',
                     options={'maxiter': 500, 'xatol': 0.001, 'fatol': 1e-8})
    optimal_temps = np.exp(result.x)

    kl_before = kl_divergence(gts, preds)
    cal_preds = temperature_scale(preds, optimal_temps)
    kl_after = kl_divergence(gts, cal_preds)

    print(f"  Global: {len(preds)} cells")
    print(f"    KL: {kl_before:.6f} -> {kl_after:.6f} ({100*(kl_after/kl_before-1):+.1f}%)")
    print(f"    Temps: {', '.join(f'{t:.3f}' for t in optimal_temps)}")

    return optimal_temps


# =================== Evaluation ===================

def evaluate_method(maps, apply_fn, label):
    """Evaluate a calibration method across all round/seed maps.

    maps: dict of (rnum, sidx) -> (pred, gt)
    apply_fn: function(pred) -> calibrated_pred
    """
    round_scores_base = defaultdict(list)
    round_scores_cal = defaultdict(list)

    for (rnum, sidx), (pred, gt) in sorted(maps.items()):
        base_pred = apply_floor(pred.copy())
        s_base = score_prediction(base_pred, gt)
        round_scores_base[rnum].append(s_base)

        cal_pred = apply_fn(pred)
        cal_pred = apply_floor(cal_pred)
        s_cal = score_prediction(cal_pred, gt)
        round_scores_cal[rnum].append(s_cal)

    print(f"\n  --- {label} ---")
    results = {}
    for rnum in sorted(round_scores_base.keys()):
        avg_base = np.mean(round_scores_base[rnum])
        avg_cal = np.mean(round_scores_cal[rnum])
        delta = avg_cal - avg_base
        results[rnum] = (avg_base, avg_cal, delta)
        marker = "+" if delta > 0.05 else ("-" if delta < -0.05 else "=")
        print(f"    R{rnum:2d}: base={avg_base:.2f}  cal={avg_cal:.2f}  d={delta:+.2f} {marker}")

    base_avg = np.mean([v[0] for v in results.values()])
    cal_avg = np.mean([v[1] for v in results.values()])
    delta_avg = cal_avg - base_avg
    print(f"    LORO avg: base={base_avg:.2f}  cal={cal_avg:.2f}  delta={delta_avg:+.2f}")

    improved = sum(1 for v in results.values() if v[2] > 0.05)
    regressed = sum(1 for v in results.values() if v[2] < -0.05)
    print(f"    Improved: {improved}/{len(results)}, Regressed: {regressed}/{len(results)}")

    return results, cal_avg


def main():
    import time
    t0 = time.time()

    print("Loading cached LORO predictions...")
    preds, gts, meta = load_cached_predictions()
    maps = group_by_round_seed(preds, gts, meta)
    print(f"  {len(preds)} cells, {len(maps)} maps")

    # METHOD 1: Per-bucket temperature scaling (KL-optimized)
    print("\n" + "=" * 60)
    print("METHOD 1: Per-Bucket Temperature Scaling")
    print("=" * 60)
    bucket_temps = optimize_bucket_temperatures(preds, gts)

    def apply_m1(pred):
        return apply_bucket_temperatures(pred, bucket_temps)

    r1, s1 = evaluate_method(maps, apply_m1, "Per-Bucket Temps")

    # METHOD 2: Per-bucket multiplicative factors
    print("\n" + "=" * 60)
    print("METHOD 2: Per-Bucket Multiplicative Factors")
    print("=" * 60)
    bucket_factors = optimize_bucket_factors(preds, gts)

    def apply_m2(pred):
        return apply_bucket_factors(pred, bucket_factors)

    r2, s2 = evaluate_method(maps, apply_m2, "Per-Bucket Factors")

    # METHOD 3: Global temperature (no bucketing baseline)
    print("\n" + "=" * 60)
    print("METHOD 3: Global Temperature (no bucketing)")
    print("=" * 60)
    global_temps = optimize_global_temperature(preds, gts)

    def apply_m3(pred):
        return temperature_scale(pred.reshape(-1, NUM_CLASSES), global_temps).reshape(pred.shape)

    r3, s3 = evaluate_method(maps, apply_m3, "Global Temps")

    # METHOD 4: Per-bucket temperature, but only for high-entropy cells
    print("\n" + "=" * 60)
    print("METHOD 4: Temperature only on high-entropy cells")
    print("=" * 60)
    high_only_bounds = [(0.5, 2.0, "high")]
    high_temps = optimize_bucket_temperatures(preds, gts, high_only_bounds)

    def apply_m4(pred):
        return apply_bucket_temperatures(pred, high_temps, high_only_bounds)

    r4, s4 = evaluate_method(maps, apply_m4, "High-Entropy Only Temps")

    # METHOD 5: Smaller temperature adjustments — regularized toward 1.0
    print("\n" + "=" * 60)
    print("METHOD 5: Regularized per-bucket temperatures")
    print("=" * 60)

    ent = predicted_entropy(preds)
    reg_temps = {}

    for lo, hi, name in ENTROPY_BUCKETS:
        mask = (ent >= lo) & (ent < hi)
        n = mask.sum()
        if n < 200:
            reg_temps[name] = np.ones(NUM_CLASSES)
            continue

        bucket_preds = preds[mask]
        bucket_gts = gts[mask]

        # KL + L2 regularization toward temp=1.0
        def objective(log_temps):
            temps = np.exp(log_temps)
            cal = temperature_scale(bucket_preds, temps)
            kl = kl_divergence(bucket_gts, cal)
            # Regularize: penalize deviation from 1.0
            reg = 0.1 * np.sum((temps - 1.0) ** 2)
            return kl + reg

        x0 = np.zeros(NUM_CLASSES)
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 0.001, 'fatol': 1e-8})
        optimal_temps = np.exp(result.x)
        reg_temps[name] = optimal_temps

        kl_before = kl_divergence(bucket_gts, bucket_preds)
        cal_preds = temperature_scale(bucket_preds, optimal_temps)
        kl_after = kl_divergence(bucket_gts, cal_preds)
        print(f"  Bucket '{name}': KL {kl_before:.6f} -> {kl_after:.6f}, Temps: {', '.join(f'{t:.3f}' for t in optimal_temps)}")

    def apply_m5(pred):
        return apply_bucket_temperatures(pred, reg_temps)

    r5, s5 = evaluate_method(maps, apply_m5, "Regularized Bucket Temps")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    base_avg = np.mean([v[0] for v in r1.values()])
    methods = [
        ("Baseline (no cal)", base_avg),
        ("M1: Per-Bucket Temps", s1),
        ("M2: Per-Bucket Factors", s2),
        ("M3: Global Temps", s3),
        ("M4: High-Entropy Only", s4),
        ("M5: Regularized Bucket", s5),
    ]
    for name, score in sorted(methods, key=lambda x: -x[1]):
        delta = score - base_avg
        print(f"  {name:30s}: {score:.2f} ({delta:+.2f})")

    # Save best method artifacts
    best_method = max(methods[1:], key=lambda x: x[1])
    print(f"\n  Best: {best_method[0]} ({best_method[1]:.2f})")

    save_data = {
        "bucket_temps": bucket_temps,
        "bucket_factors": bucket_factors,
        "global_temps": global_temps,
        "high_only_temps": high_temps,
        "reg_temps": reg_temps,
        "bucket_bounds": ENTROPY_BUCKETS,
    }
    save_path = DATA_DIR / "kl_calibration_params.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"  Saved all calibration params to {save_path}")

    elapsed = time.time() - t0
    print(f"\n  Wall-clock: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
