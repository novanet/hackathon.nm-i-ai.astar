"""Quick sweep of calibration factors and floor on cached LORO predictions.
Uses LGB-only LORO cache (same as calibrate_kl.py), applies M5 temps + calibration + floor.
"""
import numpy as np
import pickle
from pathlib import Path
from itertools import product

from astar.model import (
    apply_floor, PROB_FLOOR, ENTROPY_BUCKET_BOUNDS, ENTROPY_BUCKET_TEMPS,
    entropy_bucket_temperature_scale,
)
from astar.submit import score_prediction
from astar.replay import NUM_CLASSES

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


def apply_cal_and_score(preds_by_round, gts_by_round, cal_factors, floor_val):
    """Apply calibration → M5 temps → floor → score per round."""
    scores = {}
    cal = np.array(cal_factors, dtype=np.float64)
    for rnum in preds_by_round:
        round_scores = []
        for pred, gt in zip(preds_by_round[rnum], gts_by_round[rnum]):
            # Calibration
            p = pred * cal[np.newaxis, np.newaxis, :]
            p = p / p.sum(axis=-1, keepdims=True)
            # M5 bucket temps
            p = entropy_bucket_temperature_scale(p)
            # Floor
            p = np.maximum(p, floor_val)
            p = p / p.sum(axis=-1, keepdims=True)
            round_scores.append(score_prediction(p, gt))
        scores[rnum] = np.mean(round_scores)
    return scores


def main():
    # Load cached LORO predictions
    cache_path = DATA_DIR / "loro_preds_cache.npz"
    meta_path = DATA_DIR / "loro_meta_cache.pkl"
    cached = np.load(cache_path)
    preds_flat, gts_flat = cached["preds"], cached["gts"]
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # Organize by round and seed → reshape to (H, W, C)
    map_w, map_h = 40, 40
    cells_per_map = map_w * map_h
    preds_by_round = {}
    gts_by_round = {}

    idx = 0
    while idx < len(meta):
        rnum, seed = meta[idx]
        pred_chunk = preds_flat[idx:idx+cells_per_map].reshape(map_h, map_w, NUM_CLASSES)
        gt_chunk = gts_flat[idx:idx+cells_per_map].reshape(map_h, map_w, NUM_CLASSES)
        preds_by_round.setdefault(rnum, []).append(pred_chunk)
        gts_by_round.setdefault(rnum, []).append(gt_chunk)
        idx += cells_per_map

    print(f"Loaded {len(preds_by_round)} rounds, {sum(len(v) for v in preds_by_round.values())} maps")

    # Current baseline: cal=[1,1,1,1,0.95,1], floor=0.0001
    baseline = apply_cal_and_score(preds_by_round, gts_by_round,
                                    [1.0, 1.0, 1.0, 1.0, 0.95, 1.0], 0.0001)
    base_avg = np.mean(list(baseline.values()))
    print(f"\nBaseline (F=0.95, floor=0.0001): LORO avg = {base_avg:.3f}")
    for r in sorted(baseline):
        print(f"  R{r:2d}: {baseline[r]:.2f}")

    # Sweep calibration factor for Forest
    print("\n=== FOREST CALIBRATION SWEEP ===")
    for f_cal in [0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 1.0]:
        result = apply_cal_and_score(preds_by_round, gts_by_round,
                                      [1.0, 1.0, 1.0, 1.0, f_cal, 1.0], 0.0001)
        avg = np.mean(list(result.values()))
        delta = avg - base_avg
        wins = sum(1 for r in result if result[r] > baseline[r])
        print(f"  F={f_cal:.2f}: avg={avg:.3f} (Δ={delta:+.3f}, {wins}/15 wins)")

    # Sweep Empty calibration too
    print("\n=== EMPTY CALIBRATION SWEEP ===")
    for e_cal in [0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03]:
        result = apply_cal_and_score(preds_by_round, gts_by_round,
                                      [e_cal, 1.0, 1.0, 1.0, 0.95, 1.0], 0.0001)
        avg = np.mean(list(result.values()))
        delta = avg - base_avg
        wins = sum(1 for r in result if result[r] > baseline[r])
        print(f"  E={e_cal:.2f}: avg={avg:.3f} (Δ={delta:+.3f}, {wins}/15 wins)")

    # Sweep Settlement calibration
    print("\n=== SETTLEMENT CALIBRATION SWEEP ===")
    for s_cal in [0.90, 0.92, 0.95, 0.98, 1.0, 1.02, 1.05]:
        result = apply_cal_and_score(preds_by_round, gts_by_round,
                                      [1.0, s_cal, 1.0, 1.0, 0.95, 1.0], 0.0001)
        avg = np.mean(list(result.values()))
        delta = avg - base_avg
        wins = sum(1 for r in result if result[r] > baseline[r])
        print(f"  S={s_cal:.2f}: avg={avg:.3f} (Δ={delta:+.3f}, {wins}/15 wins)")

    # Sweep floor
    print("\n=== FLOOR SWEEP ===")
    for floor in [0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.001]:
        result = apply_cal_and_score(preds_by_round, gts_by_round,
                                      [1.0, 1.0, 1.0, 1.0, 0.95, 1.0], floor)
        avg = np.mean(list(result.values()))
        delta = avg - base_avg
        wins = sum(1 for r in result if result[r] > baseline[r])
        print(f"  floor={floor:.5f}: avg={avg:.3f} (Δ={delta:+.3f}, {wins}/15 wins)")

    # Test no calibration at all
    print("\n=== NO CALIBRATION ===")
    result = apply_cal_and_score(preds_by_round, gts_by_round,
                                  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0001)
    avg = np.mean(list(result.values()))
    delta = avg - base_avg
    wins = sum(1 for r in result if result[r] > baseline[r])
    print(f"  No cal: avg={avg:.3f} (Δ={delta:+.3f}, {wins}/15 wins)")


if __name__ == "__main__":
    main()
