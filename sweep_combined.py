"""Quick combined sweep of best calibration + floor combos."""
import numpy as np
import pickle
from pathlib import Path
from astar.model import entropy_bucket_temperature_scale
from astar.submit import score_prediction
from astar.replay import NUM_CLASSES

DATA_DIR = Path("data")

def apply_cal_and_score(preds_by_round, gts_by_round, cal_factors, floor_val):
    cal = np.array(cal_factors, dtype=np.float64)
    scores = {}
    for rnum in preds_by_round:
        round_scores = []
        for pred, gt in zip(preds_by_round[rnum], gts_by_round[rnum]):
            p = pred * cal[np.newaxis, np.newaxis, :]
            p = p / p.sum(axis=-1, keepdims=True)
            p = entropy_bucket_temperature_scale(p)
            p = np.maximum(p, floor_val)
            p = p / p.sum(axis=-1, keepdims=True)
            round_scores.append(score_prediction(p, gt))
        scores[rnum] = np.mean(round_scores)
    return scores

def main():
    cached = np.load(DATA_DIR / "loro_preds_cache.npz")
    preds_flat, gts_flat = cached["preds"], cached["gts"]
    with open(DATA_DIR / "loro_meta_cache.pkl", "rb") as f:
        meta = pickle.load(f)
    
    map_w, map_h = 40, 40
    cells_per_map = map_w * map_h
    preds_by_round, gts_by_round = {}, {}
    idx = 0
    while idx < len(meta):
        rnum, seed = meta[idx]
        preds_by_round.setdefault(rnum, []).append(
            preds_flat[idx:idx+cells_per_map].reshape(map_h, map_w, NUM_CLASSES))
        gts_by_round.setdefault(rnum, []).append(
            gts_flat[idx:idx+cells_per_map].reshape(map_h, map_w, NUM_CLASSES))
        idx += cells_per_map

    # Configs to test: (name, cal_factors, floor)
    configs = [
        ("Current",         [1.0, 1.0, 1.0, 1.0, 0.95, 1.0], 0.0001),
        ("F=0.98+fl0.0003", [1.0, 1.0, 1.0, 1.0, 0.98, 1.0], 0.0003),
        ("E=0.97+F=0.98+fl0.0003", [0.97, 1.0, 1.0, 1.0, 0.98, 1.0], 0.0003),
        ("NoCal+fl0.0003",  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0003),
        ("E=0.97+NoCal+fl0.0003", [0.97, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0003),
        ("F=0.97+fl0.0003", [1.0, 1.0, 1.0, 1.0, 0.97, 1.0], 0.0003),
        ("E=0.97+F=0.97+fl0.0003", [0.97, 1.0, 1.0, 1.0, 0.97, 1.0], 0.0003),
        ("Best combo: E=0.97+F=0.98+fl0.0002", [0.97, 1.0, 1.0, 1.0, 0.98, 1.0], 0.0002),
    ]

    print(f"{'Config':<35} {'Avg':>7} {'Δ':>7}  Per-round")
    print("-" * 100)

    baseline_scores = None
    baseline_avg = None
    for name, cal, floor in configs:
        scores = apply_cal_and_score(preds_by_round, gts_by_round, cal, floor)
        avg = np.mean(list(scores.values()))
        if baseline_scores is None:
            baseline_scores = scores
            baseline_avg = avg
        delta = avg - baseline_avg
        wins = sum(1 for r in scores if scores[r] > baseline_scores.get(r, 0))
        per_round = " ".join(f"{scores[r]:.1f}" for r in sorted(scores))
        print(f"{name:<35} {avg:7.3f} {delta:+7.3f}  {per_round}  ({wins}/15)")


if __name__ == "__main__":
    main()
