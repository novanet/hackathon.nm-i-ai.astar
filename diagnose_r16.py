"""Diagnose R16: compare old settings vs new settings on R16 ground truth."""
import numpy as np
from pathlib import Path
from astar.client import get_round_detail
from astar.model import build_prediction, apply_floor
from astar.replay import load_ground_truth_array
from astar.submit import score_prediction

R16_ID = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"
DATA_DIR = Path("data") / f"round_{R16_ID}"


def load_gt(seed_idx: int) -> np.ndarray:
    gt = load_ground_truth_array(R16_ID, seed_idx)
    if gt is None:
        raise FileNotFoundError(f"Missing ground truth for seed {seed_idx} in {DATA_DIR}")
    return gt


def score_with_settings(detail, seed_idx, gt, bayes_min, bayes_max, floor):
    """Score a prediction with specific overlay/floor settings."""
    import astar.model as m
    # Save originals
    orig_min = m._BAYES_MIN_PS
    orig_max = m._BAYES_MAX_PS
    orig_floor = m.PROB_FLOOR

    # Override
    m._BAYES_MIN_PS = bayes_min
    m._BAYES_MAX_PS = bayes_max
    m.PROB_FLOOR = floor

    try:
        pred = build_prediction(R16_ID, detail, seed_idx)
        score = score_prediction(pred, gt)
        return score
    finally:
        # Restore
        m._BAYES_MIN_PS = orig_min
        m._BAYES_MAX_PS = orig_max
        m.PROB_FLOOR = orig_floor


def main():
    detail = get_round_detail(R16_ID)
    n_seeds = len(detail.get("initial_states", []))

    configs = {
        "NEW (0.5/3/0.0003)":    (0.5,   3.0,   0.0003),
        "OLD (5/100/0.0001)":    (5.0,   100.0, 0.0001),
        "No overlay (999/999)":  (999.0, 999.0, 0.0003),
        "Stronger (0.3/2)":      (0.3,   2.0,   0.0003),
        "Mid (1.0/5)":           (1.0,   5.0,   0.0003),
    }

    print(f"{'Config':<25s}", end="")
    for s in range(n_seeds):
        print(f"  {'S'+str(s):>6s}", end="")
    print(f"  {'AVG':>7s}")
    print("-" * 75)

    for name, (bmin, bmax, floor) in configs.items():
        scores = []
        print(f"{name:<25s}", end="", flush=True)
        for seed_idx in range(n_seeds):
            gt = load_gt(seed_idx)
            sc = score_with_settings(detail, seed_idx, gt, bmin, bmax, floor)
            scores.append(sc)
            print(f"  {sc:6.2f}", end="", flush=True)
        avg = np.mean(scores)
        print(f"  {avg:7.2f}")

    print()
    # Also check what the submitted scores were
    print("Submitted scores: 67.75, 75.10, 74.24, 66.52, 73.94 → avg 71.51")


if __name__ == "__main__":
    main()
