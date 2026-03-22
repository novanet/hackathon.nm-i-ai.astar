"""Quick: test different configs on R16 GT to find what actually works best now."""
import numpy as np
from pathlib import Path
from astar.client import get_round_detail
from astar.model import build_prediction
from astar.replay import load_ground_truth_array
from astar.submit import score_prediction
import astar.model as m

R16_ID = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"
DATA_DIR = Path("data") / f"round_{R16_ID}"


def load_gt(seed_idx):
    gt = load_ground_truth_array(R16_ID, seed_idx)
    if gt is None:
        raise FileNotFoundError(f"Missing ground truth for seed {seed_idx} in {DATA_DIR}")
    return gt


def test_config(detail, n_seeds, label, bayes_min, bayes_max, floor, unet_w=None):
    orig = (m._BAYES_MIN_PS, m._BAYES_MAX_PS, m.PROB_FLOOR)
    if unet_w is not None:
        orig_unet = m.UNET_BLEND_WEIGHT
        m.UNET_BLEND_WEIGHT = unet_w
    m._BAYES_MIN_PS, m._BAYES_MAX_PS, m.PROB_FLOOR = bayes_min, bayes_max, floor
    scores = []
    for s in range(n_seeds):
        gt = load_gt(s)
        pred = build_prediction(R16_ID, detail, s)
        scores.append(score_prediction(pred, gt))
    m._BAYES_MIN_PS, m._BAYES_MAX_PS, m.PROB_FLOOR = orig
    if unet_w is not None:
        m.UNET_BLEND_WEIGHT = orig_unet
    avg = np.mean(scores)
    seeds_str = " ".join(f"{s:5.1f}" for s in scores)
    print(f"{label:<35s}  {seeds_str}  avg={avg:.2f}")
    return avg


def main():
    detail = get_round_detail(R16_ID)
    n_seeds = len(detail.get("initial_states", []))

    print("Config                                  S0    S1    S2    S3    S4  avg")
    print("-" * 85)

    # Current reverted settings
    test_config(detail, n_seeds, "Reverted (5/100/0.0001)", 5, 100, 0.0001)

    # Floor variants
    test_config(detail, n_seeds, "Floor=0.0003 (5/100)", 5, 100, 0.0003)
    test_config(detail, n_seeds, "Floor=0.001  (5/100)", 5, 100, 0.001)
    test_config(detail, n_seeds, "Floor=0.0005 (5/100)", 5, 100, 0.0005)

    # No overlay at all
    test_config(detail, n_seeds, "No overlay (floor=0.0001)", 999, 999, 0.0001)
    test_config(detail, n_seeds, "No overlay (floor=0.0003)", 999, 999, 0.0003)

    # U-Net weight variants
    if hasattr(m, "UNET_BLEND_WEIGHT"):
        print()
        curr_uw = m.UNET_BLEND_WEIGHT
        print(f"(Current UNET_BLEND_WEIGHT={curr_uw})")
        for uw in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
            test_config(detail, n_seeds, f"U-Net={uw:.0%} (5/100/0.0001)", 5, 100, 0.0001, unet_w=uw)


if __name__ == "__main__":
    main()
