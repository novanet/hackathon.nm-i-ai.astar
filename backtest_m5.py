"""Backtest M5 entropy-conditional temps vs legacy temps through full build_prediction() pipeline.
Runs on ALL rounds with ground truth, toggling USE_ENTROPY_TEMPS on/off."""

import numpy as np
import json
import os
import sys

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


def load_round_data(rnum: int, rid: str):
    """Load round detail and ground truths."""
    import glob
    from pathlib import Path
    base = Path(f"data/round_{rid}")
    
    # Find round detail (may have timestamp in filename)
    detail_files = sorted(base.glob("round_detail_*.json"))
    if not detail_files:
        return None, None

    with open(detail_files[-1], encoding="utf-8") as f:
        detail = json.load(f)

    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = base / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            return detail, None
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
        gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts


def main():
    from astar.submit import score_prediction
    import astar.model as model_mod
    from astar.model import build_prediction

    # Detect which rounds have the observation data needed for build_prediction
    print("=== M5 Entropy Temps vs Legacy Temps Backtest ===\n")

    results_m5 = {}
    results_legacy = {}

    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rnum, rid)
        if detail is None or gts is None:
            print(f"  R{rnum}: skipped (no data)")
            continue

        n_seeds = len(gts)
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)

        scores_m5 = []
        scores_legacy = []

        for s in range(n_seeds):
            gt = gts[s]

            # M5 mode
            model_mod.USE_ENTROPY_TEMPS = True
            pred_m5 = build_prediction(rid, detail, s, map_w, map_h)
            scores_m5.append(score_prediction(pred_m5, gt))

            # Legacy mode
            model_mod.USE_ENTROPY_TEMPS = False
            pred_leg = build_prediction(rid, detail, s, map_w, map_h)
            scores_legacy.append(score_prediction(pred_leg, gt))

        avg_m5 = np.mean(scores_m5)
        avg_leg = np.mean(scores_legacy)
        delta = avg_m5 - avg_leg
        results_m5[rnum] = avg_m5
        results_legacy[rnum] = avg_leg
        marker = "+" if delta > 0 else "-" if delta < 0 else "="
        print(f"  R{rnum:2d}: M5={avg_m5:.2f}  legacy={avg_leg:.2f}  delta={delta:+.2f} {marker}")

    print()
    if results_m5:
        avg_m5_all = np.mean(list(results_m5.values()))
        avg_leg_all = np.mean(list(results_legacy.values()))
        delta_all = avg_m5_all - avg_leg_all
        wins = sum(1 for r in results_m5 if results_m5[r] > results_legacy[r])
        losses = sum(1 for r in results_m5 if results_m5[r] < results_legacy[r])
        ties = len(results_m5) - wins - losses
        print(f"  Average: M5={avg_m5_all:.2f}  legacy={avg_leg_all:.2f}  delta={delta_all:+.2f}")
        print(f"  Wins: {wins}, Losses: {losses}, Ties: {ties}")

    # Reset to M5 if it wins
    if results_m5:
        model_mod.USE_ENTROPY_TEMPS = (avg_m5_all >= avg_leg_all)
        print(f"\n  -> USE_ENTROPY_TEMPS = {model_mod.USE_ENTROPY_TEMPS}")


if __name__ == "__main__":
    main()
