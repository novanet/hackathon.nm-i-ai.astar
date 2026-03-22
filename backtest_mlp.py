"""Quick backtest: run full build_prediction() pipeline on rounds with sim+GT."""
import numpy as np
import json
import sys
from pathlib import Path

# Force reload of model module to pick up U-Net changes
if "astar.model" in sys.modules:
    del sys.modules["astar.model"]

from astar.model import build_prediction, apply_floor


def score_prediction(pred, gt):
    p = np.clip(pred, 1e-10, 1.0)
    t = np.clip(gt, 1e-10, 1.0)
    kl = np.sum(t * np.log(t / p), axis=-1)
    ent = -np.sum(t * np.log(t), axis=-1)
    w = np.maximum(ent, 0.01)
    wkl = np.sum(w * kl) / np.sum(w)
    return 100.0 * np.exp(-3.0 * wkl)


ROUNDS = {
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
}

all_scores = []
for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path("data") / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        print(f"R{rnum}: no detail file")
        continue
    detail = json.loads(detail_files[0].read_text())
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    scores = []
    for s in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text())
        gt = np.array(gt_data["ground_truth"], dtype=np.float64)
        pred = build_prediction(rid, detail, s, map_w, map_h)
        pred = apply_floor(pred)
        scores.append(score_prediction(pred, gt))

    if scores:
        avg = np.mean(scores)
        all_scores.append(avg)
        print(f"R{rnum}: avg={avg:.2f}  seeds: {[f'{s:.1f}' for s in scores]}")

print(f"\nOverall avg: {np.mean(all_scores):.2f}")
