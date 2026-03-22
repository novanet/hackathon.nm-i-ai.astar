"""Sweep U-Net blend weight on R1-R16 LORO using build_prediction pipeline.
Tests blend ratios from 0% to 60% in steps of 5%.
"""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction
import astar.model as m

DATA_DIR = Path("data")

ROUND_IDS = {
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
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}


def load_gt(round_id: str, seed: int) -> np.ndarray:
    rdir = DATA_DIR / f"round_{round_id}"
    data = json.loads((rdir / f"ground_truth_s{seed}.json").read_text())
    return np.array(data["ground_truth"], dtype=np.float64)


def score_round(round_id: str, detail: dict, n_seeds: int) -> float:
    scores = []
    for s in range(n_seeds):
        gt = load_gt(round_id, s)
        pred = build_prediction(round_id, detail, s)
        scores.append(score_prediction(pred, gt))
    return np.mean(scores)


def main():
    print("=" * 80)
    print("U-NET BLEND WEIGHT SWEEP")
    print("=" * 80)
    
    for weight in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        m.UNET_BLEND_WEIGHT = weight
        
        scores = {}
        for rnum, rid in sorted(ROUND_IDS.items()):
            rdir = DATA_DIR / f"round_{rid}"
            detail_files = sorted(rdir.glob("round_detail_*.json"))
            if not detail_files:
                continue
            detail = json.loads(detail_files[0].read_text())
            n_seeds = len(detail.get("initial_states", []))
            sim_files = list(rdir.glob("sim_*.json"))
            if not sim_files:
                continue
            try:
                scores[rnum] = score_round(rid, detail, n_seeds)
            except Exception as e:
                print(f"  Skip R{rnum}: {e}")
        
        avg = np.mean(list(scores.values()))
        per_round = " ".join(f"R{r}={s:.1f}" for r, s in sorted(scores.items()))
        print(f"UNet={weight:.2f}  avg={avg:.2f}  {per_round}")
    
    # Reset to default
    m.UNET_BLEND_WEIGHT = 0.40


if __name__ == "__main__":
    main()
