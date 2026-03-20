"""Quick backtest of new model on all rounds."""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
}

import traceback, sys

print("=== IN-SAMPLE SCORES WITH NEW MODEL (R1-R8 training) ===")
sys.stdout.flush()
for rnum, rid in sorted(ROUND_IDS.items()):
    try:
        data_dir = Path("data") / f"round_{rid}"
        detail = json.loads(sorted(data_dir.glob("round_detail_*.json"))[-1].read_text(encoding="utf-8"))
        scores = []
        for s in range(5):
            gtp = data_dir / f"ground_truth_s{s}.json"
            if not gtp.exists():
                break
            gt = np.array(json.loads(gtp.read_text(encoding="utf-8"))["ground_truth"], dtype=np.float64)
            pred = build_prediction(rid, detail, s)
            scores.append(score_prediction(pred, gt))
        if scores:
            w = 1.05 ** rnum
            seed_strs = [f"{s:.1f}" for s in scores]
            print(f"  R{rnum}: avg={np.mean(scores):.2f}  weighted={np.mean(scores)*w:.2f}  seeds={seed_strs}")
            sys.stdout.flush()
    except Exception as e:
        print(f"  R{rnum}: ERROR - {e}")
        traceback.print_exc()
        sys.stdout.flush()
