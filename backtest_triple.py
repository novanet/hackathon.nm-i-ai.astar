"""Quick full backtest with triple-blend model."""
import importlib, json, numpy as np
from pathlib import Path

import astar.model as mod
mod._spatial_model = None
importlib.reload(mod)

from astar.model import build_prediction, load_spatial_model
from astar.submit import score_prediction

model = load_spatial_model()

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

all_scores = []
for rnum, rid in sorted(ROUND_IDS.items()):
    rdir = Path(f"data/round_{rid}")
    detail = json.loads(sorted(rdir.glob("round_detail_*.json"))[-1].read_text(encoding="utf-8"))
    scores = []
    for s in range(len(detail["initial_states"])):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding="utf-8"))["ground_truth"])
        pred = build_prediction(rid, detail, s)
        sc = score_prediction(pred, gt)
        scores.append(sc)
    avg = np.mean(scores)
    all_scores.append(avg)
    print(f"R{rnum}: {avg:.2f}  seeds={[f'{s:.1f}' for s in scores]}")
print(f"Overall avg: {np.mean(all_scores):.2f}")
