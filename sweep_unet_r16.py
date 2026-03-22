"""Sweep U-Net blend weight on R16 GT."""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction
import astar.model as m

R16 = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"
DATA = Path("data") / f"round_{R16}"

detail = json.loads(next(DATA.glob("round_detail_*.json")).read_text())
n = len(detail.get("initial_states", []))

def gt(s):
    return np.array(json.loads((DATA / f"ground_truth_s{s}.json").read_text())["ground_truth"], dtype=np.float64)

print("U-Net weight sweep on R16 GT (overlay=5/100, floor=0.0001)")
print("-" * 60)
for uw in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6]:
    m.UNET_BLEND_WEIGHT = uw
    scores = [score_prediction(build_prediction(R16, detail, s), gt(s)) for s in range(n)]
    avg = np.mean(scores)
    seeds_str = " ".join(f"{s:5.1f}" for s in scores)
    print(f"U-Net={uw:.0%}:  {seeds_str}  avg={avg:.2f}")
m.UNET_BLEND_WEIGHT = 0.4  # restore
