"""Backtest the full pipeline (V4: adaptive temps + settlement stats + R1-R8 calibration)."""
import json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from astar.model import build_prediction, apply_floor, _spatial_model
from astar.submit import score_prediction

# Force reload of model by clearing cache
import astar.model as _m
_m._spatial_model = None

DATA_DIR = Path('data')
ROUND_IDS = {
    1: '71451d74-be9f-471f-aacd-a41f3b68a9cd',
    2: '76909e29-f664-4b2f-b16b-61b7507277e9',
    3: 'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
    4: '8e839974-b13b-407b-a5e7-fc749d877195',
    5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    6: 'ae78003a-4efe-425a-881a-d16a39bca0ad',
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
}

# Previous best scores for comparison
OLD_SCORES = {
    1: 82.16, 2: 86.27, 3: 83.34, 4: 90.88,
    5: 80.35, 6: 81.41, 7: 63.82, 8: 71.51,
}

print("=== V4 BACKTEST: Adaptive temps + settlement stats + R1-R8 calibration ===")
all_scores = []
for rnum, rid in sorted(ROUND_IDS.items()):
    rdir = DATA_DIR / f'round_{rid}'
    detail_files = sorted(rdir.glob('round_detail_*.json'))
    detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)
    scores = []
    for s in range(len(detail.get('initial_states', []))):
        gt_path = rdir / f'ground_truth_s{s}.json'
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text(encoding='utf-8'))
        gt = np.array(gt_data['ground_truth'], dtype=np.float64)
        pred = build_prediction(rid, detail, s, map_w, map_h)
        sc = score_prediction(pred, gt)
        scores.append(sc)
    if scores:
        avg = np.mean(scores)
        all_scores.append((rnum, avg))
        old = OLD_SCORES.get(rnum, 0)
        delta = avg - old
        weighted = avg * 1.05**rnum
        print(f"  R{rnum}: {avg:.2f} (old={old:.2f}, delta={delta:+.2f}, weighted={weighted:.2f})")

if all_scores:
    total = np.mean([s for _, s in all_scores])
    old_total = np.mean(list(OLD_SCORES.values()))
    best_weighted = max(s * 1.05**r for r, s in all_scores)
    print(f"\n  TOTAL AVG: {total:.2f} (old={old_total:.2f}, delta={total-old_total:+.2f})")
    print(f"  Best weighted: {best_weighted:.2f}")
