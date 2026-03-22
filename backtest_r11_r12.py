"""Backtest R11+R12 with current model."""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction

DATA_DIR = Path('data')

rounds = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

for rnum, rid in rounds.items():
    rdir = DATA_DIR / f'round_{rid}'
    detail_files = sorted(rdir.glob('round_detail_*.json'))
    detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
    n_seeds = len(detail.get('initial_states', []))

    scores = []
    for s in range(n_seeds):
        gt_path = rdir / f'ground_truth_s{s}.json'
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text(encoding='utf-8'))
        gt = np.array(gt_data['ground_truth'], dtype=np.float64)
        pred = build_prediction(rid, detail, s)
        sc = score_prediction(pred, gt)
        scores.append(sc)

    avg = np.mean(scores)
    weight = 1.05 ** rnum
    seeds_str = ", ".join(f"{s:.1f}" for s in scores)
    print(f'R{rnum}: avg={avg:.2f} (weighted={avg*weight:.2f}) seeds=[{seeds_str}]')
