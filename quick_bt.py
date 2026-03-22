import json, numpy as np
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction

rounds = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}
for rnum, rid in sorted(rounds.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    scores = []
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
        pred = build_prediction(rid, detail, seed)
        scores.append(score_prediction(pred, gt))
    avg = np.mean(scores)
    seeds_str = ", ".join(f"{s:.1f}" for s in scores)
    print(f"R{rnum}: avg={avg:.2f} seeds=[{seeds_str}]")
