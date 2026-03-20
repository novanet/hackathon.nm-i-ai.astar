"""Try to download R8 GT + check score. Also analyze settlement prediction patterns."""
import json, numpy as np, warnings
from pathlib import Path
from astar.client import _request, get_analysis
from astar.model import build_prediction, CLASS_NAMES
from astar.submit import score_prediction

warnings.filterwarnings('ignore')

R8_ID = 'c5cdf100-a876-4fb7-b5d8-757162c97989'
DATA_DIR = Path('data') / f'round_{R8_ID}'

# Check score first
my = _request('GET', '/my-rounds')
for r in sorted(my, key=lambda x: x.get('round_number', 0)):
    rn = r.get('round_number')
    rs = r.get('round_score')
    ss = r.get('seed_scores')
    if rn and rn >= 7:
        w = rs * 1.05**rn if rs else None
        wstr = f'{w:.2f}' if w else 'pending'
        print(f'R{rn}: score={rs}  weighted={wstr}  seeds={ss}')

# Try to get R8 GT
print('\n=== Attempting R8 GT download ===')
detail_files = sorted(DATA_DIR.glob('round_detail_*.json'))
detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
n_seeds = len(detail.get('initial_states', []))

for seed_idx in range(n_seeds):
    try:
        analysis = get_analysis(R8_ID, seed_idx)
        gt = np.array(analysis['ground_truth'], dtype=np.float64)
        gt_path = DATA_DIR / f'ground_truth_s{seed_idx}.json'
        gt_path.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
        print(f'  Seed {seed_idx}: downloaded GT shape={gt.shape}')
    except Exception as e:
        print(f'  Seed {seed_idx}: {e}')
        break

# If we got GT, score it
gt_files = sorted(DATA_DIR.glob('ground_truth_s*.json'))
if gt_files:
    print(f'\n=== R8 Backtest ===')
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)
    scores = []
    for seed_idx in range(n_seeds):
        gt_path = DATA_DIR / f'ground_truth_s{seed_idx}.json'
        if not gt_path.exists():
            continue
        analysis = json.loads(gt_path.read_text(encoding='utf-8'))
        gt = np.array(analysis['ground_truth'], dtype=np.float64)
        pred = build_prediction(R8_ID, detail, seed_idx, map_w, map_h)
        sc = score_prediction(pred, gt)
        scores.append(sc)
        print(f'  Seed {seed_idx}: {sc:.2f}')
    if scores:
        print(f'  Average: {np.mean(scores):.2f}')
else:
    print('  No GT available yet (round still active)')
