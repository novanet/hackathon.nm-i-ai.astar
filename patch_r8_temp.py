"""Resubmit R8 with temperature scaling (T=1.15) applied."""
import json, numpy as np, warnings
from pathlib import Path
from astar.client import submit
from astar.model import build_prediction, prediction_to_list

warnings.filterwarnings('ignore')

R8_ID = 'c5cdf100-a876-4fb7-b5d8-757162c97989'
DATA_DIR = Path('data') / f'round_{R8_ID}'

detail_files = sorted(DATA_DIR.glob('round_detail_*.json'))
detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
n_seeds = len(detail['initial_states'])

print(f'Resubmitting R8 with temperature scaling T=1.15')
for seed_idx in range(n_seeds):
    pred = build_prediction(R8_ID, detail, seed_idx)
    resp = submit(R8_ID, seed_idx, prediction_to_list(pred))
    status = resp.get('status', '?')
    score = resp.get('score', '?')
    print(f'  Seed {seed_idx}: {status} score={score}')

print('Done')
