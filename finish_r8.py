"""Finish R8 submission: seeds 3-4 + extra queries + resubmit all."""
import json, numpy as np, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')
from astar.client import _request, submit, get_budget, simulate, save_round_summary
from astar.model import build_prediction, prediction_to_list, observation_calibrated_transitions, HISTORICAL_TRANSITIONS
from astar.replay import CLASS_NAMES

R8_ID = 'c5cdf100-a876-4fb7-b5d8-757162c97989'
DATA_DIR = Path('data') / f'round_{R8_ID}'

detail_files = sorted(DATA_DIR.glob('round_detail_*.json'))
detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
n_seeds = len(detail.get('initial_states', []))
map_w = detail.get('map_width', 40)
map_h = detail.get('map_height', 40)

# Submit seeds 3 and 4
for seed_idx in [3, 4]:
    print(f'Predicting seed {seed_idx}...')
    pred = build_prediction(R8_ID, detail, seed_idx, map_w, map_h)
    print(f'  shape={pred.shape}, min={pred.min():.6f}, max={pred.max():.4f}')
    resp = submit(R8_ID, seed_idx, prediction_to_list(pred))
    status = resp.get('status', '?')
    score = resp.get('score', '?')
    print(f'  Seed {seed_idx}: {status} score={score}')

# Check budget for extra queries
budget = get_budget()
remaining = budget['queries_max'] - budget['queries_used']
print(f'\nBudget: {budget["queries_used"]}/{budget["queries_max"]} ({remaining} remaining)')

# Spend extra queries on settlement-dense seeds
if remaining > 0:
    extra_viewports = [
        (7, 7, 15, 15),
        (7, 20, 15, 15),
        (20, 7, 15, 15),
        (20, 20, 15, 15),
        (3, 3, 15, 15),
    ]
    seed_dynamics = []
    for i, state in enumerate(detail['initial_states']):
        n_setts = len([s for s in state['settlements'] if s['alive']])
        seed_dynamics.append((n_setts, i))
    seed_dynamics.sort(reverse=True)

    count = 0
    vp_idx = 0
    for _, seed_idx in seed_dynamics:
        if count >= remaining:
            break
        vx, vy, w, h = extra_viewports[vp_idx % len(extra_viewports)]
        w = min(w, map_w - vx)
        h = min(h, map_h - vy)
        try:
            simulate(R8_ID, seed_idx, vx, vy, w, h)
            print(f'  Extra query: seed {seed_idx} ({vx},{vy})')
            count += 1
        except Exception as e:
            print(f'  Extra failed: {e}')
            break
        time.sleep(0.05)
        vp_idx += 1
    print(f'  Used {count} extra queries')

    # Resubmit all seeds with extra observations
    print('\n=== Resubmitting all seeds (pass 2) ===')
    for seed_idx in range(n_seeds):
        pred = build_prediction(R8_ID, detail, seed_idx, map_w, map_h)
        resp = submit(R8_ID, seed_idx, prediction_to_list(pred))
        status = resp.get('status', '?')
        score = resp.get('score', '?')
        print(f'  Seed {seed_idx}: {status} score={score}')

# Final status
budget = get_budget()
print(f'\nFinal budget: {budget["queries_used"]}/{budget["queries_max"]}')

my = _request('GET', '/my-rounds')
for r in my:
    if r.get('round_number') == 8:
        rs = r.get('round_score')
        ss = r.get('seed_scores')
        print(f'R8 score: {rs}')
        if ss:
            print(f'Seeds: {ss}')
        if rs:
            print(f'Weighted: {rs * 1.05**8:.2f} (weight={1.05**8:.4f})')
        break

save_round_summary(R8_ID)
print('Round summary saved')
