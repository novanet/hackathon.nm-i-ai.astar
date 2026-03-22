import json, numpy as np
from pathlib import Path
from astar.model import (observation_calibrated_transitions, debias_transitions,
    compute_round_features, HISTORICAL_TRANSITIONS, SHRINKAGE_MATRIX)

rid = '795bfb1f-54bd-4f39-a526-9868b36f7ebd'
rdir = Path(f'data/round_{rid}')
detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))

cal = observation_calibrated_transitions(rid, detail)
debiased = debias_transitions(cal)

print('=== S->S comparison ===')
print(f'  Historical:  {HISTORICAL_TRANSITIONS[1,1]:.4f}')
print(f'  Calibrated:  {cal[1,1]:.4f}')
print(f'  Debiased:    {debiased[1,1]:.4f}')
print()

# Now check what round features the model gets
rf = compute_round_features(debiased, detail)
rf_cal = compute_round_features(cal, detail)
rf_names = ['E_E', 'S_S', 'F_F', 'E_S', 'sett_density', 'food', 'wealth', 'defense']
print('=== Round features ===')
for i, n in enumerate(rf_names):
    print(f'  {n:15s}: debiased={rf[i]:.4f}  raw_cal={rf_cal[i]:.4f}  delta={rf[i]-rf_cal[i]:+.4f}')

# R12 ground truth S->S for comparison
from astar.replay import TERRAIN_TO_CLASS
gt_ss_list = []
for seed in range(5):
    gt_path = rdir / f'ground_truth_s{seed}.json'
    if not gt_path.exists():
        continue
    gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
    ig = np.vectorize(lambda x: TERRAIN_TO_CLASS.get(x, 0))(np.array(detail['initial_states'][seed]['grid'], dtype=int))
    gt_ss = gt[ig==1].mean(axis=0)[1] if (ig==1).sum() > 0 else 0
    gt_ss_list.append(gt_ss)
print(f'\n  GT S->S per seed: {[f"{s:.3f}" for s in gt_ss_list]}')
print(f'  GT S->S mean: {np.mean(gt_ss_list):.4f}')
print(f'  Cal S->S: {cal[1,1]:.4f}, Debiased S->S: {debiased[1,1]:.4f}')
print(f'  Diff: cal-GT={cal[1,1]-np.mean(gt_ss_list):+.4f}, deb-GT={debiased[1,1]-np.mean(gt_ss_list):+.4f}')
