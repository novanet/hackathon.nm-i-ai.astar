import json, numpy as np
from pathlib import Path
from astar.model import build_prediction, observation_calibrated_transitions, apply_floor, PROB_FLOOR, NUM_CLASSES
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction

R12_ID = '795bfb1f-54bd-4f39-a526-9868b36f7ebd'
rdir = Path(f'data/round_{R12_ID}')
detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))

all_scores = {n: [] for n in ['model','cal_trans','oracle','b50_cal','b70_cal','oracle_blend']}

for seed in range(5):
    gt = np.array(json.loads((rdir / f'ground_truth_s{seed}.json').read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
    H, W, C = gt.shape
    ig = np.vectorize(lambda x: TERRAIN_TO_CLASS.get(x, 0))(np.array(detail['initial_states'][seed]['grid'], dtype=int))

    pred = build_prediction(R12_ID, detail, seed)
    cal = observation_calibrated_transitions(R12_ID, detail, smoothing=5.0)

    # oracle transitions from GT
    gt_t = np.zeros((C, C))
    for ic in range(C):
        m = ig == ic
        gt_t[ic] = gt[m].mean(axis=0) if m.sum() > 0 else np.eye(C)[ic]

    # Build transition-based predictions
    p_cal = np.zeros_like(gt)
    p_ora = np.zeros_like(gt)
    for y in range(H):
        for x in range(W):
            p_cal[y, x] = cal[ig[y, x]]
            p_ora[y, x] = gt_t[ig[y, x]]

    p_cal = apply_floor(p_cal, PROB_FLOOR)
    p_ora = apply_floor(p_ora, PROB_FLOOR)

    results = {
        'model': score_prediction(pred, gt),
        'cal_trans': score_prediction(p_cal, gt),
        'oracle': score_prediction(p_ora, gt),
        'b50_cal': score_prediction(apply_floor(0.5*pred + 0.5*p_cal, PROB_FLOOR), gt),
        'b70_cal': score_prediction(apply_floor(0.3*pred + 0.7*p_cal, PROB_FLOOR), gt),
        'oracle_blend': score_prediction(apply_floor(0.5*pred + 0.5*p_ora, PROB_FLOOR), gt),
    }
    
    gt_ss = gt[ig==1].mean(axis=0)[1] if (ig==1).sum() > 0 else 0
    print(f"S{seed}: " + " | ".join(f"{k}={v:.1f}" for k, v in results.items()) + f" | GT_SS={gt_ss:.3f} Cal_SS={cal[1,1]:.3f}")
    for k, v in results.items():
        all_scores[k].append(v)

print()
w = 1.05**12
for k, v in all_scores.items():
    a = np.mean(v)
    print(f"  {k:15s}: avg={a:.2f}  weighted={a*w:.2f}")
