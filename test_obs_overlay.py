"""
Test direct observation overlay: blend model predictions with per-cell observations.
Our 9 viewports cover the FULL 40x40 map, but we only use observations for 
aggregate transition calibration. Direct cell-level use could dramatically improve scores.
"""
import json, numpy as np
from pathlib import Path
from astar.model import (
    build_prediction, bayesian_update, apply_floor, PROB_FLOOR, NUM_CLASSES, PRIOR_STRENGTH
)
from astar.replay import TERRAIN_TO_CLASS, build_observation_grid
from astar.submit import score_prediction

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    
    results = {k: [] for k in ['base', 'bayes_1', 'bayes_3', 'bayes_5', 'bayes_10', 'bayes_20', 'bayes_50']}
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
        
        pred = build_prediction(rid, detail, seed)
        s_base = score_prediction(pred, gt)
        results['base'].append(s_base)
        
        # Test different prior strengths for Bayesian update
        for ps in [1, 3, 5, 10, 20, 50]:
            pred_bay = bayesian_update(rid, seed, prior=pred, prior_strength=ps)
            pred_bay = apply_floor(pred_bay, PROB_FLOOR)
            s = score_prediction(pred_bay, gt)
            results[f'bayes_{ps}'].append(s)
        
        # Count observations per cell
        obs = build_observation_grid(rid, seed)
        n_obs = sum(1 for y in range(40) for x in range(40) if obs[y][x])
        n_multi = sum(1 for y in range(40) for x in range(40) if len(obs[y][x]) > 1)
        
        if seed == 0:
            print(f"R{rnum} S0: {n_obs}/1600 cells observed, {n_multi} multi-obs")
    
    print(f"R{rnum}:")
    for name in ['base', 'bayes_1', 'bayes_3', 'bayes_5', 'bayes_10', 'bayes_20', 'bayes_50']:
        avg = np.mean(results[name])
        delta = avg - np.mean(results['base'])
        print(f"  {name:12s}: {avg:.2f} ({delta:+.2f})")
    print()
