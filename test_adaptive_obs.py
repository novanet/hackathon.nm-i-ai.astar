"""
Test adaptive Bayesian observation overlay: scale prior_strength per cell based on
model prediction entropy. Uncertain cells → trust observation more.
"""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction, apply_floor, PROB_FLOOR, NUM_CLASSES
from astar.replay import TERRAIN_TO_CLASS, build_observation_grid
from astar.submit import score_prediction


def adaptive_bayesian_update(round_id, seed, pred, base_ps=10.0, min_ps=3.0, max_ps=100.0):
    """Per-cell adaptive Bayesian update. More uncertain cells get lower prior_strength."""
    obs_grid = build_observation_grid(round_id, seed)
    result = pred.copy()
    H, W, C = pred.shape
    
    # Compute per-cell entropy
    eps = 1e-10
    p = np.clip(pred, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)  # (H, W)
    max_ent = np.log(C)  # max possible entropy
    
    for y in range(H):
        for x in range(W):
            obs = obs_grid[y][x]
            if not obs:
                continue
            
            # Scale prior_strength by inverse entropy (confident → high ps, uncertain → low ps)
            norm_ent = entropy[y, x] / max_ent  # 0 = certain, 1 = max uncertain
            # ps = max_ps when entropy=0, min_ps when entropy=max
            ps = max_ps - (max_ps - min_ps) * norm_ent
            
            alpha = result[y, x] * ps
            for cls in obs:
                alpha[cls] += 1.0
            result[y, x] = alpha / alpha.sum()
    
    return result


ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

# Test different (min_ps, max_ps) combos
configs = [
    (3, 50),
    (3, 100),
    (5, 50),
    (5, 100),
    (10, 100),
    (10, 200),
    (5, 200),
]

for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    
    results = {'base': []}
    for min_ps, max_ps in configs:
        results[f'{min_ps}_{max_ps}'] = []
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
        
        pred = build_prediction(rid, detail, seed)
        results['base'].append(score_prediction(pred, gt))
        
        for min_ps, max_ps in configs:
            pred_adap = adaptive_bayesian_update(rid, seed, pred, base_ps=10, min_ps=min_ps, max_ps=max_ps)
            pred_adap = apply_floor(pred_adap, PROB_FLOOR)
            results[f'{min_ps}_{max_ps}'].append(score_prediction(pred_adap, gt))
    
    print(f"R{rnum}:")
    base_avg = np.mean(results['base'])
    print(f"  {'base':12s}: {base_avg:.2f}")
    for min_ps, max_ps in configs:
        key = f'{min_ps}_{max_ps}'
        avg = np.mean(results[key])
        print(f"  ps={key:8s}: {avg:.2f} ({avg-base_avg:+.2f})")
    print()
