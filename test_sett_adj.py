"""Test post-hoc settlement attribute adjustment.
For cells observed as settlements, adjust P(Settlement) based on wealth/pop.
Applied AFTER the adaptive Bayesian overlay."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict
from astar.model import build_prediction
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

def get_settlement_features(round_id, seed, map_w=40, map_h=40):
    """Extract per-cell settlement attributes from observation data."""
    rdir = Path(f'data/round_{round_id}')
    # cell -> list of (wealth, pop, defense, food)
    cell_attrs = defaultdict(list)
    
    for sim_file in sorted(rdir.glob(f'sim_s{seed}_*.json')):
        data = json.loads(sim_file.read_text())
        resp = data.get('response', data)
        for sett in resp.get('settlements', []):
            x, y = sett['x'], sett['y']
            if 0 <= x < map_w and 0 <= y < map_h:
                cell_attrs[(x, y)].append({
                    'wealth': sett['wealth'],
                    'pop': sett['population'],
                    'defense': sett['defense'],
                    'food': sett['food'],
                })
    
    # Aggregate: mean wealth, pop per cell
    result = {}
    for (x, y), attrs in cell_attrs.items():
        result[(x, y)] = {
            'mean_wealth': np.mean([a['wealth'] for a in attrs]),
            'mean_pop': np.mean([a['pop'] for a in attrs]),
            'mean_defense': np.mean([a['defense'] for a in attrs]),
            'n_obs': len(attrs),
        }
    return result

def apply_sett_adjustment(pred, sett_feats, strength=0.3):
    """
    For cells with observed settlement attributes, adjust P(Settlement) 
    based on wealth. Higher wealth → more P(Settlement).
    """
    pred = pred.copy()
    for (x, y), attrs in sett_feats.items():
        # wealth ranges roughly 0-1
        w = attrs['mean_wealth']
        p = attrs['mean_pop']
        
        # Score: higher = more likely to survive
        # wealth is the strongest predictor (21.4% → 34.1% survival Q1→Q4)
        survival_score = w * 0.5 + min(p, 2.0) / 2.0 * 0.3 + attrs['mean_defense'] * 0.2
        
        # Map to multiplier: low score → <1, high score → >1
        # Center around the mean score
        multiplier = 1.0 + strength * (survival_score - 0.3)
        multiplier = max(0.5, min(2.0, multiplier))  # clip
        
        # Apply to P(Settlement) at (x, y)
        pred[y, x, 1] *= multiplier
        # Renormalize
        pred[y, x] /= pred[y, x].sum()
    
    return pred

# Test
for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text())
    
    scores_base = []
    scores_adj = {s: [] for s in [0.0, 0.2, 0.3, 0.5, 0.8]}
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists(): continue
        gt = np.array(json.loads(gt_path.read_text())['ground_truth'], dtype=np.float64)
        
        pred = build_prediction(rid, detail, seed, 40, 40)
        scores_base.append(score_prediction(pred, gt))
        
        sett_feats = get_settlement_features(rid, seed)
        for s in [0.0, 0.2, 0.3, 0.5, 0.8]:
            adj = apply_sett_adjustment(pred, sett_feats, strength=s)
            scores_adj[s].append(score_prediction(adj, gt))
    
    print(f"R{rnum}:")
    base = np.mean(scores_base)
    print(f"  baseline: {base:.2f}")
    for s in [0.0, 0.2, 0.3, 0.5, 0.8]:
        avg = np.mean(scores_adj[s])
        delta = avg - base
        print(f"  strength={s}: {avg:.2f} ({delta:+.2f})")
    print()
