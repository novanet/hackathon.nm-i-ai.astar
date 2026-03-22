"""Analyze whether per-settlement attributes (food, wealth, defense, population, alive) 
predict final-state survival. If so, adding these as features could dramatically improve predictions."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict

ROUNDS = {
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

# Collect (settlement_attrs, final_class) pairs
records = []
for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text())['ground_truth'], dtype=np.float64)
        gt_class = gt.argmax(axis=-1)  # (H, W)
        
        # Collect all settlement observations for this seed
        sett_obs = defaultdict(list)  # (x,y) -> [attr_dicts]
        for sim_file in sorted(rdir.glob(f'sim_s{seed}_*.json')):
            data = json.loads(sim_file.read_text())
            resp = data.get('response', data)
            vp = resp.get('viewport', {})
            vx, vy = vp.get('x', 0), vp.get('y', 0)
            
            for sett in resp.get('settlements', []):
                # Settlement coordinates are viewport-relative? or absolute?
                sx, sy = sett['x'], sett['y']
                # Check if they look viewport-relative or absolute
                sett_obs[(sx, sy)].append({
                    'pop': sett['population'],
                    'food': sett['food'],
                    'wealth': sett['wealth'],
                    'defense': sett['defense'],
                    'has_port': sett['has_port'],
                    'alive': sett['alive'],
                    'round': rnum,
                    'seed': seed,
                    'vp_x': vx, 'vp_y': vy,
                })
        
        # Match to ground truth
        for (sx, sy), obs_list in sett_obs.items():
            if 0 <= sy < gt_class.shape[0] and 0 <= sx < gt_class.shape[1]:
                final_cls = gt_class[sy, sx]
                for obs in obs_list:
                    records.append({**obs, 'final_class': final_cls, 'x': sx, 'y': sy})

print(f"Total settlement observation records: {len(records)}")
print()

# Check if coordinates are absolute or viewport-relative
sample = records[:10]
for r in sample:
    print(f"  ({r['x']},{r['y']}) vp=({r['vp_x']},{r['vp_y']}) alive={r['alive']} pop={r['pop']} food={r['food']:.1f} → class={r['final_class']}")
print()

# Statistics by alive status
alive_records = [r for r in records if r['alive']]
dead_records = [r for r in records if not r['alive']]
print(f"Alive settlements: {len(alive_records)}")
print(f"Dead settlements: {len(dead_records)}")

if alive_records:
    alive_finals = [r['final_class'] for r in alive_records]
    print(f"  Alive → Settlement (cls 1): {sum(1 for c in alive_finals if c==1)/len(alive_finals)*100:.1f}%")
    print(f"  Alive → Empty (cls 0): {sum(1 for c in alive_finals if c==0)/len(alive_finals)*100:.1f}%")
    print(f"  Alive → Port (cls 2): {sum(1 for c in alive_finals if c==2)/len(alive_finals)*100:.1f}%")

if dead_records:
    dead_finals = [r['final_class'] for r in dead_records]
    print(f"  Dead → Settlement (cls 1): {sum(1 for c in dead_finals if c==1)/len(dead_finals)*100:.1f}%")
    print(f"  Dead → Empty (cls 0): {sum(1 for c in dead_finals if c==0)/len(dead_finals)*100:.1f}%")

print()
# Survival rate by food/wealth quantiles (for alive settlements only)
if alive_records:
    foods = [r['food'] for r in alive_records]
    finals = [r['final_class'] for r in alive_records]
    
    for metric_name in ['food', 'wealth', 'defense', 'pop']:
        vals = [r[metric_name] for r in alive_records]
        if max(vals) == min(vals):
            print(f"{metric_name}: all same value ({vals[0]})")
            continue
        
        # Split into quartiles
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        for label, lo, hi in [('Q1', min(vals)-1, q25), ('Q2', q25, q50), ('Q3', q50, q75), ('Q4', q75, max(vals)+1)]:
            subset = [r for r, v in zip(alive_records, vals) if lo < v <= hi]
            if not subset:
                continue
            surv = sum(1 for r in subset if r['final_class'] == 1) / len(subset) * 100
            port = sum(1 for r in subset if r['final_class'] == 2) / len(subset) * 100
            empty = sum(1 for r in subset if r['final_class'] == 0) / len(subset) * 100
            print(f"  {metric_name} {label} ({lo:.1f}-{hi:.1f}, n={len(subset)}): "
                  f"→S={surv:.1f}% →P={port:.1f}% →E={empty:.1f}%")
        print()
