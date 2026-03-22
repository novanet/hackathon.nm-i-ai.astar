"""Dry-run R17 predictions on all 5 seeds to verify correctness."""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction

rid = '3eb0c25d-28fa-48ca-b8e1-fc249e3918e9'
rdir = Path(f'data/round_{rid}')
detail = json.loads(list(rdir.glob('round_detail_*.json'))[0].read_text())
W, H = detail['map_width'], detail['map_height']
ns = detail.get('num_simulations', detail.get('seeds_count', 5))
print(f'Map: {W}x{H}, seeds: {ns}')

all_ok = True
for s in range(ns):
    pred = build_prediction(rid, detail, s)
    shape_ok = pred.shape == (H, W, 6)
    sums = pred.sum(axis=2)
    sum_ok = np.allclose(sums, 1.0, atol=1e-5)
    min_val = pred.min()
    max_val = pred.max()
    floor_ok = min_val >= 0.0001 - 1e-8
    no_nan = not np.any(np.isnan(pred))
    argmax_dist = np.bincount(pred.argmax(axis=2).ravel(), minlength=6)
    status = "OK" if (shape_ok and sum_ok and floor_ok and no_nan) else "FAIL"
    print(f'Seed {s}: {status} | shape={pred.shape} sums_to_1={sum_ok} min={min_val:.6f} no_nan={no_nan}')
    print(f'  argmax: E={argmax_dist[0]} S={argmax_dist[1]} P={argmax_dist[2]} R={argmax_dist[3]} F={argmax_dist[4]} M={argmax_dist[5]}')
    if status == "FAIL":
        all_ok = False
        print('  *** PROBLEM ***')

print(f'\nAll seeds valid: {all_ok}')
