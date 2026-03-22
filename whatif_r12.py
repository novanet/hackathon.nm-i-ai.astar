"""What-if analysis: what scores would we get with different strategies on R12?"""
import json, numpy as np
from pathlib import Path
from astar.model import (
    build_prediction, observation_calibrated_transitions,
    load_simulations, HISTORICAL_TRANSITIONS, RECENT_TRANSITIONS,
    SHRINKAGE_MATRIX, NUM_CLASSES, SIM_BLEND_ALPHA,
    _count_transitions, apply_floor, PROB_FLOOR
)
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction

DATA_DIR = Path('data')
R12_ID = '795bfb1f-54bd-4f39-a526-9868b36f7ebd'
rdir = DATA_DIR / f'round_{R12_ID}'

detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
CLASS_NAMES = ['Empty', 'Settlement', 'Port', 'Ruin', 'Forest', 'Mountain']

scores_all = {}

for seed in range(5):
    gt_data = json.loads((rdir / f'ground_truth_s{seed}.json').read_text(encoding='utf-8'))
    gt = np.array(gt_data['ground_truth'], dtype=np.float64)
    H, W, C = gt.shape

    init_data = detail['initial_states'][seed]
    init_grid_raw = np.array(init_data['grid'], dtype=int)
    init_grid = np.vectorize(lambda x: TERRAIN_TO_CLASS.get(x, 0))(init_grid_raw)

    # 1. Current model (V8+ with recency-weighted transitions)
    pred_current = build_prediction(R12_ID, detail, seed)
    s1 = score_prediction(pred_current, gt)

    # 2. Pure observation-calibrated transitions (oracle-ish: bypass spatial model)
    cal_trans = observation_calibrated_transitions(R12_ID, detail, seed, smoothing=5.0)
    pred_cal = np.zeros_like(gt)
    for y in range(H):
        for x in range(W):
            ic = init_grid[y, x]
            pred_cal[y, x] = cal_trans[ic]
    pred_cal = apply_floor(pred_cal, PROB_FLOOR)
    s2 = score_prediction(pred_cal, gt)

    # 3. Oracle transitions (from GT) — ceiling for transition-based approach
    gt_trans = np.zeros((C, C), dtype=np.float64)
    for ic in range(C):
        mask = init_grid == ic
        if mask.sum() > 0:
            gt_trans[ic] = gt[mask].mean(axis=0)
        else:
            gt_trans[ic, ic] = 1.0
    pred_oracle = np.zeros_like(gt)
    for y in range(H):
        for x in range(W):
            ic = init_grid[y, x]
            pred_oracle[y, x] = gt_trans[ic]
    pred_oracle = apply_floor(pred_oracle, PROB_FLOOR)
    s3 = score_prediction(pred_oracle, gt)

    # 4. High sim blend (50% spatial + 50% sim instead of 15%)
    # Recompute sim part manually
    from astar.model import _run_stochastic_sim
    sim_pred = _run_stochastic_sim(R12_ID, detail, seed, cal_trans, n_sims=50)
    pred_high_sim = 0.50 * pred_current / (1 - SIM_BLEND_ALPHA) * (1 - 0.50) + 0.50 * sim_pred
    # Simpler: just blend
    spatial_only = (pred_current - SIM_BLEND_ALPHA * sim_pred) / (1 - SIM_BLEND_ALPHA)
    pred_high_sim = 0.50 * spatial_only + 0.50 * sim_pred
    pred_high_sim = apply_floor(pred_high_sim, PROB_FLOOR)
    s4 = score_prediction(pred_high_sim, gt)

    # 5. Pure sim (100% stochastic simulator with calibrated transitions)
    pred_pure_sim = apply_floor(sim_pred, PROB_FLOOR)
    s5 = score_prediction(pred_pure_sim, gt)

    # 6. Oracle transitions + spatial correction (use GT transitions but add spatial model residuals)
    # Blend: 50% spatial + 50% oracle_trans
    pred_blend = 0.50 * spatial_only + 0.50 * pred_oracle
    pred_blend = apply_floor(pred_blend, PROB_FLOOR)
    s6 = score_prediction(pred_blend, gt)

    # 7. What if we knew GT S→S exactly and used it as round feature? (Oracle round features)
    # Hard to do without retraining, but we can estimate by interpolating

    print(f"=== Seed {seed} ===")
    print(f"  1. Current model (V8+):        {s1:.2f}")
    print(f"  2. Pure cal_trans (no spatial): {s2:.2f}")
    print(f"  3. Oracle transitions (GT):     {s3:.2f}")
    print(f"  4. 50% sim blend:               {s4:.2f}")
    print(f"  5. Pure sim (100%):             {s5:.2f}")
    print(f"  6. 50% spatial + 50% oracle:    {s6:.2f}")

    for name, sc in [('current', s1), ('cal_trans', s2), ('oracle_trans', s3),
                     ('high_sim', s4), ('pure_sim', s5), ('blend_oracle', s6)]:
        scores_all.setdefault(name, []).append(sc)

    # Also print GT and calibrated transition S->S comparison
    gt_ss = gt[init_grid == 1].mean(axis=0)[1] if (init_grid == 1).sum() > 0 else 0
    cal_ss = cal_trans[1, 1]
    print(f"  GT S→S: {gt_ss:.3f}, Calibrated S→S: {cal_ss:.3f}, Delta: {cal_ss-gt_ss:+.3f}")

print()
print("=== Averages ===")
for name, scores in scores_all.items():
    avg = np.mean(scores)
    weight = 1.05 ** 12
    print(f"  {name:20s}: {avg:.2f} (weighted {avg*weight:.2f})")
