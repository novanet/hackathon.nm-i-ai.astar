"""Test different SIM_BLEND_ALPHA values on R9-R12."""
import json, numpy as np
from pathlib import Path
from astar.model import (
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, spatial_prior, apply_floor,
    PROB_FLOOR, HISTORICAL_TRANSITIONS, NUM_CLASSES,
    _extract_settlement_stats, per_class_temperature_scale,
    PER_CLASS_TEMPS, CALIBRATION_FACTORS, _adaptive_bayesian_overlay
)
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction
from simulator import params_from_transition_matrix, simulate_monte_carlo_vectorized, grid_to_numpy

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    map_w, map_h = 40, 40
    
    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    deb = debias_transitions(cal) if cal is not None else None
    sett_stats = _extract_settlement_stats(rid, detail)
    rf = compute_round_features(deb, detail, settlement_stats=sett_stats)
    
    trans_for_sim = deb if deb is not None else HISTORICAL_TRANSITIONS
    sim_params = params_from_transition_matrix(trans_for_sim)
    sim_params = sim_params._replace(expansion_base=sim_params.expansion_base * 1.3)
    
    ss_rate = rf[1]
    if ss_rate < 0.15:
        temps = np.array([1.05, 1.05, 1.05, 1.0, 1.05, 1.0])
    elif ss_rate > 0.40:
        temps = np.array([1.10, 0.95, 1.10, 1.0, 1.10, 1.0])
    else:
        temps = PER_CLASS_TEMPS
    
    scores = {a: [] for a in [0.0, 0.15, 0.30, 0.50]}
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
        
        spatial = spatial_prior(detail, seed, map_w, map_h, round_features=rf)
        
        init_grid = grid_to_numpy(detail['initial_states'][seed]['grid'], map_h, map_w)
        sim_pred = simulate_monte_carlo_vectorized(init_grid, sim_params, n_sims=200, H=map_h, W=map_w, seed=42+seed)
        
        for alpha in [0.0, 0.15, 0.30, 0.50]:
            pred = (1 - alpha) * spatial + alpha * sim_pred
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            
            pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
            pred = pred / pred.sum(axis=-1, keepdims=True)
            
            pred = per_class_temperature_scale(pred, detail, seed, temps=temps, map_w=map_w, map_h=map_h)
            pred = _adaptive_bayesian_overlay(rid, seed, pred, map_w, map_h)
            pred = apply_floor(pred)
            
            scores[alpha].append(score_prediction(pred, gt))
    
    print(f"R{rnum} (S->S={rf[1]:.3f}):")
    for a in [0.0, 0.15, 0.30, 0.50]:
        avg = np.mean(scores[a])
        delta = avg - np.mean(scores[0.15])
        print(f"  alpha={a:.2f}: {avg:.2f} ({delta:+.2f} vs 0.15)")
    print()
