"""Test if calibration factors and temperature scaling still help with the overlay."""
import json, numpy as np
from pathlib import Path
from astar.model import (
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, spatial_prior, apply_floor,
    PROB_FLOOR, NUM_CLASSES, _extract_settlement_stats,
    per_class_temperature_scale, PER_CLASS_TEMPS, _adaptive_bayesian_overlay
)
from astar.submit import score_prediction

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

CAL_OPTIONS = {
    'cal=1.0':   np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # no calibration
    'cal=0.95F': np.array([1.0, 1.0, 1.0, 1.0, 0.95, 1.0]),  # current
    'cal=0.90F': np.array([1.0, 1.0, 1.0, 1.0, 0.90, 1.0]),
}

TEMP_OPTIONS = {
    'no_temp': False,
    'with_temp': True,
}

for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    
    cal = observation_calibrated_transitions(rid, detail, 40, 40)
    deb = debias_transitions(cal) if cal is not None else None
    sett_stats = _extract_settlement_stats(rid, detail)
    rf = compute_round_features(deb, detail, settlement_stats=sett_stats)
    
    ss_rate = rf[1]
    if ss_rate < 0.15:
        temps = np.array([1.05, 1.05, 1.05, 1.0, 1.05, 1.0])
    elif ss_rate > 0.40:
        temps = np.array([1.10, 0.95, 1.10, 1.0, 1.10, 1.0])
    else:
        temps = PER_CLASS_TEMPS
    
    results = {}
    for cal_name, cal_factors in CAL_OPTIONS.items():
        for temp_name, use_temp in TEMP_OPTIONS.items():
            key = f"{cal_name}|{temp_name}"
            scores = []
            for seed in range(5):
                gt_path = rdir / f'ground_truth_s{seed}.json'
                if not gt_path.exists(): continue
                gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
                
                pred = spatial_prior(detail, seed, 40, 40, round_features=rf)
                
                # Calibration
                pred = pred * cal_factors[np.newaxis, np.newaxis, :]
                pred = pred / pred.sum(axis=-1, keepdims=True)
                
                # Temperature
                if use_temp:
                    pred = per_class_temperature_scale(pred, detail, seed, temps=temps, map_w=40, map_h=40)
                
                # Overlay
                pred = _adaptive_bayesian_overlay(rid, seed, pred, 40, 40)
                pred = apply_floor(pred)
                scores.append(score_prediction(pred, gt))
            results[key] = np.mean(scores)
    
    print(f"R{rnum} (S->S={rf[1]:.3f}):")
    for key, val in results.items():
        print(f"  {key}: {val:.2f}")
    print()
