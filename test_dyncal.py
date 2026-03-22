"""
Test per-round dynamic calibration: correct model output using observed transitions.
The spatial model was trained on S→S ≤ 0.42, so cannot extrapolate to 0.59.
Fix: compute model's expected transition rates, compare to observed, apply correction.
"""
import json, numpy as np
from pathlib import Path
from astar.model import (build_prediction, observation_calibrated_transitions, 
                          apply_floor, PROB_FLOOR, NUM_CLASSES, debias_transitions,
                          HISTORICAL_TRANSITIONS, SHRINKAGE_MATRIX)
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction

def dynamic_calibrate(pred, init_grid, cal_trans, strength=1.0):
    """
    Correct model predictions to match observed transition rates.
    
    For each initial class, compute model's average predicted class distribution.
    Compare to calibrated transitions. Apply multiplicative correction.
    """
    H, W, C = pred.shape
    corrected = pred.copy()
    
    for ic in range(C):
        mask = init_grid == ic
        n = mask.sum()
        if n < 5:
            continue
        
        # Model's average prediction for this initial class
        model_avg = pred[mask].mean(axis=0)  # (C,)
        # Target: calibrated transition rates
        target = cal_trans[ic]  # (C,)
        
        # Compute per-class correction factor
        correction = np.ones(C)
        for fc in range(C):
            if model_avg[fc] > 0.001:
                raw_factor = target[fc] / model_avg[fc]
                # Dampen toward 1.0 based on strength
                correction[fc] = 1.0 + strength * (raw_factor - 1.0)
        
        # Apply correction to all cells of this initial class
        corrected[mask] *= correction[np.newaxis, :]
    
    # Renormalize
    corrected = np.maximum(corrected, 1e-10)
    corrected = corrected / corrected.sum(axis=-1, keepdims=True)
    return corrected


def test_on_round(round_id, round_num, rdir, detail):
    """Test dynamic calibration on one round."""
    scores = {}
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
        H, W, C = gt.shape
        ig = np.vectorize(lambda x: TERRAIN_TO_CLASS.get(x, 0))(
            np.array(detail['initial_states'][seed]['grid'], dtype=int))
        
        pred = build_prediction(round_id, detail, seed)
        cal = observation_calibrated_transitions(round_id, detail, smoothing=5.0)
        if cal is None:
            cal = HISTORICAL_TRANSITIONS
        
        s_base = score_prediction(pred, gt)
        
        # Test different calibration strengths
        for strength in [0.3, 0.5, 0.7, 1.0]:
            pred_cal = dynamic_calibrate(pred, ig, cal, strength=strength)
            pred_cal = apply_floor(pred_cal, PROB_FLOOR)
            s = score_prediction(pred_cal, gt)
            scores.setdefault(f'str_{strength}', []).append(s)
        
        scores.setdefault('base', []).append(s_base)
    
    print(f"R{round_num} ({round_id[:8]}):")
    for name in ['base'] + [f'str_{s}' for s in [0.3, 0.5, 0.7, 1.0]]:
        avg = np.mean(scores[name])
        delta = avg - np.mean(scores['base'])
        print(f"  {name:12s}: {avg:.2f} ({delta:+.2f})")
    return scores


# Test on R11 and R12 (high S→S rounds)
rounds = {
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
    # Also test on R9, R10 to check for regression
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
}

for rnum in sorted(rounds):
    rid = rounds[rnum]
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    test_on_round(rid, rnum, rdir, detail)
    print()
