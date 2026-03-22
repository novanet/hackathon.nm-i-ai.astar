"""Quick test: does 9×9 ring help even more? 
Test by adding features to compute_cell_features temporarily."""
import json, numpy as np
from pathlib import Path
from collections import defaultdict
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from astar.submit import score_prediction
from astar.model import (
    compute_cell_features, apply_floor, observation_calibrated_transitions,
    debias_transitions, compute_round_features, _extract_settlement_stats,
    _adaptive_bayesian_overlay, per_class_temperature_scale, PER_CLASS_TEMPS,
    CALIBRATION_FACTORS
)

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

def compute_9x9_ring(init_grid, map_w, map_h):
    """Compute 9×9 outer ring class fractions (excluding 7×7 inner)."""
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    result = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    if abs(dy) <= 3 and abs(dx) <= 3:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < map_h and 0 <= nx < map_w:
                        result[y, x, cls_grid[ny, nx]] += 1.0
            total = result[y, x].sum()
            if total > 0:
                result[y, x] /= total
    return result

# Load model
from astar.model import load_spatial_model
import joblib, torch

loaded = joblib.load(Path("data") / "spatial_model.pkl")
lgb_model = loaded["lgb"]
xgb_model = loaded["xgb"]
lgb_w = loaded.get("lgb_weight", 0.7)
from astar.model import _load_mlp_model
mlp_model = _load_mlp_model()

for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path(f'data/round_{rid}')
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text())
    
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
    
    scores_37 = []
    scores_43 = []
    
    for seed in range(5):
        gt_path = rdir / f'ground_truth_s{seed}.json'
        if not gt_path.exists(): continue
        gt = np.array(json.loads(gt_path.read_text())['ground_truth'], dtype=np.float64)
        
        init_grid = detail['initial_states'][seed]['grid']
        feats_37 = compute_cell_features(init_grid, 40, 40, round_features=rf)
        
        # 37-feat prediction
        X37 = feats_37.reshape(-1, 37)
        gbm_pred = lgb_w * lgb_model.predict(X37) + (1 - lgb_w) * xgb_model.predict(X37)
        device = next(mlp_model.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(X37, dtype=torch.float32, device=device)
            mlp_pred = torch.exp(mlp_model(X_t)).cpu().numpy()
        raw37 = 0.5 * gbm_pred + 0.5 * mlp_pred
        pred37 = raw37.reshape(40, 40, NUM_CLASSES)
        pred37 = np.maximum(pred37, 1e-10)
        pred37 = pred37 / pred37.sum(axis=-1, keepdims=True)
        pred37 = pred37 * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
        pred37 = pred37 / pred37.sum(axis=-1, keepdims=True)
        pred37 = per_class_temperature_scale(pred37, detail, seed, temps=temps, map_w=40, map_h=40)
        pred37 = _adaptive_bayesian_overlay(rid, seed, pred37, 40, 40)
        pred37 = apply_floor(pred37)
        scores_37.append(score_prediction(pred37, gt))
        
        # 43-feat: append 9×9 ring (6 features)
        ring9 = compute_9x9_ring(init_grid, 40, 40)
        feats_43 = np.concatenate([feats_37[:, :, :29], ring9, feats_37[:, :, 29:]], axis=-1)
        # Can't predict with 37-feat model on 43-feat input. 
        # Instead, test JUST the 9×9 features by training a simple model on the fly.
        # This is too slow for a quick test. Let me just check correlation.
        scores_43.append(0)  # placeholder
    
    print(f"R{rnum}: 37-feat avg={np.mean(scores_37):.2f}")
