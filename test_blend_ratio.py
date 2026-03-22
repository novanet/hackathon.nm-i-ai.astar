"""Test different GBM/MLP blend ratios. Currently 50/50."""
import json, numpy as np, torch
from pathlib import Path
from astar.model import (
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, compute_cell_features, apply_floor,
    NUM_CLASSES, _extract_settlement_stats, per_class_temperature_scale,
    PER_CLASS_TEMPS, CALIBRATION_FACTORS, _adaptive_bayesian_overlay,
    _load_mlp_model, load_spatial_model
)
from astar.replay import TERRAIN_TO_CLASS
from astar.submit import score_prediction

ROUNDS = {
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
}

# Load models separately
import joblib
loaded = joblib.load(Path("data") / "spatial_model.pkl")
lgb_model = loaded["lgb"]
xgb_model = loaded["xgb"]
lgb_w = loaded.get("lgb_weight", 0.7)
mlp_model = _load_mlp_model()

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
    
    scores_by_ratio = {}
    GBM_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for gbm_w in GBM_WEIGHTS:
        seed_scores = []
        for seed in range(5):
            gt_path = rdir / f'ground_truth_s{seed}.json'
            if not gt_path.exists(): continue
            gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
            
            feats_3d = compute_cell_features(detail['initial_states'][seed]['grid'], 40, 40, round_features=rf)
            X = feats_3d.reshape(-1, feats_3d.shape[-1])
            
            # GBM prediction
            lgb_pred = lgb_model.predict(X)
            xgb_pred = xgb_model.predict(X)
            gbm_pred = lgb_w * lgb_pred + (1 - lgb_w) * xgb_pred
            
            # MLP prediction
            device = next(mlp_model.parameters()).device
            with torch.no_grad():
                X_t = torch.tensor(X, dtype=torch.float32, device=device)
                log_pred = mlp_model(X_t)
                mlp_pred = torch.exp(log_pred).cpu().numpy()
            
            # Blend
            raw = gbm_w * gbm_pred + (1 - gbm_w) * mlp_pred
            pred = raw.reshape(40, 40, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            
            # Post-processing
            pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = per_class_temperature_scale(pred, detail, seed, temps=temps, map_w=40, map_h=40)
            pred = _adaptive_bayesian_overlay(rid, seed, pred, 40, 40)
            pred = apply_floor(pred)
            
            seed_scores.append(score_prediction(pred, gt))
        scores_by_ratio[gbm_w] = np.mean(seed_scores)
    
    print(f"R{rnum} (S->S={rf[1]:.3f}):")
    for w in GBM_WEIGHTS:
        delta = scores_by_ratio[w] - scores_by_ratio[0.5]
        print(f"  GBM={w:.1f} MLP={1-w:.1f}: {scores_by_ratio[w]:.2f} ({delta:+.2f})")
    print()
