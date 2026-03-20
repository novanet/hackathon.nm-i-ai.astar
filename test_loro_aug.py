"""LORO CV for augmented model only."""
import json, numpy as np, pickle, warnings
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    apply_floor, observation_calibrated_transitions, debias_transitions,
    compute_round_features, compute_cell_features, HISTORICAL_TRANSITIONS,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from train_spatial import compute_gt_round_features, load_round_data, ROUND_IDS

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')
obs_std = np.array([0.0048, 0.0140, 0.0070, 0.0048, 0.0000])


def build_augmented_data(round_ids, noise_std, n_augments=3):
    X_parts, Y_parts = [], []
    rng = np.random.default_rng(123)
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not detail or not gts:
            continue
        map_w = detail.get('map_width', 40)
        map_h = detail.get('map_height', 40)
        gt_feats = compute_gt_round_features(detail, gts)
        for aug in range(n_augments):
            feats = gt_feats if aug == 0 else np.clip(gt_feats + rng.normal(0, 1, size=len(gt_feats)) * noise_std, 0.01, 0.99)
            for seed_idx, gt in enumerate(gts):
                init_grid = detail['initial_states'][seed_idx]['grid']
                cell_feat = compute_cell_features(init_grid, map_w, map_h, round_features=feats)
                X_parts.append(cell_feat.reshape(-1, cell_feat.shape[-1]))
                Y_parts.append(gt.reshape(-1, gt.shape[-1]))
    return np.vstack(X_parts), np.vstack(Y_parts)


# Collect all available data
all_data = {}
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if detail and gts:
        all_data[rnum] = (rid, detail, gts)
        print(f'R{rnum}: {len(gts)} seeds')

# Also run LORO for original model for fair comparison
print('\n=== LORO CV: ORIGINAL vs AUGMENTED ===')
loro_orig = {}
loro_aug = {}
for test_rnum in sorted(all_data.keys()):
    train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
    
    rid, detail, gts = all_data[test_rnum]
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    # Get test features
    gt_feats = compute_gt_round_features(detail, gts)
    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    # Train original model
    X_tr_parts, Y_tr_parts = [], []
    for rn in sorted(train_ids):
        r_id = train_ids[rn]
        d, gs = load_round_data(r_id)
        if not d or not gs:
            continue
        mw = d.get('map_width', 40)
        mh = d.get('map_height', 40)
        rf = compute_gt_round_features(d, gs)
        for si, gt in enumerate(gs):
            ig = d['initial_states'][si]['grid']
            cf = compute_cell_features(ig, mw, mh, round_features=rf)
            X_tr_parts.append(cf.reshape(-1, cf.shape[-1]))
            Y_tr_parts.append(gt.reshape(-1, gt.shape[-1]))
    X_orig = np.vstack(X_tr_parts)
    Y_orig = np.vstack(Y_tr_parts)

    model_orig = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            verbosity=-1,
        ), n_jobs=-1,
    )
    model_orig.fit(X_orig, Y_orig)

    # Train augmented model
    X_aug, Y_aug = build_augmented_data(train_ids, obs_std, n_augments=3)
    model_aug = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            verbosity=-1,
        ), n_jobs=-1,
    )
    model_aug.fit(X_aug, Y_aug)

    # Evaluate both with obs features (realistic setting)
    orig_scores = []
    aug_scores = []
    for s in range(len(gts)):
        init_grid = detail['initial_states'][s]['grid']
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=obs_feats)
        flat = feat.reshape(-1, feat.shape[-1])

        # Original
        pred = model_orig.predict(flat).reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        orig_scores.append(score_prediction(apply_floor(pred), gts[s]))

        # Augmented
        pred2 = model_aug.predict(flat).reshape(map_h, map_w, NUM_CLASSES)
        pred2 = np.maximum(pred2, 1e-10)
        pred2 = pred2 / pred2.sum(axis=-1, keepdims=True)
        aug_scores.append(score_prediction(apply_floor(pred2), gts[s]))

    oavg = np.mean(orig_scores)
    aavg = np.mean(aug_scores)
    loro_orig[test_rnum] = oavg
    loro_aug[test_rnum] = aavg
    delta = aavg - oavg
    marker = '***' if delta > 1 else '  *' if delta > 0 else ''
    print(f'  R{test_rnum}: orig={oavg:.2f}  aug={aavg:.2f}  Δ={delta:+.2f} {marker}  seeds_aug={[f"{s:.1f}" for s in aug_scores]}')

oavg_all = np.mean(list(loro_orig.values()))
aavg_all = np.mean(list(loro_aug.values()))
print(f'\n  LORO average: orig={oavg_all:.2f}  aug={aavg_all:.2f}  Δ={aavg_all - oavg_all:+.2f}')

# Weighted by 1.05^round for leaderboard relevance
oweighted = sum(loro_orig[r] * 1.05**r for r in loro_orig) / sum(1.05**r for r in loro_orig)
aweighted = sum(loro_aug[r] * 1.05**r for r in loro_aug) / sum(1.05**r for r in loro_aug)
print(f'  LORO weighted: orig={oweighted:.2f}  aug={aweighted:.2f}  Δ={aweighted - oweighted:+.2f}')
