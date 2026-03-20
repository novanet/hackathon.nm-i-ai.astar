"""
Refined ensemble test with correct error std (obs-available rounds only).
And test training with noisy round features.
"""
import json, numpy as np, pickle, warnings
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    spatial_prior, apply_floor, observation_calibrated_transitions,
    debias_transitions, compute_round_features, load_spatial_model,
    compute_cell_features, HISTORICAL_TRANSITIONS,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from train_spatial import compute_gt_round_features, load_round_data, ROUND_IDS, build_training_data_multi

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

# Rounds where we had actual observations (not falling back to historical)
OBS_ROUNDS = {2, 5, 6, 7}

# Compute correct error statistics from obs-available rounds only
print('=== OBS-ONLY FEATURE ERRORS ===')
errors_obs = []
for rnum in sorted(OBS_ROUNDS):
    rid = ROUND_IDS[rnum]
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)
    gt_feats = compute_gt_round_features(detail, gts)
    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal)
    obs_feats = compute_round_features(debiased, detail)
    errs = obs_feats - gt_feats
    errors_obs.append(errs)
    print(f'  R{rnum}: err=[{", ".join(f"{e:+.4f}" for e in errs)}]')

errors_obs = np.array(errors_obs)
obs_std = np.std(errors_obs, axis=0)
obs_mean = np.mean(errors_obs, axis=0)
feature_names = ['E→E', 'S→S', 'F→F', 'E→S', 'density']
print(f'\nObs-only error std:  [{", ".join(f"{s:.4f}" for s in obs_std)}]')
print(f'Obs-only error mean: [{", ".join(f"{m:+.4f}" for m in obs_mean)}]')


def ensemble_prediction_v2(detail, seed_idx, round_feats_mean, round_feats_std,
                           n_samples=20, map_w=40, map_h=40):
    """Ensemble over feature uncertainty with correct std."""
    model = load_spatial_model()
    if model is None:
        return None

    init_grid = detail['initial_states'][seed_idx]['grid']
    preds = []

    rng = np.random.default_rng(42)
    for i in range(n_samples):
        if i == 0:
            feats = round_feats_mean  # include mean itself
        else:
            feats = round_feats_mean + rng.normal(0, 1, size=len(round_feats_mean)) * round_feats_std
            feats = np.clip(feats, 0.01, 0.99)

        cell_feat = compute_cell_features(init_grid, map_w, map_h, round_features=feats)
        flat = model.predict(cell_feat.reshape(-1, cell_feat.shape[-1]))
        pred = flat.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        preds.append(pred)

    avg_pred = np.mean(preds, axis=0)
    avg_pred = avg_pred / avg_pred.sum(axis=-1, keepdims=True)
    return avg_pred


# Test corrected ensemble
print('\n\n=== CORRECTED ENSEMBLE (obs-only std) ===')
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    point_scores = []
    ens_scores = []

    for s in range(len(gts)):
        sp = spatial_prior(detail, s, map_w, map_h, round_features=obs_feats)
        point_pred = apply_floor(sp)
        point_scores.append(score_prediction(point_pred, gts[s]))

        ens_pred = ensemble_prediction_v2(detail, s, obs_feats, obs_std, n_samples=20, map_w=map_w, map_h=map_h)
        ens_pred = apply_floor(ens_pred)
        ens_scores.append(score_prediction(ens_pred, gts[s]))

    pavg = np.mean(point_scores)
    eavg = np.mean(ens_scores)
    delta = eavg - pavg
    marker = '***' if delta > 1 else '  *' if delta > 0 else ''
    print(f'  R{rnum}: point={pavg:.2f}  ensemble={eavg:.2f}  Δ={delta:+.2f} {marker}')


# Test: train with AUGMENTED round features (noise injection during training)
print('\n\n=== TRAINING WITH NOISY ROUND FEATURES (augmentation) ===')
print('Training 3x augmented model...')


def build_augmented_training_data(round_ids: dict[int, str], noise_std: np.ndarray, n_augments: int = 3):
    """Build training data with N augmented copies using noisy round features."""
    X_parts, Y_parts = [], []
    rng = np.random.default_rng(123)

    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            continue
        map_w = detail.get('map_width', 40)
        map_h = detail.get('map_height', 40)
        gt_feats = compute_gt_round_features(detail, gts)

        for aug in range(n_augments):
            if aug == 0:
                feats = gt_feats  # original
            else:
                # Add noise matching the empirical estimation error
                feats = gt_feats + rng.normal(0, 1, size=len(gt_feats)) * noise_std
                feats = np.clip(feats, 0.01, 0.99)

            for seed_idx, gt in enumerate(gts):
                init_grid = detail['initial_states'][seed_idx]['grid']
                cell_feat = compute_cell_features(init_grid, map_w, map_h, round_features=feats)
                X_parts.append(cell_feat.reshape(-1, cell_feat.shape[-1]))
                Y_parts.append(gt.reshape(-1, gt.shape[-1]))

    return np.vstack(X_parts), np.vstack(Y_parts)


X_aug, Y_aug = build_augmented_training_data(ROUND_IDS, obs_std, n_augments=3)
print(f'  Augmented training: {X_aug.shape[0]} samples, {X_aug.shape[1]} features')

aug_model = MultiOutputRegressor(
    lgb.LGBMRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        verbosity=-1,
    ),
    n_jobs=-1,
)
aug_model.fit(X_aug, Y_aug)

# Test augmented model on all rounds (using obs features = realistic prediction-time features)
print('\n  Augmented model vs original:')
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    orig_scores = []
    aug_scores = []

    for s in range(len(gts)):
        init_grid = detail['initial_states'][s]['grid']
        cell_feat = compute_cell_features(init_grid, map_w, map_h, round_features=obs_feats)
        flat_feat = cell_feat.reshape(-1, cell_feat.shape[-1])

        # Original model
        sp = load_spatial_model().predict(flat_feat).reshape(map_h, map_w, NUM_CLASSES)
        sp = np.maximum(sp, 1e-10)
        sp = sp / sp.sum(axis=-1, keepdims=True)
        orig_scores.append(score_prediction(apply_floor(sp), gts[s]))

        # Augmented model
        sp2 = aug_model.predict(flat_feat).reshape(map_h, map_w, NUM_CLASSES)
        sp2 = np.maximum(sp2, 1e-10)
        sp2 = sp2 / sp2.sum(axis=-1, keepdims=True)
        aug_scores.append(score_prediction(apply_floor(sp2), gts[s]))

    oavg = np.mean(orig_scores)
    aavg = np.mean(aug_scores)
    delta = aavg - oavg
    marker = '***' if delta > 1 else '  *' if delta > 0 else ''
    print(f'    R{rnum}: original={oavg:.2f}  augmented={aavg:.2f}  Δ={delta:+.2f} {marker}')

# LORO CV for augmented model
print('\n  LORO CV for augmented model:')
loro_results = {}
for test_rnum in sorted(ROUND_IDS.keys()):
    if test_rnum not in {r for r, (_, _, gts) in [(k, load_round_data(ROUND_IDS[k])) for k in ROUND_IDS] if gts}:
        continue
    detail, gts = load_round_data(ROUND_IDS[test_rnum])
    if not gts:
        continue

    train_ids = {r: ROUND_IDS[r] for r in ROUND_IDS if r != test_rnum}
    X_tr, Y_tr = build_augmented_training_data(train_ids, obs_std, n_augments=3)

    model_loro = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            verbosity=-1,
        ),
        n_jobs=-1,
    )
    model_loro.fit(X_tr, Y_tr)

    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)
    gt_feats = compute_gt_round_features(detail, gts)

    cal = observation_calibrated_transitions(ROUND_IDS[test_rnum], detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    scores_gt = []
    scores_obs = []
    for s in range(len(gts)):
        init_grid = detail['initial_states'][s]['grid']

        # With GT features
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=gt_feats)
        flat = model_loro.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        scores_gt.append(score_prediction(apply_floor(pred), gts[s]))

        # With obs features (realistic)
        feat2 = compute_cell_features(init_grid, map_w, map_h, round_features=obs_feats)
        flat2 = model_loro.predict(feat2.reshape(-1, feat2.shape[-1]))
        pred2 = flat2.reshape(map_h, map_w, NUM_CLASSES)
        pred2 = np.maximum(pred2, 1e-10)
        pred2 = pred2 / pred2.sum(axis=-1, keepdims=True)
        scores_obs.append(score_prediction(apply_floor(pred2), gts[s]))

    gt_avg = np.mean(scores_gt)
    obs_avg = np.mean(scores_obs)
    gap = gt_avg - obs_avg
    loro_results[test_rnum] = obs_avg
    print(f'    R{test_rnum}: obs_features={obs_avg:.2f}  gt_features={gt_avg:.2f}  gap={gap:.2f}')

avg_loro = np.mean(list(loro_results.values()))
print(f'    LORO average (obs features): {avg_loro:.2f}')
