"""
Test feature uncertainty ensemble: instead of a single prediction with 
point-estimate round features, sample N predictions with perturbed features
and average them. This hedges against feature estimation errors.

Also test: training with noisy round features (augmentation) to make
the model more robust to feature estimation errors.
"""
import json, numpy as np, warnings
from pathlib import Path
from astar.model import (
    spatial_prior, apply_floor, observation_calibrated_transitions,
    debias_transitions, compute_round_features, load_spatial_model,
    compute_cell_features, HISTORICAL_TRANSITIONS,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from train_spatial import compute_gt_round_features, load_round_data, ROUND_IDS

warnings.filterwarnings('ignore')

DATA_DIR = Path('data')


def ensemble_prediction(detail, seed_idx, round_feats_mean, round_feats_std,
                        n_samples=20, map_w=40, map_h=40):
    """
    Make N predictions with perturbed round features and average them.
    """
    model = load_spatial_model()
    if model is None:
        return None

    init_grid = detail['initial_states'][seed_idx]['grid']
    preds = []

    # Include the mean prediction
    feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats_mean)
    flat = model.predict(feat.reshape(-1, feat.shape[-1]))
    pred = flat.reshape(map_h, map_w, NUM_CLASSES)
    pred = np.maximum(pred, 1e-10)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    preds.append(pred)

    # Sample perturbed features
    rng = np.random.default_rng(42)
    for _ in range(n_samples - 1):
        perturbed = round_feats_mean + rng.normal(0, 1, size=len(round_feats_mean)) * round_feats_std
        # Clip to reasonable ranges
        perturbed = np.clip(perturbed, 0.01, 0.99)
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=perturbed)
        flat = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        preds.append(pred)

    # Average all predictions
    avg_pred = np.mean(preds, axis=0)
    avg_pred = avg_pred / avg_pred.sum(axis=-1, keepdims=True)
    return avg_pred


# First, compute empirical feature estimation errors across all rounds
# This tells us how uncertain our estimates are
print('=== FEATURE ESTIMATION ERRORS ACROSS ROUNDS ===')
errors = {i: [] for i in range(5)}
feature_names = ['E→E', 'S→S', 'F→F', 'E→S', 'density']

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    # GT features (truth)
    gt_feats = compute_gt_round_features(detail, gts)

    # Observed features (what we'd estimate at prediction time)
    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    errs = obs_feats - gt_feats
    for i in range(5):
        errors[i].append(errs[i])

    print(f'  R{rnum}: obs=[{", ".join(f"{obs_feats[i]:.4f}" for i in range(5))}]  '
          f'gt=[{", ".join(f"{gt_feats[i]:.4f}" for i in range(5))}]  '
          f'err=[{", ".join(f"{errs[i]:+.4f}" for i in range(5))}]')

print('\nError statistics:')
error_means = np.zeros(5)
error_stds = np.zeros(5)
for i in range(5):
    err_arr = np.array(errors[i])
    error_means[i] = np.mean(err_arr)
    error_stds[i] = np.std(err_arr)
    print(f'  {feature_names[i]:>8}: mean={error_means[i]:+.4f}, std={error_stds[i]:.4f}, '
          f'range=[{err_arr.min():+.4f}, {err_arr.max():+.4f}]')

# Test ensemble on each round
print('\n\n=== ENSEMBLE VS POINT ESTIMATE ===')
print('Testing N=20 samples with empirical error std')

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    # Observed features
    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    point_scores = []
    ens_scores = []

    for s in range(len(gts)):
        # Point estimate (current approach)
        sp = spatial_prior(detail, s, map_w, map_h, round_features=obs_feats)
        point_pred = apply_floor(sp)
        point_scores.append(score_prediction(point_pred, gts[s]))

        # Ensemble
        ens_pred = ensemble_prediction(detail, s, obs_feats, error_stds, n_samples=20, map_w=map_w, map_h=map_h)
        ens_pred = apply_floor(ens_pred)
        ens_scores.append(score_prediction(ens_pred, gts[s]))

    pavg = np.mean(point_scores)
    eavg = np.mean(ens_scores)
    print(f'  R{rnum}: point={pavg:.2f}  ensemble={eavg:.2f}  Δ={eavg-pavg:+.2f}')

# Also test: correct the systematic bias in features
# If we know the mean error, we can subtract it
print('\n\n=== BIAS-CORRECTED FEATURES ===')
print(f'Correction: subtract mean error per feature')

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    obs_feats = compute_round_features(debiased, detail)

    # Bias-corrected (leave-one-out: use mean error from OTHER rounds)
    loo_errors = {i: [] for i in range(5)}
    for other_rnum in ROUND_IDS:
        if other_rnum == rnum:
            continue
        for i in range(5):
            loo_errors[i].append(errors[i][list(ROUND_IDS.keys()).index(other_rnum)])
    loo_means = np.array([np.mean(loo_errors[i]) for i in range(5)])

    corrected_feats = obs_feats - loo_means
    corrected_feats = np.clip(corrected_feats, 0.01, 0.99)

    point_scores = []
    corr_scores = []
    for s in range(len(gts)):
        sp = spatial_prior(detail, s, map_w, map_h, round_features=obs_feats)
        point_pred = apply_floor(sp)
        point_scores.append(score_prediction(point_pred, gts[s]))

        sp2 = spatial_prior(detail, s, map_w, map_h, round_features=corrected_feats)
        corr_pred = apply_floor(sp2)
        corr_scores.append(score_prediction(corr_pred, gts[s]))

    pavg = np.mean(point_scores)
    cavg = np.mean(corr_scores)
    print(f'  R{rnum}: point={pavg:.2f}  corrected={cavg:.2f}  Δ={cavg-pavg:+.2f}')
