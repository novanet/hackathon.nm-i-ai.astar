"""
Deep R7 analysis: Would observation blending help?
Test 1: Current model (spatial only, no obs)
Test 2: Blend spatial model with per-cell observation data
Test 3: What if we could perfectly know round features? (oracle test)
"""
import json, numpy as np, warnings
from pathlib import Path
from astar.model import (
    build_prediction, compute_cell_features, load_spatial_model,
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, _cell_observation_frequencies,
    apply_floor, _apply_transition_matrix, HISTORICAL_TRANSITIONS,
    spatial_prior,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES, CLASS_NAMES

warnings.filterwarnings('ignore')

R7_ID = '36e581f1-73f8-453f-ab98-cbe3052b701b'
R6_ID = 'ae78003a-4efe-425a-881a-d16a39bca0ad'
DATA_DIR = Path('data')


def load_round(round_id):
    rdir = DATA_DIR / f'round_{round_id}'
    detail_files = sorted(rdir.glob('round_detail_*.json'))
    detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
    gts = []
    for s in range(len(detail['initial_states'])):
        gt_path = rdir / f'ground_truth_s{s}.json'
        if gt_path.exists():
            gts.append(np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64))
    return detail, gts


def test_obs_blending(round_id, detail, gts, label):
    """Test blending spatial model + observation data at various alphas."""
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)
    n_seeds = len(gts)

    # Get calibrated transitions
    cal = observation_calibrated_transitions(round_id, detail, map_w, map_h)
    debiased = debias_transitions(cal) if cal is not None else None
    round_feats = compute_round_features(debiased, detail)

    print(f'\n{"="*60}')
    print(f'{label}: {n_seeds} seeds')
    print(f'Round features: E→E={round_feats[0]:.4f}, S→S={round_feats[1]:.4f}, '
          f'F→F={round_feats[2]:.4f}, E→S={round_feats[3]:.4f}, density={round_feats[4]:.6f}')
    print(f'{"="*60}')

    # Test different strategies
    strategies = {
        'spatial_only': {},          # Current approach
        'obs_0.3': {'alpha': 0.3},   # 30% obs weight
        'obs_0.5': {'alpha': 0.5},   # 50% obs weight
        'obs_0.7': {'alpha': 0.7},   # 70% obs weight
        'obs_1.0': {'alpha': 1.0},   # pure obs (where available)
        'obs_adaptive': {'adaptive': True},  # weight by observation count
    }

    for strat_name, params in strategies.items():
        scores = []
        obs_cells_total = 0
        for seed_idx in range(n_seeds):
            # Get spatial prediction
            sp = spatial_prior(detail, seed_idx, map_w, map_h, round_features=round_feats)
            if sp is None:
                continue

            # Get cell observations
            obs_freq, obs_count = _cell_observation_frequencies(
                round_id, detail, seed_idx, map_w, map_h)

            if strat_name == 'spatial_only':
                pred = sp
            else:
                pred = sp.copy()
                obs_mask = obs_count > 0
                obs_cells_total += obs_mask.sum()

                if obs_mask.any():
                    # Normalize obs to probabilities
                    obs_prob = obs_freq.copy()
                    obs_sums = obs_count[..., np.newaxis]
                    obs_sums = np.maximum(obs_sums, 1.0)
                    obs_prob = obs_freq / obs_sums

                    if params.get('adaptive'):
                        # Weight by how many times we observed the cell
                        # 1 obs → 30% obs weight, 2 obs → 50%, 3+ → 70%
                        weight = np.minimum(obs_count * 0.2 + 0.1, 0.7)
                        weight = weight[..., np.newaxis]
                        pred[obs_mask] = (
                            (1 - weight[obs_mask]) * sp[obs_mask] +
                            weight[obs_mask] * obs_prob[obs_mask]
                        )
                    else:
                        alpha = params['alpha']
                        pred[obs_mask] = (1 - alpha) * sp[obs_mask] + alpha * obs_prob[obs_mask]

            pred = apply_floor(pred)
            sc = score_prediction(pred, gts[seed_idx])
            scores.append(sc)

        avg = np.mean(scores) if scores else 0
        obs_pct = obs_cells_total / (n_seeds * map_w * map_h) * 100 if n_seeds else 0
        print(f'  {strat_name:>15}: {avg:.2f}  seeds={[f"{s:.1f}" for s in scores]}')
        if strat_name == 'spatial_only':
            print(f'                    obs coverage: {obs_pct:.1f}% of cells')


# Test on R6 (good) and R7 (bad)
for rid, label in [(R6_ID, 'R6 (good, 77.9 official)'), (R7_ID, 'R7 (bad, 56.9 official)')]:
    detail, gts = load_round(rid)
    if gts:
        test_obs_blending(rid, detail, gts, label)

# Oracle test: what if we use GT round features for R7?
print('\n\n=== ORACLE TEST: GT round features for R7 ===')
from train_spatial import compute_gt_round_features
r7_detail, r7_gts = load_round(R7_ID)
gt_feats = compute_gt_round_features(r7_detail, r7_gts)

# Compare debiased vs GT round features
cal = observation_calibrated_transitions(R7_ID, r7_detail)
debiased = debias_transitions(cal) if cal is not None else None
obs_feats = compute_round_features(debiased, r7_detail)

print(f'  Obs features:    E→E={obs_feats[0]:.4f}, S→S={obs_feats[1]:.4f}, '
      f'F→F={obs_feats[2]:.4f}, E→S={obs_feats[3]:.4f}')
print(f'  GT  features:    E→E={gt_feats[0]:.4f}, S→S={gt_feats[1]:.4f}, '
      f'F→F={gt_feats[2]:.4f}, E→S={gt_feats[3]:.4f}')

map_w = r7_detail.get('map_width', 40)
map_h = r7_detail.get('map_height', 40)

oracle_scores = []
obs_scores = []
for s in range(len(r7_gts)):
    # Oracle: use GT features
    sp = spatial_prior(r7_detail, s, map_w, map_h, round_features=gt_feats)
    pred = apply_floor(sp)
    oracle_scores.append(score_prediction(pred, r7_gts[s]))

    # Observed: use debiased features (our current approach)
    sp2 = spatial_prior(r7_detail, s, map_w, map_h, round_features=obs_feats)
    pred2 = apply_floor(sp2)
    obs_scores.append(score_prediction(pred2, r7_gts[s]))

print(f'  With obs features:  {np.mean(obs_scores):.2f}  seeds={[f"{s:.1f}" for s in obs_scores]}')
print(f'  With GT features:   {np.mean(oracle_scores):.2f}  seeds={[f"{s:.1f}" for s in oracle_scores]}')
print(f'  Δ = {np.mean(oracle_scores) - np.mean(obs_scores):+.2f} (how much better GT features are)')
