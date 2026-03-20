"""Resubmit R8 with ensemble - single-threaded to avoid Windows multiprocessing hang."""
import json, numpy as np, pickle, warnings
from pathlib import Path
from astar.client import submit
from astar.model import (
    compute_cell_features, apply_floor,
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, prediction_to_list,
)
from astar.replay import NUM_CLASSES

warnings.filterwarnings('ignore')

R8_ID = 'c5cdf100-a876-4fb7-b5d8-757162c97989'
DATA_DIR = Path('data') / f'round_{R8_ID}'

detail_files = sorted(DATA_DIR.glob('round_detail_*.json'))
detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
n_seeds = len(detail['initial_states'])
map_w = detail.get('map_width', 40)
map_h = detail.get('map_height', 40)

# Load model and force single-threaded
model = pickle.loads(Path('data/spatial_model.pkl').read_bytes())
# Patch n_jobs on MultiOutputRegressor and each child estimator
model.n_jobs = 1
for est in model.estimators_:
    if hasattr(est, 'n_jobs'):
        est.n_jobs = 1

# Get round features
cal = observation_calibrated_transitions(R8_ID, detail, map_w, map_h)
debiased = debias_transitions(cal) if cal is not None else None
obs_feats = compute_round_features(debiased, detail)
print(f'R8 features: E->E={obs_feats[0]:.4f}, S->S={obs_feats[1]:.4f}, '
      f'F->F={obs_feats[2]:.4f}, E->S={obs_feats[3]:.4f}')

# Feature estimation std from obs-available rounds
obs_std = np.array([0.0048, 0.0140, 0.0070, 0.0048, 0.0000])
N_SAMPLES = 20

for seed_idx in range(n_seeds):
    init_grid = detail['initial_states'][seed_idx]['grid']
    preds = []

    rng = np.random.default_rng(42)
    for i in range(N_SAMPLES):
        if i == 0:
            feats = obs_feats
        else:
            feats = obs_feats + rng.normal(0, 1, size=len(obs_feats)) * obs_std
            feats = np.clip(feats, 0.01, 0.99)

        cell_feat = compute_cell_features(init_grid, map_w, map_h, round_features=feats)
        flat = model.predict(cell_feat.reshape(-1, cell_feat.shape[-1]))
        pred = flat.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        preds.append(pred)
        if i % 5 == 0:
            print(f'  Seed {seed_idx}: sample {i}/{N_SAMPLES}')

    ensemble = np.mean(preds, axis=0)
    ensemble = ensemble / ensemble.sum(axis=-1, keepdims=True)
    ensemble = apply_floor(ensemble)

    resp = submit(R8_ID, seed_idx, prediction_to_list(ensemble))
    status = resp.get('status', '?')
    score = resp.get('score', '?')
    print(f'  Seed {seed_idx}: {status} score={score}')

print('Done - R8 patched with ensemble predictions')
