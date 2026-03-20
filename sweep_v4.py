"""Quick temperature and floor sweep on the V4 model."""
import json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import astar.model as _m
_m._spatial_model = None

from astar.model import (
    build_prediction, apply_floor, spatial_prior, per_class_temperature_scale,
    observation_calibrated_transitions, debias_transitions, compute_round_features,
    _extract_settlement_stats, _apply_transition_matrix, HISTORICAL_TRANSITIONS,
    PER_CLASS_TEMPS, compute_cell_features, load_spatial_model
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

DATA_DIR = Path('data')
ROUND_IDS = {
    2: '76909e29-f664-4b2f-b16b-61b7507277e9',
    5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    6: 'ae78003a-4efe-425a-881a-d16a39bca0ad',
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
}

def backtest_pipeline(rounds, temp_override=None, floor_override=None):
    """Run pipeline with optional temp/floor overrides. Only on rounds with obs."""
    scores = []
    for rnum, rid in sorted(rounds.items()):
        rdir = DATA_DIR / f'round_{rid}'
        detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
        map_w, map_h = detail.get('map_width', 40), detail.get('map_height', 40)
        for s in range(len(detail.get('initial_states', []))):
            gt_path = rdir / f'ground_truth_s{s}.json'
            if not gt_path.exists(): continue
            gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)

            # Replicate build_prediction manually for flexibility
            cal = observation_calibrated_transitions(rid, detail, map_w, map_h)
            deb = debias_transitions(cal) if cal is not None else None
            stats = _extract_settlement_stats(rid, detail)
            feats = compute_round_features(deb, detail, settlement_stats=stats)

            pred = spatial_prior(detail, s, map_w, map_h, round_features=feats)
            if pred is None:
                trans = deb if deb is not None else HISTORICAL_TRANSITIONS
                pred = _apply_transition_matrix(detail, s, trans, map_w, map_h)

            # Temperature (with collapse detection)
            ss_rate = feats[1]
            if temp_override is not None:
                temps = temp_override
            elif ss_rate < 0.15:
                temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
            else:
                temps = PER_CLASS_TEMPS
            pred = per_class_temperature_scale(pred, detail, s, temps=temps, map_w=map_w, map_h=map_h)

            floor = floor_override if floor_override is not None else 0.0003
            floored = np.maximum(pred, floor)
            pred = floored / floored.sum(axis=-1, keepdims=True)
            scores.append(score_prediction(pred, gt))
    return np.mean(scores)

# Baseline
base = backtest_pipeline(ROUND_IDS)
print(f"Baseline (current): {base:.3f}")

# Temperature experiments (only for normal rounds, not collapse)
print("\n=== NORMAL TEMPS (for rounds with obs, S->S > 0.15) ===")
temp_configs = [
    ("Current [1.15, 0.80, 0.80, 0.80, 1.15, 1.0]", None),
    ("All 1.0 (no temps)", np.ones(6)),
    ("Softer [1.20, 0.90, 0.90, 0.90, 1.20, 1.0]", np.array([1.20, 0.90, 0.90, 0.90, 1.20, 1.0])),
    ("Sharper [1.10, 0.70, 0.70, 0.70, 1.10, 1.0]", np.array([1.10, 0.70, 0.70, 0.70, 1.10, 1.0])),
    ("Moderate [1.15, 0.85, 0.85, 0.85, 1.15, 1.0]", np.array([1.15, 0.85, 0.85, 0.85, 1.15, 1.0])),
    ("Extra soft [1.20, 1.0, 1.0, 1.0, 1.20, 1.0]", np.array([1.20, 1.0, 1.0, 1.0, 1.20, 1.0])),
]
for name, t in temp_configs:
    score = backtest_pipeline(ROUND_IDS, temp_override=t)
    delta = score - base
    print(f"  {name}: {score:.3f} ({delta:+.3f})")

# Floor experiments
print("\n=== PROBABILITY FLOOR ===")
for floor in [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.005]:
    score = backtest_pipeline(ROUND_IDS, floor_override=floor)
    delta = score - base
    print(f"  floor={floor:.4f}: {score:.3f} ({delta:+.3f})")
