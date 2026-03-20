"""Extended search in the softer direction."""
import json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import astar.model as _m
_m._spatial_model = None

from astar.model import (
    spatial_prior, per_class_temperature_scale,
    observation_calibrated_transitions, debias_transitions, compute_round_features,
    _extract_settlement_stats, _apply_transition_matrix, HISTORICAL_TRANSITIONS
)
from astar.submit import score_prediction

DATA_DIR = Path('data')
ROUND_IDS = {
    2: '76909e29-f664-4b2f-b16b-61b7507277e9',
    5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    6: 'ae78003a-4efe-425a-881a-d16a39bca0ad',
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
}

_cache = {}
def get_round_data():
    if _cache:
        return _cache
    for rnum, rid in sorted(ROUND_IDS.items()):
        rdir = DATA_DIR / f'round_{rid}'
        detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
        mw, mh = detail.get('map_width', 40), detail.get('map_height', 40)
        cal = observation_calibrated_transitions(rid, detail, mw, mh)
        deb = debias_transitions(cal) if cal is not None else None
        stats = _extract_settlement_stats(rid, detail)
        feats = compute_round_features(deb, detail, settlement_stats=stats)
        seeds = []
        for s in range(len(detail.get('initial_states', []))):
            gt_path = rdir / f'ground_truth_s{s}.json'
            if not gt_path.exists(): continue
            gt = np.array(json.loads(gt_path.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64)
            pred_base = spatial_prior(detail, s, mw, mh, round_features=feats)
            if pred_base is None:
                trans = deb if deb is not None else HISTORICAL_TRANSITIONS
                pred_base = _apply_transition_matrix(detail, s, trans, mw, mh)
            seeds.append((detail, s, pred_base, gt, mw, mh, feats))
        _cache[rnum] = seeds
    return _cache

def evaluate(sft_temps, floor=0.0002):
    data = get_round_data()
    scores = []
    for rnum, seeds in sorted(data.items()):
        for (detail, s, pred_base, gt, mw, mh, feats) in seeds:
            ss_rate = feats[1]
            if ss_rate < 0.15:
                temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
            else:
                temps = sft_temps
            pred = per_class_temperature_scale(pred_base.copy(), detail, s, temps=temps, map_w=mw, map_h=mh)
            floored = np.maximum(pred, floor)
            pred = floored / floored.sum(axis=-1, keepdims=True)
            scores.append(score_prediction(pred, gt))
    return np.mean(scores)

print("=== EXTENDED SOFT SWEEP ===")
for e_temp in [1.15, 1.20, 1.25, 1.30]:
    for sfp_temp in [0.90, 0.92, 0.95, 0.98, 1.00]:
        temps = np.array([e_temp, sfp_temp, sfp_temp, sfp_temp, e_temp, 1.0])
        score = evaluate(temps)
        print(f"  E/W={e_temp:.2f} SFP={sfp_temp:.2f}: {score:.3f}")

print("\n=== WATER TEMP VARIATIONS ===")
for w_temp in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]:
    temps = np.array([1.20, 0.90, 0.90, 0.90, w_temp, 1.0])
    score = evaluate(temps)
    print(f"  W={w_temp:.2f}: {score:.3f}")
