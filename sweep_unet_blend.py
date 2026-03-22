"""Sweep U-Net blend ratio in the full pipeline backtest."""
import numpy as np
import json
import sys
from pathlib import Path

# Prevent cached module — force fresh U-Net load
for mod in list(sys.modules):
    if mod.startswith("astar"):
        del sys.modules[mod]
import astar.model as _m
_m._unet_model = None  # force reload of new .pt file

from astar.model import (
    observation_calibrated_transitions, debias_transitions,
    _extract_settlement_stats, compute_round_features, correct_round_features,
    spatial_prior, _apply_transition_matrix, per_class_temperature_scale,
    _adaptive_bayesian_overlay, apply_floor, _load_unet_model,
    HISTORICAL_TRANSITIONS, CALIBRATION_FACTORS, PER_CLASS_TEMPS,
)


def score_prediction(pred, gt):
    p = np.clip(pred, 1e-10, 1.0)
    t = np.clip(gt, 1e-10, 1.0)
    kl = np.sum(t * np.log(t / p), axis=-1)
    ent = -np.sum(t * np.log(t), axis=-1)
    w = np.maximum(ent, 0.01)
    wkl = np.sum(w * kl) / np.sum(w)
    return 100.0 * np.exp(-3.0 * wkl)


ROUNDS = {
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
}


def build_prediction_with_blend(rid, detail, seed_index, map_w, map_h, unet_weight):
    """Replicate build_prediction but with configurable unet_weight."""
    calibrated_trans = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased_trans = debias_transitions(calibrated_trans) if calibrated_trans is not None else None
    sett_stats = _extract_settlement_stats(rid, detail)
    round_feats = compute_round_features(debiased_trans, detail, settlement_stats=sett_stats)
    round_feats = correct_round_features(round_feats)

    pred = spatial_prior(detail, seed_index, map_w, map_h, round_features=round_feats)

    # U-Net blend with configurable weight
    if unet_weight > 0:
        unet = _load_unet_model()
        if unet is not None and pred is not None:
            from astar.unet import predict_unet_with_tta
            init_grid = detail["initial_states"][seed_index]["grid"]
            unet_pred = predict_unet_with_tta(unet, init_grid, map_w, map_h, round_feats)
            unet_pred = np.maximum(unet_pred, 1e-10)
            unet_pred = unet_pred / unet_pred.sum(axis=-1, keepdims=True)
            pred = (1 - unet_weight) * pred + unet_weight * unet_pred
            pred = pred / pred.sum(axis=-1, keepdims=True)

    if pred is None:
        trans = debiased_trans if debiased_trans is not None else HISTORICAL_TRANSITIONS
        pred = _apply_transition_matrix(detail, seed_index, trans, map_w, map_h)

    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)

    ss_rate = round_feats[1]
    if ss_rate < 0.15:
        adaptive_temps = np.array([1.05, 1.05, 1.05, 1.0, 1.05, 1.0])
    elif ss_rate > 0.40:
        adaptive_temps = np.array([1.10, 0.95, 1.10, 1.0, 1.10, 1.0])
    else:
        adaptive_temps = PER_CLASS_TEMPS
    pred = per_class_temperature_scale(pred, detail, seed_index,
                                       temps=adaptive_temps, map_w=map_w, map_h=map_h)
    pred = _adaptive_bayesian_overlay(rid, seed_index, pred, map_w, map_h)
    return apply_floor(pred)


# Pre-load data
print("Loading round data...")
round_data = {}
for rnum, rid in sorted(ROUNDS.items()):
    rdir = Path("data") / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        continue
    detail = json.loads(detail_files[0].read_text())
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gts = {}
    for s in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if gt_path.exists():
            gt_data = json.loads(gt_path.read_text())
            gts[s] = np.array(gt_data["ground_truth"], dtype=np.float64)
    if gts:
        round_data[rnum] = (rid, detail, map_w, map_h, gts)

print(f"Loaded {len(round_data)} rounds\n")

# Sweep
WEIGHTS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
results = {}

for w in WEIGHTS:
    all_scores = []
    per_round = {}
    for rnum, (rid, detail, map_w, map_h, gts) in sorted(round_data.items()):
        scores = []
        for s, gt in gts.items():
            pred = build_prediction_with_blend(rid, detail, s, map_w, map_h, w)
            scores.append(score_prediction(pred, gt))
        avg = np.mean(scores)
        all_scores.append(avg)
        per_round[rnum] = avg
    overall = np.mean(all_scores)
    results[w] = (overall, per_round)
    print(f"unet_weight={w:.2f}  avg={overall:.2f}  " +
          "  ".join(f"R{r}={s:.1f}" for r, s in sorted(per_round.items())))

print("\n=== SUMMARY ===")
best_w = max(results, key=lambda w: results[w][0])
for w in WEIGHTS:
    marker = " <-- BEST" if w == best_w else ""
    print(f"  unet_weight={w:.2f}  avg={results[w][0]:.2f}{marker}")
print(f"\nOptimal blend: {best_w:.0%} U-Net / {1-best_w:.0%} GBM+MLP")
