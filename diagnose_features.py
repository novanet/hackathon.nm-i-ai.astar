"""Diagnose train/test feature mismatch for food/wealth/defense."""
import importlib
import json
import numpy as np
from pathlib import Path

from train_mlp import load_round_data, compute_gt_round_features, ROUND_IDS
from astar.model import (
    compute_cell_features, compute_round_features,
    observation_calibrated_transitions, debias_transitions,
    _extract_settlement_stats, apply_floor,
    CALIBRATION_FACTORS, PER_CLASS_TEMPS,
    per_class_temperature_scale,
)
from astar.submit import score_prediction

import astar.model as mod
mod._spatial_model = None
importlib.reload(mod)
from astar.model import load_spatial_model
model = load_spatial_model()

print("=== TRAIN/TEST FEATURE MISMATCH ANALYSIS ===\n")

names = ["E->E", "S->S", "F->F", "E->S", "sett_dens", "food", "wealth", "defense"]

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if not gts:
        continue
    map_w, map_h = 40, 40

    gt_feats = compute_gt_round_features(detail, gts)

    cal = observation_calibrated_transitions(rid, detail)
    deb = debias_transitions(cal) if cal is not None else None
    stats = _extract_settlement_stats(rid, detail)

    est_with = compute_round_features(deb, detail, settlement_stats=stats)
    est_without = compute_round_features(deb, detail, settlement_stats=None)

    print(f"--- R{rnum} ---")
    # Show food/wealth/defense mismatch
    for i in [5, 6, 7]:  # food, wealth, defense
        g, w, n = gt_feats[i], est_with[i], est_without[i]
        ew = abs(g - w) / max(abs(g), 1e-10) * 100
        en = abs(g - n) / max(abs(g), 1e-10) * 100
        print(f"  {names[i]:>8}: GT={g:.4f}, w/stats={w:.4f}({ew:.0f}%), no_stats={n:.4f}({en:.0f}%)")

    # Score comparison
    for feat_label, feats in [("GT", gt_feats), ("w/stats", est_with), ("no_stats", est_without)]:
        scores = []
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            features = compute_cell_features(init_grid, map_w, map_h, feats)
            flat = features.reshape(-1, features.shape[-1])
            raw = model.predict(flat).reshape(map_h, map_w, 6)
            raw = np.maximum(raw, 1e-10)
            raw = raw / raw.sum(axis=-1, keepdims=True)
            raw = raw * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
            raw = raw / raw.sum(axis=-1, keepdims=True)
            ss_rate = feats[1]
            if ss_rate < 0.15:
                temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
            else:
                temps = PER_CLASS_TEMPS
            raw = per_class_temperature_scale(raw, detail, s, temps=temps)
            raw = apply_floor(raw)
            scores.append(score_prediction(raw, gt))
        print(f"    {feat_label:>8}: avg={np.mean(scores):.2f}")
    print()
