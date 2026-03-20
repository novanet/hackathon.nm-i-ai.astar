"""
Refined tests:
1. Fine-grained entropy weight powers (0.1-0.5 range sweet spot)
2. Calibration tested in LORO mode (not in-sample)
3. Combination of best entropy weighting + calibration
"""
import json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from astar.model import (
    compute_cell_features, apply_floor, per_class_temperature_scale,
    PER_CLASS_TEMPS, NUM_CLASSES
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS

DATA_DIR = Path("data")
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
}

def load_round_data(round_id):
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files: return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists(): break
        gts.append(np.array(json.loads(gt_path.read_text(encoding="utf-8"))["ground_truth"], dtype=np.float64))
    return detail, gts

def compute_gt_round_features(detail, gts):
    states = detail.get("initial_states", [])
    mw, mh = detail.get("map_width", 40), detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = 0; total = 0
    for s in range(len(gts)):
        for y in range(mh):
            for x in range(mw):
                init_cls = TERRAIN_TO_CLASS.get(states[s]["grid"][y][x], 0)
                gt_counts[init_cls] += gts[s][y, x]
                if init_cls == 1: n_sett += 1
                total += 1
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / row_sums
    ss = gt_trans[1,1]
    return np.array([gt_trans[0,0], ss, gt_trans[4,4], gt_trans[0,1],
                     n_sett/max(total,1), 0.3+0.7*ss, gt_trans[0,1]*0.3, 1.0-gt_trans[1,0]])

def build_data(round_ids, return_weights=False):
    X_parts, Y_parts, W_parts = [], [], []
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts: continue
        mw, mh = detail.get("map_width", 40), detail.get("map_height", 40)
        rf = compute_gt_round_features(detail, gts)
        for s, gt in enumerate(gts):
            feat = compute_cell_features(detail["initial_states"][s]["grid"], mw, mh, round_features=rf)
            X_parts.append(feat.reshape(-1, feat.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            if return_weights:
                p = np.clip(gt, 1e-10, 1.0)
                W_parts.append(-np.sum(p * np.log(p), axis=-1).flatten())
    X, Y = np.vstack(X_parts), np.vstack(Y_parts)
    W = np.concatenate(W_parts) if return_weights else None
    return X, Y, W

# Load all round data once
ALL_DATA = {}
for rn, rid in sorted(ROUND_IDS.items()):
    d, g = load_round_data(rid)
    if g: ALL_DATA[rn] = (rid, d, g)

def run_loro(model_fn, calib_factors=None, use_temps=True):
    cf = np.array(calib_factors) if calib_factors else None
    results = {}
    for test_rnum in sorted(ALL_DATA.keys()):
        train_ids = {r: ROUND_IDS[r] for r in ALL_DATA if r != test_rnum}
        model = model_fn(train_ids)
        _, test_detail, test_gts = ALL_DATA[test_rnum]
        mw, mh = test_detail.get("map_width", 40), test_detail.get("map_height", 40)
        rf = compute_gt_round_features(test_detail, test_gts)
        scores = []
        for s, gt in enumerate(test_gts):
            feat = compute_cell_features(test_detail["initial_states"][s]["grid"], mw, mh, round_features=rf)
            pred = model.predict(feat.reshape(-1, feat.shape[-1])).reshape(mh, mw, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            if cf is not None:
                pred = pred * cf[np.newaxis, np.newaxis, :]
                pred = pred / pred.sum(axis=-1, keepdims=True)
            if use_temps:
                ss_rate = rf[1]
                temps = np.array([1.15,1.15,1.15,1.0,1.15,1.0]) if ss_rate < 0.15 else PER_CLASS_TEMPS
                pred = per_class_temperature_scale(pred, test_detail, s, temps=temps, map_w=mw, map_h=mh)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))
        results[test_rnum] = np.mean(scores)
    return results

LGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
    num_leaves=15, min_child_samples=50, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0, verbosity=-1)

def make_model(train_ids, weight_power=None):
    X, Y, W = build_data(train_ids, return_weights=(weight_power is not None))
    m = MultiOutputRegressor(lgb.LGBMRegressor(**LGB_PARAMS), n_jobs=1)
    if weight_power is not None:
        W_final = np.power(W + 0.01, weight_power)
        m.fit(X, Y, sample_weight=W_final)
    else:
        m.fit(X, Y)
    return m

# ============================================================
print("=" * 70)
print("1. FINE-GRAINED ENTROPY WEIGHT SWEEP (LORO + temps)")
print("=" * 70)
baseline = run_loro(lambda ids: make_model(ids))
bl_avg = np.mean(list(baseline.values()))
per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(baseline.items()))
print(f"  power=0.0 (baseline): AVG={bl_avg:.2f} | {per_round}")

for wp in [0.15, 0.25, 0.35, 0.50, 0.75]:
    results = run_loro(lambda ids, w=wp: make_model(ids, weight_power=w))
    avg = np.mean(list(results.values()))
    delta = avg - bl_avg
    per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(results.items()))
    print(f"  power={wp:.2f}: AVG={avg:.2f} (Δ={delta:+.2f}) | {per_round}")

# ============================================================
print("\n" + "=" * 70)
print("2. CALIBRATION IN LORO MODE (not in-sample)")
print("=" * 70)
for name, cf in [
    ("No calib", None),
    ("S/P/R 0.95", [1.02, 0.95, 0.95, 0.95, 1.0, 1.0]),
    ("S/P/R 0.90", [1.04, 0.90, 0.90, 0.90, 1.0, 1.0]),
    ("S/P/R 0.92", [1.03, 0.92, 0.92, 0.92, 1.0, 1.0]),
    ("S 0.93 P/R 0.80", [1.03, 0.93, 0.80, 0.80, 1.0, 1.0]),
]:
    results = run_loro(lambda ids: make_model(ids), calib_factors=cf)
    avg = np.mean(list(results.values()))
    delta = avg - bl_avg
    per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(results.items()))
    print(f"  {name}: AVG={avg:.2f} (Δ={delta:+.2f}) | {per_round}")

# ============================================================
print("\n" + "=" * 70)
print("3. COMBINED: Best entropy weighting + best calibration")
print("=" * 70)
for wp in [0.25, 0.35, 0.50]:
    for name, cf in [("+ no calib", None), ("+ S/P/R 0.95", [1.02, 0.95, 0.95, 0.95, 1.0, 1.0]),
                      ("+ S/P/R 0.92", [1.03, 0.92, 0.92, 0.92, 1.0, 1.0])]:
        results = run_loro(lambda ids, w=wp: make_model(ids, weight_power=w), calib_factors=cf)
        avg = np.mean(list(results.values()))
        delta = avg - bl_avg
        per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(results.items()))
        print(f"  ew={wp:.2f} {name}: AVG={avg:.2f} (Δ={delta:+.2f}) | {per_round}")
