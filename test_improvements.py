"""
Test two improvement hypotheses:
1. Entropy-weighted training (focus model on high-entropy cells)
2. Post-model calibration (correct systematic class bias)

Both tested via LORO cross-validation.
"""
import json, pickle, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from astar.model import (
    compute_cell_features, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, NUM_CLASSES, per_class_temperature_scale, PER_CLASS_TEMPS
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
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts

def compute_gt_round_features(detail, gts):
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = 0; total = 0
    for s in range(len(gts)):
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                gt_counts[init_cls] += gts[s][y, x]
                if init_cls == 1: n_sett += 1
                total += 1
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / row_sums
    sett_density = n_sett / max(total, 1)
    ss_rate = gt_trans[1, 1]
    return np.array([gt_trans[0,0], gt_trans[1,1], gt_trans[4,4], gt_trans[0,1],
                     sett_density,
                     0.3 + 0.7 * ss_rate,
                     gt_trans[0,1] * 0.3,
                     1.0 - gt_trans[1,0]], dtype=np.float64)

def build_data(round_ids, return_weights=False):
    X_parts, Y_parts, W_parts = [], [], []
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts: continue
        mw, mh = detail.get("map_width", 40), detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, mw, mh, round_features=round_feats)
            X_parts.append(feat.reshape(-1, feat.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            if return_weights:
                eps = 1e-10
                p = np.clip(gt, eps, 1.0)
                entropy = -np.sum(p * np.log(p), axis=-1).flatten()
                W_parts.append(entropy)
    X = np.vstack(X_parts)
    Y = np.vstack(Y_parts)
    W = np.concatenate(W_parts) if return_weights else None
    return X, Y, W

def score_loro(model_fn, use_temps=True):
    """Run LORO with a model factory function. Returns per-round scores."""
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts: all_data[rnum] = (rid, detail, gts)

    results = {}
    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        model = model_fn(train_ids)

        test_rid, test_detail, test_gts = all_data[test_rnum]
        mw, mh = test_detail.get("map_width", 40), test_detail.get("map_height", 40)
        test_feats = compute_gt_round_features(test_detail, test_gts)

        scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, mw, mh, round_features=test_feats)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(mh, mw, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            if use_temps:
                ss_rate = test_feats[1]
                if ss_rate < 0.15:
                    temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
                else:
                    temps = PER_CLASS_TEMPS
                pred = per_class_temperature_scale(pred, test_detail, s, temps=temps, map_w=mw, map_h=mh)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))
        results[test_rnum] = np.mean(scores)
    return results

# ============================================================
# Baseline: current model (no entropy weighting, with temps)
# ============================================================
print("=" * 60)
print("BASELINE: Current model with temps")
print("=" * 60)

def make_baseline(train_ids):
    X, Y, _ = build_data(train_ids)
    m = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0, verbosity=-1),
        n_jobs=1)
    m.fit(X, Y)
    return m

baseline = score_loro(make_baseline, use_temps=True)
for rnum, sc in sorted(baseline.items()):
    print(f"  R{rnum}: {sc:.2f}")
bl_avg = np.mean(list(baseline.values()))
print(f"  AVG: {bl_avg:.2f}")

# ============================================================
# Test 1: Entropy-weighted training
# ============================================================
print("\n" + "=" * 60)
print("TEST 1: Entropy-weighted training")
print("=" * 60)

for weight_power in [0.5, 1.0, 2.0]:
    def make_entropy_weighted(train_ids, wp=weight_power):
        X, Y, W = build_data(train_ids, return_weights=True)
        # Transform weights: w = entropy^power + small baseline
        W_final = np.power(W + 0.01, wp)  # avoid log(0) issues
        # LightGBM doesn't directly support sample_weight in MultiOutputRegressor,
        # so we need to use fit_params
        m = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0, verbosity=-1),
            n_jobs=1)
        # MultiOutputRegressor passes sample_weight to each estimator
        m.fit(X, Y, sample_weight=W_final)
        return m

    results = score_loro(make_entropy_weighted, use_temps=True)
    avg = np.mean(list(results.values()))
    delta = avg - bl_avg
    per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(results.items()))
    print(f"  power={weight_power:.1f}: AVG={avg:.2f} (Δ={delta:+.2f}) | {per_round}")

# ============================================================
# Test 2: Post-model calibration (class scaling)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Post-model calibration factors")
print("=" * 60)
print("(Settlement is over-predicted, Empty under-predicted)")

# Test different calibration strategies
import astar.model as _m
_m._spatial_model = None  # force reload
from astar.model import load_spatial_model

model = load_spatial_model()
all_data = {}
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if gts: all_data[rnum] = (rid, detail, gts)

def test_calibration(calib_factors):
    """Apply per-class scaling after model prediction."""
    cf = np.array(calib_factors)
    scores_by_round = {}
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        mw, mh = detail.get("map_width", 40), detail.get("map_height", 40)
        rf = compute_gt_round_features(detail, gts)
        scores = []
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, mw, mh, round_features=rf)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(mh, mw, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            # Apply calibration
            pred = pred * cf[np.newaxis, np.newaxis, :]
            pred = pred / pred.sum(axis=-1, keepdims=True)
            # Apply temps
            ss_rate = rf[1]
            if ss_rate < 0.15:
                temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
            else:
                temps = PER_CLASS_TEMPS
            pred = per_class_temperature_scale(pred, detail, s, temps=temps, map_w=mw, map_h=mh)
            pred = apply_floor(pred)
            scores.append(score_prediction(pred, gt))
        scores_by_round[rnum] = np.mean(scores)
    return scores_by_round

# Note: this uses in-sample (all data), not LORO — just to check direction
base_insample = test_calibration([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
base_avg_is = np.mean(list(base_insample.values()))
print(f"  Baseline (in-sample): {base_avg_is:.2f}")

# Shrink settlement/port/ruin toward empty
configs = [
    ("Shrink S/P/R 5%", [1.02, 0.95, 0.95, 0.95, 1.0, 1.0]),
    ("Shrink S/P/R 10%", [1.04, 0.90, 0.90, 0.90, 1.0, 1.0]),
    ("Shrink S/P/R 15%", [1.06, 0.85, 0.85, 0.85, 1.0, 1.0]),
    ("Shrink S/P/R 20%", [1.08, 0.80, 0.80, 0.80, 1.0, 1.0]),
    ("Boost E 5%, shrink S 10%", [1.05, 0.90, 0.90, 0.90, 1.0, 1.0]),
    ("Boost E 3%, shrink SPR 5%", [1.03, 0.95, 0.95, 0.95, 1.0, 1.0]),
]
for name, cf in configs:
    results = test_calibration(cf)
    avg = np.mean(list(results.values()))
    delta = avg - base_avg_is
    per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(results.items()))
    print(f"  {name}: {avg:.2f} (Δ={delta:+.2f}) | {per_round}")

# ============================================================
# Test 3: KL-divergence-based loss (approximate with weighted MSE)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Target transformation — log-odds")
print("=" * 60)
print("If model predicts log(p) instead of p, errors in small probs matter more")

def make_log_target(train_ids):
    """Train on log-transformed targets, predict, then exp-transform back."""
    X, Y, _ = build_data(train_ids)
    # Log transform targets (with floor)
    Y_log = np.log(np.maximum(Y, 1e-6))
    m = MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0, verbosity=-1),
        n_jobs=1)
    m.fit(X, Y_log)
    
    class LogWrapper:
        def __init__(self, inner):
            self.inner = inner
        def predict(self, X):
            log_pred = self.inner.predict(X)
            return np.exp(log_pred)
    
    return LogWrapper(m)

log_results = score_loro(make_log_target, use_temps=True)
log_avg = np.mean(list(log_results.values()))
delta = log_avg - bl_avg
per_round = " ".join(f"R{r}:{s:.1f}" for r, s in sorted(log_results.items()))
print(f"  Log-target: AVG={log_avg:.2f} (Δ={delta:+.2f}) | {per_round}")
