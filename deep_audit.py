"""
Deep analysis: Where does LORO lose points? What can we still improve?

Checks:
1. Per-round, per-class KL contribution (which cells hurt most)
2. Entropy-weighted training potential
3. LORO with realistic inference conditions (debiased obs + temps)
4. Per-seed features vs per-round features
5. Calibration analysis (predicted vs GT class probabilities)
"""
import json, numpy as np, warnings, pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Reset spatial model to force fresh load
import astar.model as _m
_m._spatial_model = None

from astar.model import (
    compute_cell_features, apply_floor, HISTORICAL_TRANSITIONS,
    NUM_CLASSES, per_class_temperature_scale, PER_CLASS_TEMPS,
    observation_calibrated_transitions, debias_transitions,
    compute_round_features, _extract_settlement_stats, spatial_prior,
    _apply_transition_matrix, build_prediction
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS

DATA_DIR = Path('data')

ROUND_IDS = {
    1: '71451d74-be9f-471f-aacd-a41f3b68a9cd',
    2: '76909e29-f664-4b2f-b16b-61b7507277e9',
    3: 'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
    4: '8e839974-b13b-407b-a5e7-fc749d877195',
    5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    6: 'ae78003a-4efe-425a-881a-d16a39bca0ad',
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

def load_round(rid):
    rdir = DATA_DIR / f'round_{rid}'
    detail = json.loads(sorted(rdir.glob('round_detail_*.json'))[-1].read_text(encoding='utf-8'))
    gts = []
    for s in range(len(detail.get('initial_states', []))):
        gtp = rdir / f'ground_truth_s{s}.json'
        if gtp.exists():
            gts.append(np.array(json.loads(gtp.read_text(encoding='utf-8'))['ground_truth'], dtype=np.float64))
    return detail, gts

def compute_gt_round_features(detail, gts):
    """Same as in train_spatial.py"""
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    states = detail.get("initial_states", [])
    
    trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = 0
    total = 0
    for s_idx, gt in enumerate(gts):
        init_grid = states[s_idx]["grid"]
        gt_argmax = gt.argmax(axis=-1)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                for c in range(NUM_CLASSES):
                    trans_counts[init_cls, c] += gt[y, x, c]
                if init_cls == 1:
                    n_sett += 1
                total += 1
    
    row_sums = np.maximum(trans_counts.sum(axis=1, keepdims=True), 1e-10)
    trans = trans_counts / row_sums
    sett_density = n_sett / max(total, 1)
    
    ss_rate = trans[1, 1]
    es_rate = trans[0, 1]
    se_rate = trans[1, 0]
    mean_food = 0.3 + 0.7 * ss_rate
    mean_wealth = es_rate * 0.3
    mean_defense = 1.0 - se_rate
    
    return np.array([trans[0,0], trans[1,1], trans[4,4], trans[0,1],
                     sett_density, mean_food, mean_wealth, mean_defense], dtype=np.float64)

# ============================================================
# 1. Per-round, per-class KL contribution analysis
# ============================================================
print("=" * 70)
print("1. PER-ROUND, PER-CLASS KL CONTRIBUTION (using full pipeline)")
print("=" * 70)
eps = 1e-10

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        continue
    map_w, map_h = detail.get('map_width', 40), detail.get('map_height', 40)
    round_feats = compute_gt_round_features(detail, gts)
    
    total_kl_by_class = np.zeros(NUM_CLASSES)
    total_entropy_by_class = np.zeros(NUM_CLASSES)
    total_cells_by_class = np.zeros(NUM_CLASSES)
    all_scores = []
    
    for s, gt in enumerate(gts):
        init_grid = detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
        model = _m.load_spatial_model()
        flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        
        # Apply temps
        ss_rate = round_feats[1]
        if ss_rate < 0.15:
            adaptive_temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
        else:
            adaptive_temps = PER_CLASS_TEMPS
        pred = per_class_temperature_scale(pred, detail, s, temps=adaptive_temps,
                                           map_w=map_w, map_h=map_h)
        pred = apply_floor(pred)
        
        p = np.clip(gt, eps, 1.0)
        q = np.clip(pred, eps, 1.0)
        kl = np.sum(p * np.log(p / q), axis=-1)
        entropy = -np.sum(p * np.log(p), axis=-1)
        
        # Break down by initial class
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                total_kl_by_class[init_cls] += entropy[y, x] * kl[y, x]
                total_entropy_by_class[init_cls] += entropy[y, x]
                total_cells_by_class[init_cls] += 1
        
        all_scores.append(score_prediction(pred, gt))
    
    total_weighted = total_entropy_by_class.sum()
    print(f"\nR{rnum}: avg_score={np.mean(all_scores):.2f}")
    for c in range(NUM_CLASSES):
        if total_cells_by_class[c] == 0:
            continue
        frac = total_kl_by_class[c] / max(total_weighted, eps) * 100
        mean_kl = total_kl_by_class[c] / max(total_entropy_by_class[c], eps)
        print(f"  {CLASS_NAMES[c]:>10}: {frac:5.1f}% of loss | "
              f"weighted_kl={mean_kl:.4f} | "
              f"n_cells={int(total_cells_by_class[c])}")

# ============================================================
# 2. Entropy distribution analysis
# ============================================================
print("\n" + "=" * 70)
print("2. ENTROPY DISTRIBUTION: How many cells matter?")
print("=" * 70)

all_entropies = []
all_kls = []
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    for gt in gts:
        p = np.clip(gt, eps, 1.0)
        entropy = -np.sum(p * np.log(p), axis=-1)
        all_entropies.append(entropy.flatten())

all_e = np.concatenate(all_entropies)
print(f"Total cells across all rounds: {len(all_e)}")
for threshold in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    pct = (all_e > threshold).mean() * 100
    mean_e = all_e[all_e > threshold].mean() if (all_e > threshold).any() else 0
    print(f"  entropy > {threshold:.2f}: {pct:.1f}% of cells, mean entropy={mean_e:.3f}")

# ============================================================
# 3. Calibration analysis: predicted vs GT mean probabilities
# ============================================================
print("\n" + "=" * 70)
print("3. CALIBRATION: Predicted vs GT mean class probabilities (in-sample)")
print("=" * 70)

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        continue
    map_w, map_h = detail.get('map_width', 40), detail.get('map_height', 40)
    round_feats = compute_gt_round_features(detail, gts)
    
    all_pred = []
    all_gt = []
    for s, gt in enumerate(gts):
        init_grid = detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
        model = _m.load_spatial_model()
        flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        ss_rate = round_feats[1]
        if ss_rate < 0.15:
            adaptive_temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
        else:
            adaptive_temps = PER_CLASS_TEMPS
        pred = per_class_temperature_scale(pred, detail, s, temps=adaptive_temps,
                                           map_w=map_w, map_h=map_h)
        pred = apply_floor(pred)
        
        all_pred.append(pred.reshape(-1, NUM_CLASSES))
        all_gt.append(gt.reshape(-1, NUM_CLASSES))
    
    pred_cat = np.vstack(all_pred)
    gt_cat = np.vstack(all_gt)
    pred_mean = pred_cat.mean(axis=0)
    gt_mean = gt_cat.mean(axis=0)
    
    print(f"\nR{rnum}:")
    print(f"  {'Class':>10} | {'GT mean':>8} | {'Pred mean':>9} | {'Ratio':>6} | {'Bias':>7}")
    for c in range(NUM_CLASSES):
        ratio = pred_mean[c] / max(gt_mean[c], eps)
        bias = pred_mean[c] - gt_mean[c]
        print(f"  {CLASS_NAMES[c]:>10} | {gt_mean[c]:8.4f} | {pred_mean[c]:9.4f} | {ratio:6.3f} | {bias:+7.4f}")

# ============================================================
# 4. LORO with realistic pipeline (debiased obs + temps) 
#    vs current LORO (GT features, no temps)
# ============================================================
print("\n" + "=" * 70)
print("4. LORO COMPARISON: GT features vs debiased obs features")
print("=" * 70)
print("(Simulating what happens at inference time)")

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# We can approximate this: for each test round, use build_prediction()
# which uses debiased obs features. Compare to GT-feature LORO.
# Note: rounds without observations (R1, R3, R4) will use HISTORICAL fallback.

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        continue
    map_w, map_h = detail.get('map_width', 40), detail.get('map_height', 40)
    
    # GT feature scores (current LORO approach)
    round_feats_gt = compute_gt_round_features(detail, gts)
    model = _m.load_spatial_model()
    
    gt_scores = []
    pipeline_scores = []
    for s, gt in enumerate(gts):
        # GT features path
        init_grid = detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats_gt)
        flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred_gt = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred_gt = np.maximum(pred_gt, 1e-10)
        pred_gt = pred_gt / pred_gt.sum(axis=-1, keepdims=True)
        ss_rate = round_feats_gt[1]
        if ss_rate < 0.15:
            adaptive_temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
        else:
            adaptive_temps = PER_CLASS_TEMPS
        pred_gt = per_class_temperature_scale(pred_gt, detail, s, temps=adaptive_temps,
                                              map_w=map_w, map_h=map_h)
        pred_gt = apply_floor(pred_gt)
        gt_scores.append(score_prediction(pred_gt, gt))
        
        # Full pipeline path (uses debiased obs features)
        pred_pipe = build_prediction(rid, detail, s, map_w, map_h)
        pipeline_scores.append(score_prediction(pred_pipe, gt))
    
    gt_avg = np.mean(gt_scores)
    pipe_avg = np.mean(pipeline_scores)
    delta = pipe_avg - gt_avg
    print(f"  R{rnum}: GT_feats={gt_avg:.2f}  pipeline={pipe_avg:.2f}  Δ={delta:+.2f}")

# ============================================================
# 5. Check: Would sample_weight by entropy help?
# ============================================================
print("\n" + "=" * 70)
print("5. ENTROPY-WEIGHTED TRAINING POTENTIAL")
print("=" * 70)
print("Checking how much score is driven by high-entropy cells...")

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        continue
    
    high_entropy_kl = 0.0
    high_entropy_weight = 0.0
    low_entropy_kl = 0.0
    low_entropy_weight = 0.0
    total_weighted_kl = 0.0
    total_weight = 0.0
    
    round_feats = compute_gt_round_features(detail, gts)
    map_w, map_h = detail.get('map_width', 40), detail.get('map_height', 40)
    model = _m.load_spatial_model()
    
    for s, gt in enumerate(gts):
        init_grid = detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
        flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
        pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
        pred = np.maximum(pred, 1e-10)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        pred = apply_floor(pred)
        
        p = np.clip(gt, eps, 1.0)
        q = np.clip(pred, eps, 1.0)
        kl = np.sum(p * np.log(p / q), axis=-1)
        entropy = -np.sum(p * np.log(p), axis=-1)
        
        high_mask = entropy > 0.5
        low_mask = (entropy > 0.01) & (entropy <= 0.5)
        
        high_entropy_kl += np.sum(entropy[high_mask] * kl[high_mask])
        high_entropy_weight += np.sum(entropy[high_mask])
        low_entropy_kl += np.sum(entropy[low_mask] * kl[low_mask])
        low_entropy_weight += np.sum(entropy[low_mask])
        total_weighted_kl += np.sum(entropy * kl)
        total_weight += np.sum(entropy)
    
    if total_weight > 0:
        high_frac = high_entropy_kl / total_weighted_kl * 100
        low_frac = low_entropy_kl / total_weighted_kl * 100
        high_mean = high_entropy_kl / max(high_entropy_weight, eps)
        low_mean = low_entropy_kl / max(low_entropy_weight, eps)
        print(f"  R{rnum}: high_entropy(>0.5) = {high_frac:.1f}% of loss (mean_kl={high_mean:.4f}), "
              f"low_entropy(0.01-0.5) = {low_frac:.1f}% (mean_kl={low_mean:.4f})")
