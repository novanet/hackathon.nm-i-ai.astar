"""Deep analysis of R5 performance and all-round model assessment."""
import json, numpy as np
from pathlib import Path

DATA_DIR = Path("data")
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}

TERRAIN_TO_CLASS = {0: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0}
NUM_CLASSES = 6

def score_prediction(prediction, ground_truth):
    eps = 1e-10
    p = np.clip(ground_truth, eps, 1.0)
    q = np.clip(prediction, eps, 1.0)
    kl = np.sum(p * np.log(p / q), axis=-1)
    entropy = -np.sum(p * np.log(p), axis=-1)
    total_entropy = np.sum(entropy)
    if total_entropy < eps:
        return 100.0
    weighted_kl = np.sum(entropy * kl) / total_entropy
    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))

def load_round(rid):
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts

# HISTORICAL_TRANSITIONS from model.py
HISTORICAL_TRANSITIONS = np.array([
    [0.8674, 0.0902, 0.0074, 0.0083, 0.0267, 0.0000],
    [0.4794, 0.2700, 0.0042, 0.0227, 0.2237, 0.0000],
    [0.4911, 0.0757, 0.1821, 0.0214, 0.2296, 0.0000],
    [0.5000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
    [0.0694, 0.1116, 0.0084, 0.0102, 0.8005, 0.0000],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])

def apply_floor(pred, floor=0.001):
    floored = np.maximum(pred, floor)
    return floored / floored.sum(axis=-1, keepdims=True)

# ===== Score accumulation analysis =====
# The competition scores = sum of round_weight * seed_score across ALL rounds
print("=" * 70)
print("SCORING CONTEXT: How total score is calculated")
print("=" * 70)
print()
print("Total = sum(round_weight * avg_seed_score) across all rounds")
print("Each round weight increases: 1.05, 1.1025, 1.1576, 1.2155, 1.2763, ...")
print()

# Check what we scored vs what's possible
round_weights = {}
for rnum in range(1, 20):
    round_weights[rnum] = 1.05 ** rnum

print(f"{'Round':>6} {'Weight':>8} {'Our Score':>10} {'Weighted':>10} {'Best(113.9)':>12}")
print("-" * 50)
our_scores = {1: 0, 2: 3.02, 3: 0, 4: 0, 5: 75.38}
total_us = 0
total_best_theoretical = 0
for rnum in range(1, 6):
    w = round_weights[rnum]
    our = our_scores.get(rnum, 0)
    total_us += w * our
    # Best team cumulative: 113.9
    print(f"  R{rnum:>3}  {w:>8.4f}  {our:>10.2f}  {w*our:>10.2f}")

print(f"\nOur total: {total_us:.2f}")
print(f"Best team: 113.9 (we don't know per-round breakdown)")
print()

# ===== Per-round model evaluation (oracle - what's theoretically possible) =====
print("=" * 70)
print("PER-ROUND ORACLE ANALYSIS: What's achievable per round?")
print("=" * 70)
print()

for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        print(f"R{rnum}: No ground truth available")
        continue
    
    n_seeds = len(gts)
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    
    # Baseline: historical transitions
    scores_hist = []
    for s in range(n_seeds):
        init_grid = detail["initial_states"][s]["grid"]
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = HISTORICAL_TRANSITIONS[init_cls]
        pred = apply_floor(pred)
        scores_hist.append(score_prediction(pred, gts[s]))
    
    # Oracle: per-round transition matrix
    transition_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s in range(n_seeds):
        init_grid = detail["initial_states"][s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                transition_counts[init_cls] += gts[s][y, x]
    row_sums = np.maximum(transition_counts.sum(axis=1, keepdims=True), 1.0)
    oracle_trans = transition_counts / row_sums
    
    scores_oracle = []
    for s in range(n_seeds):
        init_grid = detail["initial_states"][s]["grid"]
        pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                pred[y, x] = oracle_trans[init_cls]
        pred = apply_floor(pred)
        scores_oracle.append(score_prediction(pred, gts[s]))
    
    # Ground truth entropy analysis
    all_ent = []
    for s in range(n_seeds):
        entropy = -np.sum(gts[s] * np.log(np.clip(gts[s], 1e-10, 1.0)), axis=-1)
        all_ent.append(entropy)
    mean_ent = np.mean([e.mean() for e in all_ent])
    max_ent = np.mean([e.max() for e in all_ent])
    pct_dynamic = np.mean([(e > 0.01).mean() for e in all_ent])
    
    print(f"R{rnum}: Historical={np.mean(scores_hist):.1f}, Oracle_trans={np.mean(scores_oracle):.1f}, "
          f"Mean_entropy={mean_ent:.3f}, %Dynamic={pct_dynamic:.1%}")
    print(f"   Oracle transition matrix (key rows):")
    for cls_idx in [0, 1, 4]:  # Empty, Settlement, Forest
        row = oracle_trans[cls_idx]
        names = ['Emp', 'Set', 'Prt', 'Rui', 'For', 'Mtn']
        row_str = ", ".join([f"{names[i]}:{v:.3f}" for i, v in enumerate(row) if v > 0.005])
        print(f"     {names[cls_idx]:>3} -> {row_str}")
    print()

# ===== R5 specifically: why did we score 75.38 and where do we lose? =====
print("=" * 70)
print("R5 DEEP DIVE: Where are we losing points?")
print("=" * 70)
print()

detail, gts = load_round(ROUND_IDS[5])
n_seeds = len(gts)
map_w = detail.get("map_width", 40)
map_h = detail.get("map_height", 40)

for s in range(n_seeds):
    init_grid = detail["initial_states"][s]["grid"]
    gt = gts[s]
    
    # Our model's prediction (historical transition as proxy for zero-query)
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            pred[y, x] = HISTORICAL_TRANSITIONS[init_cls]
    pred = apply_floor(pred)
    
    eps = 1e-10
    p = np.clip(gt, eps, 1.0)
    q = np.clip(pred, eps, 1.0)
    kl = np.sum(p * np.log(p / q), axis=-1)
    entropy = -np.sum(p * np.log(p), axis=-1)
    
    # Score loss by initial class
    initial_classes = np.zeros((map_h, map_w), dtype=int)
    for y in range(map_h):
        for x in range(map_w):
            initial_classes[y, x] = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
    
    total_entropy = entropy.sum()
    names = ['Empty', 'Settlement', 'Port', 'Ruin', 'Forest', 'Mountain']
    
    if s == 0:
        print(f"Seed {s} score loss breakdown by initial class:")
        for cls in range(NUM_CLASSES):
            mask = initial_classes == cls
            if mask.any():
                cls_kl = (entropy[mask] * kl[mask]).sum()
                pct = cls_kl / total_entropy * 100 if total_entropy > 0 else 0
                mean_kl = kl[mask].mean()
                print(f"  {names[cls]:>10}: {mask.sum():>4} cells, {pct:>5.1f}% of loss, mean_kl={mean_kl:.4f}")
        print()

# ===== What would a perfectly calibrated round-specific transition matrix score? =====
print("=" * 70)
print("LEAVE-ONE-SEED-OUT WITHIN R5: Same-round transition quality")
print("=" * 70)
print()

detail, gts = load_round(ROUND_IDS[5])
n_seeds = len(gts)
map_w, map_h = 40, 40

for target_s in range(n_seeds):
    # Learn transitions from other 4 seeds
    tc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s in range(n_seeds):
        if s == target_s:
            continue
        init_grid = detail["initial_states"][s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                tc[init_cls] += gts[s][y, x]
    rs = np.maximum(tc.sum(axis=1, keepdims=True), 1.0)
    trans = tc / rs
    
    pred = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    init_grid = detail["initial_states"][target_s]["grid"]
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            pred[y, x] = trans[init_cls]
    pred = apply_floor(pred)
    sc = score_prediction(pred, gts[target_s])
    print(f"  Seed {target_s}: LOSO transitions = {sc:.1f}")

print()

# ===== The BIG question: what's the ceiling for different model types? =====
print("=" * 70)
print("MODEL CEILING ANALYSIS: Perfect knowledge of different types")
print("=" * 70)
print()

# For R5, what if we knew transitions perfectly?
detail5, gts5 = load_round(ROUND_IDS[5])

# Oracle full-round transitions
tc5 = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
for s in range(len(gts5)):
    for y in range(40):
        for x in range(40):
            init_cls = TERRAIN_TO_CLASS.get(detail5["initial_states"][s]["grid"][y][x], 0)
            tc5[init_cls] += gts5[s][y, x]
rs = np.maximum(tc5.sum(axis=1, keepdims=True), 1.0)
oracle5 = tc5 / rs

# Oracle score
scores5 = []
for s in range(len(gts5)):
    pred = np.zeros((40, 40, NUM_CLASSES))
    for y in range(40):
        for x in range(40):
            init_cls = TERRAIN_TO_CLASS.get(detail5["initial_states"][s]["grid"][y][x], 0)
            pred[y, x] = oracle5[init_cls]
    pred = apply_floor(pred)
    scores5.append(score_prediction(pred, gts5[s]))

print(f"R5 Oracle transitions: {np.mean(scores5):.1f} (seeds: {[f'{s:.1f}' for s in scores5]})")

# Perfect per-cell prediction would score 100 -- what kills us is spatial variation
# Let's see: per-POSITION (pooled across seeds) oracle
scores_pos = []
for target_s in range(len(gts5)):
    # Average GT across all 5 seeds as prediction for each cell
    avg_gt = np.mean(gts5, axis=0)
    avg_gt = apply_floor(avg_gt)
    scores_pos.append(score_prediction(avg_gt, gts5[target_s]))
print(f"R5 Cross-seed avg GT as pred: {np.mean(scores_pos):.1f} (seeds: {[f'{s:.1f}' for s in scores_pos]})")

# Perfect per-cell per-seed = 100, but what does "per-initial-class-per-position" look like?
# i.e., if we could learn position-specific biases
print()

# Across ALL rounds: what's the sum of (weight * oracle_score)?
print("=" * 70)
print("THEORETICAL CEILING: Perfect oracle transitions per round")
print("=" * 70)
total_oracle = 0
total_hist = 0
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round(rid)
    if not gts:
        continue
    w = 1.05 ** rnum
    n_seeds = len(gts)
    
    # Oracle trans
    tc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s in range(n_seeds):
        for y in range(40):
            for x in range(40):
                init_cls = TERRAIN_TO_CLASS.get(detail["initial_states"][s]["grid"][y][x], 0)
                tc[init_cls] += gts[s][y, x]
    rs = np.maximum(tc.sum(axis=1, keepdims=True), 1.0)
    ot = tc / rs
    
    oracle_scores = []
    hist_scores = []
    for s in range(n_seeds):
        pred_o = np.zeros((40, 40, NUM_CLASSES))
        pred_h = np.zeros((40, 40, NUM_CLASSES))
        for y in range(40):
            for x in range(40):
                init_cls = TERRAIN_TO_CLASS.get(detail["initial_states"][s]["grid"][y][x], 0)
                pred_o[y, x] = ot[init_cls]
                pred_h[y, x] = HISTORICAL_TRANSITIONS[init_cls]
        oracle_scores.append(score_prediction(apply_floor(pred_o), gts[s]))
        hist_scores.append(score_prediction(apply_floor(pred_h), gts[s]))
    
    o_avg = np.mean(oracle_scores)
    h_avg = np.mean(hist_scores)
    total_oracle += w * o_avg
    total_hist += w * h_avg
    print(f"  R{rnum} (w={w:.4f}): Oracle={o_avg:.1f} (weighted={w*o_avg:.1f}), Hist={h_avg:.1f} (weighted={w*h_avg:.1f})")

print(f"\n  Total with oracle transitions: {total_oracle:.1f}")
print(f"  Total with historical transitions: {total_hist:.1f}")
print(f"  Best team score: 113.9")
print()
print("Gap analysis:")
print(f"  Oracle - Best team: {total_oracle - 113.9:.1f}")
print(f"  Our score: 96.2")
print(f"  Points lost from missed rounds R1/R3/R4: estimated from oracle baselines")
