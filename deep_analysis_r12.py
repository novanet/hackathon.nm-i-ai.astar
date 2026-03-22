"""Deep error analysis on R12 — where is the 20pt gap vs leaders?"""
import json, numpy as np
from pathlib import Path
from astar.model import build_prediction, HISTORICAL_TRANSITIONS, RECENT_TRANSITIONS
from astar.submit import score_prediction

DATA_DIR = Path('data')
R12_ID = '795bfb1f-54bd-4f39-a526-9868b36f7ebd'
rdir = DATA_DIR / f'round_{R12_ID}'

detail_files = sorted(rdir.glob('round_detail_*.json'))
detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))

# Analyze seed 0 in detail
seed = 0
gt_data = json.loads((rdir / f'ground_truth_s{seed}.json').read_text(encoding='utf-8'))
gt = np.array(gt_data['ground_truth'], dtype=np.float64)
pred = build_prediction(R12_ID, detail, seed)

H, W, C = gt.shape
print(f"Map: {H}x{W}x{C}")
print(f"Score: {score_prediction(pred, gt):.2f}")
print()

# Initial state
init_data = detail['initial_states'][seed]
init_grid = np.array(init_data.get('terrain') or init_data.get('grid'), dtype=int)
CLASS_NAMES = ['Empty', 'Settlement', 'Port', 'Ruin', 'Forest', 'Mountain']

# GT argmax distribution
gt_argmax = gt.argmax(axis=2)
print("=== GT argmax distribution ===")
for c in range(C):
    cnt = (gt_argmax == c).sum()
    pct = cnt / (H * W) * 100
    print(f"  {CLASS_NAMES[c]:12s}: {cnt:4d} ({pct:5.1f}%)")
print()

# GT mean probability distribution
gt_mean = gt.mean(axis=(0, 1))
print("=== GT mean probability ===")
for c in range(C):
    print(f"  {CLASS_NAMES[c]:12s}: {gt_mean[c]:.4f}")
print()

# Pred mean probability distribution
pred_mean = pred.mean(axis=(0, 1))
print("=== Pred mean probability ===")
for c in range(C):
    print(f"  {CLASS_NAMES[c]:12s}: {pred_mean[c]:.4f} (GT: {gt_mean[c]:.4f}, ratio: {pred_mean[c]/max(gt_mean[c],1e-10):.3f})")
print()

# Per-cell entropy and KL
gt_safe = np.clip(gt, 1e-10, 1)
pred_safe = np.clip(pred, 1e-10, 1)
entropy = -np.sum(gt_safe * np.log(gt_safe), axis=2)
kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=2)

# Weight by entropy
total_entropy = entropy.sum()
weighted_kl = np.sum(entropy * kl) / total_entropy
print(f"=== Scoring breakdown ===")
print(f"  Mean entropy: {entropy.mean():.4f}")
print(f"  Total entropy: {total_entropy:.1f}")
print(f"  Mean KL: {kl.mean():.6f}")
print(f"  Weighted KL: {weighted_kl:.6f}")
print(f"  Score: {100 * np.exp(-3 * weighted_kl):.2f}")
print()

# Per initial-class breakdown
print("=== Loss by initial class ===")
for c in range(C):
    mask = init_grid == c
    if mask.sum() == 0:
        continue
    e = entropy[mask]
    k = kl[mask]
    wk = (e * k).sum() / max(total_entropy, 1e-10) * 100  # % of total weighted KL
    print(f"  {CLASS_NAMES[c]:12s}: n={mask.sum():4d}, mean_ent={e.mean():.3f}, mean_kl={k.mean():.4f}, "
          f"weighted_kl_pct={wk:.1f}%, max_kl={k.max():.4f}")
print()

# Per GT-argmax class breakdown (where does loss accumulate by final class?)
print("=== Loss by GT argmax class ===")
for c in range(C):
    mask = gt_argmax == c
    if mask.sum() == 0:
        continue
    e = entropy[mask]
    k = kl[mask]
    wk = (e * k).sum() / max(total_entropy, 1e-10) * 100
    # Also show our pred avg for this GT class
    pred_c = pred[mask][:, c].mean()
    gt_c = gt[mask][:, c].mean()
    print(f"  {CLASS_NAMES[c]:12s}: n={mask.sum():4d}, mean_ent={e.mean():.3f}, mean_kl={k.mean():.4f}, "
          f"weighted_kl_pct={wk:.1f}%, pred_class_avg={pred_c:.3f}, gt_class_avg={gt_c:.3f}")
print()

# Top 20 worst cells
flat_wk = entropy * kl
worst_idx = np.argsort(flat_wk.flatten())[::-1][:20]
print("=== Top 20 worst cells (highest weighted KL) ===")
for idx in worst_idx:
    y, x = divmod(idx, W)
    init_c = int(init_grid[y, x])
    gt_c = int(gt_argmax[y, x])
    ent = entropy[y, x]
    k = kl[y, x]
    gt_probs = gt[y, x]
    pred_probs = pred[y, x]
    gt_str = " ".join(f"{p:.2f}" for p in gt_probs)
    pred_str = " ".join(f"{p:.2f}" for p in pred_probs)
    init_name = CLASS_NAMES[init_c] if init_c < len(CLASS_NAMES) else f"cls{init_c}"
    gt_name = CLASS_NAMES[gt_c] if gt_c < len(CLASS_NAMES) else f"cls{gt_c}"
    print(f"  ({y:2d},{x:2d}) init={init_name:8s} → gt={gt_name:8s} "
          f"ent={ent:.3f} kl={k:.4f} gt=[{gt_str}] pred=[{pred_str}]")

print()

# Transition matrix from GT
print("=== GT transition matrix (R12 S0) ===")
for ic in range(C):
    mask = init_grid == ic
    if mask.sum() == 0:
        continue
    row = gt[mask].mean(axis=0)
    probs = " ".join(f"{p:.3f}" for p in row)
    print(f"  {CLASS_NAMES[ic]:12s} → [{probs}]")

print()

# What are our observation-calibrated transitions?
from astar.model import observation_calibrated_transitions, load_simulations, SHRINKAGE_MATRIX
sims = load_simulations(R12_ID)
seed_sims = [s for s in sims if s['seed'] == seed]
print(f"Loaded {len(seed_sims)} simulations for seed {seed}")

# Get calibrated transitions
cal_trans = observation_calibrated_transitions(R12_ID, detail, seed, smoothing=5.0)
print("=== Our calibrated transitions (S0) ===")
for ic in range(C):
    row = cal_trans[ic]
    probs = " ".join(f"{p:.3f}" for p in row)
    print(f"  {CLASS_NAMES[ic]:12s} → [{probs}]")
print()

# Compare debiased obs transitions (raw obs × shrinkage)
print("=== Debiased transitions (obs × shrinkage) ===")
from astar.model import _count_transitions
obs_counts = _count_transitions(R12_ID, detail, seed)
obs_row_sums = np.maximum(obs_counts.sum(axis=1, keepdims=True), 1)
obs_trans = obs_counts / obs_row_sums
debiased = obs_trans * SHRINKAGE_MATRIX
debiased_row_sums = np.maximum(debiased.sum(axis=1, keepdims=True), 1e-10)
debiased_norm = debiased / debiased_row_sums
for ic in range(C):
    if obs_counts[ic].sum() == 0:
        continue
    row = debiased_norm[ic]
    probs = " ".join(f"{p:.3f}" for p in row)
    print(f"  {CLASS_NAMES[ic]:12s} → [{probs}]  (n={int(obs_counts[ic].sum())})")

print()
print("=== Historical vs Recent transitions (Settlement row) ===")
print(f"  Historical S→S: {HISTORICAL_TRANSITIONS[1,1]:.3f}")
print(f"  Recent     S→S: {RECENT_TRANSITIONS[1,1]:.3f}")
print(f"  GT R12     S→S: {gt[init_grid==1].mean(axis=0)[1]:.3f}")
print(f"  Calibrated S→S: {cal_trans[1,1]:.3f}")
