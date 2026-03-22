"""Sweep Bayesian overlay params (MIN_PS, MAX_PS) on R13-R16 with 80% U-Net blend.
Current: MIN_PS=5, MAX_PS=100.
"""
import json, numpy as np
from pathlib import Path
from astar.submit import score_prediction
import astar.model as model

ROUND_IDS = {
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}

# Grid: (min_ps, max_ps) combinations
PARAMS = [
    (1, 50),   (1, 100),  (1, 200),
    (3, 50),   (3, 100),  (3, 200),
    (5, 50),   (5, 100),  (5, 200),  (5, 500),
    (10, 50),  (10, 100), (10, 200), (10, 500),
    (20, 100), (20, 200), (20, 500),
    (50, 200), (50, 500),
    # Also test overlay disabled
    (0, 0),
]

# Load rounds
rounds = {}
for rnum, rid in ROUND_IDS.items():
    rdir = Path(f"data/round_{rid}")
    if not rdir.exists():
        continue
    detail_files = list(rdir.glob("round_detail_*.json"))
    gt_files = list(rdir.glob("ground_truth_s*.json"))
    if not detail_files or not gt_files:
        continue
    detail = json.loads(detail_files[0].read_text())
    gts = {}
    for gf in gt_files:
        s = int(gf.stem.split("_s")[1])
        gts[s] = np.array(json.loads(gf.read_text())["ground_truth"])
    rounds[rnum] = (rid, detail, gts)

print(f"Loaded {len(rounds)} rounds: {sorted(rounds.keys())}")
print(f"Testing {len(PARAMS)} param combinations")
print(f"UNET_BLEND_W = {model.UNET_BLEND_W}")

results = {}

for min_ps, max_ps in PARAMS:
    if min_ps == 0 and max_ps == 0:
        # Disable overlay
        orig_overlay = model._adaptive_bayesian_overlay
        model._adaptive_bayesian_overlay = lambda *a, **k: a[2]
    else:
        model._BAYES_MIN_PS = float(min_ps)
        model._BAYES_MAX_PS = float(max_ps)

    scores_by_round = {}
    for rnum in sorted(rounds.keys()):
        rid, detail, gts = rounds[rnum]
        seed_scores = []
        for seed_idx in sorted(gts.keys()):
            gt = gts[seed_idx]
            pred = model.build_prediction(rid, detail, seed_idx)
            score = score_prediction(pred, gt)
            seed_scores.append(score)
        scores_by_round[rnum] = np.mean(seed_scores)

    avg = np.mean(list(scores_by_round.values()))
    results[(min_ps, max_ps)] = (avg, scores_by_round)

    label = f"({min_ps:3d},{max_ps:3d})" if max_ps > 0 else "DISABLED"
    per_r = "  ".join(f"R{r}={s:.2f}" for r, s in sorted(scores_by_round.items()))
    print(f"  {label}  avg={avg:.3f}  {per_r}", flush=True)

    if min_ps == 0 and max_ps == 0:
        model._adaptive_bayesian_overlay = orig_overlay

# Restore defaults
model._BAYES_MIN_PS = 5.0
model._BAYES_MAX_PS = 100.0

print("\n=== RANKED RESULTS ===")
ranked = sorted(results.items(), key=lambda x: -x[1][0])
for i, ((min_ps, max_ps), (avg, by_round)) in enumerate(ranked[:10]):
    label = f"({min_ps},{max_ps})" if max_ps > 0 else "DISABLED"
    marker = " <<<" if min_ps == 5 and max_ps == 100 else ""
    per_r = "  ".join(f"R{r}={s:.2f}" for r, s in sorted(by_round.items()))
    print(f"  #{i+1} {label:>10}  avg={avg:.3f}  {per_r}{marker}")

curr = results.get((5, 100), (0, {}))[0]
best_params, (best_avg, _) = ranked[0]
print(f"\nCurrent (5,100): {curr:.3f}")
print(f"Best {best_params}: {best_avg:.3f}")
print(f"Delta: {best_avg - curr:+.3f}")
