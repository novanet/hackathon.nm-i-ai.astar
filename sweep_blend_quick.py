"""Quick sweep of U-Net blend weight on R9-R16 using saved models.
Sets model.UNET_BLEND_W and calls build_prediction() directly.
"""
import json, numpy as np
from pathlib import Path
from astar.submit import score_prediction
import astar.model as model

ROUND_IDS = {
    9: "2a341ace-1065-4acb-b64d-67c4e4f22857",
    10: "75e625c3-b0dc-4e09-a653-98be9e72e3a0",
    11: "324fde07-4978-484e-8f71-27c317f8b910",
    12: "795bfb1f-f8c4-4030-ad63-fa28da8ee32e",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}

UNET_WEIGHTS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

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

results = {w: {} for w in UNET_WEIGHTS}

for rnum in sorted(rounds.keys()):
    rid, detail, gts = rounds[rnum]
    map_w = detail["map_width"]
    map_h = detail["map_height"]
    
    scores_by_w = {w: [] for w in UNET_WEIGHTS}
    
    for seed_idx in sorted(gts.keys()):
        gt = gts[seed_idx]
        
        for w in UNET_WEIGHTS:
            model.UNET_BLEND_W = w
            pred = model.build_prediction(rid, detail, seed_idx)
            score = score_prediction(pred, gt)
            scores_by_w[w].append(score)
    
    line = f"R{rnum:2d}:"
    for w in UNET_WEIGHTS:
        avg = np.mean(scores_by_w[w])
        results[w][rnum] = avg
        line += f"  {int(w*100)}%={avg:.2f}"
    print(line, flush=True)

# Restore default
model.UNET_BLEND_W = 0.40

print("\n=== SUMMARY ===")
header = "      " + "  ".join(f"{int(w*100):>5}%" for w in UNET_WEIGHTS)
print(header)
for rnum in sorted(rounds.keys()):
    line = f"R{rnum:2d}  "
    for w in UNET_WEIGHTS:
        line += f"  {results[w].get(rnum, 0):6.2f}"
    print(line)

avgs = []
for w in UNET_WEIGHTS:
    avg = np.mean([results[w][r] for r in rounds])
    avgs.append(avg)
    
line = "AVG  "
for a in avgs:
    line += f"  {a:6.2f}"
print(line)

best_idx = int(np.argmax(avgs))
curr_idx = UNET_WEIGHTS.index(0.40)
print(f"\nBest: {int(UNET_WEIGHTS[best_idx]*100)}% U-Net (avg={avgs[best_idx]:.3f})")
print(f"Current (40%): avg={avgs[curr_idx]:.3f}")
print(f"Delta: +{avgs[best_idx] - avgs[curr_idx]:.3f}")
