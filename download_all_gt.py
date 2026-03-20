"""Download ground truth for all completed rounds and check scores."""
import json
import numpy as np
from pathlib import Path
from astar.client import _request, get_round_detail, get_rounds

# Check our scores
print("=== OUR SCORES ===")
my = _request("GET", "/my-rounds")
for r in sorted(my, key=lambda x: x["round_number"]):
    rn = r["round_number"]
    score = r.get("round_score")
    rank = r.get("rank")
    queries = r.get("queries_used")
    seeds = r.get("seed_scores") or {}
    print(f"  R{rn}: score={score}, rank={rank}, queries={queries}, seeds={len(seeds)}")
    if seeds:
        for i, ss in enumerate(seeds if isinstance(seeds, list) else seeds.values()):
            print(f"    seed {i}: {ss}")

# Get all completed rounds
rounds = get_rounds()
completed = [r for r in rounds if r["status"] == "completed"]
print(f"\n=== DOWNLOADING GROUND TRUTH ({len(completed)} completed rounds) ===")

for r in sorted(completed, key=lambda x: x["round_number"]):
    rid = r["id"]
    rn = r["round_number"]
    data_dir = Path("data") / f"round_{rid}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if we already have ground truth
    existing_gt = list(data_dir.glob("ground_truth_s*.json"))
    if len(existing_gt) >= 5:
        print(f"  R{rn} ({rid[:8]}): already have {len(existing_gt)} ground truth files, skipping")
        continue

    print(f"  R{rn} ({rid[:8]}): downloading...", end="")
    detail = get_round_detail(rid)
    n_seeds = len(detail.get("initial_states", []))

    for seed_idx in range(n_seeds):
        gt_path = data_dir / f"ground_truth_s{seed_idx}.json"
        if gt_path.exists():
            continue
        try:
            analysis = _request("GET", f"/analysis/{rid}/{seed_idx}")
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
        except Exception as e:
            print(f" ERROR seed {seed_idx}: {e}", end="")

    new_gt = list(data_dir.glob("ground_truth_s*.json"))
    print(f" {len(new_gt)} seeds downloaded")
    
    # Quick stats
    if new_gt:
        gt0 = np.array(json.loads(new_gt[0].read_text(encoding="utf-8"))["ground_truth"], dtype=np.float64)
        print(f"    Map: {detail['map_width']}x{detail['map_height']}, shape={gt0.shape}")
