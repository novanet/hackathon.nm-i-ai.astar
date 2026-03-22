"""Verify R14 data is complete and save round detail if missing."""
import json
from pathlib import Path
from astar.client import get_round_detail

rid = "d0a2c894-2162-4d49-86cf-435b9013f3b8"
rdir = Path("data") / f"round_{rid}"
detail_files = list(rdir.glob("round_detail_*.json"))
if not detail_files:
    detail = get_round_detail(rid)
    dp = rdir / "round_detail_20260321.json"
    dp.write_text(json.dumps(detail, indent=2))
    print(f"Saved round detail to {dp}")
else:
    print(f"Already have detail: {detail_files[0].name}")
    detail = json.loads(detail_files[0].read_text())

print(f"  Seeds: {len(detail.get('initial_states', []))}")
print(f"  Map: {detail.get('map_width')}x{detail.get('map_height')}")

for s in range(5):
    gt_path = rdir / f"ground_truth_s{s}.json"
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())
        print(f"  GT s{s}: ok, keys={list(gt.keys())[:3]}")
    else:
        print(f"  GT s{s}: MISSING!")
