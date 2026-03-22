"""
Poll for R12 completion every 5 minutes, then:
1. Download GT for all seeds
2. Add R12 to training data
3. Retrain model on R1-R12
4. Update model constants (HISTORICAL_TRANSITIONS, SHRINKAGE_MATRIX)
5. Report results

Does NOT start R13.
"""

import json
import time
import subprocess
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from astar.client import _request, get_round_detail, get_analysis

R12_ID = "795bfb1f-54bd-4f39-a526-9868b36f7ebd"
R12_NUM = 12
DATA_DIR = Path("data")
POLL_INTERVAL = 300  # 5 minutes


def poll_for_completion() -> dict:
    """Poll until R12 is completed and score is available."""
    print("=== Polling for R12 completion (every 5 min) ===")
    while True:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        try:
            my = _request("GET", "/my-rounds")
            r12 = next((r for r in my if r.get("round_number") == R12_NUM), None)
            if r12:
                status = r12.get("status", "?")
                score = r12.get("round_score")
                print(f"[{now}] R12: status={status}, score={score}")
                if status == "completed" and score is not None:
                    print(f"\nR12 COMPLETED!")
                    print(f"  Score: {score}")
                    print(f"  Seed scores: {r12.get('seed_scores')}")
                    weight = 1.05 ** R12_NUM
                    print(f"  Weighted: {score * weight:.2f}")
                    return r12
            else:
                print(f"[{now}] R12 not in my-rounds yet")
        except Exception as e:
            print(f"[{now}] Error: {e}")

        sys.stdout.flush()
        time.sleep(POLL_INTERVAL)


def download_gt() -> int:
    """Download R12 ground truth for all seeds. Returns number of seeds."""
    print("\n=== Downloading R12 Ground Truth ===")
    rdir = DATA_DIR / f"round_{R12_ID}"
    rdir.mkdir(parents=True, exist_ok=True)

    detail = get_round_detail(R12_ID)
    detail_path = rdir / f"round_detail_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    n_seeds = len(detail.get("initial_states", []))
    for seed_idx in range(n_seeds):
        try:
            analysis = get_analysis(R12_ID, seed_idx)
            gt = np.array(analysis["ground_truth"], dtype=np.float64)
            gt_path = rdir / f"ground_truth_s{seed_idx}.json"
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(f"  Seed {seed_idx}: GT shape={gt.shape}, saved")
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}")

    return n_seeds


def add_r12_to_training():
    """Add R12 to ROUND_IDS in train_spatial.py and compute_shrinkage.py."""
    print("\n=== Adding R12 to training files ===")

    for fname in ["train_spatial.py", "compute_shrinkage.py"]:
        fpath = Path(fname)
        content = fpath.read_text(encoding="utf-8")
        old = '    11: "324fde07-1670-4202-b199-7aa92ecb40ee",\n}'
        new = f'    11: "324fde07-1670-4202-b199-7aa92ecb40ee",\n    12: "{R12_ID}",\n}}'
        if f'12: "{R12_ID}"' in content:
            print(f"  {fname}: R12 already present")
        elif old in content:
            content = content.replace(old, new)
            fpath.write_text(content, encoding="utf-8")
            print(f"  {fname}: R12 added")
        else:
            print(f"  {fname}: WARNING - could not find insertion point")


def retrain():
    """Run train_spatial.py to retrain on R1-R12."""
    print("\n=== Retraining on R1-R12 ===")
    result = subprocess.run(
        [sys.executable, "train_spatial.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        # Filter out warnings, show errors
        for line in result.stderr.split("\n"):
            if "error" in line.lower() or "traceback" in line.lower():
                print(f"  STDERR: {line}")
    if result.returncode != 0:
        print(f"  WARNING: train_spatial.py exited with code {result.returncode}")
    return result.returncode == 0


def compute_shrinkage():
    """Run compute_shrinkage.py to update shrinkage matrix."""
    print("\n=== Computing Shrinkage Matrix ===")
    result = subprocess.run(
        [sys.executable, "compute_shrinkage.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  WARNING: compute_shrinkage.py exited with code {result.returncode}")
    return result.returncode == 0


def main():
    print(f"Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"R12 ID: {R12_ID}")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print()
    sys.stdout.flush()

    # Step 1: Wait for completion
    r12_result = poll_for_completion()

    # Step 2: Download GT
    n_seeds = download_gt()

    # Step 3: Add R12 to training
    add_r12_to_training()

    # Step 4: Retrain
    retrain()

    # Step 5: Compute shrinkage
    compute_shrinkage()

    # Step 6: Summary
    print("\n" + "=" * 60)
    print("=== R12 POST-ROUND PIPELINE COMPLETE ===")
    print("=" * 60)
    score = r12_result.get("round_score", "?")
    seeds = r12_result.get("seed_scores", [])
    weight = 1.05 ** R12_NUM
    weighted = score * weight if isinstance(score, (int, float)) else "?"
    print(f"  R12 Score: {score}")
    print(f"  Seed scores: {seeds}")
    print(f"  Weighted: {weighted}")
    print(f"  Weight multiplier: {weight:.4f}")
    print(f"  Seeds downloaded: {n_seeds}")
    print(f"  Model retrained on R1-R12")
    print()
    print("NOT starting R13 (as requested).")
    print("Ready for manual R13 execution when you decide.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
