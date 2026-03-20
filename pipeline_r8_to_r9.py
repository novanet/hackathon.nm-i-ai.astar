"""
End-to-end pipeline: Download R8 GT → Retrain model on R1-R8 → Run R9.

Usage:
  $env:ASTAR_TOKEN = "..."; python pipeline_r8_to_r9.py

Polls until R8 GT is available, then runs all steps automatically.
"""

import json
import time
import subprocess
import sys
import numpy as np
from pathlib import Path

from astar.client import _request, get_round_detail, get_analysis, get_rounds

R8_ID = "c5cdf100-a876-4fb7-b5d8-757162c97989"
DATA_DIR = Path("data")


def wait_for_r8_completed(poll_interval: int = 30) -> None:
    """Poll until R8 status is 'completed'."""
    print("=== Waiting for R8 to close ===")
    while True:
        rounds = get_rounds()
        r8 = next((r for r in rounds if r["id"] == R8_ID), None)
        if r8 and r8.get("status") == "completed":
            score = r8.get("round_score")
            print(f"  R8 completed! Score: {score}")
            return
        status = r8.get("status", "?") if r8 else "not found"
        closes = r8.get("closes_at", "?") if r8 else "?"
        print(f"  R8 status: {status}, closes: {closes} — retrying in {poll_interval}s")
        time.sleep(poll_interval)


def download_r8_gt() -> bool:
    """Download R8 ground truth for all seeds. Returns True if successful."""
    print("\n=== Downloading R8 Ground Truth ===")
    rdir = DATA_DIR / f"round_{R8_ID}"
    rdir.mkdir(parents=True, exist_ok=True)

    # Get detail
    detail = get_round_detail(R8_ID)
    n_seeds = len(detail.get("initial_states", []))

    success = True
    for seed_idx in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{seed_idx}.json"
        if gt_path.exists():
            print(f"  Seed {seed_idx}: already exists, skipping")
            continue
        try:
            analysis = get_analysis(R8_ID, seed_idx)
            gt = np.array(analysis["ground_truth"], dtype=np.float64)
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(f"  Seed {seed_idx}: saved (shape={gt.shape})")
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}")
            success = False

    return success


def print_r8_score():
    """Print R8 official score + leaderboard position."""
    print("\n=== R8 Score ===")
    my = _request("GET", "/my-rounds")
    r8 = next((r for r in my if r.get("round_number") == 8), None)
    if r8:
        score = r8.get("round_score")
        seeds = r8.get("seed_scores")
        rank = r8.get("rank")
        total = r8.get("total_teams")
        print(f"  Score: {score}")
        print(f"  Seeds: {seeds}")
        print(f"  Rank: {rank}/{total}")
        if score:
            weighted = score * 1.05 ** 8
            print(f"  Weighted: {weighted:.2f}")
    else:
        print("  R8 not found in my rounds")

    # Leaderboard
    lb = _request("GET", "/leaderboard")
    print("\nLeaderboard top 10:")
    for i, e in enumerate(lb[:10]):
        print(f"  {i+1}. {e.get('team_name', '?')}: {e.get('score', 0):.2f}")
    for i, e in enumerate(lb):
        if "novanet" in str(e).lower():
            print(f"  Us: rank {i+1}, score {e.get('score', 0):.2f}")
            break


def retrain_model():
    """Run train_spatial.py (which now includes R8 in ROUND_IDS)."""
    print("\n=== Retraining Model (R1-R8) ===")
    result = subprocess.run(
        [sys.executable, "train_spatial.py"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Training failed with exit code {result.returncode}")
        return False
    return True


def run_r9():
    """Run the generic next-round solver for R9."""
    print("\n=== Running R9 ===")
    result = subprocess.run(
        [sys.executable, "run_next_round.py"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f"  R9 solver failed with exit code {result.returncode}")
        return False
    return True


def main():
    # Step 1: Wait for R8 to close
    wait_for_r8_completed(poll_interval=30)

    # Step 2: Download R8 GT + print score
    download_r8_gt()
    print_r8_score()

    # Step 3: Retrain model
    if not retrain_model():
        print("ERROR: Training failed, aborting")
        return

    # Step 4: Run R9
    print("\n" + "=" * 60)
    print("Ready to run R9. Launching solver...")
    print("=" * 60)
    run_r9()


if __name__ == "__main__":
    main()
