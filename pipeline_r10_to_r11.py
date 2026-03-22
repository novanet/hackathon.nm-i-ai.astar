"""
End-to-end pipeline: Wait for R10 → Download GT → Retrain on R1-R10 → Backtest.

Usage:
  $env:ASTAR_TOKEN = "..."; python pipeline_r10_to_r11.py

Polls until R10 is completed, downloads GT, retrains GBM+MLP,
updates SHRINKAGE_MATRIX, and backtests. R11 must be run manually.
"""

import json
import time
import subprocess
import sys
import numpy as np
from pathlib import Path

from astar.client import _request, get_round_detail, get_analysis
from astar.model import build_prediction, prediction_to_list, apply_floor
from astar.submit import score_prediction

R10_ID = "75e625c3-60cb-4392-af3e-c86a98bde8c2"
DATA_DIR = Path("data")


def get_rounds():
    return _request("GET", "/rounds")


def wait_for_round_completed(round_id: str, label: str = "R10",
                              poll_interval: int = 30) -> dict:
    """Poll until round status is 'completed'. Returns round info."""
    print(f"=== Waiting for {label} to close ===")
    while True:
        rounds = get_rounds()
        r = next((r for r in rounds if r["id"] == round_id), None)
        if r and r.get("status") == "completed":
            score = r.get("round_score")
            print(f"  {label} completed! Score: {score}")
            return r
        status = r.get("status", "?") if r else "not found"
        closes = r.get("closes_at", "?") if r else "?"
        print(f"  {label} status: {status}, closes: {closes} — retrying in {poll_interval}s")
        time.sleep(poll_interval)


def download_gt(round_id: str, label: str = "R10") -> bool:
    """Download ground truth for all seeds. Returns True if successful."""
    print(f"\n=== Downloading {label} Ground Truth ===")
    rdir = DATA_DIR / f"round_{round_id}"
    rdir.mkdir(parents=True, exist_ok=True)

    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))

    success = True
    for seed_idx in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{seed_idx}.json"
        if gt_path.exists():
            print(f"  Seed {seed_idx}: already exists, skipping")
            continue
        try:
            analysis = get_analysis(round_id, seed_idx)
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            gt = np.array(analysis["ground_truth"], dtype=np.float64)
            print(f"  Seed {seed_idx}: saved (shape={gt.shape})")
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}")
            success = False
    return success


def retrain_models() -> bool:
    """Retrain GBM + MLP on all available rounds. Returns True if successful."""
    print("\n=== Retraining Models (train_spatial.py) ===")
    result = subprocess.run(
        [sys.executable, "train_spatial.py"],
        capture_output=False, text=True,
    )
    if result.returncode != 0:
        print(f"  Training FAILED (exit code {result.returncode})")
        return False
    print("  Training complete!")
    return True


def backtest_pipeline() -> float:
    """Run full pipeline backtest. Returns average score."""
    print("\n=== Running Pipeline Backtest ===")
    result = subprocess.run(
        [sys.executable, "backtest_pipeline.py"],
        capture_output=False, text=True,
    )
    return result.returncode == 0


def check_r10_score() -> float | None:
    """Check R10 official score."""
    my = _request("GET", "/my-rounds")
    r10 = next((r for r in my if r.get("round_number") == 10), None)
    if r10:
        score = r10.get("round_score")
        seeds = r10.get("seed_scores")
        print(f"\n=== R10 Official Score ===")
        print(f"  Round score: {score}")
        print(f"  Seed scores: {seeds}")
        if score:
            weighted = score * 1.6289
            print(f"  Weighted: {weighted:.2f}")
        return score
    return None


def backtest_r10_with_current_model(round_id: str) -> float:
    """Backtest R10 with current model to see potential improvement."""
    print("\n=== Backtesting R10 with current model ===")
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    n_seeds = len(detail.get("initial_states", []))

    scores = []
    for seed_idx in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{seed_idx}.json"
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gt = np.array(gt_data["ground_truth"], dtype=np.float64)
        pred = build_prediction(round_id, detail, seed_idx)
        score = score_prediction(pred, gt)
        scores.append(score)
        print(f"  Seed {seed_idx}: {score:.2f}")

    avg = np.mean(scores) if scores else 0.0
    print(f"  R10 backtest avg: {avg:.2f} (weighted: {avg * 1.6289:.2f})")
    return avg


def main():
    print("=" * 60)
    print("  PIPELINE: R10 GT → Retrain → R11")
    print("=" * 60)

    # Step 1: Wait for R10 to complete
    r10_info = wait_for_round_completed(R10_ID, "R10")

    # Step 2: Check R10 score
    check_r10_score()

    # Step 3: Download R10 ground truth
    if not download_gt(R10_ID, "R10"):
        print("WARNING: Some GT downloads failed, continuing anyway...")

    # Step 4: Backtest R10 with current model (before retrain)
    backtest_r10_with_current_model(R10_ID)

    # Step 5: Update train_spatial.py ROUND_IDS to include R10
    # (We need to add R10 to the training set)
    print("\n=== Adding R10 to training set ===")
    train_file = Path("train_spatial.py")
    content = train_file.read_text(encoding="utf-8")
    if R10_ID not in content:
        old = '    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",\n}'
        new = '    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",\n    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",\n}'
        content = content.replace(old, new)
        train_file.write_text(content, encoding="utf-8")
        print("  Added R10 to ROUND_IDS")
    else:
        print("  R10 already in ROUND_IDS")

    # Step 6: Retrain models
    if not retrain_models():
        print("ERROR: Training failed. Leaving the current model unchanged.")
        print("  R11 must still be evaluated and run manually.")
        return

    # Step 7: Run full backtest
    backtest_pipeline()

    # Done — R11 must be run manually
    print("\n✓ Pipeline complete. Models retrained on R1-R10.")
    print("  To run R11: python run_next_round.py")

    # Step 8: Check leaderboard
    print("\n=== Leaderboard ===")
    lb = _request("GET", "/leaderboard")
    for i, e in enumerate(lb[:10]):
        marker = " ←" if "novanet" in str(e).lower() else ""
        print(f"  {i+1}. {e.get('team_name', '?')}: {e.get('score', 0):.2f}{marker}")
    for i, e in enumerate(lb):
        if "novanet" in str(e).lower():
            print(f"\n  Our rank: #{i+1}, score: {e.get('score', 0):.2f}")
            break


if __name__ == "__main__":
    main()
