"""
Poll for R11, run it when active, then poll for completion and download GT.
Pre-approved by user for autonomous execution.
"""
import time
import json
import subprocess
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from astar.client import _request, get_round_detail, get_analysis

POLL_INTERVAL = 1800  # 30 minutes


def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def download_ground_truth(round_id: str):
    """Download ground truth for all seeds of a completed round."""
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    data_dir = Path("data") / f"round_{round_id}"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading ground truth...")
    for s in range(n_seeds):
        try:
            a = get_analysis(round_id, s)
            gt = np.array(a["ground_truth"], dtype=np.float64)
            gt_path = data_dir / f"ground_truth_s{s}.json"
            gt_path.write_text(json.dumps(a, indent=2), encoding="utf-8")
            print(f"  Seed {s}: GT shape={gt.shape}")
        except Exception as e:
            print(f"  Seed {s}: ERROR - {e}")
    print("GT download complete.")


def main():
    print("=== R11 Poller Started ===")
    print(f"Polling every {POLL_INTERVAL // 60} minutes for active round...")
    print(f"Started at: {now_str()}")
    print()
    sys.stdout.flush()

    # Phase 1: Poll for active round
    while True:
        try:
            rounds = _request("GET", "/rounds")
            active = [r for r in rounds if r.get("status") == "active"]
            if active:
                r = active[0]
                rn = r.get("round_number")
                rid = r["id"]
                closes = r.get("closes_at", "?")
                print(f"[{now_str()}] ROUND {rn} IS ACTIVE!")
                print(f"  ID: {rid}")
                print(f"  Closes at: {closes}")
                print()
                print(">>> RUNNING run_next_round.py <<<")
                sys.stdout.flush()

                result = subprocess.run(
                    [sys.executable, "run_next_round.py"],
                    capture_output=False,
                )
                print(f"\nrun_next_round.py exited with code {result.returncode}")
                print()
                sys.stdout.flush()

                # Phase 2: Poll for completion
                print(f"=== Polling for R{rn} completion (every {POLL_INTERVAL // 60} min) ===")
                sys.stdout.flush()

                while True:
                    time.sleep(POLL_INTERVAL)
                    try:
                        my = _request("GET", "/my-rounds")
                        this_round = [x for x in my if x.get("round_number") == rn]
                        if this_round:
                            status = this_round[0].get("status", "?")
                            score = this_round[0].get("round_score")
                            print(f"[{now_str()}] R{rn}: status={status}, score={score}")
                            sys.stdout.flush()
                            if status == "completed" and score is not None:
                                seeds = this_round[0].get("seed_scores")
                                weight = 1.05 ** rn
                                print(f"\nR{rn} COMPLETED!")
                                print(f"  Score: {score}")
                                print(f"  Seeds: {seeds}")
                                print(f"  Weighted: {score * weight:.2f}")
                                print()
                                download_ground_truth(rid)
                                print()
                                print("=== READY FOR RETRAINING ===")
                                print(f"  Round ID: {rid}")
                                print(f"  Add to ROUND_IDS in train_spatial.py and retrain.")
                                sys.stdout.flush()
                                return
                        else:
                            print(f"[{now_str()}] R{rn} not in my-rounds yet")
                            sys.stdout.flush()
                    except Exception as e:
                        print(f"[{now_str()}] Poll error: {e}")
                        sys.stdout.flush()

                return  # Done

            else:
                n_completed = len([r for r in rounds if r.get("status") == "completed"])
                print(f"[{now_str()}] No active round. {n_completed} completed. Sleeping 30min...")
                sys.stdout.flush()
        except Exception as e:
            print(f"[{now_str()}] Error: {e}")
            sys.stdout.flush()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
