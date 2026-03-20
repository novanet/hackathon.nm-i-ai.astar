"""
Generic next-round solver: find active round, query, predict, submit, iterate.
Replaces per-round scripts (run_r7.py, run_r8.py, etc).

Usage: $env:ASTAR_TOKEN = "your-jwt"; python run_next_round.py
       $env:ASTAR_TOKEN = "your-jwt"; python run_next_round.py --round-id <id>
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path

from astar.client import (
    _request, get_round_detail, get_budget, simulate, get_analysis, submit
)
from astar.model import (
    build_prediction, prediction_to_list, apply_floor,
    observation_calibrated_transitions, HISTORICAL_TRANSITIONS,
)
from astar.submit import score_prediction
from astar.replay import load_simulations, TERRAIN_TO_CLASS, CLASS_NAMES

DATA_DIR = Path("data")

GRID_POSITIONS = [
    (0, 0), (13, 0), (26, 0),
    (0, 13), (13, 13), (26, 13),
    (0, 26), (13, 26), (26, 26),
]


def find_active_round() -> dict | None:
    """Find the current active round."""
    rounds = _request("GET", "/rounds")
    for r in rounds:
        if r.get("status") == "active":
            return r
    return None


def print_round_info(detail: dict, round_number: int, round_id: str, closes_at: str):
    """Print round summary."""
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    print(f"Round {round_number} | ID: {round_id}")
    print(f"Map: {map_w}×{map_h}, {n_seeds} seeds")
    print(f"Closes at: {closes_at}")
    for i, state in enumerate(detail["initial_states"]):
        setts = len([s for s in state["settlements"] if s["alive"]])
        ports = len([s for s in state["settlements"] if s.get("has_port")])
        print(f"  Seed {i}: {setts} settlements, {ports} ports")


def query_grid(round_id: str, n_seeds: int, map_w: int, map_h: int) -> int:
    """Run 9-viewport grid on all seeds. Returns number of queries used."""
    count = 0
    for seed_idx in range(n_seeds):
        for vx, vy in GRID_POSITIONS:
            w = min(15, map_w - vx)
            h = min(15, map_h - vy)
            try:
                simulate(round_id, seed_idx, vx, vy, w, h)
                count += 1
            except Exception as e:
                print(f"  ERROR seed {seed_idx} ({vx},{vy}): {e}")
            time.sleep(0.05)
        print(f"  Seed {seed_idx}: 9 viewports done")
    return count


def diagnose_observations(round_id: str, detail: dict, map_w: int, map_h: int):
    """Print diagnostic info about observed transitions vs historical."""
    cal = observation_calibrated_transitions(round_id, detail, map_w, map_h)
    if cal is None:
        print("  No calibrated transitions available")
        return

    hist = HISTORICAL_TRANSITIONS
    print("\n=== Transition Diagnostics (calibrated vs historical) ===")
    for c in range(6):
        if cal[c].sum() < 0.01:
            continue
        deltas = cal[c] - hist[c]
        big = [(CLASS_NAMES[j], deltas[j]) for j in range(6) if abs(deltas[j]) > 0.02]
        if big:
            changes = ", ".join(f"{n}:{d:+.1%}" for n, d in big)
            print(f"  {CLASS_NAMES[c]:>10} → {changes}")

    # Activity detection
    diag = cal[0, 0] - hist[0, 0]  # Empty→Empty delta
    activity = max(0, -diag)
    mode = "HIGH-ACTIVITY" if activity >= 0.10 else "NORMAL"
    print(f"  Activity: {activity:.2%} → {mode} mode")


def submit_all(round_id: str, detail: dict, n_seeds: int,
               map_w: int, map_h: int, label: str = "pass 1") -> list[dict]:
    """Submit predictions for all seeds. Returns responses."""
    print(f"\n=== Submitting ({label}) ===")
    results = []
    for seed_idx in range(n_seeds):
        try:
            pred = build_prediction(round_id, detail, seed_idx, map_w, map_h)
            resp = submit(round_id, seed_idx, prediction_to_list(pred))
            status = resp.get("status", "?")
            score = resp.get("score", "?")
            print(f"  Seed {seed_idx}: {status} score={score}")
            results.append(resp)
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}")
            results.append({"status": "error", "error": str(e)})
    return results


def spend_extra_queries(round_id: str, detail: dict, remaining: int,
                        map_w: int, map_h: int) -> int:
    """Spend remaining budget on extra observations. Returns count used."""
    if remaining <= 0:
        return 0

    n_seeds = len(detail.get("initial_states", []))
    # Sort seeds by settlement count (most dynamic first)
    seed_dynamics = []
    for i, state in enumerate(detail["initial_states"]):
        n_setts = len([s for s in state["settlements"] if s["alive"]])
        seed_dynamics.append((n_setts, i))
    seed_dynamics.sort(reverse=True)

    # Offset viewports that overlap between standard grid positions
    extra_viewports = [
        (7, 7, 15, 15),   # center-shifted
        (7, 20, 15, 15),  # mid-bottom-shifted
        (20, 7, 15, 15),  # right-mid-shifted
        (20, 20, 15, 15), # bottom-right shifted
        (3, 3, 15, 15),   # top-left interior
    ]

    count = 0
    vp_idx = 0
    for _, seed_idx in seed_dynamics:
        if count >= remaining:
            break
        vx, vy, w, h = extra_viewports[vp_idx % len(extra_viewports)]
        w = min(w, map_w - vx)
        h = min(h, map_h - vy)
        try:
            simulate(round_id, seed_idx, vx, vy, w, h)
            print(f"  Extra: seed {seed_idx} ({vx},{vy})")
            count += 1
        except Exception as e:
            print(f"  Extra failed seed {seed_idx}: {e}")
            break
        time.sleep(0.05)
        vp_idx += 1

    return count


def print_final_summary(round_number: int, round_id: str):
    """Print final budget and score info."""
    budget = get_budget()
    print(f"\n=== Final: {budget['queries_used']}/{budget['queries_max']} queries ===")

    my = _request("GET", "/my-rounds")
    for r in my:
        if r.get("round_number") == round_number:
            rs = r.get("round_score")
            ss = r.get("seed_scores")
            print(f"Score: {rs}")
            if ss:
                print(f"Seeds: {ss}")
            if rs:
                weight = 1.05 ** round_number
                print(f"Weighted: {rs * weight:.2f} (weight={weight:.4f})")
            break

    print(f"\nRound ID: {round_id}")


def main():
    parser = argparse.ArgumentParser(description="Solve the active Astar Island round")
    parser.add_argument("--round-id", help="Override round ID (otherwise finds active)")
    parser.add_argument("--skip-queries", action="store_true",
                        help="Skip querying, use existing observations")
    args = parser.parse_args()

    # Find round
    if args.round_id:
        round_id = args.round_id
        rounds = _request("GET", "/rounds")
        r_info = next((r for r in rounds if r["id"] == round_id), None)
        round_number = r_info["round_number"] if r_info else 0
        closes_at = (r_info or {}).get("closes_at", "?")
    else:
        print("=== Finding active round ===")
        r_info = find_active_round()
        if not r_info:
            print("No active round found!")
            rounds = _request("GET", "/rounds")
            for r in rounds:
                print(f"  R{r.get('round_number')}: {r.get('status')} ({r['id'][:8]})")
            return
        round_id = r_info["id"]
        round_number = r_info["round_number"]
        closes_at = r_info.get("closes_at", "?")

    # Get round detail
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    print_round_info(detail, round_number, round_id, closes_at)

    # Save round detail
    rdir = DATA_DIR / f"round_{round_id}"
    rdir.mkdir(parents=True, exist_ok=True)

    # Check budget
    budget = get_budget()
    used = budget["queries_used"]
    total = budget["queries_max"]
    print(f"\nBudget: {used}/{total} queries used")

    queries_needed = 9 * n_seeds
    remaining = total - used

    # Phase 1: Grid queries
    if not args.skip_queries and remaining >= queries_needed:
        print(f"\n=== Querying {n_seeds} seeds (9 viewports each) ===")
        query_grid(round_id, n_seeds, map_w, map_h)
        budget = get_budget()
        remaining = budget["queries_max"] - budget["queries_used"]
        print(f"Budget after grid: {budget['queries_used']}/{budget['queries_max']}")
    elif args.skip_queries:
        print("Skipping queries (--skip-queries)")
    else:
        print(f"Skipping grid — only {remaining} remaining (need {queries_needed})")

    # Diagnostics
    diagnose_observations(round_id, detail, map_w, map_h)

    # Phase 2: Submit pass 1
    submit_all(round_id, detail, n_seeds, map_w, map_h, "pass 1")

    # Phase 3: Extra queries with remaining budget
    budget = get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining > 0 and not args.skip_queries:
        print(f"\n=== Extra queries: {remaining} remaining ===")
        spend_extra_queries(round_id, detail, remaining, map_w, map_h)

        # Resubmit with extra observations
        submit_all(round_id, detail, n_seeds, map_w, map_h, "pass 2 + extra obs")

    # Final summary
    print_final_summary(round_number, round_id)


if __name__ == "__main__":
    main()
