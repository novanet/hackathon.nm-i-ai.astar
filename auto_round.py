"""
Auto-pilot: poll for new rounds every hour and auto-submit predictions.

Runs unattended overnight. When a new active round is detected:
1. Submit zero-query baseline immediately (all seeds)
2. Run 9-viewport grid coverage for all seeds (45 queries)
3. Spend remaining queries on repeat observations of dynamic seeds
4. Resubmit with full model predictions

Usage:
    $env:ASTAR_TOKEN = "your-jwt"
    python auto_round.py
"""

import time
import json
import logging
import traceback
from datetime import datetime, timezone

from astar.client import get_rounds, get_round_detail, get_budget, simulate
from astar.submit import submit_round

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto_round.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

POLL_INTERVAL = 3600  # 1 hour
SEEN_ROUNDS: set[str] = set()

# Viewport grid for full 40x40 coverage with 15x15 viewports
GRID_POSITIONS = [
    (0, 0), (13, 0), (26, 0),
    (0, 13), (13, 13), (26, 13),
    (0, 26), (13, 26), (26, 26),
]


def find_active_round() -> dict | None:
    """Find the current active round, if any."""
    rounds = get_rounds()
    for r in rounds:
        if r["status"] == "active":
            return r
    return None


def run_grid_queries(round_id: str, seed_index: int, map_w: int = 40, map_h: int = 40) -> int:
    """Run 9-viewport grid coverage for one seed. Returns queries used."""
    count = 0
    for vx, vy in GRID_POSITIONS:
        w = min(15, map_w - vx)
        h = min(15, map_h - vy)
        if w < 5 or h < 5:
            continue
        try:
            result = simulate(round_id, seed_index, vx, vy, w, h)
            used = result.get("queries_used", "?")
            log.info(f"  Seed {seed_index} ({vx},{vy}) {w}x{h} -> used={used}")
            count += 1
        except Exception as e:
            log.warning(f"  Seed {seed_index} ({vx},{vy}) failed: {e}")
            break
    return count


def run_extra_queries(round_id: str, detail: dict) -> None:
    """Spend remaining queries on repeat observations of most dynamic seeds."""
    budget = get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining <= 0:
        log.info("No remaining queries for extras")
        return

    # Sort seeds by settlement count (most dynamic first)
    states = detail.get("initial_states", [])
    seed_dynamics = []
    for i, state in enumerate(states):
        n_setts = len([s for s in state["settlements"] if s["alive"]])
        seed_dynamics.append((n_setts, i))
    seed_dynamics.sort(reverse=True)

    # Center-map viewports for repeat observations
    extra_viewports = [(10, 10, 15, 15), (10, 20, 15, 15)]

    queries_left = remaining
    for _, seed_idx in seed_dynamics:
        if queries_left <= 0:
            break
        for vx, vy, w, h in extra_viewports:
            if queries_left <= 0:
                break
            try:
                result = simulate(round_id, seed_idx, vx, vy, w, h)
                used = result.get("queries_used", "?")
                log.info(f"  Extra: seed {seed_idx} ({vx},{vy}) -> used={used}")
                queries_left -= 1
            except Exception as e:
                log.warning(f"  Extra query failed: {e}")
                queries_left = 0
                break


def handle_round(round_info: dict) -> None:
    """Full pipeline for one round."""
    round_id = round_info["id"]
    round_num = round_info["round_number"]
    closes_at = round_info["closes_at"]

    log.info(f"=== ROUND {round_num} DETECTED === (closes: {closes_at})")
    log.info(f"Round ID: {round_id}")

    # Step 1: Get round details
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    log.info(f"Map: {map_w}x{map_h}, {n_seeds} seeds")

    for i, state in enumerate(detail["initial_states"]):
        setts = len([s for s in state["settlements"] if s["alive"]])
        ports = len([s for s in state["settlements"] if s.get("has_port")])
        log.info(f"  Seed {i}: {setts} settlements, {ports} ports")

    # Step 2: Submit zero-query baseline (safety net)
    log.info("Submitting zero-query baseline...")
    try:
        submit_round(round_id)
        log.info("Baseline submitted for all seeds")
    except Exception as e:
        log.error(f"Baseline submission failed: {e}")

    # Step 3: Full grid coverage for all seeds
    log.info("Running grid coverage queries...")
    budget = get_budget()
    queries_available = budget["queries_max"] - budget["queries_used"]
    queries_needed = 9 * n_seeds
    log.info(f"Budget: {budget['queries_used']}/{budget['queries_max']} used, need {queries_needed} for full coverage")

    for seed_idx in range(n_seeds):
        budget = get_budget()
        if budget["queries_used"] >= budget["queries_max"]:
            log.warning(f"Budget exhausted at seed {seed_idx}")
            break
        log.info(f"Querying seed {seed_idx}...")
        run_grid_queries(round_id, seed_idx, map_w, map_h)

    # Step 4: Spend remaining queries on repeat observations
    log.info("Running extra queries on dynamic seeds...")
    run_extra_queries(round_id, detail)

    # Step 5: Resubmit with full model
    log.info("Resubmitting with observation-informed model...")
    try:
        submit_round(round_id)
        log.info("Final predictions submitted for all seeds")
    except Exception as e:
        log.error(f"Final submission failed: {e}")

    budget = get_budget()
    log.info(f"Round {round_num} complete: {budget['queries_used']}/{budget['queries_max']} queries used")
    log.info(f"=== ROUND {round_num} DONE ===\n")


def main():
    log.info("Auto-pilot started. Polling every %d seconds.", POLL_INTERVAL)

    # Seed SEEN_ROUNDS with already-known rounds so we don't re-process them
    try:
        rounds = get_rounds()
        for r in rounds:
            SEEN_ROUNDS.add(r["id"])
            log.info(f"Existing round {r['round_number']} ({r['status']}): {r['id']}")
    except Exception as e:
        log.error(f"Failed to fetch initial rounds: {e}")

    # Wait before first poll to avoid rate-limiting from startup
    log.info("Waiting 60s before first poll cycle...")
    time.sleep(60)

    while True:
        try:
            active = find_active_round()
            if active and active["id"] not in SEEN_ROUNDS:
                SEEN_ROUNDS.add(active["id"])
                handle_round(active)
            elif active:
                log.info(f"Round {active['round_number']} still active (already processed). Closes: {active['closes_at']}")
            else:
                log.info("No active round. Waiting...")
        except Exception:
            log.error("Poll cycle failed:\n%s", traceback.format_exc())

        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        log.info(f"Next poll at +{POLL_INTERVAL}s (now: {now})")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
