"""
Astar Island /solve endpoint for Cloud Run deployment.

Receives round info from competition validators, runs simulation queries
for full map coverage, builds predictions with the trained model, and submits.
"""

import logging
import time

from fastapi import FastAPI

from astar.client import get_round_detail, get_budget, simulate_grid, submit
from astar.model import build_prediction, prediction_to_list

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/solve")
def solve_get():
    return {"status": "ready", "method": "POST required"}


@app.post("/solve")
async def solve(request: dict):
    """
    Called by competition validators with round info.
    Queries the simulator for observations, builds predictions, and submits.
    """
    round_id = request.get("round_id")
    if not round_id:
        return {"status": "error", "message": "missing round_id"}

    log.info("Solving round %s", round_id)

    try:
        detail = get_round_detail(round_id)
    except Exception:
        log.exception("Failed to get round detail")
        return {"status": "error", "message": "failed to get round detail"}

    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    # Phase 1: Query simulator if budget allows (9 queries/seed × 5 seeds = 45)
    queries_needed = n_seeds * 9
    try:
        budget = get_budget()
        remaining = budget["queries_max"] - budget["queries_used"]
        log.info("Budget: %d/%d used, %d remaining, need %d",
                 budget["queries_used"], budget["queries_max"], remaining, queries_needed)
    except Exception:
        log.exception("Failed to check budget")
        remaining = 0

    if remaining >= queries_needed:
        log.info("Querying %d seeds with grid coverage", n_seeds)
        for seed_idx in range(n_seeds):
            try:
                simulate_grid(round_id, seed_idx, map_w, map_h, delay=0.05)
                log.info("Seed %d: observations collected", seed_idx)
            except Exception:
                log.exception("Failed to query seed %d", seed_idx)
    else:
        log.info("Skipping queries — only %d remaining (need %d)", remaining, queries_needed)

    # Phase 1b: Submit initial predictions
    results = {}
    for seed_idx in range(n_seeds):
        try:
            pred = build_prediction(round_id, detail, seed_idx, map_w, map_h)
            resp = submit(round_id, seed_idx, prediction_to_list(pred))
            results[seed_idx] = resp
            log.info("Seed %d pass 1: %s", seed_idx, resp.get("status"))
        except Exception:
            log.exception("Failed seed %d pass 1", seed_idx)
            results[seed_idx] = {"status": "error"}

    # Phase 2: Spend remaining budget on extra observations + resubmit
    try:
        budget = get_budget()
        extra_remaining = budget["queries_max"] - budget["queries_used"]
    except Exception:
        extra_remaining = 0

    if extra_remaining > 0:
        log.info("Spending %d extra queries", extra_remaining)
        extra_vps = [(7, 7, 15, 15), (7, 20, 15, 15), (20, 7, 15, 15),
                     (20, 20, 15, 15), (3, 3, 15, 15)]
        # Sort seeds by settlement count
        seed_order = sorted(
            range(n_seeds),
            key=lambda i: len([s for s in detail["initial_states"][i]["settlements"]
                               if s["alive"]]),
            reverse=True,
        )
        used = 0
        for seed_idx in seed_order:
            if used >= extra_remaining:
                break
            vx, vy, vw, vh = extra_vps[used % len(extra_vps)]
            try:
                from astar.client import simulate
                simulate(round_id, seed_idx, vx, vy,
                         min(vw, map_w - vx), min(vh, map_h - vy))
                used += 1
            except Exception:
                log.exception("Extra query failed, stopping")
                break
            time.sleep(0.05)

        # Resubmit with extra observations
        for seed_idx in range(n_seeds):
            try:
                pred = build_prediction(round_id, detail, seed_idx, map_w, map_h)
                resp = submit(round_id, seed_idx, prediction_to_list(pred))
                results[seed_idx] = resp
                log.info("Seed %d pass 2: %s", seed_idx, resp.get("status"))
            except Exception:
                log.exception("Failed seed %d pass 2", seed_idx)

    return {"status": "completed", "seeds_submitted": n_seeds, "results": results}
