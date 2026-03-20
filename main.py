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

    # Phase 2: Build predictions and submit
    results = {}
    for seed_idx in range(n_seeds):
        try:
            pred = build_prediction(round_id, detail, seed_idx, map_w, map_h)
            resp = submit(round_id, seed_idx, prediction_to_list(pred))
            results[seed_idx] = resp
            log.info("Seed %d submitted: %s", seed_idx, resp.get("status"))
        except Exception:
            log.exception("Failed seed %d", seed_idx)
            results[seed_idx] = {"status": "error"}

    return {"status": "completed", "seeds_submitted": n_seeds, "results": results}
