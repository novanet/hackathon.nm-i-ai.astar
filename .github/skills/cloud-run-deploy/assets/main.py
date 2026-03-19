"""
Astar Island /solve endpoint for Cloud Run deployment.

This is a template — adapt the solve() function to use your actual model.
Copy this file to the project root as main.py before deploying.
"""

import logging

from fastapi import FastAPI

from astar.client import get_round_detail, submit
from astar.model import build_prediction, prediction_to_list

log = logging.getLogger(__name__)
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: dict):
    """
    Called by competition validators with round info.
    Must query the simulator, build predictions, and submit for all seeds.
    """
    round_id = request.get("round_id")
    if not round_id:
        return {"status": "error", "message": "missing round_id"}

    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))

    results = {}
    for seed_idx in range(n_seeds):
        try:
            pred = build_prediction(round_id, detail, seed_idx)
            resp = submit(round_id, seed_idx, prediction_to_list(pred))
            results[seed_idx] = resp
        except Exception:
            log.exception("Failed seed %d", seed_idx)
            results[seed_idx] = {"status": "error"}

    return {"status": "completed", "seeds_submitted": n_seeds, "results": results}
