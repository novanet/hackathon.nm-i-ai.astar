"""
API client for Astar Island with automatic response logging.

Every API response is saved to disk so runs can be replayed locally
without making network requests.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://api.ainm.no/astar-island"
DATA_DIR = Path(__file__).parent.parent / "data"


def _get_token() -> str:
    token = os.environ.get("ASTAR_TOKEN")
    if not token:
        raise RuntimeError("Set ASTAR_TOKEN environment variable (JWT from app.ainm.no cookie)")
    return token


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_token()}",
        "Content-Type": "application/json",
    }


def _round_dir(round_id: str) -> Path:
    d = DATA_DIR / f"round_{round_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _log_response(round_id: str, category: str, data: dict, extra_key: str = "") -> None:
    """Save an API response as a timestamped JSON file."""
    d = _round_dir(round_id)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = f"_{extra_key}" if extra_key else ""
    path = d / f"{category}{suffix}_{ts}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _request(method: str, path: str, json_body: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    resp = requests.request(method, url, headers=_headers(), json=json_body, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ── Public API ──────────────────────────────────────────────────────────────


def get_rounds() -> list[dict]:
    """List all rounds."""
    return _request("GET", "/rounds")


def get_round_detail(round_id: str) -> dict:
    """Get round details including initial states for all seeds."""
    data = _request("GET", f"/rounds/{round_id}")
    _log_response(round_id, "round_detail", data)
    return data


def get_budget() -> dict:
    """Check remaining query budget for the active round."""
    return _request("GET", "/budget")


def simulate(round_id: str, seed_index: int,
             viewport_x: int = 0, viewport_y: int = 0,
             viewport_w: int = 15, viewport_h: int = 15) -> dict:
    """
    Run one stochastic simulation and observe through a viewport.
    Costs 1 query from budget. Response is logged to disk.
    """
    body = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": viewport_x,
        "viewport_y": viewport_y,
        "viewport_w": viewport_w,
        "viewport_h": viewport_h,
    }
    data = _request("POST", "/simulate", body)

    # Log with query metadata
    logged = {
        "request": body,
        "response": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _log_response(round_id, f"sim_s{seed_index}", logged,
                  extra_key=f"x{viewport_x}_y{viewport_y}")
    return data


def submit(round_id: str, seed_index: int, prediction: list) -> dict:
    """
    Submit prediction for one seed. Overwrites any previous submission.
    prediction: H×W×6 nested list of probabilities.
    """
    body = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    }
    data = _request("POST", "/submit", body)
    _log_response(round_id, f"submit_s{seed_index}", {
        "seed_index": seed_index,
        "response": data,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return data


def get_my_rounds() -> list[dict]:
    """Get all rounds with your team's scores and budget info."""
    return _request("GET", "/my-rounds")


def get_my_predictions(round_id: str) -> list[dict]:
    """Get your submitted predictions with argmax/confidence grids."""
    return _request("GET", f"/my-predictions/{round_id}")


def get_analysis(round_id: str, seed_index: int) -> dict:
    """Post-round: get ground truth comparison for one seed."""
    data = _request("GET", f"/analysis/{round_id}/{seed_index}")
    _log_response(round_id, f"analysis_s{seed_index}", data)
    return data


def get_leaderboard() -> list[dict]:
    """Public leaderboard."""
    return _request("GET", "/leaderboard")


def simulate_grid(round_id: str, seed_index: int, map_w: int = 40, map_h: int = 40,
                  vp_size: int = 15, delay: float = 0.25) -> list[dict]:
    """
    Cover the full map with a systematic grid of viewports.
    Returns list of simulation responses. Respects rate limit with delay.
    """
    results = []
    for vy in range(0, map_h, vp_size - 1):  # overlap by 1
        for vx in range(0, map_w, vp_size - 1):
            w = min(vp_size, map_w - vx)
            h = min(vp_size, map_h - vy)
            if w < 5 or h < 5:
                continue
            data = simulate(round_id, seed_index, vx, vy, w, h)
            results.append(data)
            if delay > 0:
                time.sleep(delay)
    return results
