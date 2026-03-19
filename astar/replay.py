"""
Local replay store for Astar Island.

Loads logged API responses from disk so you can build models
and test predictions without making network requests.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Terrain code → prediction class index
TERRAIN_TO_CLASS = {
    10: 0,  # Ocean → Empty
    11: 0,  # Plains → Empty
    0: 0,   # Empty → Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
NUM_CLASSES = 6


def round_dir(round_id: str) -> Path:
    return DATA_DIR / f"round_{round_id}"


def list_rounds() -> list[str]:
    """List all round IDs that have logged data."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name.removeprefix("round_")
        for d in DATA_DIR.iterdir()
        if d.is_dir() and d.name.startswith("round_")
    )


def load_round_detail(round_id: str) -> dict | None:
    """Load the saved round detail (initial states, map size, etc.)."""
    d = round_dir(round_id)
    files = sorted(d.glob("round_detail_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_simulations(round_id: str, seed_index: int | None = None) -> list[dict]:
    """
    Load all saved simulation responses for a round.
    Each entry has 'request', 'response', 'timestamp'.
    Optionally filter to a specific seed.
    """
    d = round_dir(round_id)
    pattern = f"sim_s{seed_index}_*.json" if seed_index is not None else "sim_s*_*.json"
    results = []
    for f in sorted(d.glob(pattern)):
        results.append(json.loads(f.read_text(encoding="utf-8")))
    return results


def load_analysis(round_id: str, seed_index: int) -> dict | None:
    """Load saved ground truth analysis for a seed."""
    d = round_dir(round_id)
    files = sorted(d.glob(f"analysis_s{seed_index}_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def build_observation_grid(round_id: str, seed_index: int,
                           map_w: int = 40, map_h: int = 40) -> list[list[list[int]]]:
    """
    Aggregate all simulation observations for a seed into a per-cell
    list of observed terrain classes.

    Returns: grid[y][x] = [class_idx, class_idx, ...]  (one per observation)
    Unobserved cells have an empty list.
    """
    grid = [[[] for _ in range(map_w)] for _ in range(map_h)]

    for sim in load_simulations(round_id, seed_index):
        req = sim["request"]
        resp = sim["response"]
        vx, vy = req["viewport_x"], req["viewport_y"]
        rows = resp["grid"]
        for dy, row in enumerate(rows):
            for dx, terrain_code in enumerate(row):
                cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                grid[vy + dy][vx + dx].append(cls)

    return grid


def build_empirical_distribution(round_id: str, seed_index: int,
                                 map_w: int = 40, map_h: int = 40) -> list[list[list[float]]]:
    """
    Build per-cell empirical probability distribution from observations.

    Returns: dist[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
    Unobserved cells get None (caller should fill with prior).
    """
    obs_grid = build_observation_grid(round_id, seed_index, map_w, map_h)
    dist = [[None for _ in range(map_w)] for _ in range(map_h)]

    for y in range(map_h):
        for x in range(map_w):
            obs = obs_grid[y][x]
            if not obs:
                continue
            counts = [0] * NUM_CLASSES
            for cls in obs:
                counts[cls] += 1
            total = len(obs)
            dist[y][x] = [c / total for c in counts]

    return dist
