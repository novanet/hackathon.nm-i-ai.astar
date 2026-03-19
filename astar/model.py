"""
Prediction model for Astar Island.

Builds a 40×40×6 probability tensor from:
  1. Static prior (initial terrain state)
  2. Bayesian update with simulation observations
  3. Neighbor-based inference for unobserved cells
  4. Probability floor enforcement (0.01)
"""

import numpy as np
from .replay import (
    TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES,
    load_round_detail, build_observation_grid, build_empirical_distribution,
)

PROB_FLOOR = 0.01


def initial_prior(round_detail: dict, seed_index: int,
                  map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """
    Build a prior from the initial terrain state of a seed.
    Returns (H, W, 6) array with one-hot vectors for the initial class.
    """
    states = round_detail.get("initial_states", [])
    grid = states[seed_index]["grid"]

    prior = np.full((map_h, map_w, NUM_CLASSES), PROB_FLOOR, dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            cls = TERRAIN_TO_CLASS.get(grid[y][x], 0)
            prior[y, x, cls] = 1.0 - PROB_FLOOR * (NUM_CLASSES - 1)

    return prior


def empirical_model(round_id: str, seed_index: int,
                    map_w: int = 40, map_h: int = 40,
                    prior: np.ndarray | None = None) -> np.ndarray:
    """
    Build prediction from empirical observation frequencies.

    For observed cells: use frequency distribution.
    For unobserved cells: fall back to prior (if given) or uniform.
    """
    emp = build_empirical_distribution(round_id, seed_index, map_w, map_h)
    pred = np.full((map_h, map_w, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    if prior is not None:
        pred[:] = prior

    for y in range(map_h):
        for x in range(map_w):
            if emp[y][x] is not None:
                pred[y, x] = emp[y][x]

    return pred


def neighbor_inference(pred: np.ndarray, obs_grid: list[list[list[int]]],
                       radius: int = 1) -> np.ndarray:
    """
    For unobserved cells, average the predictions of observed neighbors.
    Only modifies cells that have no direct observations.
    """
    h, w = pred.shape[:2]
    result = pred.copy()

    for y in range(h):
        for x in range(w):
            if obs_grid[y][x]:
                continue  # already observed
            neighbor_preds = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and obs_grid[ny][nx]:
                        neighbor_preds.append(pred[ny, nx])
            if neighbor_preds:
                result[y, x] = np.mean(neighbor_preds, axis=0)

    return result


def apply_floor(pred: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """
    Enforce minimum probability floor and renormalize.
    Prevents KL divergence from exploding on near-zero predictions.
    """
    floored = np.maximum(pred, floor)
    sums = floored.sum(axis=-1, keepdims=True)
    return floored / sums


def build_prediction(round_id: str, round_detail: dict, seed_index: int,
                     map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """
    Full prediction pipeline:
      1. Initial state prior
      2. Empirical update from observations
      3. Neighbor inference for unobserved cells
      4. Floor enforcement

    Returns: (H, W, 6) probability tensor ready for submission.
    """
    prior = initial_prior(round_detail, seed_index, map_w, map_h)
    pred = empirical_model(round_id, seed_index, map_w, map_h, prior=prior)

    obs_grid = build_observation_grid(round_id, seed_index, map_w, map_h)
    pred = neighbor_inference(pred, obs_grid)

    return apply_floor(pred)


def prediction_to_list(pred: np.ndarray) -> list:
    """Convert numpy prediction to nested list for API submission."""
    return pred.tolist()
