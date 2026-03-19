"""
Prediction model for Astar Island.

Builds a 40×40×6 probability tensor from:
  1. Static prior (initial terrain state)
  2. Bayesian update with simulation observations
  3. Spatial smoothing (scipy gaussian filter) for unobserved cells
  4. Cross-seed inference (shared hidden parameters)
  5. Neighbor-based inference for remaining gaps
  6. Probability floor enforcement (0.01)
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from .replay import (
    TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES,
    load_round_detail, load_simulations,
    build_observation_grid, build_empirical_distribution,
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


def spatial_smooth(pred: np.ndarray, obs_mask: np.ndarray,
                   sigma: float = 1.5) -> np.ndarray:
    """
    Gaussian-smooth predictions into unobserved regions.

    Observed cells keep their empirical values. Unobserved cells get
    spatially interpolated probabilities from nearby observed cells,
    weighted by a Gaussian kernel.

    Args:
        pred: (H, W, 6) prediction array
        obs_mask: (H, W) boolean — True where cell was observed
        sigma: Gaussian kernel width (larger = smoother)
    """
    h, w = pred.shape[:2]
    result = pred.copy()

    # Build a weight map: observed cells have weight 1, unobserved 0
    weights = obs_mask.astype(np.float64)

    # Smooth each class channel independently, weighted by observation mask
    smoothed_weights = gaussian_filter(weights, sigma=sigma)
    smoothed_weights = np.maximum(smoothed_weights, 1e-10)  # avoid division by zero

    for c in range(NUM_CLASSES):
        weighted_vals = pred[:, :, c] * weights
        smoothed_vals = gaussian_filter(weighted_vals, sigma=sigma)
        # Only update unobserved cells
        mask = ~obs_mask
        result[:, :, c] = np.where(mask, smoothed_vals / smoothed_weights, pred[:, :, c])

    # Renormalize smoothed cells
    unobs = ~obs_mask
    sums = result.sum(axis=-1, keepdims=True)
    sums = np.maximum(sums, 1e-10)
    result[unobs] = result[unobs] / sums[unobs][..., np.newaxis] \
        if unobs.any() else result[unobs]
    # Simpler renormalization for all cells
    sums = result.sum(axis=-1, keepdims=True)
    result = result / np.maximum(sums, 1e-10)

    return result


def cross_seed_transition_prior(round_id: str, round_detail: dict,
                                target_seed: int,
                                map_w: int = 40, map_h: int = 40) -> np.ndarray | None:
    """
    Build a transition-based prior by pooling observations from ALL seeds.

    Since all seeds in a round share the same hidden parameters, we can
    learn initial_class → final_class transition probabilities from
    observed seeds and apply them to unobserved regions of the target seed.

    Returns (H, W, 6) prior or None if no cross-seed data available.
    """
    n_seeds = len(round_detail.get("initial_states", []))
    states = round_detail["initial_states"]

    # Count transitions: initial_class → final_class across all seeds
    transition_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for seed_idx in range(n_seeds):
        sims = load_simulations(round_id, seed_idx)
        if not sims:
            continue
        init_grid = states[seed_idx]["grid"]
        for sim in sims:
            req = sim["request"]
            resp = sim["response"]
            vx, vy = req["viewport_x"], req["viewport_y"]
            for dy, row in enumerate(resp["grid"]):
                for dx, terrain_code in enumerate(row):
                    iy, ix = vy + dy, vx + dx
                    if iy >= map_h or ix >= map_w:
                        continue
                    init_cls = TERRAIN_TO_CLASS.get(init_grid[iy][ix], 0)
                    final_cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                    transition_counts[init_cls, final_cls] += 1

    # Normalize rows to get transition probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    if row_sums.sum() == 0:
        return None
    # Avoid division by zero for unseen initial classes
    row_sums = np.maximum(row_sums, 1.0)
    transition_probs = transition_counts / row_sums

    # Apply transition matrix to target seed's initial state
    init_grid = states[target_seed]["grid"]
    prior = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            prior[y, x] = transition_probs[init_cls]

    return prior


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
      2. Cross-seed transition prior (shared hidden params)
      3. Empirical update from observations
      4. Spatial smoothing (Gaussian) for unobserved cells
      5. Neighbor inference for remaining gaps
      6. Floor enforcement

    Returns: (H, W, 6) probability tensor ready for submission.
    """
    # 1. Prior from initial terrain
    prior = initial_prior(round_detail, seed_index, map_w, map_h)

    # 2. Cross-seed transition prior (blend with initial prior)
    xseed = cross_seed_transition_prior(round_id, round_detail, seed_index, map_w, map_h)
    if xseed is not None:
        # Blend: 60% cross-seed transitions, 40% initial state prior
        prior = 0.6 * xseed + 0.4 * prior
        sums = prior.sum(axis=-1, keepdims=True)
        prior = prior / np.maximum(sums, 1e-10)

    # 3. Empirical update from this seed's observations
    pred = empirical_model(round_id, seed_index, map_w, map_h, prior=prior)

    # 4. Build observation mask and apply spatial smoothing
    obs_grid = build_observation_grid(round_id, seed_index, map_w, map_h)
    obs_mask = np.array([[bool(obs_grid[y][x]) for x in range(map_w)]
                         for y in range(map_h)])

    if obs_mask.any():
        pred = spatial_smooth(pred, obs_mask, sigma=1.5)

    # 5. Neighbor inference for any remaining unobserved cells
    pred = neighbor_inference(pred, obs_grid)

    # 6. Floor enforcement
    return apply_floor(pred)


def prediction_to_list(pred: np.ndarray) -> list:
    """Convert numpy prediction to nested list for API submission."""
    return pred.tolist()
