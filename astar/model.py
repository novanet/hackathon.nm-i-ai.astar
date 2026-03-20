"""
Prediction model for Astar Island.

Builds a 40×40×6 probability tensor from:
  1. Transition prior (historical + cross-seed observations)
  2. Bayesian update with simulation observations
  3. Spatial smoothing (scipy gaussian filter) for unobserved cells
  4. Neighbor-based inference for remaining gaps
  5. Probability floor enforcement (0.001)
"""

import warnings
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", message="X does not have valid feature names")
from .replay import (
    TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES,
    load_round_detail, load_simulations,
    build_observation_grid, build_empirical_distribution,
)

PROB_FLOOR = 0.0002

# Historical transition matrix derived from all rounds backtesting.
# Used as fallback when no observations are available for the current round.
# Rows = initial class, Cols = final class.  (calibrated on R1-R8)
# Order: Empty, Settlement, Port, Ruin, Forest, Mountain
HISTORICAL_TRANSITIONS = np.array([
    [0.8535, 0.0997, 0.0075, 0.0103, 0.0290, 0.0000],  # Empty →
    [0.4617, 0.2930, 0.0037, 0.0260, 0.2156, 0.0000],  # Settlement →
    [0.4842, 0.0886, 0.1733, 0.0217, 0.2322, 0.0000],  # Port →
    [0.5000, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],  # Ruin → (no data)
    [0.0792, 0.1275, 0.0088, 0.0130, 0.7715, 0.0000],  # Forest →
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # Mountain →
])

# Shrinkage matrix: GT/obs ratio per transition, computed from R1-R8.
# Multiplied with observed transitions to debias toward GT distribution.
# Values >1 mean GT is higher than single-run observations (more stable).
SHRINKAGE_MATRIX = np.array([
    [1.0410, 0.8048, 1.0383, 0.7179, 0.8362, 1.0000],  # Empty →
    [1.0407, 0.9193, 1.9426, 0.8194, 1.0567, 1.0000],  # Settlement →
    [1.3142, 0.8416, 0.4938, 1.2387, 1.4709, 1.0000],  # Port →
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],  # Ruin →
    [0.8973, 0.7965, 1.0213, 0.6904, 1.0653, 1.0000],  # Forest →
    [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],  # Mountain →
])


def debias_transitions(observed_trans: np.ndarray) -> np.ndarray:
    """Apply shrinkage to observed transitions to correct for single-run bias."""
    debiased = observed_trans * SHRINKAGE_MATRIX
    row_sums = np.maximum(debiased.sum(axis=1, keepdims=True), 1e-10)
    return debiased / row_sums


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


PRIOR_STRENGTH = 20.0  # Dirichlet concentration — how much to trust the prior vs observations


def bayesian_update(round_id: str, seed_index: int,
                    map_w: int = 40, map_h: int = 40,
                    prior: np.ndarray | None = None,
                    prior_strength: float = PRIOR_STRENGTH) -> np.ndarray:
    """
    Bayesian Dirichlet-Multinomial update: blend prior with observations.

    Instead of replacing the prior with raw frequencies (which makes 1 obs
    snap to 100% confidence), we treat the prior as a Dirichlet with
    concentration α = prior × prior_strength, then add observation counts.

    posterior_mean = (α + obs_counts) / (sum(α) + n_obs)

    For unobserved cells: returns the prior unchanged.
    """
    obs_grid = build_observation_grid(round_id, seed_index, map_w, map_h)
    pred = np.full((map_h, map_w, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)

    if prior is not None:
        pred[:] = prior

    for y in range(map_h):
        for x in range(map_w):
            obs = obs_grid[y][x]
            if not obs:
                continue
            # Dirichlet prior: α = prior_probs × prior_strength
            alpha = pred[y, x] * prior_strength
            # Add observation counts
            for cls in obs:
                alpha[cls] += 1.0
            # Posterior mean
            pred[y, x] = alpha / alpha.sum()

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


def _apply_transition_matrix(round_detail: dict, seed_index: int,
                             transition_matrix: np.ndarray,
                             map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """Apply a transition matrix to a seed's initial state to produce a prior."""
    init_grid = round_detail["initial_states"][seed_index]["grid"]
    prior = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    for y in range(map_h):
        for x in range(map_w):
            init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
            prior[y, x] = transition_matrix[init_cls]
    return prior


def apply_floor(pred: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """
    Enforce minimum probability floor and renormalize.
    Prevents KL divergence from exploding on near-zero predictions.
    """
    floored = np.maximum(pred, floor)
    sums = floored.sum(axis=-1, keepdims=True)
    return floored / sums


# ── Spatial Conditional Model ──────────────────────────────────────────────

def compute_cell_features(init_grid: list[list[int]],
                          map_w: int = 40, map_h: int = 40,
                          round_features: np.ndarray | None = None) -> np.ndarray:
    """
    Compute spatial features for every cell from the initial state.

    Features per cell (22 spatial + up to 8 round-level = 30 total):
      - initial class one-hot (6)
      - 3×3 neighborhood class fractions (6), normalized
      - 5×5 outer ring class fractions (6), normalized
      - distance to nearest settlement (1), normalized by map size
      - distance to nearest forest (1), normalized
      - distance to nearest port (1), normalized
      - count of settlements within radius 5 (1), normalized
      - [optional] round-level features (8): E→E, S→S, F→F, E→S, settlement_density,
        mean_food, mean_wealth, mean_defense

    Returns: (H, W, 22 or 30) feature array
    """
    from scipy.ndimage import distance_transform_edt

    n_spatial = 6 + 6 + 6 + 4  # 22 spatial features
    n_round = len(round_features) if round_features is not None else 0
    n_feat = n_spatial + n_round
    features = np.zeros((map_h, map_w, n_feat), dtype=np.float64)

    # Convert grid to class indices
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])

    # Precompute distance transforms for key classes
    max_dist = float(map_w + map_h)  # normalization constant
    settlement_mask = (cls_grid == 1) | (cls_grid == 2)  # settlements + ports
    forest_mask = cls_grid == 4
    port_mask = cls_grid == 2

    dist_settlement = distance_transform_edt(~settlement_mask) if settlement_mask.any() else np.full((map_h, map_w), max_dist)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), max_dist)
    dist_port = distance_transform_edt(~port_mask) if port_mask.any() else np.full((map_h, map_w), max_dist)

    # Count settlements within radius 5
    sett_count_r5 = np.zeros((map_h, map_w), dtype=np.float64)
    sett_positions = np.argwhere(settlement_mask)
    for y in range(map_h):
        for x in range(map_w):
            for sy, sx in sett_positions:
                if abs(sy - y) <= 5 and abs(sx - x) <= 5:
                    sett_count_r5[y, x] += 1.0

    for y in range(map_h):
        for x in range(map_w):
            idx = 0
            # One-hot initial class
            features[y, x, cls_grid[y, x]] = 1.0
            idx = 6

            # 3×3 neighborhood counts (excluding self)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < map_h and 0 <= nx < map_w:
                        features[y, x, idx + cls_grid[ny, nx]] += 1.0
            n3 = features[y, x, idx:idx+6].sum()
            if n3 > 0:
                features[y, x, idx:idx+6] /= n3
            idx += 6

            # 5×5 outer ring counts (excluding 3×3 inner)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if abs(dy) <= 1 and abs(dx) <= 1:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < map_h and 0 <= nx < map_w:
                        features[y, x, idx + cls_grid[ny, nx]] += 1.0
            n5 = features[y, x, idx:idx+6].sum()
            if n5 > 0:
                features[y, x, idx:idx+6] /= n5
            idx += 6

            # Distance features (normalized to [0, 1])
            features[y, x, idx] = dist_settlement[y, x] / max_dist
            features[y, x, idx + 1] = dist_forest[y, x] / max_dist
            features[y, x, idx + 2] = dist_port[y, x] / max_dist
            features[y, x, idx + 3] = sett_count_r5[y, x] / max(1.0, sett_count_r5.max())

    # Append round-level features (same for every cell)
    if round_features is not None:
        for i, val in enumerate(round_features):
            features[:, :, n_spatial + i] = val

    return features


_spatial_model = None  # cached trained model


def load_spatial_model():
    """Load the trained spatial model from disk, or return None.
    Supports both single model and ensemble dict {lgb, xgb, lgb_weight}."""
    global _spatial_model
    if _spatial_model is not None:
        return _spatial_model
    import pickle
    model_path = Path(__file__).parent.parent / "data" / "spatial_model.pkl"
    if model_path.exists():
        loaded = pickle.loads(model_path.read_bytes())
        if isinstance(loaded, dict) and "lgb" in loaded:
            # Ensemble format
            class EnsemblePredictor:
                def __init__(self, lgb_m, xgb_m, w=0.7):
                    self.lgb = lgb_m
                    self.xgb = xgb_m
                    self.w = w
                    self.n_jobs = 1
                def predict(self, X):
                    return self.w * self.lgb.predict(X) + (1 - self.w) * self.xgb.predict(X)
            for m in [loaded["lgb"], loaded["xgb"]]:
                if hasattr(m, 'n_jobs'):
                    m.n_jobs = 1
            _spatial_model = EnsemblePredictor(loaded["lgb"], loaded["xgb"], loaded.get("lgb_weight", 0.7))
        else:
            # Single model (legacy)
            if hasattr(loaded, 'n_jobs'):
                loaded.n_jobs = 1
            _spatial_model = loaded
        return _spatial_model
    return None


def spatial_prior(round_detail: dict, seed_index: int,
                  map_w: int = 40, map_h: int = 40,
                  round_features: np.ndarray | None = None) -> np.ndarray | None:
    """
    Use trained spatial model to predict per-cell distributions from initial state features.
    Returns (H, W, 6) or None if no model available.
    """
    model = load_spatial_model()
    if model is None:
        return None

    init_grid = round_detail["initial_states"][seed_index]["grid"]
    features = compute_cell_features(init_grid, map_w, map_h, round_features)

    # Flatten to (H*W, n_features), predict, reshape
    flat_features = features.reshape(-1, features.shape[-1])
    flat_pred = model.predict(flat_features)  # (H*W, 6)
    pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)

    # Ensure valid probabilities
    pred = np.maximum(pred, 1e-10)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def observation_calibrated_transitions(round_id: str, round_detail: dict,
                                       map_w: int = 40, map_h: int = 40,
                                       smoothing: float = 5.0) -> np.ndarray | None:
    """
    Learn round-specific transition matrix from ALL observations across all seeds.
    Blends with historical transitions using Bayesian smoothing.

    Returns: (NUM_CLASSES, NUM_CLASSES) calibrated transition matrix, or None if no obs.
    """
    n_seeds = len(round_detail.get("initial_states", []))
    states = round_detail["initial_states"]

    # Count transitions from observations
    obs_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    total_obs = 0

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
                    obs_counts[init_cls, final_cls] += 1
                    total_obs += 1

    if total_obs == 0:
        return None

    # Bayesian blend: historical prior (weighted by smoothing) + observations
    prior_counts = HISTORICAL_TRANSITIONS * smoothing
    blended = prior_counts + obs_counts
    row_sums = np.maximum(blended.sum(axis=1, keepdims=True), 1e-10)
    return blended / row_sums


def _extract_settlement_stats(round_id: str, round_detail: dict) -> dict:
    """
    Extract aggregate settlement statistics from all simulation observations.
    Returns dict with: mean_pop, mean_food, mean_wealth, mean_defense,
    n_alive, n_dead, n_ports, n_factions, settlement_density.
    These are signals for hidden parameters (expansion, winter, raiding, trade).
    """
    n_seeds = len(round_detail.get("initial_states", []))
    all_pop, all_food, all_wealth, all_defense = [], [], [], []
    factions: set[int] = set()
    n_alive = n_dead = n_ports = 0

    for seed_idx in range(n_seeds):
        sims = load_simulations(round_id, seed_idx)
        for sim in sims:
            resp = sim.get("response", sim)
            for s in resp.get("settlements", []):
                if s.get("alive", True):
                    all_pop.append(s.get("population", 0))
                    all_food.append(s.get("food", 0))
                    all_wealth.append(s.get("wealth", 0))
                    all_defense.append(s.get("defense", 0))
                    factions.add(s.get("owner_id", 0))
                    n_alive += 1
                    if s.get("has_port"):
                        n_ports += 1
                else:
                    n_dead += 1

    if not all_pop:
        return {}

    # Settlement density: alive settlements per observation viewport
    total_viewports = sum(len(load_simulations(round_id, s)) for s in range(n_seeds))
    density = n_alive / max(total_viewports, 1)

    return {
        "mean_pop": float(np.mean(all_pop)),
        "mean_food": float(np.mean(all_food)),
        "mean_wealth": float(np.mean(all_wealth)),
        "mean_defense": float(np.mean(all_defense)),
        "n_alive": n_alive,
        "n_dead": n_dead,
        "n_ports": n_ports,
        "n_factions": len(factions),
        "density": density,
    }


def compute_round_features(calibrated_trans: np.ndarray | None,
                            round_detail: dict,
                            settlement_stats: dict | None = None) -> np.ndarray:
    """
    Compute round-level features for the spatial model.
    8 features: E→E, S→S, F→F, E→S, settlement_density, mean_food, mean_wealth, mean_defense.
    Falls back to historical averages if no calibrated transitions available.
    """
    states = round_detail.get("initial_states", [])
    map_w = round_detail.get("map_width", 40)
    map_h = round_detail.get("map_height", 40)

    # Settlement density from initial states
    n_sett = 0
    total = 0
    for s in states:
        init_grid = s["grid"]
        for y in range(map_h):
            for x in range(map_w):
                cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                if cls == 1:
                    n_sett += 1
                total += 1
    sett_density = n_sett / max(total, 1)

    if calibrated_trans is not None:
        trans = calibrated_trans
    else:
        trans = HISTORICAL_TRANSITIONS

    # Settlement stats (from observations at inference, from GT proxies at training)
    # When no observations available, derive proxies from transition matrix
    # using the same formulas as compute_gt_round_features() in train_spatial.py
    ss_rate = trans[1, 1]  # S→S
    es_rate = trans[0, 1]  # E→S
    se_rate = trans[1, 0]  # S→E
    mean_food = 0.3 + 0.7 * ss_rate  # food ~ settlement survival
    mean_wealth = es_rate * 0.3       # wealth ~ expansion rate
    mean_defense = 1.0 - se_rate      # defense ~ 1 - collapse rate
    if settlement_stats:
        mean_food = settlement_stats.get("mean_food", mean_food)
        mean_wealth = settlement_stats.get("mean_wealth", mean_wealth)
        mean_defense = settlement_stats.get("mean_defense", mean_defense)

    return np.array([
        trans[0, 0],  # E→E
        trans[1, 1],  # S→S
        trans[4, 4],  # F→F
        trans[0, 1],  # E→S
        sett_density,
        mean_food,
        mean_wealth,
        mean_defense,
    ], dtype=np.float64)


def _detect_round_activity(calibrated_trans: np.ndarray | None,
                           settlement_stats: dict | None = None) -> float:
    """
    Detect how "active" a round is compared to historical averages.
    Uses two complementary signals:
      1. Transition matrix divergence (primary, when available)
      2. Settlement stats (secondary, catches dynamics even with few observations)
    Returns a divergence score: 0 = matches historical, higher = more different.
    """
    trans_activity = 0.0
    if calibrated_trans is not None:
        ee_hist = HISTORICAL_TRANSITIONS[0, 0]  # ~0.865
        ff_hist = HISTORICAL_TRANSITIONS[4, 4]  # ~0.794
        ee_obs = calibrated_trans[0, 0]
        ff_obs = calibrated_trans[4, 4]
        ee_div = max(0, ee_hist - ee_obs) / ee_hist
        ff_div = max(0, ff_hist - ff_obs) / ff_hist
        trans_activity = (ee_div + ff_div) / 2.0

    # Settlement stats provide a secondary signal:
    # High pop + low food + high defense = aggressive expansion + conflict
    # Baseline from R5 (calm round): pop~1.06, food~0.71, def~0.31
    stats_activity = 0.0
    if settlement_stats and settlement_stats.get("mean_pop", 0) > 0:
        pop_signal = max(0, settlement_stats["mean_pop"] - 1.06) / 1.06
        food_signal = max(0, 0.71 - settlement_stats["mean_food"]) / 0.71
        def_signal = max(0, settlement_stats["mean_defense"] - 0.31) / 0.31
        stats_activity = (pop_signal + food_signal + def_signal) / 3.0

    # Combine: transition matrix is primary when available, stats are secondary
    if calibrated_trans is not None:
        return 0.75 * trans_activity + 0.25 * stats_activity
    else:
        return stats_activity


def _proximity_conditioned_transitions(round_id: str, round_detail: dict,
                                       map_w: int = 40, map_h: int = 40,
                                       smoothing: float = 0.5) -> tuple | None:
    """
    Build distance-binned, forest-conditioned transition matrices from observations.
    Returns (bin_trans_near_forest, bin_trans_far_forest, DIST_BINS, SIGMA, FOREST_THRESH)
    or None if no observations.
    """
    from scipy.ndimage import distance_transform_edt

    DIST_BINS = [0, 2, 4, 7, 999]
    n_bins = len(DIST_BINS) - 1
    SIGMA = 1.5
    FOREST_THRESH = 3.0

    n_seeds = len(round_detail.get("initial_states", []))
    states = round_detail["initial_states"]

    nf_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    ff_counts = [np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64) for _ in range(n_bins)]
    total_obs = 0

    for seed_idx in range(n_seeds):
        sims = load_simulations(round_id, seed_idx)
        if not sims:
            continue
        init_grid = states[seed_idx]["grid"]
        cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                              for x in range(map_w)] for y in range(map_h)])
        sett_mask = (cls_grid == 1) | (cls_grid == 2)
        forest_mask = (cls_grid == 4)
        dist_sett = distance_transform_edt(~sett_mask) if sett_mask.any() else np.full((map_h, map_w), 999.0)
        dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), 999.0)

        for sim in sims:
            req = sim["request"]
            resp = sim["response"]
            vx, vy = req["viewport_x"], req["viewport_y"]
            for dy, row in enumerate(resp["grid"]):
                for dx, terrain_code in enumerate(row):
                    iy, ix = vy + dy, vx + dx
                    if iy >= map_h or ix >= map_w:
                        continue
                    init_cls = cls_grid[iy, ix]
                    final_cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                    d = dist_sett[iy, ix]
                    df = dist_forest[iy, ix]
                    for b in range(n_bins):
                        if DIST_BINS[b] <= d < DIST_BINS[b + 1]:
                            if df <= FOREST_THRESH:
                                nf_counts[b][init_cls, final_cls] += 1
                            else:
                                ff_counts[b][init_cls, final_cls] += 1
                            break
                    total_obs += 1

    if total_obs == 0:
        return None

    nf_trans = []
    ff_trans = []
    for b in range(n_bins):
        prior = HISTORICAL_TRANSITIONS * smoothing
        nft = (prior + nf_counts[b]) / np.maximum((prior + nf_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        fft = (prior + ff_counts[b]) / np.maximum((prior + ff_counts[b]).sum(axis=1, keepdims=True), 1e-10)
        nf_trans.append(nft)
        ff_trans.append(fft)

    return nf_trans, ff_trans, DIST_BINS, SIGMA, FOREST_THRESH


def _cell_observation_frequencies(round_id: str, round_detail: dict,
                                  seed_index: int,
                                  map_w: int = 40, map_h: int = 40) -> tuple:
    """
    Build per-cell empirical observation distributions for one seed.
    Returns (obs_freq, obs_count) arrays of shape (H,W,C) and (H,W).
    """
    sims = load_simulations(round_id, seed_index)
    obs_freq = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float64)
    obs_count = np.zeros((map_h, map_w), dtype=np.float64)

    for sim in sims:
        req = sim["request"]
        resp = sim["response"]
        vx, vy = req["viewport_x"], req["viewport_y"]
        for dy, row in enumerate(resp["grid"]):
            for dx, terrain_code in enumerate(row):
                iy, ix = vy + dy, vx + dx
                if iy >= map_h or ix >= map_w:
                    continue
                final_cls = TERRAIN_TO_CLASS.get(terrain_code, 0)
                obs_freq[iy, ix, final_cls] += 1.0
                obs_count[iy, ix] += 1.0

    return obs_freq, obs_count


def build_prediction(round_id: str, round_detail: dict, seed_index: int,
                     map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """
    Full prediction pipeline with round-conditioned spatial model:
      1. Observation-calibrated transitions → debias via shrinkage matrix
      2. Settlement stats from observations
      3. Round-level features for spatial model (includes settlement stats)
      4. Spatial model prediction with round features
      5. Fallback to debiased transition prior if no spatial model
      6. Adaptive per-class temperature scaling (collapse detection)
      7. Floor enforcement

    Returns: (H, W, 6) probability tensor ready for submission.
    """
    # 1. Round-calibrated transitions → debias
    calibrated_trans = observation_calibrated_transitions(
        round_id, round_detail, map_w, map_h)
    debiased_trans = debias_transitions(calibrated_trans) if calibrated_trans is not None else None

    # 2. Settlement stats from observations
    sett_stats = _extract_settlement_stats(round_id, round_detail)

    # 3. Round-level features for spatial model (now includes settlement stats)
    round_feats = compute_round_features(debiased_trans, round_detail,
                                         settlement_stats=sett_stats)

    # 4. Spatial model (pure — subsumes transition blending)
    pred = spatial_prior(round_detail, seed_index, map_w, map_h,
                         round_features=round_feats)

    # 5. Fallback if no spatial model
    if pred is None:
        trans = debiased_trans if debiased_trans is not None else HISTORICAL_TRANSITIONS
        pred = _apply_transition_matrix(round_detail, seed_index, trans, map_w, map_h)

    # 5b. Post-model calibration: correct systematic Settlement/Port/Ruin overestimation
    #     Model consistently over-predicts S/P/R and under-predicts E (verified in LORO audit)
    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)

    # 6. Adaptive per-class temperature scaling
    #    Detect collapse rounds (S→S < 0.15) and use softer temps for settlements
    ss_rate = round_feats[1]  # S→S feature
    if ss_rate < 0.15:
        # Collapse round: settlements dying — don't sharpen, soften instead
        adaptive_temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
    else:
        adaptive_temps = PER_CLASS_TEMPS
    pred = per_class_temperature_scale(pred, round_detail, seed_index,
                                       temps=adaptive_temps,
                                       map_w=map_w, map_h=map_h)

    # 7. Floor enforcement
    return apply_floor(pred)


TEMPERATURE = 1.10  # calibrated via LORO CV; T>1 softens predictions

# Per-class temperature: calibrated via LORO. E/W need slight softening.
# Order: Empty, Settlement, Port, Ruin, Forest, Mountain
PER_CLASS_TEMPS = np.array([1.20, 1.0, 1.0, 1.0, 1.20, 1.0])

# Post-model calibration: correct systematic S/P/R overestimation, E underestimation
# Model consistently over-predicts settlement/port/ruin by 13-54%, under-predicts empty by 3-5%
# Validated via LORO: ew=0.25 + S/P/R 0.92 gives best LORO (80.66, Δ=+0.88)
CALIBRATION_FACTORS = np.array([1.03, 0.92, 0.92, 0.92, 1.0, 1.0])


def temperature_scale(pred: np.ndarray, T: float | None = None) -> np.ndarray:
    """Apply temperature scaling: q_i = p_i^(1/T) / sum(p_j^(1/T)).
    T>1 softens (less confident), T<1 sharpens."""
    if T is None:
        T = TEMPERATURE
    if T == 1.0:
        return pred
    scaled = np.power(np.maximum(pred, 1e-30), 1.0 / T)
    scaled = scaled / scaled.sum(axis=-1, keepdims=True)
    return scaled


def per_class_temperature_scale(pred: np.ndarray, round_detail: dict,
                                seed_index: int,
                                temps: np.ndarray | None = None,
                                map_w: int = 40, map_h: int = 40) -> np.ndarray:
    """Apply different temperature per initial class of each cell."""
    if temps is None:
        temps = PER_CLASS_TEMPS
    init_grid = round_detail["initial_states"][seed_index]["grid"]
    result = pred.copy()
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])
    for c in range(NUM_CLASSES):
        mask = (cls_grid == c)
        if not mask.any() or temps[c] == 1.0:
            continue
        cells = result[mask]
        scaled = np.power(np.maximum(cells, 1e-30), 1.0 / temps[c])
        result[mask] = scaled / scaled.sum(axis=-1, keepdims=True)
    return result


def prediction_to_list(pred: np.ndarray) -> list:
    """Convert numpy prediction to nested list for API submission."""
    return pred.tolist()
