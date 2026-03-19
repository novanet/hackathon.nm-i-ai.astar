"""
Visualization for Astar Island grids, predictions, and comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from .replay import (
    TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES,
    load_round_detail, load_simulations, build_observation_grid,
)

# Color scheme for terrain classes
CLASS_COLORS = [
    "#4a90d9",  # 0 Empty (blue-gray for ocean/plains)
    "#e8a435",  # 1 Settlement (amber)
    "#2ecc71",  # 2 Port (green)
    "#95a5a6",  # 3 Ruin (gray)
    "#27ae60",  # 4 Forest (dark green)
    "#8b7355",  # 5 Mountain (brown)
]

TERRAIN_CMAP = mcolors.ListedColormap(CLASS_COLORS)
TERRAIN_NORM = mcolors.BoundaryNorm(range(NUM_CLASSES + 1), NUM_CLASSES)


def _class_legend() -> list[Patch]:
    return [Patch(facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]


def plot_terrain_grid(grid: list[list[int]], title: str = "Terrain", ax=None,
                      show: bool = True) -> None:
    """
    Plot a terrain grid (raw terrain codes) as a colored heatmap.
    Converts terrain codes to class indices automatically.
    """
    arr = np.array(grid)
    class_grid = np.vectorize(lambda v: TERRAIN_TO_CLASS.get(v, 0))(arr)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(class_grid, cmap=TERRAIN_CMAP, norm=TERRAIN_NORM, interpolation="nearest")
    ax.set_title(title)
    ax.legend(handles=_class_legend(), loc="upper right", fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()


def plot_initial_states(round_detail: dict, show: bool = True) -> None:
    """Plot initial terrain grids for all seeds in a round."""
    states = round_detail.get("initial_states", [])
    n = len(states)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for i, state in enumerate(states):
        plot_terrain_grid(state["grid"], title=f"Seed {i} — Initial", ax=axes[i], show=False)
    fig.suptitle(f"Round {round_detail.get('round_number', '?')} — Initial States", fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


def plot_simulation(sim_data: dict, title: str = "Simulation Result", show: bool = True) -> None:
    """Plot a single simulation response (viewport grid)."""
    plot_terrain_grid(sim_data["response"]["grid"], title=title, show=show)


def plot_observation_coverage(round_id: str, seed_index: int,
                              map_w: int = 40, map_h: int = 40,
                              show: bool = True) -> None:
    """
    Heatmap showing how many times each cell has been observed.
    Brighter = more observations.
    """
    obs_grid = build_observation_grid(round_id, seed_index, map_w, map_h)
    counts = np.array([[len(obs_grid[y][x]) for x in range(map_w)] for y in range(map_h)])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(counts, cmap="YlOrRd", interpolation="nearest")
    ax.set_title(f"Observation Coverage — Seed {seed_index} (max={counts.max()})")
    plt.colorbar(im, ax=ax, label="# observations")
    plt.tight_layout()
    if show:
        plt.show()


def plot_prediction(prediction: np.ndarray, title: str = "Prediction",
                    show: bool = True) -> None:
    """
    Plot a prediction tensor (H×W×6).
    Shows argmax class and confidence side by side.
    """
    argmax = np.argmax(prediction, axis=-1)
    confidence = np.max(prediction, axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(argmax, cmap=TERRAIN_CMAP, norm=TERRAIN_NORM, interpolation="nearest")
    ax1.set_title(f"{title} — Argmax Class")
    ax1.legend(handles=_class_legend(), loc="upper right", fontsize=7)

    im = ax2.imshow(confidence, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title(f"{title} — Confidence")
    plt.colorbar(im, ax=ax2, label="Max probability")

    plt.tight_layout()
    if show:
        plt.show()


def plot_prediction_vs_truth(prediction: np.ndarray, ground_truth: np.ndarray,
                             title: str = "Prediction vs Ground Truth",
                             show: bool = True) -> None:
    """
    Side-by-side comparison of prediction and ground truth.
    Shows argmax for both plus a difference map.
    """
    pred_argmax = np.argmax(prediction, axis=-1)
    truth_argmax = np.argmax(ground_truth, axis=-1)
    match = (pred_argmax == truth_argmax).astype(float)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(pred_argmax, cmap=TERRAIN_CMAP, norm=TERRAIN_NORM, interpolation="nearest")
    ax1.set_title("Prediction (argmax)")
    ax1.legend(handles=_class_legend(), loc="upper right", fontsize=7)

    ax2.imshow(truth_argmax, cmap=TERRAIN_CMAP, norm=TERRAIN_NORM, interpolation="nearest")
    ax2.set_title("Ground Truth (argmax)")
    ax2.legend(handles=_class_legend(), loc="upper right", fontsize=7)

    ax3.imshow(match, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax3.set_title(f"Match ({match.mean():.1%} correct)")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()


def plot_entropy_map(ground_truth: np.ndarray, title: str = "Cell Entropy",
                     show: bool = True) -> None:
    """
    Show per-cell entropy from ground truth. High-entropy cells are where
    scoring concentrates — these are the cells that matter most.
    """
    eps = 1e-10
    p = np.clip(ground_truth, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(entropy, cmap="hot", interpolation="nearest")
    ax.set_title(f"{title} (total={entropy.sum():.1f})")
    plt.colorbar(im, ax=ax, label="Entropy (nats)")
    plt.tight_layout()
    if show:
        plt.show()


def plot_class_distribution(prediction: np.ndarray, y: int, x: int,
                            ground_truth: np.ndarray | None = None,
                            show: bool = True) -> None:
    """Bar chart comparing prediction vs ground truth for a single cell."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    xs = np.arange(NUM_CLASSES)
    width = 0.35

    ax.bar(xs - width / 2, prediction[y, x], width, label="Prediction", color="#3498db")
    if ground_truth is not None:
        ax.bar(xs + width / 2, ground_truth[y, x], width, label="Ground Truth", color="#e74c3c")

    ax.set_xticks(xs)
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_ylabel("Probability")
    ax.set_title(f"Cell ({y}, {x}) — Class Distribution")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if show:
        plt.show()
