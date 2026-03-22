"""
Submission orchestrator for Astar Island.

Builds predictions for all seeds in a round and submits them.
Can also do dry-run with local scoring if ground truth is available.
"""

import sys
import numpy as np

from .client import get_round_detail, submit
from .model import build_prediction, prediction_to_list, apply_floor
from .replay import load_ground_truth_array


def score_prediction(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Local scorer: entropy-weighted KL divergence → score on [0, 100].
    Matches the official scoring formula:
        score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    """
    eps = 1e-10
    p = np.clip(ground_truth, eps, 1.0)   # ground truth
    q = np.clip(prediction, eps, 1.0)     # your prediction

    # Per-cell KL divergence: KL(p || q)
    kl = np.sum(p * np.log(p / q), axis=-1)

    # Per-cell entropy (weights)
    entropy = -np.sum(p * np.log(p), axis=-1)

    total_entropy = np.sum(entropy)
    if total_entropy < eps:
        return 100.0  # all cells static → perfect by default

    weighted_kl = np.sum(entropy * kl) / total_entropy

    return max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))


def submit_round(round_id: str, seeds: list[int] | None = None,
                 dry_run: bool = False) -> dict[int, float | None]:
    """
    Build and submit predictions for specified seeds (default: all 5).

    Returns dict mapping seed_index → score (if ground truth available)
    or None (if submitted without local scoring).
    """
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    if seeds is None:
        seeds = list(range(n_seeds))

    results = {}
    for seed_idx in seeds:
        print(f"  Building prediction for seed {seed_idx}...")
        pred = build_prediction(round_id, detail, seed_idx)

        # Local score if ground truth available
        gt = load_ground_truth_array(round_id, seed_idx)
        if gt is not None:
            score = score_prediction(pred, gt)
            results[seed_idx] = score
            print(f"  Seed {seed_idx}: local score = {score:.2f}")
        else:
            results[seed_idx] = None

        if not dry_run:
            print(f"  Submitting seed {seed_idx}...")
            resp = submit(round_id, seed_idx, prediction_to_list(pred))
            print(f"  → {resp}")
        else:
            print(f"  (dry run — not submitted)")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m astar.submit <round_id> [--dry-run] [--seeds 0,1,2]")
        sys.exit(1)

    round_id = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    seeds = None

    for i, arg in enumerate(sys.argv):
        if arg == "--seeds" and i + 1 < len(sys.argv):
            seeds = [int(s) for s in sys.argv[i + 1].split(",")]

    print(f"Round: {round_id}, dry_run={dry_run}, seeds={seeds or 'all'}")
    results = submit_round(round_id, seeds=seeds, dry_run=dry_run)

    print("\nResults:")
    for seed_idx, score in sorted(results.items()):
        label = f"{score:.2f}" if score is not None else "submitted (no local score)"
        print(f"  Seed {seed_idx}: {label}")


if __name__ == "__main__":
    main()
