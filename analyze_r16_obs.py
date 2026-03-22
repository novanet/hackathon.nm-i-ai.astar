"""Analyze R16 observations: coverage, consistency, and Bayesian overlay impact.
Identify where observations can most improve predictions."""
import numpy as np
from collections import Counter
from astar.replay import (load_round_detail, build_observation_grid,
                          CLASS_NAMES, NUM_CLASSES, TERRAIN_TO_CLASS)
from astar.model import (build_prediction, _adaptive_bayesian_overlay,
                          observation_calibrated_transitions, HISTORICAL_TRANSITIONS,
                          _BAYES_MIN_PS, _BAYES_MAX_PS, PROB_FLOOR, apply_floor)

ROUND_ID = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"


def analyze_seed(round_detail, seed_idx, map_w=40, map_h=40):
    obs_grid = build_observation_grid(ROUND_ID, seed_idx, map_w, map_h)

    # Coverage stats
    obs_counts = np.zeros((map_h, map_w), dtype=int)
    for y in range(map_h):
        for x in range(map_w):
            obs_counts[y, x] = len(obs_grid[y][x])

    total_cells = map_w * map_h
    observed = (obs_counts > 0).sum()
    multi_obs = (obs_counts > 1).sum()
    max_obs = obs_counts.max()

    print(f"\n  Seed {seed_idx}: {observed}/{total_cells} cells observed "
          f"({multi_obs} with 2+ obs, max {max_obs})")

    # For cells with 2+ observations, check consistency
    consistent = 0
    inconsistent = 0
    inconsistent_cells = []
    for y in range(map_h):
        for x in range(map_w):
            if obs_counts[y, x] < 2:
                continue
            obs = obs_grid[y][x]
            if len(set(obs)) == 1:
                consistent += 1
            else:
                inconsistent += 1
                counts = Counter(obs)
                inconsistent_cells.append((y, x, counts, obs_counts[y, x]))

    if multi_obs > 0:
        print(f"  Multi-obs cells: {consistent} consistent, {inconsistent} inconsistent "
              f"({100*consistent/multi_obs:.0f}% agree)")

    if inconsistent_cells:
        print(f"  Inconsistent cells (obs disagree):")
        for y, x, counts, n in inconsistent_cells[:10]:
            labels = ", ".join(f"{CLASS_NAMES[c]}:{n}" for c, n in counts.most_common())
            print(f"    ({x},{y}): {labels} [{n} obs]")

    return obs_counts, obs_grid, inconsistent_cells


def measure_overlay_impact(round_detail, seed_idx, map_w=40, map_h=40):
    """Compare predictions with and without Bayesian overlay."""
    # Build prediction up to the overlay step
    pred_full = build_prediction(ROUND_ID, round_detail, seed_idx, map_w, map_h)

    # Build without overlay: re-run but skip overlay
    # We can't easily skip it inside build_prediction, so let's measure the overlay's effect
    # by comparing pred before/after overlay in the existing pipeline
    obs_grid = build_observation_grid(ROUND_ID, seed_idx, map_w, map_h)

    # Count cells where overlay changed prediction significantly
    obs_counts = np.zeros((map_h, map_w), dtype=int)
    for y in range(map_h):
        for x in range(map_w):
            obs_counts[y, x] = len(obs_grid[y][x])

    # Compute KL between model argmax and observation argmax for observed cells
    model_argmax = pred_full.argmax(axis=-1)
    obs_argmax = np.full((map_h, map_w), -1, dtype=int)
    obs_confidence = np.zeros((map_h, map_w))

    for y in range(map_h):
        for x in range(map_w):
            obs = obs_grid[y][x]
            if not obs:
                continue
            counts = Counter(obs)
            most_common_cls, most_common_n = counts.most_common(1)[0]
            obs_argmax[y, x] = most_common_cls
            obs_confidence[y, x] = most_common_n / len(obs)

    observed_mask = obs_counts > 0
    agree = (model_argmax == obs_argmax) & observed_mask
    disagree = (model_argmax != obs_argmax) & observed_mask

    n_agree = agree.sum()
    n_disagree = disagree.sum()
    print(f"\n  Model vs Obs argmax: {n_agree} agree, {n_disagree} disagree "
          f"({100*n_disagree/(n_agree+n_disagree):.1f}% disagree)")

    # Show biggest disagreements
    if n_disagree > 0:
        disagree_cells = []
        for y in range(map_h):
            for x in range(map_w):
                if disagree[y, x]:
                    model_cls = model_argmax[y, x]
                    obs_cls = obs_argmax[y, x]
                    model_p = pred_full[y, x, model_cls]
                    obs_p = obs_confidence[y, x]
                    disagree_cells.append((y, x, model_cls, model_p, obs_cls, obs_p, obs_counts[y, x]))

        # Sort by observation confidence (high confidence disagrees are most interesting)
        disagree_cells.sort(key=lambda t: -t[5])
        print(f"  Top disagreements (obs confident but model differs):")
        for y, x, m_cls, m_p, o_cls, o_p, n_obs in disagree_cells[:15]:
            print(f"    ({x:2d},{y:2d}): model={CLASS_NAMES[m_cls]}({m_p:.2f}) "
                  f"obs={CLASS_NAMES[o_cls]}({o_p:.0%}) [{n_obs} obs]")

    return n_agree, n_disagree


def test_stronger_overlay(round_detail, seed_idx, map_w=40, map_h=40):
    """Test what happens with different Bayesian prior strengths."""
    from astar.model import (spatial_prior, compute_round_features, correct_round_features,
                             debias_transitions, _extract_settlement_stats, _load_unet_model,
                             entropy_bucket_temperature_scale, CALIBRATION_FACTORS,
                             USE_ENTROPY_TEMPS)

    # Rebuild prediction up to overlay step
    calibrated_trans = observation_calibrated_transitions(ROUND_ID, round_detail, map_w, map_h)
    debiased_trans = debias_transitions(calibrated_trans) if calibrated_trans is not None else None
    sett_stats = _extract_settlement_stats(ROUND_ID, round_detail)
    round_feats = compute_round_features(debiased_trans, round_detail, settlement_stats=sett_stats)
    round_feats = correct_round_features(round_feats)

    pred = spatial_prior(round_detail, seed_idx, map_w, map_h, round_features=round_feats)
    unet = _load_unet_model()
    if unet is not None and pred is not None:
        from astar.unet import predict_unet_with_tta
        init_grid = round_detail["initial_states"][seed_idx]["grid"]
        unet_pred = predict_unet_with_tta(unet, init_grid, map_w, map_h, round_feats)
        unet_pred = np.maximum(unet_pred, 1e-10)
        unet_pred = unet_pred / unet_pred.sum(axis=-1, keepdims=True)
        pred = 0.60 * pred + 0.40 * unet_pred
        pred = pred / pred.sum(axis=-1, keepdims=True)

    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = entropy_bucket_temperature_scale(pred)

    # Pre-overlay prediction saved
    pred_before = pred.copy()

    # Test different prior strengths
    obs_grid = build_observation_grid(ROUND_ID, seed_idx, map_w, map_h)
    max_ent = np.log(NUM_CLASSES)
    eps = 1e-10
    p_clip = np.clip(pred_before, eps, 1.0)
    entropy = -np.sum(p_clip * np.log(p_clip), axis=-1)

    configs = [
        ("Current (5-100)", 5.0, 100.0),
        ("Medium (3-50)", 3.0, 50.0),
        ("Aggressive (2-20)", 2.0, 20.0),
        ("Very aggressive (1-10)", 1.0, 10.0),
        ("Empirical direct (0.5-5)", 0.5, 5.0),
    ]

    print(f"\n  Bayesian overlay strength comparison (seed {seed_idx}):")
    for label, min_ps, max_ps in configs:
        test_pred = pred_before.copy()
        H, W, C = test_pred.shape
        for y in range(H):
            for x in range(W):
                obs = obs_grid[y][x]
                if not obs:
                    continue
                norm_ent = entropy[y, x] / max_ent
                ps = max_ps - (max_ps - min_ps) * norm_ent
                alpha = test_pred[y, x] * ps
                for cls in obs:
                    alpha[cls] += 1.0
                test_pred[y, x] = alpha / alpha.sum()
        test_pred = apply_floor(test_pred)

        # Measure how much the overlay changed predictions
        delta = np.abs(test_pred - pred_before)
        avg_shift = delta[delta > 0.001].mean() if (delta > 0.001).any() else 0
        max_shift = delta.max()
        cells_changed = (delta.max(axis=-1) > 0.01).sum()
        print(f"    {label:30s}: avg_shift={avg_shift:.4f}, max_shift={max_shift:.4f}, "
              f"cells_changed={cells_changed}")

    return pred_before


def main():
    round_detail = load_round_detail(ROUND_ID)
    if not round_detail:
        print("No round detail found for R16")
        return

    n_seeds = len(round_detail.get("initial_states", []))
    map_w = round_detail.get("map_width", 40)
    map_h = round_detail.get("map_height", 40)

    print(f"=== R16 Observation Analysis ===")
    print(f"Map: {map_w}x{map_h}, {n_seeds} seeds")

    for s in range(n_seeds):
        analyze_seed(round_detail, s, map_w, map_h)
        measure_overlay_impact(round_detail, s, map_w, map_h)

    # Test overlay strengths on seed 0 (most observed)
    test_stronger_overlay(round_detail, 0, map_w, map_h)
    # And one representative middle seed
    test_stronger_overlay(round_detail, 2, map_w, map_h)


if __name__ == "__main__":
    main()
