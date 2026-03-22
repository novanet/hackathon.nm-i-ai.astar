"""Backtest Bayesian overlay strength on past rounds with ground truth.
Tests different prior_strength settings and measures impact on score."""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from astar.replay import (load_round_detail, load_simulations, load_analysis,
                          TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES,
                          build_observation_grid)
from astar.model import (build_prediction, apply_floor, PROB_FLOOR,
                          observation_calibrated_transitions, debias_transitions,
                          _extract_settlement_stats, compute_round_features,
                          correct_round_features, spatial_prior, _load_unet_model,
                          entropy_bucket_temperature_scale, CALIBRATION_FACTORS,
                          USE_ENTROPY_TEMPS)
from astar.submit import score_prediction

DATA_DIR = Path("data")

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
}


def load_ground_truth(round_id: str, seed_index: int, map_w: int, map_h: int):
    """Load ground truth as (H, W) class grid."""
    rdir = DATA_DIR / f"round_{round_id}"
    gt_file = rdir / f"ground_truth_s{seed_index}.json"
    if not gt_file.exists():
        return None

    data = json.loads(gt_file.read_text(encoding="utf-8"))
    # GT stored as probability tensor under 'ground_truth' key (40x40x6 one-hot)
    gt_probs = data.get("ground_truth")
    if gt_probs is not None and isinstance(gt_probs, list):
        gt_arr = np.array(gt_probs)
        if gt_arr.ndim == 3:  # (H, W, 6) probability tensor
            return gt_arr.argmax(axis=-1)
        elif gt_arr.ndim == 2:  # (H, W) class grid
            return gt_arr.astype(int)

    # Fallback: try grid key with terrain codes
    gt_grid = data.get("grid") or data.get("ground_truth_grid")
    if gt_grid is None:
        return None
    gt = np.zeros((map_h, map_w), dtype=int)
    for y in range(min(map_h, len(gt_grid))):
        for x in range(min(map_w, len(gt_grid[y]))):
            gt[y, x] = TERRAIN_TO_CLASS.get(gt_grid[y][x], 0)
    return gt


def apply_overlay(pred: np.ndarray, obs_grid, min_ps: float, max_ps: float,
                  map_w: int, map_h: int):
    """Apply Bayesian overlay with given prior strengths."""
    result = pred.copy()
    H, W, C = pred.shape
    max_ent = np.log(C)
    eps = 1e-10
    p = np.clip(pred, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)

    for y in range(H):
        for x in range(W):
            obs = obs_grid[y][x]
            if not obs:
                continue
            norm_ent = entropy[y, x] / max_ent
            ps = max_ps - (max_ps - min_ps) * norm_ent
            alpha = result[y, x] * ps
            for cls in obs:
                alpha[cls] += 1.0
            result[y, x] = alpha / alpha.sum()

    return apply_floor(result)


def build_pre_overlay_pred(round_id, round_detail, seed_idx, map_w, map_h):
    """Build prediction up to but NOT including the Bayesian overlay."""
    calibrated_trans = observation_calibrated_transitions(round_id, round_detail, map_w, map_h)
    debiased_trans = debias_transitions(calibrated_trans) if calibrated_trans is not None else None
    sett_stats = _extract_settlement_stats(round_id, round_detail)
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

    if pred is None:
        from astar.model import _apply_transition_matrix, HISTORICAL_TRANSITIONS
        trans = debiased_trans if debiased_trans is not None else HISTORICAL_TRANSITIONS
        pred = _apply_transition_matrix(round_detail, seed_idx, trans, map_w, map_h)

    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)

    if USE_ENTROPY_TEMPS:
        pred = entropy_bucket_temperature_scale(pred)

    return pred


def main():
    # Overlay configs to test
    configs = [
        ("No overlay", None, None),
        ("Current (5-100)", 5.0, 100.0),
        ("Medium (3-50)", 3.0, 50.0),
        ("Aggressive (2-20)", 2.0, 20.0),
        ("V.Aggressive (1-10)", 1.0, 10.0),
        ("Empirical (0.5-5)", 0.5, 5.0),
    ]

    print("=== BAYESIAN OVERLAY STRENGTH BACKTEST ===\n")
    print(f"{'Round':>6}  {'No overlay':>12}  {'Cur(5-100)':>12}  {'Med(3-50)':>12}  "
          f"{'Agg(2-20)':>12}  {'VAgg(1-10)':>12}  {'Emp(0.5-5)':>12}")

    all_scores = {label: [] for label, _, _ in configs}

    for rnd_num in sorted(ROUND_IDS.keys()):
        rid = ROUND_IDS[rnd_num]
        detail = load_round_detail(rid)
        if detail is None:
            continue

        n_seeds = len(detail.get("initial_states", []))
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)

        # Check if we have ground truth and observations for this round
        has_gt = False
        has_obs = False
        for s in range(n_seeds):
            gt = load_ground_truth(rid, s, map_w, map_h)
            if gt is not None:
                has_gt = True
            sims = load_simulations(rid, s)
            if sims:
                has_obs = True

        if not has_gt or not has_obs:
            continue

        # Score each config
        round_scores = {}
        for label, min_ps, max_ps in configs:
            seed_scores = []
            for s in range(n_seeds):
                gt = load_ground_truth(rid, s, map_w, map_h)
                if gt is None:
                    continue

                # Build pre-overlay prediction
                pred = build_pre_overlay_pred(rid, detail, s, map_w, map_h)

                if min_ps is not None:
                    obs_grid = build_observation_grid(rid, s, map_w, map_h)
                    pred = apply_overlay(pred, obs_grid, min_ps, max_ps, map_w, map_h)
                else:
                    pred = apply_floor(pred)

                # Convert class indices to one-hot for scorer
                gt_onehot = np.zeros((map_h, map_w, NUM_CLASSES))
                gt_onehot[np.arange(map_h)[:, None], np.arange(map_w)[None, :], gt] = 1.0
                score = score_prediction(pred, gt_onehot)
                seed_scores.append(score)

            if seed_scores:
                avg = np.mean(seed_scores)
                round_scores[label] = avg
                all_scores[label].append(avg)

        if round_scores:
            vals = "  ".join(f"{round_scores.get(label, 0):12.2f}" for label, _, _ in configs)
            print(f"R{rnd_num:>4}  {vals}")

    # Summary
    print(f"\n{'AVG':>6}", end="")
    for label, _, _ in configs:
        scores = all_scores[label]
        if scores:
            avg = np.mean(scores)
            marker = " *" if avg == max(np.mean(all_scores[l]) for l, _, _ in configs if all_scores[l]) else "  "
            print(f"  {avg:10.2f}{marker}", end="")
    print()

    # Per-config wins
    print(f"\nWins analysis:")
    n_rounds = len(all_scores[configs[0][0]])
    for label, _, _ in configs:
        wins = 0
        for i in range(n_rounds):
            scores_at_i = [(l, all_scores[l][i]) for l, _, _ in configs if len(all_scores[l]) > i]
            best = max(scores_at_i, key=lambda t: t[1])
            if best[0] == label:
                wins += 1
        print(f"  {label:25s}: {wins}/{n_rounds} wins")


if __name__ == "__main__":
    main()
