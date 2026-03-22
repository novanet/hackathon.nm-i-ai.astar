"""Fine-grained sweep around the best overlay settings from initial backtest."""
import json
import numpy as np
from pathlib import Path
from astar.replay import (load_round_detail, build_observation_grid,
                          TERRAIN_TO_CLASS, NUM_CLASSES)
from astar.model import (apply_floor, observation_calibrated_transitions, debias_transitions,
                          _extract_settlement_stats, compute_round_features,
                          correct_round_features, spatial_prior, _load_unet_model,
                          entropy_bucket_temperature_scale, CALIBRATION_FACTORS,
                          USE_ENTROPY_TEMPS)
from astar.submit import score_prediction

DATA_DIR = Path("data")

ROUND_IDS = {
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
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


def load_ground_truth(round_id, seed_index, map_w, map_h):
    rdir = DATA_DIR / f"round_{round_id}"
    gt_file = rdir / f"ground_truth_s{seed_index}.json"
    if not gt_file.exists():
        return None
    data = json.loads(gt_file.read_text(encoding="utf-8"))
    gt_probs = data.get("ground_truth")
    if gt_probs is not None:
        gt_arr = np.array(gt_probs)
        if gt_arr.ndim == 3:
            return gt_arr.argmax(axis=-1)
    return None


def apply_overlay(pred, obs_grid, min_ps, max_ps, map_w, map_h):
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


def build_pre_overlay(rid, detail, seed_idx, map_w, map_h):
    calibrated_trans = observation_calibrated_transitions(rid, detail, map_w, map_h)
    debiased_trans = debias_transitions(calibrated_trans) if calibrated_trans is not None else None
    sett_stats = _extract_settlement_stats(rid, detail)
    round_feats = compute_round_features(debiased_trans, detail, settlement_stats=sett_stats)
    round_feats = correct_round_features(round_feats)

    pred = spatial_prior(detail, seed_idx, map_w, map_h, round_features=round_feats)
    unet = _load_unet_model()
    if unet is not None and pred is not None:
        from astar.unet import predict_unet_with_tta
        init_grid = detail["initial_states"][seed_idx]["grid"]
        unet_pred = predict_unet_with_tta(unet, init_grid, map_w, map_h, round_feats)
        unet_pred = np.maximum(unet_pred, 1e-10)
        unet_pred = unet_pred / unet_pred.sum(axis=-1, keepdims=True)
        pred = 0.60 * pred + 0.40 * unet_pred
        pred = pred / pred.sum(axis=-1, keepdims=True)

    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)
    if USE_ENTROPY_TEMPS:
        pred = entropy_bucket_temperature_scale(pred)
    return pred


def main():
    # Fine-grained configs around the best settings
    configs = [
        ("Current (5-100)", 5.0, 100.0),
        ("(1-10)", 1.0, 10.0),
        ("(0.5-5)", 0.5, 5.0),
        ("(0.3-3)", 0.3, 3.0),
        ("(0.2-2)", 0.2, 2.0),
        ("(0.1-1)", 0.1, 1.0),
        ("(0.5-3)", 0.5, 3.0),
        ("(0.3-5)", 0.3, 5.0),
        ("(1-5)", 1.0, 5.0),
    ]

    print("=== FINE-GRAINED OVERLAY SWEEP ===\n")
    header = f"{'Round':>6}" + "".join(f"  {label:>12}" for label, _, _ in configs)
    print(header)

    all_scores = {label: [] for label, _, _ in configs}

    for rnd_num in sorted(ROUND_IDS.keys()):
        rid = ROUND_IDS[rnd_num]
        detail = load_round_detail(rid)
        if not detail:
            continue

        n_seeds = len(detail.get("initial_states", []))
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)

        round_scores = {}
        for label, min_ps, max_ps in configs:
            seed_scores = []
            for s in range(n_seeds):
                gt = load_ground_truth(rid, s, map_w, map_h)
                if gt is None:
                    continue
                pred = build_pre_overlay(rid, detail, s, map_w, map_h)
                obs_grid = build_observation_grid(rid, s, map_w, map_h)
                pred = apply_overlay(pred, obs_grid, min_ps, max_ps, map_w, map_h)
                gt_onehot = np.zeros((map_h, map_w, NUM_CLASSES))
                gt_onehot[np.arange(map_h)[:, None], np.arange(map_w)[None, :], gt] = 1.0
                seed_scores.append(score_prediction(pred, gt_onehot))
            if seed_scores:
                avg = np.mean(seed_scores)
                round_scores[label] = avg
                all_scores[label].append(avg)

        if round_scores:
            vals = "  ".join(f"{round_scores.get(l, 0):12.2f}" for l, _, _ in configs)
            print(f"R{rnd_num:>4}  {vals}")

    print(f"\n{'AVG':>6}", end="")
    best_avg = max(np.mean(v) for v in all_scores.values() if v)
    for label, _, _ in configs:
        scores = all_scores[label]
        if scores:
            avg = np.mean(scores)
            marker = " *" if abs(avg - best_avg) < 0.01 else "  "
            print(f"  {avg:10.2f}{marker}", end="")
    print()

    # Wins
    n_rounds = len(all_scores[configs[0][0]])
    print(f"\nPer-round wins:")
    for label, _, _ in configs:
        wins = sum(1 for i in range(n_rounds)
                   if all_scores[label][i] == max(all_scores[l][i] for l, _, _ in configs))
        print(f"  {label:20s}: {wins}/{n_rounds}")


if __name__ == "__main__":
    main()
