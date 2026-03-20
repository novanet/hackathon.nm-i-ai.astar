"""
Experiment: test expanded features and hyperparams via LORO.
Candidates:
  A) Baseline (30 features, current hyperparams)
  B) +dist_mountain, +dist_edge (32 features)
  C) Baseline with more capacity (800 trees, 20 leaves, depth 5)
  D) B + C combined
"""

import json
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
from astar.submit import score_prediction
from scipy.ndimage import distance_transform_edt

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
}

ENTROPY_WEIGHT_POWER = 0.25


def load_round_data(rid: str):
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(10):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            break
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        gts.append(np.array(gt_data["ground_truth"], dtype=np.float64))
    return detail, gts


def compute_gt_round_features(detail: dict, gts: list[np.ndarray]) -> np.ndarray:
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = total = 0
    for s in range(len(gts)):
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                gt_counts[init_cls] += gts[s][y, x]
                if init_cls == 1:
                    n_sett += 1
                total += 1
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / row_sums
    sett_density = n_sett / max(total, 1)
    ss_rate = gt_trans[1, 1]
    mean_food = 0.3 + 0.7 * ss_rate
    mean_wealth = gt_trans[0, 1] * 0.3
    mean_defense = 1.0 - gt_trans[1, 0]
    return np.array([
        gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4], gt_trans[0, 1],
        sett_density, mean_food, mean_wealth, mean_defense,
    ], dtype=np.float64)


def compute_cell_features_expanded(init_grid, map_w, map_h, round_features=None,
                                     add_mountain_dist=False, add_edge_dist=False):
    """Compute features with optional expansion."""

    # Convert grid
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])

    max_dist = float(map_w + map_h)
    settlement_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = cls_grid == 4
    port_mask = cls_grid == 2
    mtn_mask = cls_grid == 5

    dist_settlement = distance_transform_edt(~settlement_mask) if settlement_mask.any() else np.full((map_h, map_w), max_dist)
    dist_forest = distance_transform_edt(~forest_mask) if forest_mask.any() else np.full((map_h, map_w), max_dist)
    dist_port = distance_transform_edt(~port_mask) if port_mask.any() else np.full((map_h, map_w), max_dist)

    extra_feats = 0
    if add_mountain_dist:
        dist_mountain = distance_transform_edt(~mtn_mask) if mtn_mask.any() else np.full((map_h, map_w), max_dist)
        extra_feats += 1
    if add_edge_dist:
        extra_feats += 1

    sett_count_r5 = np.zeros((map_h, map_w), dtype=np.float64)
    sett_positions = np.argwhere(settlement_mask)
    for y in range(map_h):
        for x in range(map_w):
            for sy, sx in sett_positions:
                if abs(sy - y) <= 5 and abs(sx - x) <= 5:
                    sett_count_r5[y, x] += 1.0

    n_spatial = 22 + extra_feats
    n_round = len(round_features) if round_features is not None else 0
    n_feat = n_spatial + n_round
    features = np.zeros((map_h, map_w, n_feat), dtype=np.float64)

    for y in range(map_h):
        for x in range(map_w):
            idx = 0
            features[y, x, cls_grid[y, x]] = 1.0
            idx = 6

            # 3x3 neighborhood
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

            # 5x5 outer ring
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

            features[y, x, idx] = dist_settlement[y, x] / max_dist
            features[y, x, idx + 1] = dist_forest[y, x] / max_dist
            features[y, x, idx + 2] = dist_port[y, x] / max_dist
            features[y, x, idx + 3] = sett_count_r5[y, x] / max(1.0, sett_count_r5.max())
            idx += 4

            if add_mountain_dist:
                features[y, x, idx] = dist_mountain[y, x] / max_dist
                idx += 1
            if add_edge_dist:
                edge_dist = min(y, x, map_h - 1 - y, map_w - 1 - x)
                features[y, x, idx] = edge_dist / (min(map_h, map_w) / 2)
                idx += 1

    if round_features is not None:
        for i, val in enumerate(round_features):
            features[:, :, n_spatial + i] = val

    return features


def build_training_data(round_ids: dict, add_mountain_dist=False, add_edge_dist=False):
    X_parts, Y_parts, W_parts, labels = [], [], [], []
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            continue
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for seed_idx, gt in enumerate(gts):
            init_grid = detail["initial_states"][seed_idx]["grid"]
            features = compute_cell_features_expanded(
                init_grid, map_w, map_h, round_features=round_feats,
                add_mountain_dist=add_mountain_dist, add_edge_dist=add_edge_dist)
            X_parts.append(features.reshape(-1, features.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            p = np.clip(gt, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p), axis=-1).flatten()
            W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))
            labels.extend([(rnum, seed_idx)] * (map_w * map_h))
    return np.vstack(X_parts), np.vstack(Y_parts), np.concatenate(W_parts), labels


def run_loro(config_name: str, lgb_params: dict,
             add_mountain_dist: bool = False, add_edge_dist: bool = False):
    """Run LORO with given config. Returns per-round and average scores."""
    print(f"\n=== {config_name} ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)

    loro_scores = {}
    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, W_train, _ = build_training_data(
            train_ids, add_mountain_dist=add_mountain_dist, add_edge_dist=add_edge_dist)

        model = MultiOutputRegressor(
            lgb.LGBMRegressor(**lgb_params, verbosity=-1),
            n_jobs=1,
        )
        model.fit(X_train, Y_train, sample_weight=W_train)

        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features_expanded(
                init_grid, map_w, map_h, round_features=test_round_feats,
                add_mountain_dist=add_mountain_dist, add_edge_dist=add_edge_dist)
            flat_pred = model.predict(feat.reshape(-1, feat.shape[-1]))
            pred = flat_pred.reshape(map_h, map_w, NUM_CLASSES)
            pred = np.maximum(pred, 1e-10)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = np.clip(pred, 0.0001, None)
            pred = pred / pred.sum(axis=-1, keepdims=True)
            s_val = score_prediction(pred, gt)
            scores.append(s_val)

        avg = np.mean(scores)
        loro_scores[test_rnum] = avg
        print(f"  R{test_rnum}: {avg:.1f}")

    overall = np.mean(list(loro_scores.values()))
    print(f"  LORO avg: {overall:.2f}")
    return loro_scores, overall


if __name__ == "__main__":
    # A) Baseline
    base_params = dict(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
    )
    scores_a, avg_a = run_loro("A) Baseline (30 feats, current params)",
                                base_params)

    # B) +mountain_dist +edge_dist (32 features)
    scores_b, avg_b = run_loro("B) +mountain_dist +edge_dist (32 feats)",
                                base_params,
                                add_mountain_dist=True, add_edge_dist=True)

    # C) More capacity
    big_params = dict(
        n_estimators=800, max_depth=5, learning_rate=0.04,
        num_leaves=20, min_child_samples=40, subsample=0.7,
        colsample_bytree=0.6, reg_alpha=0.5, reg_lambda=0.5,
    )
    scores_c, avg_c = run_loro("C) More capacity (800 trees, depth 5)",
                                big_params)

    # D) Combined
    scores_d, avg_d = run_loro("D) Combined (32 feats + big params)",
                                big_params,
                                add_mountain_dist=True, add_edge_dist=True)

    print("\n=== SUMMARY ===")
    configs = [
        ("A) Baseline", scores_a, avg_a),
        ("B) +mountain+edge", scores_b, avg_b),
        ("C) More capacity", scores_c, avg_c),
        ("D) Combined", scores_d, avg_d),
    ]
    print(f"{'Config':25s} | {'Avg':>6s} | " + " | ".join(f"R{r}" for r in sorted(ROUND_IDS)))
    for name, scores, avg in configs:
        parts = " | ".join(f"{scores.get(r, 0):.1f}" for r in sorted(ROUND_IDS))
        delta = avg - avg_a
        print(f"{name:25s} | {avg:6.2f} | {parts} | delta={delta:+.2f}")
