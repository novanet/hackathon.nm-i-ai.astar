"""
Quick LORO test: LightGBM + XGBoost ensemble vs LightGBM alone.
"""
import json
import warnings
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, testing with 2x LightGBM diversity ensemble")

from astar.model import (
    compute_cell_features, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, NUM_CLASSES, CALIBRATION_FACTORS, PER_CLASS_TEMPS,
    per_class_temperature_scale,
)
from astar.replay import TERRAIN_TO_CLASS, load_round_detail
from astar.submit import score_prediction

ENTROPY_WEIGHT_POWER = 0.25

ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
}


def compute_gt_round_features(detail, gts):
    """Compute round features from ground truth (for training)."""
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s_idx in range(n_seeds):
        init_grid = detail["initial_states"][s_idx]["grid"]
        gt = gts[s_idx]
        gt_cls = gt.argmax(axis=-1)
        for y in range(map_h):
            for x in range(map_w):
                ic = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                fc = gt_cls[y, x]
                trans_counts[ic, fc] += 1
    row_sums = np.maximum(trans_counts.sum(axis=1, keepdims=True), 1e-10)
    trans = trans_counts / row_sums

    n_sett = 0
    total = 0
    for s in detail["initial_states"]:
        for y in range(map_h):
            for x in range(map_w):
                if TERRAIN_TO_CLASS.get(s["grid"][y][x], 0) == 1:
                    n_sett += 1
                total += 1
    sett_density = n_sett / max(total, 1)

    ss_rate = trans[1, 1]
    es_rate = trans[0, 1]
    se_rate = trans[1, 0]
    mean_food = 0.3 + 0.7 * ss_rate
    mean_wealth = es_rate * 0.3
    mean_defense = 1.0 - se_rate

    return np.array([
        trans[0, 0], trans[1, 1], trans[4, 4], trans[0, 1],
        sett_density, mean_food, mean_wealth, mean_defense,
    ], dtype=np.float64)


def build_training_data_multi(round_ids: dict, compute_weights: bool = False):
    all_X, all_Y, all_W = [], [], []
    for rnum, rid in sorted(round_ids.items()):
        data_dir = Path("data") / f"round_{rid}"
        detail_files = sorted(data_dir.glob("round_detail_*.json"))
        if not detail_files:
            continue
        detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        gts = []
        for s in range(len(detail.get("initial_states", []))):
            gt_path = data_dir / f"ground_truth_s{s}.json"
            if not gt_path.exists():
                break
            gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
            gts.append(np.array(gt_data["ground_truth"], dtype=np.float64))
        if not gts:
            continue
        round_feats = compute_gt_round_features(detail, gts)
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
            flat_feat = feat.reshape(-1, feat.shape[-1])
            flat_gt = gt.reshape(-1, NUM_CLASSES)
            all_X.append(flat_feat)
            all_Y.append(flat_gt)
            if compute_weights:
                entropy = -np.sum(flat_gt * np.log(np.maximum(flat_gt, 1e-30)), axis=1)
                weights = np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER)
                all_W.append(weights)

    X = np.concatenate(all_X)
    Y = np.concatenate(all_Y)
    W = np.concatenate(all_W) if all_W else np.ones(len(X))
    return X, Y, None, W


def load_all_data():
    all_data = {}
    for rnum, rid in ROUND_IDS.items():
        data_dir = Path("data") / f"round_{rid}"
        detail_files = sorted(data_dir.glob("round_detail_*.json"))
        if not detail_files:
            continue
        detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
        gts = []
        for s in range(len(detail.get("initial_states", []))):
            gt_path = data_dir / f"ground_truth_s{s}.json"
            if not gt_path.exists():
                break
            gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
            gts.append(np.array(gt_data["ground_truth"], dtype=np.float64))
        if gts:
            all_data[rnum] = (rid, detail, gts)
    return all_data


def make_lgb_model():
    return MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
            verbosity=-1,
        ), n_jobs=1,
    )


def make_xgb_model():
    if HAS_XGB:
        return MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=1.0, verbosity=0,
            ), n_jobs=1,
        )
    else:
        # Diversity: different LGB hyperparams
        return MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=800, max_depth=5, learning_rate=0.03,
                num_leaves=20, min_child_samples=30, subsample=0.8,
                colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=2.0,
                verbosity=-1,
            ), n_jobs=1,
        )


def score_with_post_processing(pred, gt, round_detail, seed_idx, map_w, map_h):
    """Apply calibration + temps + floor (same as production pipeline)."""
    pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred = per_class_temperature_scale(pred, round_detail, seed_idx,
                                       temps=PER_CLASS_TEMPS, map_w=map_w, map_h=map_h)
    pred = apply_floor(pred)
    return score_prediction(pred, gt)


def main():
    print("=== ENSEMBLE LORO TEST ===")
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} rounds\n")

    lgb_scores = {}
    ens_scores = {}
    blend_weights = [0.5, 0.6, 0.7]

    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, _, W_train = build_training_data_multi(train_ids, compute_weights=True)

        # Train LGB
        m_lgb = make_lgb_model()
        m_lgb.fit(X_train, Y_train, sample_weight=W_train)

        # Train XGB/diverse
        m_xgb = make_xgb_model()
        if HAS_XGB:
            m_xgb.fit(X_train, Y_train, sample_weight=W_train)
        else:
            m_xgb.fit(X_train, Y_train, sample_weight=W_train)

        # Evaluate on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        lgb_round = []
        ens_round = {w: [] for w in blend_weights}

        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h, round_features=test_round_feats)
            flat = feat.reshape(-1, feat.shape[-1])

            # LGB only
            p_lgb = m_lgb.predict(flat).reshape(map_h, map_w, NUM_CLASSES)
            p_lgb = np.maximum(p_lgb, 1e-10)
            p_lgb = p_lgb / p_lgb.sum(axis=-1, keepdims=True)

            # XGB only
            p_xgb = m_xgb.predict(flat).reshape(map_h, map_w, NUM_CLASSES)
            p_xgb = np.maximum(p_xgb, 1e-10)
            p_xgb = p_xgb / p_xgb.sum(axis=-1, keepdims=True)

            sc_lgb = score_with_post_processing(p_lgb, gt, test_detail, s, map_w, map_h)
            lgb_round.append(sc_lgb)

            for w in blend_weights:
                p_ens = w * p_lgb + (1 - w) * p_xgb
                p_ens = p_ens / p_ens.sum(axis=-1, keepdims=True)
                sc_ens = score_with_post_processing(p_ens, gt, test_detail, s, map_w, map_h)
                ens_round[w].append(sc_ens)

        lgb_avg = np.mean(lgb_round)
        lgb_scores[test_rnum] = lgb_avg
        for w in blend_weights:
            ens_avg = np.mean(ens_round[w])
            if w not in ens_scores:
                ens_scores[w] = {}
            ens_scores[w][test_rnum] = ens_avg

        best_w = max(blend_weights, key=lambda w: np.mean(ens_round[w]))
        best_ens = np.mean(ens_round[best_w])
        print(f"  R{test_rnum}: LGB={lgb_avg:.2f}  Ens(best w={best_w})={best_ens:.2f}  d={best_ens-lgb_avg:+.2f}")

    print(f"\n=== SUMMARY ===")
    lgb_mean = np.mean(list(lgb_scores.values()))
    print(f"  LGB-only LORO: {lgb_mean:.2f}")
    for w in blend_weights:
        ens_mean = np.mean(list(ens_scores[w].values()))
        diff = ens_mean - lgb_mean
        print(f"  Ensemble w={w}: {ens_mean:.2f}  (d={diff:+.2f})")

    best_overall_w = max(blend_weights, key=lambda w: np.mean(list(ens_scores[w].values())))
    best_overall = np.mean(list(ens_scores[best_overall_w].values()))
    print(f"\n  Best: w={best_overall_w}, LORO={best_overall:.2f} (d={best_overall-lgb_mean:+.2f})")
    print(f"  {'XGBoost' if HAS_XGB else 'Diverse LGB'} used as second model")


if __name__ == "__main__":
    main()
