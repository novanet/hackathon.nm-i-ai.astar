"""Compute shrinkage/debiasing matrix: GT transitions vs observed transitions across all rounds."""

import json
import numpy as np
from pathlib import Path
from astar.replay import TERRAIN_TO_CLASS, CLASS_NAMES, NUM_CLASSES, load_simulations

DATA_DIR = Path("data")
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
}


def compute_gt_transitions(rid: str) -> np.ndarray | None:
    """Compute transition matrix from GT probability distributions."""
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for s in range(len(states)):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            continue
        gt = np.array(json.loads(gt_path.read_text(encoding="utf-8"))["ground_truth"], dtype=np.float64)
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                gt_counts[init_cls] += gt[y, x]
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    return gt_counts / row_sums


def compute_obs_transitions(rid: str) -> tuple[np.ndarray | None, int]:
    """Compute transition matrix from simulation observations."""
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, 0
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    obs_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    total_obs = 0
    for seed_idx in range(len(states)):
        sims = load_simulations(rid, seed_idx)
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
        return None, 0
    row_sums = np.maximum(obs_counts.sum(axis=1, keepdims=True), 1.0)
    return obs_counts / row_sums, total_obs


def compute_round_level_features(rid: str) -> dict | None:
    """Compute round-level features from GT transition matrix + settlement stats."""
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    gt_trans = compute_gt_transitions(rid)
    if gt_trans is None:
        return None

    n_settlements = 0
    total_cells = 0
    for s in range(len(states)):
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                if cls == 1:
                    n_settlements += 1
                total_cells += 1

    return {
        "ee": gt_trans[0, 0],  # Empty→Empty
        "ss": gt_trans[1, 1],  # Settlement→Settlement
        "ff": gt_trans[4, 4],  # Forest→Forest
        "es": gt_trans[0, 1],  # Empty→Settlement
        "sett_density": n_settlements / max(total_cells, 1),
    }


if __name__ == "__main__":
    all_gt_trans = []
    all_obs_trans = []

    for rnum, rid in sorted(ROUND_IDS.items()):
        gt_trans = compute_gt_transitions(rid)
        obs_trans, n_obs = compute_obs_transitions(rid)

        if gt_trans is None:
            print(f"R{rnum}: no GT")
            continue
        if obs_trans is None:
            print(f"R{rnum}: no observations")
            all_gt_trans.append(gt_trans)
            all_obs_trans.append(None)
            continue

        all_gt_trans.append(gt_trans)
        all_obs_trans.append(obs_trans)

        print(f"R{rnum}: {n_obs} obs")
        for from_c in range(NUM_CLASSES):
            for to_c in range(NUM_CLASSES):
                gt_v = gt_trans[from_c, to_c]
                obs_v = obs_trans[from_c, to_c]
                if gt_v > 0.01 and obs_v > 0.01:
                    ratio = gt_v / obs_v
                    if abs(ratio - 1.0) > 0.05:
                        print(f"  {CLASS_NAMES[from_c]}->{CLASS_NAMES[to_c]}: gt={gt_v:.3f} obs={obs_v:.3f} ratio={ratio:.3f}")

    # Compute shrinkage matrix
    print("\n=== SHRINKAGE ANALYSIS ===")
    n_valid = sum(1 for o in all_obs_trans if o is not None)
    print(f"Rounds with GT+obs data: {n_valid}")

    shrinkage = np.ones((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for from_c in range(NUM_CLASSES):
        for to_c in range(NUM_CLASSES):
            ratios = []
            for i in range(len(all_gt_trans)):
                if all_obs_trans[i] is None:
                    continue
                gt_v = all_gt_trans[i][from_c, to_c]
                obs_v = all_obs_trans[i][from_c, to_c]
                if obs_v > 0.005 and gt_v > 0.005:
                    ratios.append(gt_v / obs_v)
            if ratios:
                shrinkage[from_c, to_c] = np.median(ratios)

    header = f"{'':>12}"
    for c in range(NUM_CLASSES):
        header += f" {CLASS_NAMES[c]:>10}"
    print(header)
    for from_cls in range(NUM_CLASSES):
        row = f"{CLASS_NAMES[from_cls]:>12}"
        for to_cls in range(NUM_CLASSES):
            row += f" {shrinkage[from_cls, to_cls]:>10.4f}"
        print(row)

    # Print code-ready
    print("\n=== CODE-READY SHRINKAGE MATRIX ===")
    print("SHRINKAGE_MATRIX = np.array([")
    for from_cls in range(NUM_CLASSES):
        row = ", ".join(f"{shrinkage[from_cls, to_cls]:.4f}" for to_cls in range(NUM_CLASSES))
        print(f"    [{row}],  # {CLASS_NAMES[from_cls]} ->")
    print("])")

    # Also compute round-level features for each round
    print("\n=== ROUND-LEVEL FEATURES (from GT) ===")
    for rnum, rid in sorted(ROUND_IDS.items()):
        feats = compute_round_level_features(rid)
        if feats:
            print(f"  R{rnum}: E->E={feats['ee']:.3f} S->S={feats['ss']:.3f} "
                  f"F->F={feats['ff']:.3f} E->S={feats['es']:.3f} "
                  f"sett_dens={feats['sett_density']:.3f}")
