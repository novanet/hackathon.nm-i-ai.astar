"""
Comprehensive pipeline optimization for R10.
Sweeps calibration, temps, blend ratio, shrinkage, floor.
All evaluated via full pipeline backtest on rounds with observations.
"""
import json, pickle
import numpy as np
from pathlib import Path
from itertools import product

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

# Rounds with observations (have simulation files) — only these go through full pipeline
OBS_ROUNDS = [2, 5, 6, 7, 8, 9]


def load_gt(rnum: int, seed: int) -> np.ndarray:
    rid = ROUND_IDS[rnum]
    rdir = DATA_DIR / f"round_{rid}"
    gt_path = rdir / f"ground_truth_s{seed}.json"
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    return np.array(data["ground_truth"], dtype=np.float64)


def load_detail(rnum: int) -> dict:
    rid = ROUND_IDS[rnum]
    rdir = DATA_DIR / f"round_{rid}"
    files = sorted(rdir.glob("round_detail_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8"))


def score_prediction(pred: np.ndarray, gt: np.ndarray) -> float:
    """Score = 100 * exp(-3 * entropy_weighted_kl)"""
    h, w, c = gt.shape
    gt_entropy = -np.sum(gt * np.log(np.maximum(gt, 1e-30)), axis=-1)
    max_entropy = np.log(c)
    dynamic = gt_entropy > 0.01
    if not dynamic.any():
        return 100.0
    kl = np.sum(gt * np.log(np.maximum(gt, 1e-30) / np.maximum(pred, 1e-30)), axis=-1)
    weights = gt_entropy / max_entropy
    weighted_kl = np.sum(kl[dynamic] * weights[dynamic]) / np.sum(weights[dynamic])
    return float(max(0, min(100, 100 * np.exp(-3 * weighted_kl))))


def pipeline_score(rnum: int, seed: int,
                   calibration: np.ndarray,
                   temps_normal: np.ndarray,
                   temps_collapse: np.ndarray,
                   floor: float,
                   collapse_thresh: float = 0.15) -> float:
    """Run full build_prediction pipeline with custom params and score vs GT."""
    from astar.model import (
        observation_calibrated_transitions, debias_transitions,
        _extract_settlement_stats, compute_round_features,
        spatial_prior, _apply_transition_matrix, HISTORICAL_TRANSITIONS,
        per_class_temperature_scale, apply_floor
    )
    from astar.replay import TERRAIN_TO_CLASS

    rid = ROUND_IDS[rnum]
    detail = load_detail(rnum)
    gt = load_gt(rnum, seed)

    # 1. Calibrated transitions + debias
    cal_trans = observation_calibrated_transitions(rid, detail)
    debiased = debias_transitions(cal_trans) if cal_trans is not None else None

    # 2. Settlement stats
    sett_stats = _extract_settlement_stats(rid, detail)

    # 3. Round features
    round_feats = compute_round_features(debiased, detail, settlement_stats=sett_stats)

    # 4. Spatial model
    pred = spatial_prior(detail, seed, round_features=round_feats)
    if pred is None:
        trans = debiased if debiased is not None else HISTORICAL_TRANSITIONS
        pred = _apply_transition_matrix(detail, seed, trans)

    # 5. Calibration
    pred = pred * calibration[np.newaxis, np.newaxis, :]
    pred = pred / pred.sum(axis=-1, keepdims=True)

    # 6. Temperature
    ss_rate = round_feats[1]
    if ss_rate < collapse_thresh:
        temps = temps_collapse
    else:
        temps = temps_normal
    pred = per_class_temperature_scale(pred, detail, seed, temps=temps)

    # 7. Floor
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    return score_prediction(pred, gt)


def evaluate_config(calibration, temps_normal, temps_collapse, floor, collapse_thresh=0.15):
    """Evaluate a config across all obs rounds, all seeds. Returns mean score."""
    scores = []
    for rnum in OBS_ROUNDS:
        detail = load_detail(rnum)
        n_seeds = len(detail.get("initial_states", []))
        for seed in range(n_seeds):
            try:
                sc = pipeline_score(rnum, seed, calibration, temps_normal,
                                    temps_collapse, floor, collapse_thresh)
                scores.append(sc)
            except Exception as e:
                pass
    return np.mean(scores) if scores else 0.0


def evaluate_per_round(calibration, temps_normal, temps_collapse, floor, collapse_thresh=0.15):
    """Evaluate per round. Returns dict of round -> avg score."""
    results = {}
    for rnum in OBS_ROUNDS:
        detail = load_detail(rnum)
        n_seeds = len(detail.get("initial_states", []))
        seeds = []
        for seed in range(n_seeds):
            try:
                sc = pipeline_score(rnum, seed, calibration, temps_normal,
                                    temps_collapse, floor, collapse_thresh)
                seeds.append(sc)
            except:
                pass
        results[rnum] = np.mean(seeds) if seeds else 0.0
    return results


if __name__ == "__main__":
    import time
    
    # Force model load once
    from astar.model import load_spatial_model
    load_spatial_model()
    
    # Current baseline
    print("=== CURRENT BASELINE ===")
    base_cal = np.array([1.03, 0.92, 0.92, 0.92, 1.0, 1.0])
    base_temps_n = np.array([1.20, 1.0, 1.0, 1.0, 1.20, 1.0])
    base_temps_c = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
    base_floor = 0.0002
    
    base_results = evaluate_per_round(base_cal, base_temps_n, base_temps_c, base_floor)
    base_avg = np.mean(list(base_results.values()))
    for rnum, sc in sorted(base_results.items()):
        w = 1.05 ** rnum
        print(f"  R{rnum}: {sc:.2f} (weighted {sc*w:.1f})")
    print(f"  Avg: {base_avg:.2f}")
    print()
    
    # ── SWEEP 1: Calibration factors ──
    print("=== SWEEP: CALIBRATION FACTORS ===")
    best_cal = base_cal.copy()
    best_cal_score = base_avg
    
    # Sweep E calibration
    for e_cal in [0.98, 1.00, 1.02, 1.03, 1.05, 1.07, 1.10]:
        cal = base_cal.copy()
        cal[0] = e_cal
        sc = evaluate_config(cal, base_temps_n, base_temps_c, base_floor)
        marker = " ***" if sc > best_cal_score else ""
        print(f"  E={e_cal:.2f}: {sc:.3f}{marker}")
        if sc > best_cal_score:
            best_cal_score = sc
            best_cal = cal.copy()
    
    # Sweep S/P/R calibration
    for spr in [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.00]:
        cal = best_cal.copy()
        cal[1] = cal[2] = cal[3] = spr
        sc = evaluate_config(cal, base_temps_n, base_temps_c, base_floor)
        marker = " ***" if sc > best_cal_score else ""
        print(f"  S/P/R={spr:.2f}: {sc:.3f}{marker}")
        if sc > best_cal_score:
            best_cal_score = sc
            best_cal = cal.copy()
    
    # Sweep F calibration
    for f_cal in [0.95, 0.98, 1.00, 1.02, 1.05]:
        cal = best_cal.copy()
        cal[4] = f_cal
        sc = evaluate_config(cal, base_temps_n, base_temps_c, base_floor)
        marker = " ***" if sc > best_cal_score else ""
        print(f"  F={f_cal:.2f}: {sc:.3f}{marker}")
        if sc > best_cal_score:
            best_cal_score = sc
            best_cal = cal.copy()
    
    print(f"\n  Best calibration: {best_cal} = {best_cal_score:.3f}")
    print()
    
    # ── SWEEP 2: Normal temps ──
    print("=== SWEEP: NORMAL TEMPS ===")
    best_temps_n = base_temps_n.copy()
    best_temps_score = best_cal_score
    
    # Sweep E/F softening temp
    for ef_t in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]:
        temps = base_temps_n.copy()
        temps[0] = temps[4] = ef_t
        sc = evaluate_config(best_cal, temps, base_temps_c, base_floor)
        marker = " ***" if sc > best_temps_score else ""
        print(f"  E/F T={ef_t:.2f}: {sc:.3f}{marker}")
        if sc > best_temps_score:
            best_temps_score = sc
            best_temps_n = temps.copy()
    
    # Sweep S temp
    for s_t in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]:
        temps = best_temps_n.copy()
        temps[1] = s_t
        sc = evaluate_config(best_cal, temps, base_temps_c, base_floor)
        marker = " ***" if sc > best_temps_score else ""
        print(f"  S T={s_t:.2f}: {sc:.3f}{marker}")
        if sc > best_temps_score:
            best_temps_score = sc
            best_temps_n = temps.copy()
    
    # Sweep P temp
    for p_t in [0.80, 0.90, 1.00, 1.10, 1.20]:
        temps = best_temps_n.copy()
        temps[2] = p_t
        sc = evaluate_config(best_cal, temps, base_temps_c, base_floor)
        marker = " ***" if sc > best_temps_score else ""
        print(f"  P T={p_t:.2f}: {sc:.3f}{marker}")
        if sc > best_temps_score:
            best_temps_score = sc
            best_temps_n = temps.copy()
    
    print(f"\n  Best normal temps: {best_temps_n} = {best_temps_score:.3f}")
    print()
    
    # ── SWEEP 3: Collapse temps ──
    print("=== SWEEP: COLLAPSE TEMPS ===")
    best_temps_c = base_temps_c.copy()
    best_tc_score = best_temps_score
    
    for ct in [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        temps = np.array([ct, ct, ct, 1.0, ct, 1.0])
        sc = evaluate_config(best_cal, best_temps_n, temps, base_floor)
        marker = " ***" if sc > best_tc_score else ""
        print(f"  Collapse T={ct:.2f}: {sc:.3f}{marker}")
        if sc > best_tc_score:
            best_tc_score = sc
            best_temps_c = temps.copy()
    
    print(f"\n  Best collapse temps: {best_temps_c} = {best_tc_score:.3f}")
    print()
    
    # ── SWEEP 4: Collapse threshold ──
    print("=== SWEEP: COLLAPSE THRESHOLD ===")
    best_thresh = 0.15
    best_th_score = best_tc_score
    
    for th in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sc = evaluate_config(best_cal, best_temps_n, best_temps_c, base_floor, th)
        marker = " ***" if sc > best_th_score else ""
        print(f"  thresh={th:.2f}: {sc:.3f}{marker}")
        if sc > best_th_score:
            best_th_score = sc
            best_thresh = th
    
    print(f"\n  Best threshold: {best_thresh} = {best_th_score:.3f}")
    print()
    
    # ── SWEEP 5: Floor ──
    print("=== SWEEP: PROB FLOOR ===")
    best_floor = base_floor
    best_fl_score = best_th_score
    
    for fl in [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.005]:
        sc = evaluate_config(best_cal, best_temps_n, best_temps_c, fl, best_thresh)
        marker = " ***" if sc > best_fl_score else ""
        print(f"  floor={fl:.4f}: {sc:.3f}{marker}")
        if sc > best_fl_score:
            best_fl_score = sc
            best_floor = fl
    
    print(f"\n  Best floor: {best_floor} = {best_fl_score:.3f}")
    print()
    
    # ── Final summary ──
    print("=" * 60)
    print("FINAL OPTIMIZED CONFIG")
    print("=" * 60)
    print(f"  CALIBRATION_FACTORS = {list(best_cal)}")
    print(f"  PER_CLASS_TEMPS     = {list(best_temps_n)}")
    print(f"  COLLAPSE_TEMPS      = {list(best_temps_c)}")
    print(f"  COLLAPSE_THRESH     = {best_thresh}")
    print(f"  PROB_FLOOR          = {best_floor}")
    print()
    
    final_results = evaluate_per_round(best_cal, best_temps_n, best_temps_c, best_floor, best_thresh)
    final_avg = np.mean(list(final_results.values()))
    print("Per-round comparison (base → optimized):")
    total_gain = 0
    for rnum in OBS_ROUNDS:
        old = base_results[rnum]
        new = final_results[rnum]
        w = 1.05 ** rnum
        print(f"  R{rnum}: {old:.2f} → {new:.2f} ({new-old:+.2f}) weighted: {old*w:.1f} → {new*w:.1f}")
        total_gain += new - old
    print(f"\n  Avg: {base_avg:.2f} → {final_avg:.2f} ({final_avg-base_avg:+.2f})")
    print(f"  Total gain across {len(OBS_ROUNDS)} rounds: {total_gain:+.2f}")
