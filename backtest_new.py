"""Backtest the new pipeline (spatial+distance features, transition calibration)."""
import json
import numpy as np
from pathlib import Path
from astar.model import (
    build_prediction, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, spatial_prior, observation_calibrated_transitions
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES
import astar.model as m

# Force reload spatial model
m._spatial_model = None

DATA_DIR = Path("data")
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
}


def load_round(rid: str):
    rdir = DATA_DIR / f"round_{rid}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if gt_path.exists():
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts


def main():
    print("=" * 70)
    print("FULL PIPELINE BACKTEST: New model vs baselines")
    print("=" * 70)
    print()

    round_weights = {r: 1.05 ** r for r in range(1, 10)}
    total_new = 0
    total_hist = 0
    total_spatial_only = 0

    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round(rid)
        if not gts:
            continue
        n_seeds = len(gts)
        w = round_weights[rnum]

        scores_pipeline = []
        scores_hist = []
        scores_spatial = []

        for s in range(n_seeds):
            # New full pipeline
            pred = build_prediction(rid, detail, s)
            scores_pipeline.append(score_prediction(pred, gts[s]))

            # Historical transitions only
            hpred = _apply_transition_matrix(detail, s, HISTORICAL_TRANSITIONS)
            hpred = apply_floor(hpred)
            scores_hist.append(score_prediction(hpred, gts[s]))

            # Spatial model only
            sp = spatial_prior(detail, s)
            if sp is not None:
                sp = apply_floor(sp)
                scores_spatial.append(score_prediction(sp, gts[s]))
            else:
                scores_spatial.append(0.0)

        p_avg = np.mean(scores_pipeline)
        h_avg = np.mean(scores_hist)
        s_avg = np.mean(scores_spatial)
        total_new += w * p_avg
        total_hist += w * h_avg
        total_spatial_only += w * s_avg

        print(f"R{rnum} (w={w:.4f}):")
        print(f"  Pipeline:     {p_avg:.1f} (weighted={w*p_avg:.1f})  seeds: {[f'{s:.1f}' for s in scores_pipeline]}")
        print(f"  Hist trans:   {h_avg:.1f} (weighted={w*h_avg:.1f})")
        print(f"  Spatial only: {s_avg:.1f} (weighted={w*s_avg:.1f})")
        print()

    print("=" * 70)
    print("TOTALS (if submitted every round):")
    print(f"  New pipeline:     {total_new:.1f}")
    print(f"  Hist transitions: {total_hist:.1f}")
    print(f"  Spatial only:     {total_spatial_only:.1f}")
    print(f"  Best team:         113.9")
    print()

    # R5 with observations — what we would have scored
    print("=" * 70)
    print("R5 WITH OBSERVATIONS (simulating live submission)")
    print("=" * 70)
    rid5 = ROUND_IDS[5]
    detail5, gts5 = load_round(rid5)
    cal = observation_calibrated_transitions(rid5, detail5)
    if cal is not None:
        print("Calibrated transitions from observations:")
        from astar.replay import CLASS_NAMES
        for cls_idx in [0, 1, 4]:
            row = cal[cls_idx]
            names = ["Emp", "Set", "Prt", "Rui", "For", "Mtn"]
            row_str = ", ".join(f"{names[i]}:{v:.3f}" for i, v in enumerate(row) if v > 0.005)
            print(f"  {names[cls_idx]:>3} -> {row_str}")
        print()
        scores_cal = []
        for s in range(len(gts5)):
            pred = build_prediction(rid5, detail5, s)
            sc = score_prediction(pred, gts5[s])
            scores_cal.append(sc)
            print(f"  Seed {s}: {sc:.1f}")
        avg_cal = np.mean(scores_cal)
        print(f"  Average: {avg_cal:.1f} (was 75.38 with old pipeline)")
    else:
        print("No observations for R5.")

    # R5 blend sweep
    print()
    print("R5 spatial+histogram blend sweep (no observations):")
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        scores = []
        for s in range(len(gts5)):
            sp = spatial_prior(detail5, s)
            tp = _apply_transition_matrix(detail5, s, HISTORICAL_TRANSITIONS)
            blended = alpha * sp + (1 - alpha) * tp
            blended = blended / blended.sum(axis=-1, keepdims=True)
            blended = apply_floor(blended)
            scores.append(score_prediction(blended, gts5[s]))
        print(f"  alpha={alpha:.1f}: {np.mean(scores):.1f}  seeds: {[f'{s:.1f}' for s in scores]}")

    # R5 observation calibration smoothing sweep
    print()
    print("R5 calibration smoothing sweep (obs transition prior weight):")
    for sm in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        cal_t = observation_calibrated_transitions(rid5, detail5, smoothing=sm)
        if cal_t is None:
            continue
        scores = []
        for s in range(len(gts5)):
            sp = spatial_prior(detail5, s)
            tp = _apply_transition_matrix(detail5, s, cal_t)
            blended = 0.6 * sp + 0.4 * tp
            blended = blended / blended.sum(axis=-1, keepdims=True)
            blended = apply_floor(blended)
            scores.append(score_prediction(blended, gts5[s]))
        print(f"  smoothing={sm:.0f}: {np.mean(scores):.1f}  seeds: {[f'{s:.1f}' for s in scores]}")

    # Full pipeline blend ratio sweep for R5 with observations
    print()
    print("R5 pipeline alpha sweep (spatial vs calibrated trans, with obs):")
    for alpha in [0.4, 0.5, 0.6, 0.7, 0.8]:
        cal_t = observation_calibrated_transitions(rid5, detail5, smoothing=5.0)
        scores = []
        for s in range(len(gts5)):
            sp = spatial_prior(detail5, s)
            tp = _apply_transition_matrix(detail5, s, cal_t)
            blended = alpha * sp + (1 - alpha) * tp
            blended = blended / blended.sum(axis=-1, keepdims=True)
            blended = apply_floor(blended)
            scores.append(score_prediction(blended, gts5[s]))
        print(f"  alpha={alpha:.1f}: {np.mean(scores):.1f}  seeds: {[f'{s:.1f}' for s in scores]}")


if __name__ == "__main__":
    main()
