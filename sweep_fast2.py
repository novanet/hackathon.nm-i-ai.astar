"""FAST sweep: calibration + temps + floor + overlay on R14-R16.
Monkey-patches module constants and re-runs build_prediction for each config.
For U-Net blend: patches the build_prediction function directly.
"""
import json, numpy as np, time, types
from pathlib import Path
from astar.model import build_prediction as _orig_build_prediction
from astar.submit import score_prediction
import astar.model as m

DATA_DIR = Path("data")

TEST_ROUNDS = {
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}


def load_gt(round_id: str, seed: int) -> np.ndarray:
    rdir = DATA_DIR / f"round_{round_id}"
    data = json.loads((rdir / f"ground_truth_s{seed}.json").read_text())
    return np.array(data["ground_truth"], dtype=np.float64)


def score_rounds(label: str) -> float:
    """Score all test rounds with current model.py settings."""
    scores = {}
    for rnum, rid in sorted(TEST_ROUNDS.items()):
        rdir = DATA_DIR / f"round_{rid}"
        detail_files = sorted(rdir.glob("round_detail_*.json"))
        if not detail_files:
            continue
        detail = json.loads(detail_files[0].read_text())
        n_seeds = len(detail.get("initial_states", []))
        seed_scores = []
        for s in range(n_seeds):
            gt = load_gt(rid, s)
            pred = m.build_prediction(rid, detail, s)
            seed_scores.append(score_prediction(pred, gt))
        scores[rnum] = np.mean(seed_scores)
    
    avg = np.mean(list(scores.values()))
    per_round = " ".join(f"R{r}={s:.2f}" for r, s in sorted(scores.items()))
    print(f"{label:<45s}  avg={avg:.3f}  {per_round}", flush=True)
    return avg


def main():
    t0 = time.time()
    
    # ====== SECTION 1: CALIBRATION FACTORS ======
    print("=" * 90, flush=True)
    print("CALIBRATION FACTOR SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    orig_cal = m.CALIBRATION_FACTORS.copy()
    
    score_rounds("BASELINE [1,1,1,1,0.95,1]")
    
    for f in [0.88, 0.90, 0.93, 0.95, 0.97, 1.0]:
        m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, f, 1.0])
        score_rounds(f"Forest={f:.2f}")
    
    m.CALIBRATION_FACTORS = orig_cal.copy()
    
    for s in [0.90, 0.95, 1.00, 1.05, 1.10]:
        m.CALIBRATION_FACTORS = np.array([1.0, s, 1.0, 1.0, 0.95, 1.0])
        score_rounds(f"Sett={s:.2f},F=0.95")
    
    m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    score_rounds("No calibration")
    
    m.CALIBRATION_FACTORS = orig_cal.copy()
    
    # ====== SECTION 2: M5 TEMP SCALING ======
    print(f"\n[{time.time()-t0:.0f}s]", flush=True)
    print("=" * 90, flush=True)
    print("M5 TEMP SCALE SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    base_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    
    for scale in [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for k, v in base_temps.items():
            m.ENTROPY_BUCKET_TEMPS[k] = np.array(1.0 + scale * (v - 1.0))
        score_rounds(f"Temp scale={scale:.2f}")
    
    for k, v in base_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v.copy()
    
    # ====== SECTION 3: OVERLAY ======
    print(f"\n[{time.time()-t0:.0f}s]", flush=True)
    print("=" * 90, flush=True)
    print("OVERLAY SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    for min_ps, max_ps in [(3,50), (5,100), (10,200), (20,500)]:
        m._BAYES_MIN_PS = float(min_ps)
        m._BAYES_MAX_PS = float(max_ps)
        score_rounds(f"Overlay=({min_ps},{max_ps})")
    m._BAYES_MIN_PS = 5.0
    m._BAYES_MAX_PS = 100.0
    
    # ====== SECTION 4: PROB FLOOR ======
    print(f"\n[{time.time()-t0:.0f}s]", flush=True)
    print("=" * 90, flush=True)
    print("PROB FLOOR SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    for f in [0.00001, 0.0001, 0.0003, 0.001, 0.01]:
        m.PROB_FLOOR = f
        score_rounds(f"Floor={f}")
    m.PROB_FLOOR = 0.0001
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
