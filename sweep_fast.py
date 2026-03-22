"""FAST sweep: calibration + temps + blend on just R14-R16 (recent rounds).
Uses flush=True to show output immediately.
"""
import json, numpy as np, sys, time
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction
import astar.model as m

DATA_DIR = Path("data")

# Only test on recent rounds for speed
TEST_ROUNDS = {
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}


def load_gt(round_id: str, seed: int) -> np.ndarray:
    rdir = DATA_DIR / f"round_{round_id}"
    data = json.loads((rdir / f"ground_truth_s{seed}.json").read_text())
    return np.array(data["ground_truth"], dtype=np.float64)


def score_round(round_id: str, detail: dict, n_seeds: int) -> float:
    scores = []
    for s in range(n_seeds):
        gt = load_gt(round_id, s)
        pred = build_prediction(round_id, detail, s)
        scores.append(score_prediction(pred, gt))
    return np.mean(scores)


def run_config(label: str, cal_factors=None, temps_override=None, unet_w=None):
    """Test a config. Returns avg score."""
    orig_cal = m.CALIBRATION_FACTORS.copy()
    orig_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    orig_unet = m.UNET_BLEND_WEIGHT
    
    if cal_factors is not None:
        m.CALIBRATION_FACTORS = np.array(cal_factors)
    if temps_override is not None:
        for k, v in temps_override.items():
            m.ENTROPY_BUCKET_TEMPS[k] = np.array(v)
    if unet_w is not None:
        m.UNET_BLEND_WEIGHT = unet_w
    
    scores = {}
    for rnum, rid in sorted(TEST_ROUNDS.items()):
        rdir = DATA_DIR / f"round_{rid}"
        detail_files = sorted(rdir.glob("round_detail_*.json"))
        if not detail_files:
            continue
        detail = json.loads(detail_files[0].read_text())
        n_seeds = len(detail.get("initial_states", []))
        try:
            scores[rnum] = score_round(rid, detail, n_seeds)
        except Exception as e:
            print(f"  Skip R{rnum}: {e}", flush=True)
    
    m.CALIBRATION_FACTORS = orig_cal
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v
    m.UNET_BLEND_WEIGHT = orig_unet
    
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
    
    run_config("BASELINE [1,1,1,1,0.95,1]", [1.0, 1.0, 1.0, 1.0, 0.95, 1.0])
    run_config("No cal [1,1,1,1,1,1]", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Forest calibration
    for f in [0.88, 0.90, 0.92, 0.93, 0.95, 0.97, 1.0]:
        run_config(f"Forest={f:.2f}", [1.0, 1.0, 1.0, 1.0, f, 1.0])
    
    # Settlement calibration  
    for s in [0.90, 0.95, 1.00, 1.05, 1.10]:
        run_config(f"Sett={s:.2f},F=0.95", [1.0, s, 1.0, 1.0, 0.95, 1.0])
    
    # Empty calibration
    for e in [0.95, 0.97, 1.00, 1.03, 1.05]:
        run_config(f"Empty={e:.2f},F=0.95", [e, 1.0, 1.0, 1.0, 0.95, 1.0])
    
    # ====== SECTION 2: M5 TEMP SCALING ======
    print("\n" + "=" * 90, flush=True)
    print("M5 TEMP SCALE SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    base_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        scaled = {k: 1.0 + scale * (v - 1.0) for k, v in base_temps.items()}
        run_config(f"Temp scale={scale:.2f}", temps_override=scaled)
    
    # ====== SECTION 3: U-NET BLEND ======
    print("\n" + "=" * 90, flush=True)
    print("U-NET BLEND SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    for w in [0.0, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        run_config(f"UNet={w:.2f}", unet_w=w)
    
    # ====== SECTION 4: OVERLAY ======
    print("\n" + "=" * 90, flush=True)
    print("OVERLAY SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    for min_ps, max_ps in [(1,10), (3,50), (5,100), (10,200), (20,500)]:
        m._BAYES_MIN_PS = float(min_ps)
        m._BAYES_MAX_PS = float(max_ps)
        run_config(f"Overlay=({min_ps},{max_ps})")
    m._BAYES_MIN_PS = 5.0
    m._BAYES_MAX_PS = 100.0
    
    # No overlay
    orig_enable = m.USE_BAYESIAN_OVERLAY
    m.USE_BAYESIAN_OVERLAY = False
    run_config("No overlay")
    m.USE_BAYESIAN_OVERLAY = orig_enable    
    
    # ====== SECTION 5: PROB FLOOR ======
    print("\n" + "=" * 90, flush=True)
    print("PROB FLOOR SWEEP (R14-R16)", flush=True)
    print("=" * 90, flush=True)
    
    for f in [0.00001, 0.0001, 0.0003, 0.0005, 0.001, 0.005, 0.01]:
        m.PROB_FLOOR = f
        run_config(f"Floor={f:.5f}")
    m.PROB_FLOOR = 0.0001
    
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
