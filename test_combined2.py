"""Correct combined test: scale M5 temps to 1.0 (not disable M5 path)."""
import json, numpy as np, time
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction
import astar.model as m

DATA_DIR = Path("data")

ROUNDS = {
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
}


def load_gt(round_id: str, seed: int) -> np.ndarray:
    rdir = DATA_DIR / f"round_{round_id}"
    data = json.loads((rdir / f"ground_truth_s{seed}.json").read_text())
    return np.array(data["ground_truth"], dtype=np.float64)


def score_all(label: str) -> float:
    scores = {}
    for rnum, rid in sorted(ROUNDS.items()):
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
    print(f"{label:<50s}  avg={avg:.3f}  {per_round}", flush=True)
    return avg


def main():
    t0 = time.time()
    orig_cal = m.CALIBRATION_FACTORS.copy()
    orig_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    
    print("CORRECT COMBINED TEST (R9-R16, 8 rounds)", flush=True)
    print("=" * 110, flush=True)
    
    # A) Current baseline  
    score_all("A) BASELINE")
    
    # B) No cal only (keep temps)
    m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    score_all("B) No calibration")
    m.CALIBRATION_FACTORS = orig_cal.copy()
    
    # C) Scale M5 temps to 1.0 (CORRECT way to disable temps)
    for k in m.ENTROPY_BUCKET_TEMPS:
        m.ENTROPY_BUCKET_TEMPS[k] = np.ones(6)
    score_all("C) M5 temps=1.0 (keep USE_ENTROPY_TEMPS=True)")
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v.copy()
    
    # D) Scale M5 temps to 0.5 (halfway)
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = np.array(1.0 + 0.5 * (v - 1.0))
    score_all("D) M5 temps scaled 0.5")
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v.copy()
    
    # E) Combined: no cal + temps=1.0
    m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for k in m.ENTROPY_BUCKET_TEMPS:
        m.ENTROPY_BUCKET_TEMPS[k] = np.ones(6)
    score_all("E) COMBINED: no cal + M5 temps=1.0")
    
    # F) Combined + overlay=(10,200)
    m._BAYES_MIN_PS = 10.0
    m._BAYES_MAX_PS = 200.0
    score_all("F) E + overlay=(10,200)")
    m._BAYES_MIN_PS = 5.0
    m._BAYES_MAX_PS = 100.0
    
    # Restore everything
    m.CALIBRATION_FACTORS = orig_cal.copy()
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v.copy()
    
    print(f"\nTime: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
