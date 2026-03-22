"""Test combined optimal config: no cal + no temps. Validate on wider range R9-R16."""
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
    print(f"{label:<45s}  avg={avg:.3f}  {per_round}", flush=True)
    return avg


def main():
    t0 = time.time()
    orig_cal = m.CALIBRATION_FACTORS.copy()
    orig_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    orig_use_temps = m.USE_ENTROPY_TEMPS
    
    # 1. Current baseline
    print("WIDER VALIDATION (R9-R16)", flush=True)
    print("=" * 100, flush=True)
    score_all("A) BASELINE [cal=0.95F, temps=ON]")
    
    # 2. No calibration only
    m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    score_all("B) No calibration only")
    m.CALIBRATION_FACTORS = orig_cal.copy()
    
    # 3. No temps only
    m.USE_ENTROPY_TEMPS = False
    score_all("C) No temps only")
    m.USE_ENTROPY_TEMPS = orig_use_temps
    
    # 4. Combined: no cal + no temps
    m.CALIBRATION_FACTORS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    m.USE_ENTROPY_TEMPS = False
    score_all("D) COMBINED: no cal + no temps")
    
    # 5. Combined + higher overlay
    m._BAYES_MIN_PS = 10.0
    m._BAYES_MAX_PS = 200.0
    score_all("E) D + overlay=(10,200)")
    m._BAYES_MIN_PS = 5.0
    m._BAYES_MAX_PS = 100.0
    
    # Restore
    m.CALIBRATION_FACTORS = orig_cal.copy()
    m.USE_ENTROPY_TEMPS = orig_use_temps
    for k, v in orig_temps.items():
        m.ENTROPY_BUCKET_TEMPS[k] = v.copy()
    
    print(f"\nTime: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
