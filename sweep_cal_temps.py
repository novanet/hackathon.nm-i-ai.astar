"""Sweep calibration factors and M5 temps on R1-R16 LORO using full pipeline (build_prediction).
Tests against GT for all rounds where we have both observations and GT.
"""
import json, numpy as np, sys
from pathlib import Path
from astar.model import build_prediction
from astar.submit import score_prediction
import astar.model as m
from astar.client import get_round_detail

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


def sweep_config(label: str, cal_factors, temps_dict=None):
    """Test a config across all observed rounds."""
    orig_cal = m.CALIBRATION_FACTORS.copy()
    orig_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()} if temps_dict else None
    
    m.CALIBRATION_FACTORS = np.array(cal_factors)
    if temps_dict:
        for k, v in temps_dict.items():
            m.ENTROPY_BUCKET_TEMPS[k] = np.array(v)
    
    scores = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        rdir = DATA_DIR / f"round_{rid}"
        detail_files = sorted(rdir.glob("round_detail_*.json"))
        if not detail_files:
            continue
        detail = json.loads(detail_files[0].read_text())
        n_seeds = len(detail.get("initial_states", []))
        sim_files = list(rdir.glob("sim_*.json"))
        if not sim_files:
            continue
        try:
            scores[rnum] = score_round(rid, detail, n_seeds)
        except Exception as e:
            print(f"  Skip R{rnum}: {e}")
    
    m.CALIBRATION_FACTORS = orig_cal
    if orig_temps:
        for k, v in orig_temps.items():
            m.ENTROPY_BUCKET_TEMPS[k] = v
    
    avg = np.mean(list(scores.values()))
    per_round = " ".join(f"R{r}={s:.1f}" for r, s in sorted(scores.items()))
    print(f"{label:<40s}  avg={avg:.2f}  {per_round}")
    return avg, scores


def main():
    print("=" * 80)
    print("CALIBRATION FACTOR SWEEP")
    print("=" * 80)
    
    # Current baseline
    sweep_config("Current [1,1,1,1,0.95,1]", [1.0, 1.0, 1.0, 1.0, 0.95, 1.0])
    
    # Sweep Forest calibration
    for f_cal in [0.90, 0.92, 0.93, 0.95, 0.97, 1.0]:
        sweep_config(f"Forest={f_cal}", [1.0, 1.0, 1.0, 1.0, f_cal, 1.0])
    
    # Sweep Empty calibration
    for e_cal in [0.97, 0.98, 1.0, 1.02, 1.03]:
        sweep_config(f"Empty={e_cal},F=0.95", [e_cal, 1.0, 1.0, 1.0, 0.95, 1.0])
    
    # Sweep Settlement calibration
    for s_cal in [0.90, 0.95, 1.0, 1.05, 1.10]:
        sweep_config(f"Sett={s_cal},F=0.95", [1.0, s_cal, 1.0, 1.0, 0.95, 1.0])
    
    # No calibration
    sweep_config("No calibration", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    print()
    print("=" * 80)
    print("M5 BUCKET TEMP SWEEP (scale factor on current temps)")
    print("=" * 80)
    
    # Scale all temps toward 1.0 (shrink) or away from 1.0 (amplify)
    base_temps = {k: v.copy() for k, v in m.ENTROPY_BUCKET_TEMPS.items()}
    
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        scaled = {}
        for k, v in base_temps.items():
            # scale=1.0 means current, scale=0 means all temps=1.0
            scaled[k] = 1.0 + scale * (v - 1.0)
        sweep_config(f"Temp scale={scale:.2f}", [1.0, 1.0, 1.0, 1.0, 0.95, 1.0], scaled)
    
    # Try disabling temps entirely
    no_temps = {k: np.ones(6) for k in base_temps}
    sweep_config("No temps (all 1.0)", [1.0, 1.0, 1.0, 1.0, 0.95, 1.0], no_temps)


if __name__ == "__main__":
    main()
