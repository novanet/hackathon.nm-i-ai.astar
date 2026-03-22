"""
Auto-polling: monitor R16 close → download GT → retrain → query R17 → STOP (no submit).

Runs in background. Polls every 60s. When R16 completes:
  1. Download R16 ground truth for all seeds
  2. Retrain GBM+MLP (train_spatial.py)
  3. Retrain U-Net (train_unet.py)
  4. Re-optimize M5 bucket temps (calibrate_entropy.py --nocache)
When R17 opens:
    5. Run grid queries (45) + diagnostic repeats (5) near settlement clusters
  6. STOP and notify — do NOT submit without human approval

Usage:
    $env:ASTAR_TOKEN = "your-jwt"; python auto_poll.py
"""

import json
import time
import subprocess
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

from astar.client import (
    _request, get_rounds, get_round_detail, get_budget,
    simulate, get_analysis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auto_poll.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data")

R16_ID = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"
POLL_INTERVAL = 60  # seconds

GRID_POSITIONS = [
    (0, 0), (13, 0), (26, 0),
    (0, 13), (13, 13), (26, 13),
    (0, 26), (13, 26), (26, 26),
]


def diagnostic_viewport_for_seed(state: dict, map_w: int, map_h: int, size: int = 15) -> tuple[int, int]:
    """Pick a repeat viewport centered on the seed's active settlement cluster."""
    alive = [s for s in state.get("settlements", []) if s.get("alive")]
    if not alive:
        return min(13, max(0, map_w - size)), min(13, max(0, map_h - size))

    mean_x = sum(s["x"] for s in alive) / len(alive)
    mean_y = sum(s["y"] for s in alive) / len(alive)
    max_x = max(0, map_w - size)
    max_y = max(0, map_h - size)
    vx = min(max(int(round(mean_x)) - size // 2, 0), max_x)
    vy = min(max(int(round(mean_y)) - size // 2, 0), max_y)
    return vx, vy


# ── Phase 1: Wait for R16 to close ──────────────────────────────────────────

def check_r16_completed() -> tuple[bool, float | None]:
    """Returns (is_completed, score)."""
    my = _request("GET", "/my-rounds")
    for r in my:
        if r["id"] == R16_ID:
            if r["status"] == "completed":
                return True, r.get("round_score")
    return False, None


def download_gt(round_id: str) -> bool:
    """Download ground truth for all seeds. Returns True if all succeeded."""
    log.info(f"Downloading GT for {round_id[:8]}...")
    rdir = DATA_DIR / f"round_{round_id}"
    rdir.mkdir(parents=True, exist_ok=True)

    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    ok = True

    for seed_idx in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{seed_idx}.json"
        if gt_path.exists():
            log.info(f"  Seed {seed_idx}: already exists")
            continue
        try:
            analysis = _request("GET", f"/analysis/{round_id}/{seed_idx}")
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            log.info(f"  Seed {seed_idx}: saved")
        except Exception as e:
            log.error(f"  Seed {seed_idx}: FAILED — {e}")
            ok = False
    return ok


# ── Phase 2: Retrain models ─────────────────────────────────────────────────

def retrain_models() -> bool:
    """Retrain GBM+MLP, U-Net, and recalibrate M5 temps."""
    steps = [
        ("GBM+MLP", [sys.executable, "train_spatial.py"]),
        ("U-Net", [sys.executable, "train_unet.py"]),
        ("M5 temps", [sys.executable, "calibrate_entropy.py", "--nocache"]),
    ]
    all_ok = True
    for name, cmd in steps:
        log.info(f"Retraining {name}...")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env={**__import__("os").environ},
            )
            if result.returncode == 0:
                # Extract key metrics from output
                lines = result.stdout.strip().split("\n")
                key_lines = [l for l in lines[-20:] if any(
                    k in l.lower() for k in ["avg", "loro", "saved", "done", "best", "score"]
                )]
                log.info(f"  {name} OK. Key output:")
                for kl in key_lines[-5:]:
                    log.info(f"    {kl.strip()}")
            else:
                log.error(f"  {name} FAILED (exit {result.returncode})")
                log.error(f"  stderr: {result.stderr[-500:]}")
                all_ok = False
        except subprocess.TimeoutExpired:
            log.error(f"  {name} TIMEOUT (600s)")
            all_ok = False
        except Exception as e:
            log.error(f"  {name} ERROR: {e}")
            all_ok = False
    return all_ok


# ── Phase 3: Find and query R17 ─────────────────────────────────────────────

def find_new_active_round(exclude_ids: set[str]) -> dict | None:
    """Find an active round that isn't in exclude_ids."""
    rounds = get_rounds()
    for r in rounds:
        if r["status"] == "active" and r["id"] not in exclude_ids:
            return r
    return None


def query_round(round_id: str) -> None:
    """Run grid queries + extras on a round. Does NOT submit."""
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    log.info(f"Round {round_id[:8]} | {map_w}x{map_h}, {n_seeds} seeds")
    for i, state in enumerate(detail["initial_states"]):
        setts = len([s for s in state["settlements"] if s["alive"]])
        ports = len([s for s in state.get("settlements", []) if s.get("has_port")])
        forests = sum(1 for row in state["grid"] for c in row if c == "forest")
        log.info(f"  Seed {i}: {setts} sett, {ports} ports, {forests} forest")

    budget = get_budget()
    used = budget["queries_used"]
    total = budget["queries_max"]
    log.info(f"Budget: {used}/{total}")

    # Phase 1: Grid queries (9 per seed = 45)
    queries_needed = 9 * n_seeds
    remaining = total - used
    if remaining < queries_needed:
        log.warning(f"Not enough budget ({remaining} < {queries_needed})")
        return

    log.info(f"Querying {n_seeds} seeds (9 viewports each)...")
    for seed_idx in range(n_seeds):
        for vx, vy in GRID_POSITIONS:
            w = min(15, map_w - vx)
            h = min(15, map_h - vy)
            try:
                simulate(round_id, seed_idx, vx, vy, w, h)
            except Exception as e:
                log.error(f"  Seed {seed_idx} ({vx},{vy}): {e}")
            time.sleep(0.05)
        log.info(f"  Seed {seed_idx}: grid done")

    # Phase 2: Diagnostic repeats across seeds
    budget = get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    if remaining > 0:
        seed_targets = []
        for i, state in enumerate(detail["initial_states"]):
            n_setts = len([s for s in state["settlements"] if s["alive"]])
            vx, vy = diagnostic_viewport_for_seed(state, map_w, map_h)
            seed_targets.append((n_setts, i, vx, vy))
        seed_targets.sort(reverse=True)

        log.info("Diagnostic repeat targets:")
        for n_setts, seed_idx, vx, vy in seed_targets:
            log.info(f"  Seed {seed_idx}: {n_setts} settlements -> viewport ({vx},{vy})")

        extra = 0
        while extra < remaining and seed_targets:
            _, target_seed, vx, vy = seed_targets[extra % len(seed_targets)]
            w = min(15, map_w - vx)
            h = min(15, map_h - vy)
            try:
                simulate(round_id, target_seed, vx, vy, w, h)
                log.info(f"  Diagnostic repeat: seed {target_seed} ({vx},{vy})")
                extra += 1
            except Exception as e:
                log.error(f"  Repeat seed {target_seed} ({vx},{vy}): {e}")
                break
            time.sleep(0.05)
        log.info(f"  Extra: {extra} diagnostic repeats across {len(seed_targets)} seeds")

    budget = get_budget()
    log.info(f"Final: {budget['queries_used']}/{budget['queries_max']} queries used")

    # Print transition diagnostics
    try:
        from astar.model import observation_calibrated_transitions, HISTORICAL_TRANSITIONS
        from astar.replay import CLASS_NAMES
        cal = observation_calibrated_transitions(round_id, detail, map_w, map_h)
        if cal is not None:
            hist = HISTORICAL_TRANSITIONS
            log.info("Transition diagnostics:")
            for c in range(6):
                if cal[c].sum() < 0.01:
                    continue
                deltas = cal[c] - hist[c]
                big = [(CLASS_NAMES[j], deltas[j]) for j in range(6) if abs(deltas[j]) > 0.02]
                if big:
                    changes = ", ".join(f"{n}:{d:+.1%}" for n, d in big)
                    log.info(f"  {CLASS_NAMES[c]:>10} -> {changes}")
            ss_obs = cal[1, 1]
            ss_hist = hist[1, 1]
            activity = max(0, -(cal[0, 0] - hist[0, 0]))
            log.info(f"  Activity: {activity:.2%}, S->S: {ss_obs:.3f} (hist {ss_hist:.3f})")
    except Exception as e:
        log.warning(f"Transition diagnostics failed: {e}")


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("AUTO-POLL STARTED")
    log.info(f"  Monitoring R16={R16_ID[:8]}")
    log.info(f"  Poll interval: {POLL_INTERVAL}s")
    log.info("  Will NOT submit — human approval required")
    log.info("=" * 60)

    r16_done = False
    retrained = False
    r17_queried = False
    r17_id = None

    while True:
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # Step 1: Check R16
        if not r16_done:
            completed, score = check_r16_completed()
            if completed:
                log.info(f"*** R16 COMPLETED! Score: {score} ***")
                r16_done = True
                download_gt(R16_ID)
            else:
                log.info(f"[{now}] R16 still active...")
                time.sleep(POLL_INTERVAL)
                continue

        # Step 2: Retrain
        if not retrained:
            log.info("Starting model retraining on R1-R16...")
            retrained = retrain_models()
            if not retrained:
                log.warning("Retraining had errors — continuing anyway")
                retrained = True  # don't retry

        # Step 3: Look for R17
        if not r17_queried:
            r17 = find_new_active_round(exclude_ids={R16_ID})
            if r17:
                r17_id = r17["id"]
                rn = r17.get("round_number", "?")
                log.info(f"*** NEW ROUND DETECTED: R{rn} ({r17_id[:8]}) ***")
                query_round(r17_id)
                r17_queried = True
                log.info("=" * 60)
                log.info(f"*** R{rn} QUERIES DONE — READY FOR SUBMISSION ***")
                log.info(f"*** Round ID: {r17_id} ***")
                log.info("*** Run: python -m astar.submit <round_id> ***")
                log.info("*** WAITING FOR HUMAN APPROVAL — NOT SUBMITTING ***")
                log.info("=" * 60)
                break  # Done — stop polling
            else:
                log.info(f"[{now}] No new round yet, checking again in {POLL_INTERVAL}s...")
                time.sleep(POLL_INTERVAL)
                continue

    log.info("Auto-poll finished. All automated steps complete.")
    log.info(f"R16 score: {score}")
    if r17_id:
        log.info(f"R17 round_id: {r17_id}")
        log.info("Observations stored. Submit when ready:")
        log.info(f"  python -m astar.submit {r17_id}")


if __name__ == "__main__":
    main()
