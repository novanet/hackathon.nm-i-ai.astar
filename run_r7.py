"""
Round 7 solver: query all seeds, analyze activity, submit predictions, iterate.
Usage: $env:ASTAR_TOKEN = "your-jwt"; python run_r7.py
"""

import json
import time
import numpy as np
from pathlib import Path

from astar.client import (
    _request, get_round_detail, get_budget, simulate, get_analysis, submit
)
from astar.model import build_prediction, prediction_to_list, apply_floor
from astar.submit import score_prediction

DATA_DIR = Path("data")

# ── Step 1: Find R7 ──
print("=== Finding R7 ===")
rounds = _request("GET", "/rounds")
r7 = None
for r in rounds:
    if r.get("round_number") == 7:
        r7 = r
        break
if not r7:
    print("R7 not found! Available rounds:")
    for r in rounds:
        print("  Round %s: %s (%s)" % (r.get("round_number"), r.get("id"), r.get("status")))
    exit(1)

R7_ID = r7["id"]
print("R7 ID:", R7_ID)
print("Status:", r7.get("status"))
print("Closes at:", r7.get("closes_at", "?"))

# ── Step 2: Get round detail ──
detail = get_round_detail(R7_ID)
n_seeds = len(detail.get("initial_states", []))
map_w = detail.get("map_width", 40)
map_h = detail.get("map_height", 40)
print("Map: %dx%d, %d seeds" % (map_w, map_h, n_seeds))

for i, state in enumerate(detail["initial_states"]):
    setts = len([s for s in state["settlements"] if s["alive"]])
    ports = len([s for s in state["settlements"] if s.get("has_port")])
    print("  Seed %d: %d settlements, %d ports" % (i, setts, ports))

# Save round detail
rdir = DATA_DIR / ("round_%s" % R7_ID)
rdir.mkdir(parents=True, exist_ok=True)
ts = time.strftime("%Y%m%dT%H%M%S")
(rdir / ("round_detail_%s.json" % ts)).write_text(
    json.dumps(detail, indent=2), encoding="utf-8"
)

# ── Step 3: Check budget ──
budget = get_budget()
used = budget["queries_used"]
total = budget["queries_max"]
print("\nBudget: %d/%d queries used" % (used, total))

# ── Step 4: Run grid queries (9 viewports × 5 seeds = 45) ──
GRID_POSITIONS = [
    (0, 0), (13, 0), (26, 0),
    (0, 13), (13, 13), (26, 13),
    (0, 26), (13, 26), (26, 26),
]

queries_needed = 9 * n_seeds
remaining = total - used

if remaining >= queries_needed:
    print("\n=== Querying %d seeds (9 viewports each) ===" % n_seeds)
    for seed_idx in range(n_seeds):
        for vx, vy in GRID_POSITIONS:
            w = min(15, map_w - vx)
            h = min(15, map_h - vy)
            try:
                result = simulate(R7_ID, seed_idx, vx, vy, w, h)
            except Exception as e:
                print("  ERROR seed %d (%d,%d): %s" % (seed_idx, vx, vy, e))
            time.sleep(0.05)
        print("  Seed %d: 9 viewports done" % seed_idx)
    
    budget = get_budget()
    print("Budget after grid: %d/%d" % (budget["queries_used"], budget["queries_max"]))
else:
    print("\nSkipping grid queries — only %d remaining (need %d)" % (remaining, queries_needed))
    if used > 0:
        print("Queries already used, proceeding with existing observations")

# ── Step 5: Submit first pass ──
print("\n=== Submitting predictions (pass 1) ===")
for seed_idx in range(n_seeds):
    try:
        pred = build_prediction(R7_ID, detail, seed_idx, map_w, map_h)
        resp = submit(R7_ID, seed_idx, prediction_to_list(pred))
        status = resp.get("status", "?")
        score = resp.get("score", "?")
        print("  Seed %d: status=%s score=%s" % (seed_idx, status, score))
    except Exception as e:
        print("  Seed %d: ERROR - %s" % (seed_idx, e))

# ── Step 6: Spend remaining queries on repeat observations ──
budget = get_budget()
remaining = budget["queries_max"] - budget["queries_used"]
print("\n=== Extra queries: %d remaining ===" % remaining)

if remaining > 0:
    # Sort seeds by settlement count (most dynamic first)
    seed_dynamics = []
    for i, state in enumerate(detail["initial_states"]):
        n_setts = len([s for s in state["settlements"] if s["alive"]])
        seed_dynamics.append((n_setts, i))
    seed_dynamics.sort(reverse=True)

    extra_viewports = [(10, 10, 15, 15), (10, 20, 15, 15)]
    queries_left = remaining

    for _, seed_idx in seed_dynamics:
        if queries_left <= 0:
            break
        for vx, vy, w, h in extra_viewports:
            if queries_left <= 0:
                break
            try:
                result = simulate(R7_ID, seed_idx, vx, vy, w, h)
                print("  Extra: seed %d (%d,%d)" % (seed_idx, vx, vy))
                queries_left -= 1
            except Exception as e:
                print("  Extra failed: %s" % e)
                queries_left = 0
                break
            time.sleep(0.05)

    # Resubmit with extra observations
    print("\n=== Resubmitting (pass 2 with extra obs) ===")
    for seed_idx in range(n_seeds):
        try:
            pred = build_prediction(R7_ID, detail, seed_idx, map_w, map_h)
            resp = submit(R7_ID, seed_idx, prediction_to_list(pred))
            status = resp.get("status", "?")
            score = resp.get("score", "?")
            print("  Seed %d: status=%s score=%s" % (seed_idx, status, score))
        except Exception as e:
            print("  Seed %d: ERROR - %s" % (seed_idx, e))

# ── Step 7: Final summary ──
budget = get_budget()
print("\n=== Final Budget: %d/%d queries, subs=%s ===" % (
    budget["queries_used"], budget["queries_max"],
    budget.get("submissions_used", "?")
))

# Check our score
my = _request("GET", "/my-rounds")
for r in my:
    if r.get("round_number") == 7:
        rs = r.get("round_score")
        ss = r.get("seed_scores")
        print("R7 score: %s" % rs)
        print("Seed scores: %s" % ss)
        if rs:
            print("Weighted: %.2f" % (rs * 1.05**7))
        break

print("\nDone! Round ID: %s" % R7_ID)
