"""Analyze R14 initial grid (free metadata, no queries spent)."""
import json, numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from scipy import ndimage

now = datetime.now(timezone.utc)
closes = datetime.fromisoformat("2026-03-21T11:59:55+00:00")
remaining = closes - now
print("=== TIME CHECK ===")
print(f"UTC now: {now.strftime('%H:%M:%S')}")
print(f"R14 closes: 11:59:55 UTC")
print(f"Time remaining: {remaining}")
print()

R14_ID = "d0a2c894-2162-4d49-86cf-435b9013f3b8"
DATA_DIR = Path("data") / f"round_{R14_ID}"

# Load round detail
detail_files = sorted(DATA_DIR.glob("round_detail_*.json"))
if not detail_files:
    from astar.client import _request
    detail = _request("GET", f"/rounds/{R14_ID}")
    detail_path = DATA_DIR / f"round_detail_{now.strftime('%Y%m%dT%H%M%S')}.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")
    print("Fetched fresh round detail")
else:
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    print(f"Using cached detail: {detail_files[-1].name}")

print(f"Round: {detail.get('round_number', '?')}, Weight: {detail.get('round_weight', '?')}")
print(f"Status: {detail.get('status', '?')}")
print(f"Closes: {detail.get('closes_at', '?')}")
print()

LABELS = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}

initial_states = detail.get("initial_states", [])
n_seeds = len(initial_states)
print(f"=== R14 MAP ANALYSIS ({n_seeds} seeds) ===")
print()

all_grids = []
for s_idx, state in enumerate(initial_states):
    grid = np.array(state["grid"])
    all_grids.append(grid)
    h, w = grid.shape
    counts = Counter(grid.flatten())
    n_sett = counts.get(1, 0)
    n_port = counts.get(2, 0)
    n_ruin = counts.get(3, 0)
    n_forest = counts.get(4, 0)
    n_mountain = counts.get(5, 0)
    n_ocean = counts.get(10, 0)
    n_plains = counts.get(11, 0)
    n_empty = n_ocean + n_plains
    density = n_sett / (h * w)
    print(f"Seed {s_idx}: {h}x{w} | Sett={n_sett} Port={n_port} Ruin={n_ruin} Forest={n_forest} Mtn={n_mountain} Empty={n_empty}(oc={n_ocean},pl={n_plains}) | Density={density:.3f}")

print()

# Settlement spatial analysis
print("=== SETTLEMENT SPATIAL DISTRIBUTION ===")
for s_idx, grid in enumerate(all_grids):
    h, w = grid.shape
    sett_mask = (grid == 1)
    sett_ys, sett_xs = np.where(sett_mask)
    if len(sett_ys) > 0:
        center_y, center_x = sett_ys.mean(), sett_xs.mean()
        spread_y, spread_x = sett_ys.std(), sett_xs.std()
        edge_sett = ((sett_ys <= 2) | (sett_ys >= h - 3) | (sett_xs <= 2) | (sett_xs >= w - 3)).sum()
        labeled, n_clusters = ndimage.label(sett_mask)
        print(f"  Seed {s_idx}: {len(sett_ys)} sett, center=({center_y:.1f},{center_x:.1f}), spread=({spread_y:.1f},{spread_x:.1f}), {n_clusters} clusters, {edge_sett} near edge")

print()

# Forest distribution
print("=== FOREST SPATIAL DISTRIBUTION ===")
for s_idx, grid in enumerate(all_grids):
    forest_mask = (grid == 4)
    forest_ys, forest_xs = np.where(forest_mask)
    if len(forest_ys) > 0:
        labeled, n_clusters = ndimage.label(forest_mask)
        sizes = [np.sum(labeled == i) for i in range(1, n_clusters + 1)]
        top3 = sorted(sizes, reverse=True)[:3]
        print(f"  Seed {s_idx}: {len(forest_ys)} forest cells, {n_clusters} clusters, largest: {top3}")

print()

# Cross-seed consistency
print("=== CROSS-SEED INITIAL GRID SIMILARITY ===")
for i in range(n_seeds):
    for j in range(i + 1, n_seeds):
        gi = all_grids[i].copy()
        gj = all_grids[j].copy()
        # Normalize: 10->0, 11->0
        gi[gi == 10] = 0
        gi[gi == 11] = 0
        gj[gj == 10] = 0
        gj[gj == 11] = 0
        match = (gi == gj).mean()
        print(f"  Seed {i} vs {j}: {match:.1%} cells match")

print()

# Compare settlement count to historical
print("=== HISTORICAL COMPARISON ===")
sett_per_seed = [np.sum(g == 1) for g in all_grids]
forest_per_seed = [np.sum(g == 4) for g in all_grids]
port_per_seed = [np.sum(g == 2) for g in all_grids]
print(f"R14 avg settlements: {np.mean(sett_per_seed):.1f} (range {min(sett_per_seed)}-{max(sett_per_seed)})")
print(f"R14 avg forests: {np.mean(forest_per_seed):.1f} (range {min(forest_per_seed)}-{max(forest_per_seed)})")
print(f"R14 avg ports: {np.mean(port_per_seed):.1f} (range {min(port_per_seed)}-{max(port_per_seed)})")
print()

# S/F ratio analysis
print("=== SETTLEMENT/FOREST RATIO ===")
for s_idx, grid in enumerate(all_grids):
    n_s = np.sum(grid == 1)
    n_f = np.sum(grid == 4)
    ratio = n_s / max(n_f, 1)
    print(f"  Seed {s_idx}: S/F={ratio:.3f}")

print()
print("=== ROUND TYPE REFERENCE ===")
print("  COLLAPSE: R3(S->S=0.018), R4(0.235), R8(0.067), R10(0.058)")
print("  NORMAL:   R1(0.410), R5(0.327), R9(0.275), R13(0.260)")
print("  BOOM:     R2(0.381), R6(0.395), R7(0.605), R11(0.495), R12(0.595)")
print()
print("NOTE: Round type depends on hidden params, not initial grid.")
print("We MUST observe to detect. Our model handles all types well (LORO 82.46).")
print()

# LORO scores by round type
print("=== OUR MODEL PERFORMANCE BY TYPE ===")
print("  Normal rounds:   R9=89.74, R13=91.13 (excellent)")
print("  Collapse rounds: R3=88.67, R4=90.90, R8=79.64, R10=90.39 (good-excellent)")
print("  Boom rounds:     R11=78.55, R12=53.04 (weak)")
print("  Overall LORO:    82.46")
print()

# Timing
print("=== EXECUTION PLAN ===")
print("  1. Query grid (45 queries, ~2min)")
print("  2. Diagnose transitions (~5s)")
print("  3. Submit pass 1 (~30s)")
print("  4. Extra queries as repeats (5 queries, ~15s)")
print("  5. Submit pass 2 (~30s)")
print("  TOTAL: ~3-4 minutes")
print()
print("  To beat leader (177.1): need raw >= 89.45 on R14 (weight 1.9799)")
print("  Our LORO avg = 82.46, normal-round avg ~90+")
print("  If R14 is NORMAL: likely 89-93 raw -> 176-184 weighted -> CAN BEAT LEADER")
print("  If R14 is COLLAPSE: likely 80-91 raw -> 158-180 weighted -> MIGHT beat leader")
print("  If R14 is BOOM: likely 53-79 raw -> 105-156 weighted -> UNLIKELY to beat leader")
