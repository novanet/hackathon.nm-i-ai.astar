"""Query R16 observations only (no submission). Stores all sim files via client.py."""
import time
from astar.client import _request, get_round_detail, get_budget, simulate, save_round_summary
from astar.model import observation_calibrated_transitions, HISTORICAL_TRANSITIONS
from astar.replay import CLASS_NAMES

ROUND_ID = "8f664aed-8839-4c85-bed0-77a2cac7c6f5"

GRID_POSITIONS = [
    (0, 0), (13, 0), (26, 0),
    (0, 13), (13, 13), (26, 13),
    (0, 26), (13, 26), (26, 26),
]


def main():
    detail = get_round_detail(ROUND_ID)
    n_seeds = len(detail.get("initial_states", []))
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)

    print(f"R16 | {map_w}x{map_h}, {n_seeds} seeds")
    for i, state in enumerate(detail["initial_states"]):
        setts = len([s for s in state["settlements"] if s["alive"]])
        ports = len([s for s in state["settlements"] if s.get("has_port")])
        forests = sum(1 for row in state["grid"] for c in row if c == "forest")
        print(f"  Seed {i}: {setts} sett, {ports} ports, {forests} forest")

    budget = get_budget()
    used = budget["queries_used"]
    total = budget["queries_max"]
    print(f"\nBudget: {used}/{total}")

    # Phase 1: Grid queries (9 viewports per seed)
    queries_needed = 9 * n_seeds
    remaining = total - used
    if remaining < queries_needed:
        print(f"Not enough budget ({remaining} < {queries_needed})")
        return

    print(f"\n=== Querying {n_seeds} seeds (9 viewports each) ===")
    count = 0
    for seed_idx in range(n_seeds):
        for vx, vy in GRID_POSITIONS:
            w = min(15, map_w - vx)
            h = min(15, map_h - vy)
            try:
                simulate(ROUND_ID, seed_idx, vx, vy, w, h)
                count += 1
            except Exception as e:
                print(f"  ERROR seed {seed_idx} ({vx},{vy}): {e}")
            time.sleep(0.05)
        print(f"  Seed {seed_idx}: 9 viewports done")

    budget = get_budget()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"\nGrid done: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    # Phase 2: Extra queries — repeat on most dynamic seed
    if remaining > 0:
        seed_dynamics = []
        for i, state in enumerate(detail["initial_states"]):
            n_setts = len([s for s in state["settlements"] if s["alive"]])
            seed_dynamics.append((n_setts, i))
        seed_dynamics.sort(reverse=True)

        extra = 0
        for _, seed_idx in seed_dynamics:
            if extra >= remaining:
                break
            for vx, vy in GRID_POSITIONS:
                if extra >= remaining:
                    break
                w = min(15, map_w - vx)
                h = min(15, map_h - vy)
                try:
                    simulate(ROUND_ID, seed_idx, vx, vy, w, h)
                    print(f"  Repeat: seed {seed_idx} ({vx},{vy})")
                    extra += 1
                except Exception as e:
                    print(f"  Repeat failed: {e}")
                    break
                time.sleep(0.05)
        print(f"  Extra queries: {extra}")

    # Diagnostics
    cal = observation_calibrated_transitions(ROUND_ID, detail, map_w, map_h)
    if cal is not None:
        hist = HISTORICAL_TRANSITIONS
        print("\n=== Transition Diagnostics ===")
        for c in range(6):
            if cal[c].sum() < 0.01:
                continue
            deltas = cal[c] - hist[c]
            big = [(CLASS_NAMES[j], deltas[j]) for j in range(6) if abs(deltas[j]) > 0.02]
            if big:
                changes = ", ".join(f"{n}:{d:+.1%}" for n, d in big)
                print(f"  {CLASS_NAMES[c]:>10} -> {changes}")

        activity = max(0, -(cal[0, 0] - hist[0, 0]))
        ss_obs = cal[1, 1]
        ss_hist = hist[1, 1]
        mode = "HIGH-RETENTION" if ss_obs > 0.40 else "HIGH-ACTIVITY" if activity >= 0.10 else "NORMAL"
        print(f"  Activity: {activity:.2%}, S->S: {ss_obs:.3f} (hist {ss_hist:.3f}) -> {mode}")

    budget = get_budget()
    print(f"\nFinal: {budget['queries_used']}/{budget['queries_max']} queries used")
    print("Observations stored. Ready for submission when approved.")


if __name__ == "__main__":
    main()
