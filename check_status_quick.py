"""Quick status check."""
from astar.client import get_my_rounds, get_leaderboard

rounds = get_my_rounds()
rounds.sort(key=lambda r: r.get("round_number", 0))
for r in rounds:
    rn = r.get("round_number", "?")
    score = r.get("round_score", None)
    seeds = r.get("seeds_submitted", 0)
    status = r.get("status", "?")
    rank = r.get("rank", "?")
    if seeds > 0 or rn >= 14:
        sc = f"{score:.2f}" if score else "pending"
        print(f"R{rn}: {sc}  seeds={seeds}  status={status}  rank={rank}")

print()
lb = get_leaderboard()
for entry in lb[:10]:
    rank = entry.get("rank", "?")
    name = entry.get("team_name", entry.get("email", "?"))[:30]
    lbs = entry.get("leaderboard_score", 0)
    print(f"  {rank:>3}. {name:30s} {lbs:.2f}")
