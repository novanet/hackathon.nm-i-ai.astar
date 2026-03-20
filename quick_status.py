"""Quick status check."""
from astar.client import _request, get_budget

rounds = _request("GET", "/rounds")
for r in sorted(rounds, key=lambda x: x.get("round_number", 0)):
    rn = r.get("round_number", "?")
    st = r.get("status", "?")
    closes = r.get("closes_at", "?")
    rid = r["id"][:8]
    if rn >= 8:
        print(f"R{rn}: {st} (closes {closes}) id={rid}")

b = get_budget()
print(f"\nBudget: {b['queries_used']}/{b['queries_max']}")

my = _request("GET", "/my-rounds")
for r in sorted(my, key=lambda x: x.get("round_number", 0)):
    if r.get("round_number", 0) >= 8:
        rn = r["round_number"]
        rs = r.get("round_score")
        ss = r.get("seed_scores")
        print(f"My R{rn}: score={rs}, seeds={ss}")
