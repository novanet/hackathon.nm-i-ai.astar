"""Quick R17 status check."""
from astar.client import get_round_detail, _request

rid = "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9"
d = get_round_detail(rid)
w, h = d["map_width"], d["map_height"]
seeds = d["initial_states"]
print(f"R17 | {w}x{h}, {len(seeds)} seeds")
for i, s in enumerate(seeds):
    setts = len([x for x in s["settlements"] if x["alive"]])
    ports = len([x for x in s["settlements"] if x.get("has_port")])
    forests = sum(1 for row in s["grid"] for c in row if c == "forest")
    print(f"  Seed {i}: {setts} sett, {ports} ports, {forests} forest")

my = _request("GET", "/my-rounds")
r17 = [r for r in my if r["id"] == rid][0]
print(f"\nWeight: {r17.get('round_weight')}")
print(f"Closes: {r17.get('closes_at')}")
print(f"Queries: {r17.get('queries_used')}/{r17.get('queries_max')}")
print(f"Seeds submitted: {r17.get('seeds_submitted')}")
