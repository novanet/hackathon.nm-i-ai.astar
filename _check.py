from astar.client import get_rounds
rounds = sorted(get_rounds(), key=lambda r: r["round_number"])
for r in rounds[-3:]:
    rn = r["round_number"]
    st = r["status"]
    ca = r.get("closes_at", "?")
    print(f"R{rn}: {st}  closes={ca}")
