from astar.client import _request

my = _request('GET', '/my-rounds')
for r in sorted(my, key=lambda x: x.get('round_number', 0)):
    rn = r.get('round_number')
    if rn and rn >= 7:
        rs = r.get('round_score')
        ss = r.get('seed_scores')
        st = r.get('status', '?')
        print(f'R{rn}: score={rs}  seeds={ss}  status={st}')

print()
rounds = _request('GET', '/rounds')
for r in rounds:
    rn = r.get('round_number')
    if rn and rn >= 8:
        st = r.get('status', '?')
        closes = r.get('closes_at', '?')
        rid = r.get('id', '?')[:8]
        print(f'R{rn}: status={st}  closes={closes}  id={rid}')
