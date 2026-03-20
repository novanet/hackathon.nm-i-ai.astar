from astar.client import _request

my = _request('GET', '/my-rounds')
for r in sorted(my, key=lambda x: x.get('round_number', 0)):
    rn = r.get('round_number')
    rs = r.get('round_score')
    ss = r.get('seed_scores')
    w = rs * 1.05**rn if rs else None
    wstr = f'{w:.2f}' if w else 'pending'
    print(f'R{rn}: score={rs}  weighted={wstr}  seeds={ss}')

print()
lb = _request('GET', '/leaderboard')
print('Leaderboard top 10:')
for i, e in enumerate(lb[:10]):
    print(f'  {i+1}. {e.get("team_name", "?")}: {e.get("score", 0):.2f}')
for i, e in enumerate(lb):
    if 'novanet' in str(e).lower():
        print(f'  Us: rank {i+1}/{len(lb)}, score {e.get("score", 0):.2f}')
        break
