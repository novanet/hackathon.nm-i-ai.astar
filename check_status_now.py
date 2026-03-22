from astar.client import _request
from datetime import datetime, timezone

lb = _request('GET', '/leaderboard')
print('=== LEADERBOARD TOP 15 ===')
for i, e in enumerate(lb[:15]):
    name = e.get('team_name', '?')
    score = e.get('score', 0)
    print(f'  {i+1}. {name}: {score:.2f}')
for i, e in enumerate(lb):
    if 'novanet' in str(e.get('team_name','')).lower():
        print(f'  Us: rank {i+1}/{len(lb)}, score {e.get("score", 0):.2f}')
        break

print()
my = _request('GET', '/my-rounds')
for r in sorted(my, key=lambda x: x.get('round_number',0)):
    rn = r.get('round_number')
    sc = r.get('round_score')
    status = r.get('status')
    wt = 1.05**rn if rn else 0
    ws = sc * wt if sc else None
    closes = r.get('closes_at', '?')
    ws_str = f'{ws:.2f}' if ws else '?'
    print(f'  R{rn}: score={sc}, weighted={ws_str}, status={status}, closes={closes}')

r14 = [r for r in my if r.get('round_number') == 14]
if r14:
    print(f'\nR14 details: {r14[0]}')

now = datetime.now(timezone.utc)
print(f'\nUTC now: {now.isoformat()[:19]}')
