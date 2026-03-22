import os, json
os.environ.setdefault('ASTAR_TOKEN', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZDlkMGFkZC1mZGRhLTQyYWYtYTJjOC1jZDEyNDgyOTFhODciLCJlbWFpbCI6Imhicm90YW5AZ21haWwuY29tIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjE3NzQ1MTAxNjl9.MdfC2I-8TAannLDRNkt6NdcMos0yhwvxVRArZy4ImrY')
from astar.client import get_round_detail, get_rounds

r15_id = 'cc5442dd-bc5d-418b-911b-7eb960cb0390'
detail = get_round_detail(r15_id)
print('=== Full R15 detail keys ===')
print(json.dumps({k: str(v)[:200] if isinstance(v, (list, dict)) else v for k, v in detail.items()}, indent=2, default=str))

print('\n=== Submissions detail ===')
submissions = detail.get('submissions', [])
for i, sub in enumerate(submissions):
    print(f'Sub {i}:', json.dumps(sub, indent=2, default=str)[:500])

print('\n=== All rounds ===')
rounds = get_rounds()
for r in rounds:
    rn = r.get('round_number', '?')
    rid = r['id']
    sc = r.get('score')
    st = r.get('status', '?')
    print(f'Round {rn}: id={rid}, score={sc}, status={st}')
