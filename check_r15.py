import os, json
os.environ.setdefault('ASTAR_TOKEN', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1ZDlkMGFkZC1mZGRhLTQyYWYtYTJjOC1jZDEyNDgyOTFhODciLCJlbWFpbCI6Imhicm90YW5AZ21haWwuY29tIiwiaXNfYWRtaW4iOmZhbHNlLCJleHAiOjE3NzQ1MTAxNjl9.MdfC2I-8TAannLDRNkt6NdcMos0yhwvxVRArZy4ImrY')
from astar.client import get_round_detail, get_rounds

r15_id = 'cc5442dd-bc5d-418b-911b-7eb960cb0390'
detail = get_round_detail(r15_id)
print('R15 score:', detail.get('score'))
print('R15 status:', detail.get('status'))
print('R15 weight:', detail.get('round_weight'))

submissions = detail.get('submissions', [])
if submissions:
    for sub in submissions[-3:]:
        keys = [k for k in ['score', 'seed_scores'] if k in sub]
        print('Submission:', json.dumps({k: sub[k] for k in keys}, default=str))

rounds = get_rounds()
for r in rounds[-5:]:
    rn = r.get('round_number', '?')
    rid = r['id']
    sc = r.get('score')
    st = r.get('status', '?')
    print(f'Round {rn}: id={rid}, score={sc}, status={st}')
