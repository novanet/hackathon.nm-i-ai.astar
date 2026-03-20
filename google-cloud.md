# Google Cloud — Overnight Auto-Pilot Deployment

**Goal**: Run the solver unattended overnight so we never miss a round.  
**Deadline**: Competition ends March 22 15:00 CET. Remaining rounds R8–R14 may open at any hour.

---

## Problem

Rounds can open at any time (day or night). Each round has a limited window before it closes. If we're asleep, we miss the round entirely — no queries, no submission, no points or training data.

## Solution

Deploy `auto_round.py` to Google Cloud Run so it polls for new rounds 24/7 and auto-submits predictions.

### What `auto_round.py` does

1. Polls `/rounds` for new active rounds (every `POLL_INTERVAL` seconds)
2. On new round: submits a **zero-query baseline** immediately (safety net — guarantees _some_ score)
3. Runs full **9-viewport grid** coverage for all seeds (45 queries)
4. Spends remaining queries on **repeat observations** of dynamic seeds
5. **Resubmits** with the full V3 spatial model (`build_prediction()` with debiasing + round features)
6. Logs everything to `auto_round.log`

### Model confirmation

`auto_round.py` → calls `submit_round()` → calls `build_prediction()` — this is the **V3 model** with:
- Debiasing shrinkage (`SHRINKAGE_MATRIX`)
- Round-conditioned spatial features (27 features)
- LightGBM spatial model
- Probability floor

No model mismatch — the same pipeline used in `run_next_round.py`.

---

## Deployment Plan

### Option A: Cloud Run Job (recommended)

A Cloud Run **Job** runs a container to completion (no HTTP server needed). Perfect for our polling loop.

#### Step 1: Modify Dockerfile for auto-pilot mode

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY astar/ astar/
COPY auto_round.py .
COPY data/spatial_model.pkl data/spatial_model.pkl

CMD ["python", "-u", "auto_round.py"]
```

Key changes from current Dockerfile:
- Removed `fastapi`/`uvicorn` install (not needed for polling mode)
- Copies `auto_round.py` instead of `main.py`
- `-u` flag for unbuffered output (important for Cloud Run log streaming)

#### Step 2: Reduce poll interval

In `auto_round.py`, change:
```python
POLL_INTERVAL = 3600  # 1 hour — too slow, could miss half a round
```
to:
```python
POLL_INTERVAL = 300   # 5 minutes — fast enough, well within API rate limits
```

#### Step 3: Deploy as Cloud Run Job

From Cloud Shell or local terminal:

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/astar-autopilot --region europe-north1

# Create the job
gcloud run jobs create astar-autopilot \
  --image gcr.io/YOUR_PROJECT/astar-autopilot \
  --region europe-north1 \
  --set-env-vars ASTAR_TOKEN=YOUR_JWT_HERE \
  --task-timeout 24h \
  --max-retries 3 \
  --memory 1Gi

# Execute it
gcloud run jobs execute astar-autopilot --region europe-north1
```

#### Step 4: Monitor

```bash
# View logs
gcloud run jobs executions logs astar-autopilot --region europe-north1 --limit 100

# Check status
gcloud run jobs describe astar-autopilot --region europe-north1
```

---

### Option B: Cloud Run Service (always-on)

If Jobs don't work well (e.g. 24h timeout limit enforced), deploy as a service with `--no-cpu-throttling`:

```bash
gcloud run deploy astar-autopilot \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --set-env-vars ASTAR_TOKEN=YOUR_JWT_HERE \
  --min-instances 1 \
  --no-cpu-throttling \
  --memory 1Gi \
  --timeout 3600
```

**Critical flags**:
- `--min-instances 1` — prevents scale-to-zero (default behavior would kill our polling loop)
- `--no-cpu-throttling` — allows CPU to run between requests (otherwise Cloud Run throttles CPU when no HTTP requests are being processed)

For this option, keep the FastAPI health endpoint in `main.py` and have the Dockerfile start both:
```dockerfile
CMD ["sh", "-c", "python -u auto_round.py & uvicorn main:app --host 0.0.0.0 --port 8080"]
```

---

### Option C: Leave PC running (simplest fallback)

```powershell
$env:ASTAR_TOKEN = "your-jwt"
python auto_round.py
```

Downsides: PC sleep, network drops, Windows updates. Use as backup only.

---

## Pre-Deployment Checklist

- [ ] Reduce `POLL_INTERVAL` from 3600 → 300 in `auto_round.py`
- [ ] Verify `spatial_model.pkl` exists and is up to date (trained with V3 / LightGBM)
- [ ] Get fresh JWT token from app.ainm.no
- [ ] Test locally: `python auto_round.py` — confirm it polls and sees existing rounds
- [ ] Build and deploy to Cloud Run
- [ ] Check Cloud Run logs to confirm polling is working
- [ ] Verify token doesn't expire before March 22 (if it does, update env var)

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| JWT expires overnight | Check token expiry; may need to refresh before sleeping |
| Cloud Run Job hits 24h timeout | Use Option B (always-on service) instead |
| Model file missing in container | Dockerfile copies `data/spatial_model.pkl` explicitly |
| API rate-limiting on 5-min polls | Single GET `/rounds` every 5 min is negligible load |
| Round needs >45 queries | `auto_round.py` already handles extra queries on dynamic seeds |
| New round type breaks model | Zero-query baseline submitted first — always gets baseline score |

---

## Getting Training Data Back

The auto-pilot handles **querying + submitting**, but there are two gaps:

### Gap 1: Ephemeral filesystem loses observation data

`astar/client.py` saves every API response to `data/round_<id>/` — but on Cloud Run, the filesystem is ephemeral. When the container restarts, all saved simulation responses, round details, and prediction tensors are **lost**.

**This is fine for scoring** (predictions are submitted to the API and don't need local persistence). But it means you lose the raw observation data.

#### Why this isn't a big problem

The observations themselves aren't needed after submission. What matters for training is the **ground truth** (from `get_analysis()`), which only becomes available **after** the round closes. By then, you're awake and can download it locally.

### Gap 2: Ground truth download + retraining is manual

`auto_round.py` does not call `get_analysis()` or retrain the model. The training loop requires:

1. **Download GT** for the new round: `get_analysis(round_id, seed_idx)` for each seed
2. **Add the round** to `ROUND_IDS` in `train_spatial.py`
3. **Retrain**: `python train_spatial.py` → produces updated `spatial_model.pkl`
4. **Redeploy** with the new model

### Morning Routine (when you wake up)

Run this after each overnight round to capture learnings and retrain:

```python
# 1. Check what rounds completed overnight
python -c "
from astar.client import _request
my = _request('GET', '/my-rounds')
for r in sorted(my, key=lambda x: x.get('round_number', 0)):
    rn = r.get('round_number')
    rs = r.get('round_score')
    w = 1.05 ** rn if rn else 1
    ws = rs * w if rs else 0
    print(f'  R{rn}: score={rs}, weighted={ws:.1f}, seeds={r.get(\"seed_scores\")}')
"

# 2. Download ground truth for new round(s)
python -c "
import json, numpy as np
from pathlib import Path
from astar.client import get_round_detail, get_analysis

ROUND_ID = 'PASTE_NEW_ROUND_ID_HERE'
DATA_DIR = Path('data') / f'round_{ROUND_ID}'
DATA_DIR.mkdir(parents=True, exist_ok=True)

detail = get_round_detail(ROUND_ID)
n_seeds = len(detail.get('initial_states', []))

for seed_idx in range(n_seeds):
    analysis = get_analysis(ROUND_ID, seed_idx)
    gt_path = DATA_DIR / f'ground_truth_s{seed_idx}.json'
    gt_path.write_text(json.dumps(analysis, indent=2), encoding='utf-8')
    gt = np.array(analysis['ground_truth'], dtype=np.float64)
    print(f'  Seed {seed_idx}: gt_shape={gt.shape}')
print('Done. Now add round to ROUND_IDS in train_spatial.py and retrain.')
"

# 3. Add new round to train_spatial.py ROUND_IDS, then:
python train_spatial.py

# 4. Redeploy with updated model
# (see deployment steps above)
```

### Enhancing `auto_round.py` (optional)

To make the overnight loop smarter, add a post-round GT download step. After `handle_round()`, poll for the round to close, then download GT:

```python
def download_ground_truth(round_id: str, n_seeds: int) -> bool:
    """Try to download ground truth. Returns True if successful."""
    for seed_idx in range(n_seeds):
        try:
            get_analysis(round_id, seed_idx)
        except Exception:
            return False
    log.info(f"Ground truth downloaded for {n_seeds} seeds")
    return True
```

However, this only helps if data persists — which it doesn't on Cloud Run (unless you add a GCS bucket mount or push data to an external store). For a 2-day hackathon, the manual morning routine is simpler and more reliable.

### Summary: What happens overnight vs morning

| Step | When | Where | Automated? |
|------|------|-------|-----------|
| Detect new round | Overnight | Cloud Run | ✅ `auto_round.py` |
| Query simulator (45 viewports) | Overnight | Cloud Run | ✅ `auto_round.py` |
| Submit predictions (V3 model) | Overnight | Cloud Run | ✅ `auto_round.py` |
| **Download ground truth** | Morning | Local | ❌ Manual |
| **Update KNOWLEDGE.md** | Morning | Local | ❌ Manual |
| **Retrain with new GT** | Morning | Local | ❌ Manual |
| **Redeploy updated model** | Morning | Local | ❌ Manual |

The overnight run **locks in points** with the current V3 model. The morning routine **improves the model** for subsequent rounds.

---

## Cost

Cloud Run billing:
- **CPU**: ~$0.024/hr for 1 vCPU always-on
- **Memory**: ~$0.003/hr for 1 GiB
- **Total**: ~$0.65/day — negligible for the competition

The GCP lab account from the competition should cover this easily.
