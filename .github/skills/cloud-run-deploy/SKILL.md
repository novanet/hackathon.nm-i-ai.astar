---
name: cloud-run-deploy
description: 'Deploy the Astar Island prediction API to Google Cloud Run. Use when: setting up GCP, creating the /solve endpoint, building Docker images, deploying to Cloud Run, debugging cold starts or deployment issues, updating the live service.'
argument-hint: 'Describe what you need: initial setup, deploy update, debug logs, etc.'
---

# Deploy Astar Island API to Google Cloud Run

Astar Island **requires** a public HTTPS `/solve` endpoint that competition validators call. Google Cloud Run is the recommended hosting option — free with competition GCP accounts.

## Architecture

```
Validators (app.ainm.no)
    │
    ▼  POST /solve  { round_id, ... }
┌─────────────────────┐
│  Cloud Run service   │  ← your FastAPI container
│  (europe-north1)     │
└─────────┬───────────┘
          │  calls api.ainm.no/astar-island/*
          ▼
    Competition API
```

Your Cloud Run service receives solve requests, queries the simulator using your budget, builds predictions with your model, and returns the W×H×6 tensor.

## GCP Account Setup

1. You receive a `@gcplab.me` email + password from the competition
2. Open a **new Chrome profile** (or incognito) → [console.cloud.google.com](https://console.cloud.google.com/)
3. Sign in with the `@gcplab.me` credentials
4. Select your assigned project from the project dropdown
5. Open **Cloud Shell** (terminal icon, top-right) — it has Python, Docker, gcloud pre-installed

> Full setup docs: https://app.ainm.no/docs/google-cloud/setup

### Available Gemini Tools (in GCP console)

| Tool | Where | What |
|------|-------|------|
| Gemini Code Assist | Cloud Shell Editor | AI coding in browser IDE |
| Gemini CLI | Cloud Shell terminal | `gemini` command for CLI help |
| Gemini Cloud Assist | Console sidebar | GCP service Q&A |
| AI Studio | aistudio.google.com | Direct Gemini model access |

## Deployment Steps

### 1. Prepare the endpoint

Use the [FastAPI template](./assets/main.py) as a starting point. The `/solve` endpoint receives round info from validators and must return predictions.

### 2. Create Dockerfile

Use the [Dockerfile template](./assets/Dockerfile). It installs dependencies and runs uvicorn on port 8080.

### 3. Deploy to Cloud Run

From Cloud Shell (or local terminal with `gcloud` configured):

```bash
# Clone or upload your repo
cd ~/hackathon.nm-i-ai.astar

# One-command build + deploy
gcloud run deploy astar-solver \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300
```

This builds the Docker image, pushes it, and gives you a URL like:
```
https://astar-solver-xxxxx-lz.a.run.app
```

### 4. Submit the URL

1. Copy the Cloud Run URL
2. Go to [app.ainm.no](https://app.ainm.no/) → Astar Island submission page
3. Paste the URL and submit

### 5. Keep warm (optional but recommended)

Cold starts add latency. During active rounds:
```bash
gcloud run deploy astar-solver --min-instances 1 --region europe-north1
```

## Common Operations

### Update deployment
```bash
gcloud run deploy astar-solver --source . --region europe-north1 --allow-unauthenticated
```

### View logs
```bash
gcloud run services logs read astar-solver --region europe-north1 --limit 50
```

### Increase resources
```bash
gcloud run deploy astar-solver --memory 2Gi --cpu 2 --region europe-north1
```

## Key Constraints

- **Region**: Always use `europe-north1` — same as validators, lowest latency
- **Timeout**: Validators expect response within 60 seconds
- **Auth token**: Set `ASTAR_TOKEN` as a Cloud Run environment variable:
  ```bash
  gcloud run deploy astar-solver \
    --set-env-vars ASTAR_TOKEN=your-jwt-token \
    --region europe-north1
  ```
- **Token refresh**: JWT may expire — update the env var when you get 401s

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 401 from competition API | Refresh JWT token in env vars |
| Timeout on /solve | Reduce query count or viewport size; check cold start |
| Container won't start | Check logs; ensure port 8080; verify requirements.txt |
| Score = 0 for all seeds | Check prediction shape (H×W×6) and probability sums |

## Reference Links

- GCP Setup: https://app.ainm.no/docs/google-cloud/setup
- Cloud Run Deploy: https://app.ainm.no/docs/google-cloud/deploy
- GCP Services: https://app.ainm.no/docs/google-cloud/services
