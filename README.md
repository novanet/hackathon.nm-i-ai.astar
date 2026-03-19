# Astar Island — Viking Civilisation Prediction

**NM i AI 2026 — Challenge 2: A***

> Query a Norse island simulator and predict the world state using machine learning.

- **Competition**: March 19 18:00 CET → March 22 15:00 CET (69 hours)
- **Platform**: [app.ainm.no](https://app.ainm.no)
- **API Base URL**: `https://api.ainm.no/astar-island`
- **Metric**: KL Divergence (0–100 scale)
- **Response limit**: 60s
- **Docs**: [Overview](https://app.ainm.no/docs/astar-island/overview) · [Quickstart](https://app.ainm.no/docs/astar-island/quickstart) · [Submit](https://app.ainm.no/submit/astar-island)

---

## What is this?

Astar Island is a machine learning challenge where you observe a **black-box Norse civilisation simulator** through a limited viewport and predict the final world state. The simulator runs a procedurally generated Norse world for **50 years** — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

**Goal**: Observe, learn the world's hidden rules, and predict the **probability distribution of terrain types** across the entire map.

## How It Works

1. **A round starts** — the admin creates a round with a fixed map, many hidden parameters, and **5 random seeds**
2. **Observe through a viewport** — call `POST /astar-island/simulate` with viewport coordinates to observe one stochastic run through a window (max **15×15 cells**). You have **50 queries total** per round, shared across all 5 seeds.
3. **Learn the hidden rules** — analyze viewport observations to understand the forces that govern the world
4. **Generate predictions** — build probability distributions for the full map
5. **Submit predictions** — for each seed, submit a **W×H×6** probability tensor predicting terrain type probabilities per cell
6. **Scoring** — your prediction is compared against the ground truth using **entropy-weighted KL divergence**

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Map seed** | Determines terrain layout (fixed per seed, visible to you) |
| **Sim seed** | Random seed for each simulation run (different every query) |
| **Hidden parameters** | Values controlling the world's behavior (same for all seeds in a round) |
| **50 queries** | Your budget per round, shared across all 5 seeds |
| **Viewport** | Each query reveals a max 15×15 window of the 40×40 map |
| **W×H×6 tensor** | Your prediction — probability of each of 6 terrain classes per cell |
| **50 years** | Each simulation runs for 50 time steps |

---

## Simulation Mechanics

### Terrain Types

| Internal Code | Terrain | Class Index | Description |
|--------------|---------|-------------|-------------|
| 10 | Ocean | 0 (Empty) | Impassable water, borders the map |
| 11 | Plains | 0 (Empty) | Flat land, buildable |
| 0 | Empty | 0 | Generic empty cell |
| 1 | Settlement | 1 | Active Norse settlement |
| 2 | Port | 2 | Coastal settlement with harbour |
| 3 | Ruin | 3 | Collapsed settlement |
| 4 | Forest | 4 | Provides food to adjacent settlements |
| 5 | Mountain | 5 | Impassable terrain |

Ocean, Plains, and Empty all map to **class 0** in predictions. Mountains are static (never change). Forests are mostly static but can reclaim ruined land. The interesting cells are those that can become Settlements, Ports, or Ruins.

### Map Generation

Each map is procedurally generated from a **map seed**:
- **Ocean borders** surround the map
- **Fjords** cut inland from random edges
- **Mountain chains** form via random walks
- **Forest patches** cover land with clustered groves
- **Initial settlements** placed on land cells, spaced apart

The map seed is visible to you — you can reconstruct the initial terrain layout locally.

### Simulation Phases (per year, for 50 years)

1. **Growth** — Settlements produce food based on adjacent terrain. Prosperous settlements expand by founding new ones on nearby land. Ports develop along coastlines.
2. **Conflict** — Settlements raid each other. Longships extend raiding range. Desperate settlements (low food) raid more aggressively. Conquered settlements may change faction allegiance.
3. **Trade** — Ports within range can trade if not at war. Trade generates wealth and food; technology diffuses between partners.
4. **Winter** — Each year ends with a winter of varying severity. All settlements lose food. Settlements can collapse from starvation, raids, or harsh winters — becoming **Ruins**.
5. **Environment** — The natural world reclaims abandoned land. Nearby thriving settlements may rebuild ruined sites. Ruins not reclaimed are overtaken by forest growth or fade to open plains.

### Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (`owner_id`).

Initial states expose settlement positions and port status only. Internal stats (population, food, wealth, defense) are visible through simulation queries.

---

## API Reference

**Authentication**: JWT via `Authorization: Bearer <token>` header or `access_token` cookie.

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/astar-island/rounds` | Public | List all rounds |
| `GET` | `/astar-island/rounds/{round_id}` | Public | Round details + initial states |
| `GET` | `/astar-island/budget` | Team | Query budget for active round |
| `POST` | `/astar-island/simulate` | Team | Observe one simulation through viewport |
| `POST` | `/astar-island/submit` | Team | Submit prediction tensor |
| `GET` | `/astar-island/my-rounds` | Team | Rounds with your scores, rank, budget |
| `GET` | `/astar-island/my-predictions/{round_id}` | Team | Your predictions with argmax/confidence |
| `GET` | `/astar-island/analysis/{round_id}/{seed_index}` | Team | Post-round ground truth comparison |
| `GET` | `/astar-island/leaderboard` | Public | Astar Island leaderboard |

### POST /astar-island/simulate

The core observation endpoint. Each call runs one stochastic simulation and reveals a viewport window. Costs 1 query from your budget (50 per round).

**Request:**
```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `seed_index` | int (0–4) | Which of the 5 seeds to simulate |
| `viewport_x` | int (≥0) | Left edge of viewport |
| `viewport_y` | int (≥0) | Top edge of viewport |
| `viewport_w` | int (5–15) | Viewport width |
| `viewport_h` | int (5–15) | Viewport height |

**Response:**
```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

Each call uses a different random sim_seed → different stochastic outcome.

### POST /astar-island/submit

Submit prediction for one seed. Resubmitting overwrites the previous prediction.

**Request:**
```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05]
    ]
  ]
}
```

| Index | Class |
|-------|-------|
| 0 | Empty (Ocean, Plains, Empty) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |

Format: `prediction[y][x][class]` — H×W×6 tensor. Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance).

### Error Codes

| Status | Meaning |
|--------|---------|
| 400 | Round not active, invalid seed_index, or validation error |
| 403 | Not on a team |
| 404 | Round not found |
| 429 | Budget exhausted (50/50) or rate limit (max 5 req/sec) |

---

## Scoring

Score is based on **entropy-weighted KL divergence** between your prediction and the ground truth.

### Ground Truth

For each seed, the organizers pre-compute ground truth by running the simulation **hundreds of times** with the true hidden parameters. This produces a probability distribution for each cell.

### Formula

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)           (per cell)
entropy(cell) = -Σ pᵢ × log(pᵢ)              (only dynamic cells matter)

weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
              Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- **100** = perfect prediction
- **0** = terrible prediction
- Static cells (ocean, mountain) have near-zero entropy and are excluded

### Critical: Never assign probability 0.0

If ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`, KL divergence goes to **infinity** — destroying your score. Always enforce a minimum floor:

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

### Per-Round & Leaderboard

- **Round score** = average of 5 per-seed scores (unsubmitted seeds score **0**)
- **Leaderboard** = weighted average across all rounds (later rounds may weigh more)
- **Prediction window** = ~2h 45min per round

---

## Quick Start

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Create or join a team (max 4 members)
3. Complete Vipps verification for prize eligibility
4. When a round is active, use the API to observe and submit predictions

### Example: Observe a simulation

```bash
curl -X POST https://api.ainm.no/astar-island/simulate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": "ROUND_UUID",
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15
  }'
```

### Example: Submit a prediction

```python
import numpy as np

# Build H×W×6 prediction tensor
prediction = np.full((40, 40, 6), 1/6)  # uniform prior

# Apply minimum floor and renormalize
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)

# Submit for each seed
import requests
for seed_index in range(5):
    requests.post(
        "https://api.ainm.no/astar-island/submit",
        headers={"Authorization": "Bearer YOUR_TOKEN"},
        json={
            "round_id": "ROUND_UUID",
            "seed_index": seed_index,
            "prediction": prediction.tolist()
        }
    )
```

---

## Competition Rules

- **Prize pool**: 1,000,000 NOK (1st: 400k, 2nd: 300k, 3rd: 200k, Best U23: 100k)
- **Teams**: 1–4 members, roster locked after first submission
- **Code**: Must be open-sourced (MIT license) and submitted before deadline for prize eligibility
- **Scoring**: Overall score = average of normalized task scores (33.33% per task)
- Full rules: [app.ainm.no/rules](https://app.ainm.no/rules)

## MCP Docs Server

Connect the docs server to Claude Code for AI-assisted development:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
