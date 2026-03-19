# Astar Island — Opening Strategy

Where to start to maximize score, structured by priority.

---

## 0. The #1 Principle: Understand the Scoring Before Writing Code

The scoring formula tells you exactly what matters:

```
score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- Only **dynamic cells** (non-zero entropy) contribute — ocean, mountain, plains are free points
- Cells with **higher entropy** (more uncertain outcomes) are weighted more heavily
- A single `0.0` probability where ground truth is `>0` sends KL to **infinity** → catastrophic score loss

**Implication**: Your first submission should never be your model's raw output. Always enforce a **minimum probability floor** (0.01) and renormalize. This one line of code is worth more than hours of model tuning.

---

## 0.5 Getting Your Auth Token (Step Zero)

Nothing works without authentication. The API accepts a JWT token.

1. Sign in at [app.ainm.no](https://app.ainm.no) with Google
2. Open browser DevTools → Application → Cookies
3. Copy the `access_token` cookie value — this is your JWT
4. Use it as `Authorization: Bearer <token>` in API calls

```python
import os

# Store in environment variable, not in code
TOKEN = os.environ["ASTAR_TOKEN"]
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}
```

The token may expire — if you start getting 401s, re-extract it from your browser.

---

## 1. Immediate Baseline (First 30 Minutes)

Get a score on the board before doing anything clever.

### 1.1 Static Analysis Baseline (Zero Queries)

You can submit predictions **without using any queries**. The initial map state is given to you via `GET /astar-island/rounds/{round_id}` (the `initial_states` array with terrain grids and settlement positions).

**Baseline strategy**:
- Ocean (10) → class 0 with ~100% confidence → `[0.94, 0.01, 0.01, 0.01, 0.01, 0.02]`
- Mountain (5) → class 5 with ~100% confidence → `[0.01, 0.01, 0.01, 0.01, 0.01, 0.95]`
- Forest (4) → mostly stays forest, small chance of ruin/settlement → `[0.05, 0.03, 0.02, 0.05, 0.80, 0.05]`
- Settlement (1) → could grow, die, or become port → `[0.05, 0.35, 0.20, 0.25, 0.10, 0.05]`
- Port (2) → somewhat stable but can be ruined → `[0.05, 0.10, 0.45, 0.25, 0.10, 0.05]`
- Plains/Empty (11/0) → might get settled, forested, or stay empty → `[0.55, 0.10, 0.05, 0.05, 0.20, 0.05]`

This costs **zero queries** and gives you a non-zero score for all 5 seeds. Submit this immediately.

### 1.2 Why This Matters

- Unsubmitted seeds score **0**. Always submit something for every seed.
- The floor enforcement prevents catastrophic KL blowups.
- This baseline lets you compare future improvements against a known starting point.

---

## 2. Query Strategy (First Round — Learning Phase)

You have **50 queries** shared across **5 seeds** and a **40×40 map** with a **15×15 viewport**. That's 10 queries per seed, and each query covers at most 14% of the map. You cannot see everything. Be strategic.

### Time Budget Per Round (~2h 45min window)

| Phase | Time | Activity |
|-------|------|----------|
| **0–15 min** | Setup | Submit static baseline for all 5 seeds. Start queries. |
| **15–60 min** | Observe | Spend query budget systematically. Save all responses. |
| **60–135 min** | Model | Build/refine prediction model. Resubmit improved predictions. |
| **135–165 min** | Final | Final model run. Validate with local scorer. Submit all seeds. |

Don't run out of time with queries unspent or predictions unsubmitted. The baseline protects you — but improving it is where points come from.

### 2.1 Coverage vs. Repetition Trade-off

Two extremes:
- **Maximum coverage**: Spend all queries on different viewport positions → see more of the map, but only one stochastic sample per area
- **Maximum repetition**: Query the same viewport many times on the same seed → see how the simulation varies, but miss most of the map

The simulation is **stochastic** — same initial state produces different outcomes. For probabilistic prediction, you need **multiple samples** per area. But you also need to **see the whole map**.

**Recommended split for round 1**:
- **2-3 seeds**: Cover systematic grid (for spatial coverage)
- **2-3 seeds**: Repeat same viewport position (for distribution estimation)
- Use learnings across seeds — hidden parameters are shared within a round

### 2.2 Systematic Grid Coverage

A 40×40 map with 15×15 viewport → you need a 3×3 grid of viewports to cover the full map (with 1-cell overlap). The toolkit's `simulate_grid()` uses step 14:

```
Viewport positions for full coverage (step 14):
  (0,0)   (14,0)  (25,0)
  (0,14)  (14,14) (25,14)
  (0,25)  (14,25) (25,25)
```

The last column/row starts at 25 (= 40 - 15) to fit within bounds. That's 9 queries for one full scan of one seed. With 50 total, you can do ~5 full scans across different seeds, or a mix of full scans and repeated observations.

### 2.3 Rate Limit

The API enforces **max 5 requests/second**. With 50 queries that's a minimum of 10 seconds of API time. Add a small delay between calls to avoid 429 errors.

### 2.4 What to Record From Each Query

Each query shows the **end-of-50-years state** — the final map after the full simulation, not intermediate time steps.

For each simulation response, store:
- The **grid** (terrain codes for the viewport)
- **Settlement stats** (population, food, wealth, defense, owner_id, has_port, alive)
- The **viewport coordinates** used
- Which **seed_index** was queried

This data is your training set. Structure it for easy aggregation.

---

## 3. Building the Prediction Model (Hours 1-3)

### 3.1 Monte Carlo Empirical Distribution (Simplest Good Model)

The ground truth is itself a Monte Carlo distribution (hundreds of runs). You can approximate it:

1. For each cell position, collect all terrain observations across your queries
2. Count frequencies per class → normalize to probabilities
3. For cells you haven't observed: fall back to the static baseline from Step 1
4. Apply the probability floor (0.01) and renormalize

This is the **minimum viable model** and should be built first.

### 3.2 Neighbor-Based Inference

Cells you haven't directly observed can be inferred from neighbors:
- A cell adjacent to ocean is more likely to be a port than a cell inland
- A cell surrounded by forest is likely forest
- Settlements tend to cluster and expand from initial positions
- Ruins are former settlements — they appear where settlements were

### 3.3 Settlement Stats as Hidden Parameter Probes

The simulate response includes `population`, `food`, `wealth`, `defense`, `owner_id` per settlement. These are your window into the hidden parameters:

- **Low food across many settlements** → harsh winters or low food production
- **High wealth + active ports** → strong trade dynamics
- **Frequent faction changes (owner_id shifts)** → aggressive raiding
- **Rapid settlement expansion** → high expansion rate parameter
- **Many ruins** → high winter severity or aggressive conflict

Track these stats across multiple queries. They tell you what the hidden parameters are doing, which is the key to predicting unobserved cells.

### 3.4 Initial State as Strong Prior

The initial terrain grid is the strongest signal you have:
- **Ocean** stays ocean (100%)
- **Mountain** stays mountain (100%)
- **Forest** mostly stays forest, but may be cleared near settlements
- **Empty/Plains** — the most uncertain cells; these are where settlements expand to
- **Existing settlements** — will either survive (Settlement/Port) or die (Ruin)

Build your prior from initial state, then update with observations.

---

## 4. What to Optimize First (Priority Order)

### Priority 1: Static cells (low effort, guaranteed points)

Ocean and mountain never change. Mark these with near-100% confidence. This is free score and reduces the problem to only dynamic cells.

### Priority 2: Submit for all seeds (avoid zero scores)

Even a mediocre prediction beats 0. Submit your baseline for all 5 seeds before investing in model improvements.

### Priority 3: Probability floor (prevent catastrophic losses)

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

This single operation may be worth more than any model improvement.

### Priority 4: Per-cell empirical distributions from observations

Use your query budget to sample the map and build empirical frequency distributions.

### Priority 5: Spatial inference for unobserved cells

Propagate information from observed cells to unobserved neighbors using terrain context.

### Priority 6: Cross-seed learning

All seeds in a round share the same **hidden parameters** (growth rates, winter severity, raid aggression, trade strength, etc.), but each seed has a **different map layout**. Cross-seed learning means inferring the dynamics — not copying spatial predictions. If you observe high settlement mortality in seed 0, that tells you winter severity or raid aggression is high, and you should predict more ruins across all seeds.

---

## 5. Architecture Recommendation

```
astar/
├── __init__.py            # Package init
├── client.py              # API client with auto-logging to data/
├── replay.py              # Offline replay store (load logged responses)
├── model.py               # Prediction model (prior + observation updates + floor)
├── viz.py                 # Matplotlib visualizations (grids, heatmaps, comparisons)
├── submit.py              # Orchestrator: build predictions, local scorer, submit
data/
└── round_{id}/            # Auto-logged observations per round (flat files)
    ├── round_detail_*.json
    ├── sim_s0_x0_y0_*.json
    ├── analysis_s0_*.json
    └── ...
```

### Key Design Decisions

- **Python** is the right choice here — NumPy for tensor manipulation, easy REST API calls, fast iteration
- **Save all observations** — you can reuse data from early rounds to inform later predictions
- **Separate query strategy from prediction model** — you'll want to swap both independently
- **Automate submission** — write a script that builds predictions for all 5 seeds and submits them in one go

---

## 6. Resubmission Strategy

Resubmitting for the same seed **overwrites** your previous prediction — only the last submission counts. Use this aggressively:

1. **Submit baseline immediately** (zero queries, static priors)
2. **Run queries**, build empirical model
3. **Resubmit** with improved predictions
4. **Keep iterating** — every resubmission replaces the last, so there's no risk of making things worse by submitting again

⚠️ Note: the *last* submission counts, not the *best*. If your model regresses, your latest submission will score worse. Validate locally before resubmitting if you have a local scorer.

---

## 7. Round-Over-Round Improvement

Each completed round gives you access to **ground truth** via the analysis endpoint. This is gold:

1. **Compare your prediction vs. ground truth** per cell
2. **Identify systematic biases** — are you consistently over/under-predicting settlements?
3. **Measure per-class accuracy** — which terrain type do you predict worst?
4. **Tune your priors** using actual distributions from completed rounds
5. **Build a local scorer** that computes the same entropy-weighted KL metric — this lets you evaluate locally before submitting

### Local Scorer Implementation

```python
import numpy as np

def score_prediction(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute the entropy-weighted KL divergence score (0-100).
    prediction: H×W×6 array (your submission)
    ground_truth: H×W×6 array (from analysis endpoint)
    """
    p = np.clip(ground_truth, eps, 1.0)   # ground truth
    q = np.clip(prediction, eps, 1.0)     # your prediction
    
    # Per-cell KL divergence: KL(p || q)
    kl = np.sum(p * np.log(p / q), axis=-1)  # H×W
    
    # Per-cell entropy
    entropy = -np.sum(p * np.log(p), axis=-1)  # H×W
    
    # Weighted average (only dynamic cells contribute)
    total_entropy = np.sum(entropy)
    if total_entropy < eps:
        return 100.0  # all cells are static → perfect by default
    
    weighted_kl = np.sum(entropy * kl) / total_entropy
    
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return score
```

Use this after round 1 completes to evaluate offline before resubmitting in later rounds.

### The Feedback Loop

```
Round N: Submit prediction → Get scored → Download ground truth
         ↓
         Analyze errors → Adjust model/priors → Better prediction for Round N+1
```

Teams that iterate fastest through this loop will climb the leaderboard. Automate as much of it as possible.

### Round Weight Escalation

Later rounds may carry **higher weight** in the leaderboard calculation. Combined with the **hot streak score** (average of last 3 rounds), this means:
- Early rounds are partly for **learning** — invest in understanding the mechanics
- Later rounds **matter more** for final placement
- A strong finish is more valuable than a strong start

---

## 8. Common Pitfalls

| Pitfall | Why It Hurts | Prevention |
|---------|-------------|------------|
| Assigning 0.0 probability | KL → ∞, destroys score | Always enforce 0.01 floor |
| Not submitting all seeds | Unsubmitted = score 0 | Submit baseline for all 5 immediately |
| Wasting queries on static cells | Ocean/mountain never change | Focus queries on dynamic areas |
| Over-engineering before baseline | No score on the board | Get baseline submitted in 30 min |
| Ignoring cross-seed information | Hidden params are shared | Aggregate dynamic insights (not spatial) across seeds |
| Resubmitting without local validation | Last submission counts, not best | Build local scorer before resubmitting |
| Ignoring settlement stats | Miss hidden parameter signals | Track population/food/wealth/defense trends |
| Not respecting rate limit (5 req/s) | 429 errors waste time | Add delay between API calls |
| Not saving observations | Can't build distributions | Log every query response |
| Uniform prediction everywhere | Ignores free knowledge from initial state | Use initial terrain as strong prior |

---

## 9. Checklist

- [ ] Auth token extracted and working (`ASTAR_TOKEN` env var set)
- [ ] Can fetch active round and initial states
- [ ] Baseline prediction submitted for all 5 seeds (zero queries)
- [ ] Query execution working — can call `/simulate` and parse response
- [ ] Observations stored to disk
- [ ] Empirical distribution model built from observations
- [ ] Probability floor enforced on all submissions
- [ ] Score tracking (runs.md) started
- [ ] Post-round analysis script ready for when round completes
- [ ] Cross-seed learning implemented
