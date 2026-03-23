# Post-Mortem: Astar Island — NM i AI 2026

**Team**: Novanet  
**Final result**: Rank 23 of 100+, weighted score 258.1  
**Best leaderboard round**: 84.9 raw (official)  
**Competition period**: 19–23 March 2026  
**Winner**: Løkka Language Models — 266.6 weighted (+8.5 above Novanet)

---

## Final Standing

| Rank | Team | Rounds | Best raw | Weighted |
|---|---|---|---|---|
| 1 | Løkka Language Models | 19 | 84.3 | 266.6 |
| 2 | merge conflict | 19 | 86.9 | 265.1 |
| 3 | Laurbærene | 18 | 69.1 | 264.9 |
| 4 | Moxwold | 22 | 84.1 | 263.4 |
| 5 | Aistein | 18 | 83.7 | 263.0 |
| 6 | WinterIsComing_ | 19 | 85.0 | 262.2 |
| … | | | | |
| 13 | Stable Confusion | 18 | 85.2 | 259.9 |
| … | | | | |
| **23** | **Novanet** | **20** | **84.9** | **258.1** |
| 24 | Ave Christus Rex | 18 | 82.2 | 257.6 |
| … | | | | |
| ~100+ | (field) | | | |

Rank 23 of 100+. Our best raw score (84.9) was higher than the winner's (84.3). The gap was not model capability — it was round timing and process discipline.

---

## What Was the Challenge?

### The competition explained

NM i AI 2026 – Challenge 2 was called "Astar Island". We were given access to a black-box simulator of a Norse civilisation. The simulator ran a 40×40-cell world forward 50 simulated years, with terrain types changing based on hidden parameters (winter severity, expansion rate, raid frequency, etc.).

**The goal**: Predict the final state of the entire map — not as a single binary label, but as a *probability distribution* for each cell over every possible terrain type. The output is a 40×40×6 tensor where each cell holds a probability for the six classes: Empty, Settlement, Port, Ruin, Forest, Mountain.

**The terrain classes**:
| Class | Description |
|---|---|
| Empty | Uninhabited plains |
| Settlement | Active Norse settlement |
| Port | Coastal settlement with harbour |
| Ruin | Destroyed or abandoned settlement |
| Forest | Natural forest terrain |
| Mountain | Immutable mountain |

**The scoring formula**:
$$\text{score} = \max(0, \min(100,\ 100 \times e^{-3 \times \text{wKL}}))$$

where `wKL` is entropy-weighted KL divergence — only cells with uncertainty contribute, and the more uncertain a cell is, the more it weighs in the final score.

**The leaderboard formula**:
$$\text{leaderboard} = \max_n \left( \text{raw\_score}_n \times 1.05^n \right)$$

Only your **single best** weighted round counts. Later rounds weigh more: round 22 is worth $1.05^{22} \approx 2.93\times$ more than round 1.

### What we had to work with

- **50 queries per round**, shared across 5 seeds (stochastic starting points)
- Each query returned a **15×15 viewport** — a snapshot of the simulator's final state at a chosen position
- The simulator was **stochastic**: two queries on the same seed produced different outputs
- We never saw ground truth during the competition — only our score after submission
- We were given the initial map (which cells start as what) plus whatever we could observe through queries

### Our solution approach at a glance

```
Initial state (given for free)
        ↓
Spatial ML model (GBM + MLP + U-Net)
   → Predicts probability distribution per cell based on neighbourhood
        ↓
Observations (50 queries → 45 covering viewports + 5 repeats)
   → Calibrates round transition rates (E→S, S→S etc.)
   → Bayesian overlay: adjusts model prediction per cell
        ↓
Post-processing (temperature scaling, calibration, probability floor)
        ↓
Submission: 40×40×6 tensor per seed × 5 seeds
```

---

## Strategy

Decisions made at the strategic level determined our final result more than any single model choice. This section covers how we attacked the problem and, more importantly, where the approach broke down.

### The competitive meta-game

The leaderboard awards only your **single best** weighted round (formula above). Round 22 is worth $1.05^{22} \approx 2.93\times$ a Round 1 score. This defines which rounds you can afford to experiment on and which you cannot touch:

| Round range | Weight tier | Strategic mode |
|---|---|---|
| R1–R9 | 1.0–1.5× | Experiment freely — low cost of failure |
| R10–R15 | 1.5–2.0× | Validate before deploying — pilot on low-weight rounds first |
| R16–R22 | 2.0×+ | Conservative only — no unvalidated changes |

**We did not fully internalise this until Round 17.** For most of the competition we optimised average score across rounds, not the weighted maximum. This had downstream effects on nearly every strategic decision we made.

### How we attacked the problem

**Query approach**: 50 queries per round across 5 seeds →  ~10 per seed. A standard 9-viewport grid covers approximately 85% of the 40×40 map. Each viewport is a single stochastic snapshot of the simulator's final state at a chosen position. From these, we estimated per-round transition rates: how often does Empty→Settlement, Settlement→Settlement, Settlement→Ruin, and so on.

**Bayesian overlay**: After the spatial model produces a base prediction, the observed transition rates update it per cell. If we observed high settlement survival this round, the overlay shifted probabilities toward Settlement on settlement cells. The calibration of this overlay — how aggressively to update versus how much to trust the model prior — turned out to be the single most consequential tuning decision of the competition.

**Model philosophy**: Every model change gated by LORO (Leave-One-Round-Out) cross-validation. If a change does not produce positive LORO-delta validated against all rounds seen so far, it does not go into production. This discipline produced 10 model versions across 22 rounds with no regressions — until R16.

**The core insight**: Even with perfect transition rates from ground truth, a pure transition matrix cannot score more than ~21 points. The spatial model — which asks *which specific cells change*, based on neighbourhood context — scores 67+ on the same round. The question was never *what happens on average*. It was *which cells*.

### What worked

**LORO as a gatekeeping discipline.** We rejected multiple improvements that looked positive in-sample but failed LORO: rotation augmentation (−0.27 LORO-delta), a learned debiaser (100× worse on LOO). This kept overfitting artefacts out of production.

**Bayesian overlay as a consistent, safe gain.** An entropy-scaled overlay (prior_strength 5→100 per cell) added +0.04 to +1.41 points per round with zero regressions across all rounds — except when misconfigured in R16.

**KNOWLEDGE.md created real institutional memory.** Logging every finding, failure, and rollback in real time meant recovery after R16 was immediate — we knew exactly what to revert to and why, without wasting time on diagnosis.

**Collapse round handling turned from a weakness into a strength.** R8 (first collapse round) scored 73.68. By R19, we scored 94.84 with all 5 seeds within 0.6 points of each other. The strategy of accumulating collapse-round training data paid off completely.

**Critical bug caught under fire.** A silent train/test domain shift on settlement-stat features was killing 3+ rounds. Finding and fixing it before Round 9 was worth **+17 raw points on R9 alone** and briefly sent us to #1 on the leaderboard.

### Where the strategy failed

#### 1. R16 — the defining mistake of the competition

We changed the Bayesian overlay from `(min_ps=5, max_ps=100)` to `(min_ps=0.5, max_ps=3)` mid-Round 16, based on an in-sample backtest showing +2.72 average improvement.

The change was not LORO-validated. It was not piloted on a live round. It was deployed to a round with weight $1.05^{16} \approx 2.18\times$ — the highest-weight round we had completed to that point. It directly violated the documented rule: *no experiments mid-round*.

Round 16 scored **71.51**. With the original settings it would have scored ~87.00. The difference: **−15.5 raw points on that round**. The top team (Løkka Language Models) finished at 266.6. We finished at 258.1 — a gap of 8.5 weighted points.

Note: the leaderboard takes the best single weighted round, and our leaderboard maximum came from a later round. R16's direct leaderboard impact was therefore limited — but the mistake burned time, disrupted momentum, and established an unvalidated-deployment precedent that could have recurred on an even higher-weight round. The principle stands.

The rule existed. It was written in KNOWLEDGE.md. It was not followed, and no mechanism existed to enforce it.

#### 2. The leaderboard formula was treated as background noise

We knew the formula. We didn't act on it. Consequences:

- We ran experiments on high-weight rounds (R16 is the worst case, but not the only one)
- We optimised average score, not single-best weighted score
- We were overly cautious on late rounds where maximum weight would have rewarded boldness — and too bold on rounds where it punished failure

The correct framing: the competition is not 22 parallel rounds. It is **one attempt to produce the single best weighted score**, and every other round exists to generate training data, pilot experiments, or be discarded. The risk tier table above should have been printed on day one and consulted before every decision.

#### 3. U-Net arrived six rounds too late

U-Net became V10 at Round 15. With LORO average 83.32 vs. 82.54 for GBM+MLP from R14 data, it was clearly superior at that point. The infrastructure — 40×40 input grids, D4 symmetry TTA, entropy-weighted KL loss — was not technically exotic. It could have been prototyped during R8–R12, when round weights were still low and the cost of a failed experiment was minimal.

Lost rounds: R9 (~90.6 → ~93+), R10 (~91.6 → ~94+). Estimated lost weighted points: **15–25**.

#### 4. The first submission was deployed without a known-good baseline

Round 2 scored **3.02**. The naive empirical model replaced the prior with a single observation and produced probabilities of 0 or 1. Any sanity check — printing the output tensor, comparing it to the initial state distribution — would have caught this immediately.

The opening-strategy document described Bayesian updating correctly. The implementation was not verified before going live.

#### 5. The observation budget was misallocated for eight rounds

R6–R13: the leftover 5 queries were spent on offset viewports (new grid positions). Better: repeat the standard 9-viewport grid on the most dynamically uncertain seed. Repeated observations on the same seed provide independent stochastic samples that directly strengthen Bayesian overlay calibration. Fixed in R14; should have been fixed in R6.

### Strategic summary

| Decision | What we did | What we should have done | Estimated cost |
|---|---|---|---|
| R16 overlay change | Deployed on in-sample backtest alone | LORO-validate; pilot on a low-weight round first | −15.5 raw pts that round; indirect cost in time and precedent |
| Leaderboard formula | Internalised at R17 | Explicit risk tiers before R1 | Significant |
| U-Net architecture | Deployed at R15 | Prototype at R8–R10 | **−15–25 weighted pts** |
| R2 baseline | Deployed unvalidated | Sanity-check prediction tensor before R1 | ~−50 early weighted pts |
| Extra 5 queries R6–R13 | Offset viewports | Repeat grid on most dynamic seed | −0.5/round × 8 rounds |
| Daily structured reviews | None | `daily-review.prompt.md` every evening | Would have blocked R16 |
| Agent guardrails | Prose rules in KNOWLEDGE.md | `DEPLOYMENT_RULES.md` + `CURRENT_CONFIG.md` + checklist | Would have blocked R16 |

---

## Agent Use: What Should Have Been Done Differently with GitHub Copilot

We used **GitHub Copilot Agent Mode** as the primary implementation partner throughout the competition — for model implementation, debugging, hyperparameter sweeping, documentation, and decision support. This section covers what worked and what should have been formalised from day one.

### What worked well

The agent was effective at:
- Rapidly implementing model versions (GBM → MLP → U-Net) from conceptual descriptions
- Maintaining the KNOWLEDGE.md protocol and appending findings as they emerged
- Debugging concrete errors (e.g. `n_jobs=-1` Windows hang, U-Net key-name mismatch)
- Writing backtest and hyperparameter sweep scripts on demand

### Problem 1: No explicit experiment guardrails

The agent was asked to implement the Bayesian overlay change for R16 based on backtest results, without any explicit instruction about *what type of validation* is required before a live-round deployment. The rule existed in KNOWLEDGE.md as prose — not as a machine-readable constraint.

**What should have existed:**
```markdown
# DEPLOYMENT_RULES.md (should have existed from day 1)

## Absolute constraints
- NEVER change hyperparameters on a live round without LORO-validation
- NEVER deploy new logic untested on at least 1 historical round with GT
- NEVER rely on in-sample backtest alone for overlay tuning (documented failure: R16)
- Every parameter change proposal must include LORO-delta, not just in-sample delta

## Experiment hierarchy
1. Historical GT test (in-sample backtest) — hypothesis generation
2. LORO-validation — hypothesis testing
3. Live round with low weight (<1.7) — piloting
4. Live round with high weight (>2.0) — validated, stable settings only
```

Without this document, nothing stopped the agent — or us — from doing what we did in R16.

### Problem 2: KNOWLEDGE.md was designed for humans, not the agent

KNOWLEDGE.md was an excellent human document. As an instruction set for an agent it had weaknesses:
- **Strikethrough text** (~~old~~ → new) requires interpretation, not just reading
- **No machine-readable structure** for "current, validated configuration"
- **No priority signal**: which rules are absolute? Which are recommendations?
- **Temporal ambiguity**: a finding from R8 may have been superseded by R12, but the agent must read carefully to understand that

**What should have existed:** A `CURRENT_CONFIG.md` always reflecting the *current, validated* configuration — not the history. KNOWLEDGE.md should have been pure historical log; a separate document should have been the operative settings sheet.

```markdown
# CURRENT_CONFIG.md (always kept in sync)

## Model settings (validated as of R21)
UNET_BLEND_W = 0.90         # LORO-validated: 80%=92.06, 90%=92.37, 100%=92.52
MIN_PS = 5                  # WARNING: never below 5 (R16 catastrophe documented)
MAX_PS = 200                # LORO-insensitive (1-20 all equal), MAX_PS=200 best
PROB_FLOOR = 0.0003         # LORO-optimal, sweep done post-R17

## Change locks
These parameters require LORO-revalidation before any change:
- UNET_BLEND_W (sensitive: R13 prefers 50%, all others prefer 90%+)
- MIN_PS / MAX_PS (R16: aggressive overlay = -15.5 pts on volatile rounds)
```

### Problem 3: No behavioural constraints in copilot-instructions.md

The project's `.github/copilot-instructions.md` described context and coding conventions well, but lacked **explicit behavioural rules** for the agent — no requirement to cite LORO-delta, no warning that in-sample backtest results are misleading for overlay tuning, no rule preventing live deployment of unvalidated changes.

**Suggested additions:**
```markdown
## Experiment discipline

When proposing hyperparameter changes:
1. **Always state LORO-delta**, not just in-sample backtest delta
2. **Explicitly warn** if the proposal has not been LORO-validated
3. **Do not propose deploying to a live round** if the change:
   - Has only been tested in-sample
   - Touches overlay/floor parameters (see R16 warning in KNOWLEDGE.md)
   - Violates a change lock defined in CURRENT_CONFIG.md

Note: in-sample backtest on observation data is known to be misleading for overlay tuning.
```

### Problem 4: No pre-submit checklist

We should have had a checklist the agent was instructed to run as the final step before any submission:

```markdown
# PRE-SUBMIT CHECKLIST

- [ ] LORO-delta documented for all model changes since last submission
- [ ] CURRENT_CONFIG.md updated and reflects the configuration being deployed
- [ ] No overlay/floor changes without LORO-validation (see CURRENT_CONFIG change locks)
- [ ] All 5 seeds run locally with `build_prediction()` — no exceptions raised
- [ ] Probability floor is active (floor ≥ 0.0001)
- [ ] All seeds submitted with status=accepted confirmed
```

This checklist would have stopped the R16 deployment. It takes two minutes. It did not exist.

### Problem 5: The agent's role was not clearly defined

Without a clear role definition it was ambiguous when the agent's job was to *implement what it was asked* versus *challenge the decision*. The agent should have had an explicit instruction to:
1. **Never implement a live deployment change** without citing relevant KNOWLEDGE.md warnings
2. **Always flag** when a proposal violates documented rules in copilot-instructions.md
3. **Ask explicitly** when something appears to contradict the "no experiments mid-round" rule

---

## Google Cloud: What We Should Have Used

GCP access was provided to all teams. We used it for one thing: a **Cloud Run service** (`auto_round.py`) that polled for new rounds overnight and auto-submitted predictions so we didn't miss rounds while sleeping. That was genuinely useful. Everything else was left untouched — while locally, CPU and memory were a constant constraint throughout the competition.

### What we used

| Service | What for |
|---|---|
| Cloud Run | Overnight polling loop — auto-submit when a new round opens |

### The real problem: local resource saturation

Running training, LORO sweeps, hyperparameter searches, and backtests locally caused high CPU and memory consumption throughout the competition. This was not just inconvenient — it had direct consequences:

- **Training ran slowly**, which delayed prototyping new architectures (U-Net in particular)
- **LORO sweeps blocked the machine** for extended periods, reducing how many experiments we could run per day
- **Backtest scripts competed with each other** — running multiple in parallel risked OOM, so they were run sequentially, which was slow
- **The machine was not always available** — local compute limits meant we had to wait for jobs to finish before starting the next experiment

All of this was offloadable to GCP at low cost. We chose not to — and it cost us time at every stage.

### What the docs actually offered

The official GCP documentation at `app.ainm.no/docs/google-cloud/services` listed all of this explicitly. Nothing below required special access or configuration:

| Service | Documented purpose | Did we use it? |
|---|---|---|
| Cloud Run | Deploy containerised APIs | ✅ Polling loop only |
| Compute Engine | "Need GPU or persistent server" | ❌ |
| Cloud Shell | Free Linux VM, always authenticated, 5 GB persistent | ❌ |
| Cloud Storage | "Store datasets, model weights, logs" | ❌ |
| Vertex AI | Managed ML platform, access to models | ❌ |
| Gemini Code Assist | AI coding companion in Cloud Shell Editor | ❌ |
| Gemini CLI | AI assistant in Cloud Shell terminal | ❌ |
| NotebookLM | AI-powered research notebook | ❌ |
| MCP Server (`mcp-docs.ainm.no`) | Connect competition docs to Claude Code | ❌ |

### What we should have used

#### 1. Cloud Shell for all training and compute (highest impact, zero cost)

The most important miss — and it was free. Cloud Shell is a Linux VM built into the GCP console: Python 3, git, Docker, and `gcloud` all pre-installed with no setup. It has a 5 GB persistent home directory. It is **always available, always authenticated, and has no cost**.

Every slow operation we ran locally — U-Net training, LORO sweeps, backtest scripts, hyperparameter searches — could have been offloaded to Cloud Shell instantly. The local machine would have remained free at all times. No OOM risk. No waiting. No blocked experiments.

This requires no GCP expertise. It is a browser tab.

#### 2. Compute Engine (GPU) for U-Net training

The docs explicitly recommend Compute Engine when you "need a GPU or persistent server." A T4 GPU instance would have trained the U-Net in a fraction of the local CPU time. The docs flagged this directly — we missed it.

**This is the single highest-impact miss with a score consequence.** The U-Net arriving six rounds late cost an estimated 15–25 weighted points. Slow local training was a direct contributor to that delay.

#### 3. Parallel LORO validation on Cloud Run Jobs

LORO with 22 rounds means 22 sequential training + evaluation passes. Run locally, this blocked the machine for extended periods. Using Cloud Run Jobs, all 22 folds could have executed in parallel across independent containers — turning a multi-hour local sweep into minutes, without touching local resources.

#### 4. Cloud Storage for model artifact versioning

The docs document Cloud Storage explicitly for "store datasets, model weights, logs." Model files (`unet_model.pt`, `spatial_model.pkl`, `mlp_model.pt`) were overwritten in-place on disk. The R20 U-Net loading bug took time to debug partly because there was no prior checkpoint to roll back to. GCS versioning would have made rollback trivial.

#### 5. MCP Server for docs-connected AI development

The docs front-page documented a competition MCP server: `claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp`. This would have connected the full competition documentation directly to Claude Code so the agent could answer questions about the API, scoring formula, and task rules without manual lookups. Never used.

#### 6. Gemini tools as a second AI pair programmer

The GCP account included Gemini Code Assist in Cloud Shell Editor, a Gemini CLI in the Cloud Shell terminal, and Gemini Cloud Assist in the console sidebar. Used alongside GitHub Copilot, these were additional tools for faster implementation and debugging. Never used.

#### 7. Monte Carlo stochastic simulator as batch jobs

Running thousands of simulation runs from the initial observed state (an unexplored fix for R7/R18-class partial-stability rounds) was never attempted because it would have saturated the machine. As Cloud Run batch jobs it would have been cheap, parallelisable, and non-blocking.

### Why we didn't use these

The official GCP documentation listed all of this clearly and concisely. We didn't read it. `google-cloud.md` in this repo covered only the Cloud Run deployment pattern — it was written to answer "how do I deploy?" not "what is available?" Neither the project documentation nor `copilot-instructions.md` included a prompt to audit the full service list. The cost of local saturation was experienced as unavoidable friction rather than diagnosed as a solvable problem with available tools two clicks away.

---

## Model Development: From 3 to 94.84 Points

We went from a broken naive model (Round 2: **3.02**) to peak performance (**94.84** in Round 19) across 22 rounds.

### Key model versions

| Version | Key change | LORO gain |
|---|---|---|
| Baseline | Statistical transition matrix | 70–74 LORO |
| V3 | Round-conditioned spatial LightGBM (27 features) | +5 LORO |
| V5 | Entropy-weighted training + bias calibration | +0.88 |
| V7 | KL-loss MLP (aligns loss function with score) | +1.58 |
| V8 | Edge-distance feature | +0.46 |
| V9 | 7×7 ring features (extended neighbourhood) | ~+1.5 |
| V10 | U-Net with TTA (test-time augmentation) | +2–3 live |

**Final architecture (V10)**:
- **50% LightGBM + XGBoost ensemble** — 37 spatial features + 8 round-level features
- **50% PyTorch MLP** — trained directly with entropy-weighted KL loss
- Blended with **90% U-Net** — spatial CNN that sees the whole map at once with D4 symmetry TTA
- Adaptive Bayesian observation overlay applied after prediction

LORO (Leave-One-Round-Out) cross-validation was our primary evaluation method throughout the competition.

---

## What We Missed

### 1. R12 — the "stability surge" round we never solved
Round 12 had GT S→S = 0.595 — the highest settlement survival ever seen. The model had never encountered anything similar. Official score: **58.89**. Even after including R12 data in training, LORO only reached 73.31 for that round.

We never found a solution for stability-surge rounds — rounds where settlements are far more durable than the historical average.

### 2. R7 and R18 — the chronic "partial stability" weakness
Rounds where settlements are *moderately* stable (GT S→S ~0.5–0.6) but observations overestimate volatility (~38%) are our recurring weak point. The shrinkage matrix helps but does not fully resolve it.

Unexplored fix: a stochastic simulator generating thousands of runs from the initial state, producing the actual probability distribution directly rather than an approximation from 9 single observations.

### 3. No hierarchical model across seeds
Within a round, all 5 seeds share the same hidden parameters (winter severity, expansion rate, etc.). We used per-round transition rates as features, but never built a **hierarchical Bayesian model** that explicitly shared parameter priors across seeds within a round.

### 4. The query budget was never optimally allocated
We always spent 9 queries per seed for full coverage plus 5 leftovers. We never:
- Allocated adaptively based on which seeds were *most uncertain*
- Used partial coverage on "easy" seeds and deep repeat coverage on "hard" ones
- Prioritised cells near settlement boundaries (highest entropy)

A dynamic allocation strategy could have gained 0.5–1.0 extra raw points per round.

---

## What Was Our Potential?

### Model ceiling (from in-sample backtests)
| Round | Best in-sample | Official score | Gap |
|---|---|---|---|
| R8 | 95.7 | 73.68 | +22 |
| R9 | 93.4 | 90.59 | +2.8 |
| R15 | 94.4 | 92.53 | +1.9 |
| R19 | 95.3 | 94.84 | +0.5 |

R19 was within 0.5 points of our theoretical ceiling.

### What a clean run would have scored

| Scenario | Estimated impact |
|---|---|
| U-Net from R9 instead of R15 | **+20–30 weighted points** |
| R16 disruption cost (time, momentum, precedent) | Hard to quantify |
| Theoretical best (95 raw on last round) | Would exceed 266.6 |

**Løkka Language Models** won at 266.6 with a best raw of 84.3 — lower than our best raw of 84.9. They won on timing: their peak came on a higher-weighted round. The gap to first was **8.5 weighted points**. With U-Net deployed six rounds earlier and our late-round peak maintained, we had the model to close it.

---

## What We Didn't Fully Understand

These are things that were nominally known but whose *operational consequences* were not properly understood, investigated, or acted on. The distinction from "What We Missed" is sharpness: these are places where a clearer conceptual model at the start would have changed our behaviour from day one.

### 1. The stochastic gap: observations are single runs, ground truth is many-run average

This is the most important conceptual gap in the entire competition.

The **ground truth probability tensor** — what we were scored against — represents the probability distribution of cell states *averaged across a very large number of simulator runs* from the same initial state. The **queries we issued** each returned a *single stochastic run*. These are fundamentally different things.

A settlement that "died" in one of our observed runs had some probability of surviving in the ground truth distribution. Our observation-calibrated transition rates therefore systematically *overestimated volatility*. In Round 7, observed S→S was 38.3% but GT S→S was 60.5% — a 22-point gap caused entirely by this asymmetry.

We "knew" the simulator was stochastic. We did not operationalise what that meant for the credibility of any single observation. The correct prior for "what is the GT probability of settlement survival?" is not "what fraction of our observations showed survival?" — it is closer to "what would the mean be across 1,000 runs?" Getting from one stochastic observation to a reliable estimate of the run-average probability requires either repeated observations of the same cell or a strong model prior. We had both tools (repeat queries and the Bayesian overlay) but did not frame the problem this way until R7, and even then never derived the correct statistical model.

**What should have been understood from day 1**: each query provides one sample from a distribution. The overlapping viewport grid gave 1–2 samples per cell. Even 5 repeated observations of the same cell give a wide confidence interval on the true run-average probability. The Bayesian overlay's `min_ps` / `max_ps` parameters were implicitly a model of this uncertainty — but were never calibrated against it explicitly.

### 2. What entropy-weighted scoring actually rewards — and its implication for query placement

The scoring formula weights prediction error by cell entropy: cells where the ground truth distribution is uncertain count more. Cells that are almost certainly mountain or almost certainly stable empty plains count for almost nothing.

The direct implication — never fully exploited — is that **query placement should target high-entropy cells**. A viewport centred on an active settlement boundary, where cells are transitioning between multiple states, is worth far more than a viewport that confirms a stable forest stays forest.

Viewport positions were fixed at a uniform 9-grid from Round 2 onward. The grid was good enough, but a placement strategy that dynamically targeted settlement clusters and boundary zones from the initial map would have concentrated observations exactly where scoring penalised bad predictions most. This is a five-minute analysis of the initial state that never happened.

### 3. LORO as a biased estimator once the U-Net was introduced

LORO is only a valid estimate of live-round performance if the model is retrained per fold. For LightGBM and MLP this was true. For the U-Net it was not: a single global model was reused across all folds. This means every LORO sweep that included U-Net blend weight was **optimistically biased for U-Net** — higher U-Net weights looked better in LORO than they would in practice, because the U-Net had already seen the held-out round during global training.

KNOWLEDGE.md noted this at R17: *"treat 100% as an optimistic upper bound, with 80-90% as the safer operating range."* It should have been noted at R15 when the U-Net was introduced. All blend-ratio decisions based on LORO from R15 onward were made with a tool that was giving slightly misleading numbers for the most important dimension of the decision.

### 4. The leaderboard formula as an *operative daily constraint*, not background knowledge

We read the formula. We understood it mathematically. We did not use it to make daily decisions until Round 17.

The gap is not comprehension — it is operationalisation. Understanding the formula implies: printing round weights for all upcoming rounds, declaring a risk mode for each round, and making every experiment proposal conditional on the current round's weight. None of this happened as a daily practice. The formula was background context, not a decision tool.

The five-minute exercise of writing out R1–R22 weights and circling R16+ in red should have happened before the competition started. We did the equivalent at R17 and changed our behaviour immediately. We just needed to do it sixteen rounds earlier.

### 5. GCP as a compute environment, not just a hosting target

Covered in depth in *Google Cloud: What We Should Have Used*. The short version: local CPU and memory were a binding constraint on training speed, experiment parallelism, and the feasibility of computationally expensive approaches (stochastic simulation, parallel LORO). The GCP account gave unlimited access to Cloud Shell (free Linux VM, always available), Compute Engine with GPUs, and Cloud Run Jobs for parallel workloads. We used none of these. The local bottleneck was felt as unavoidable friction rather than diagnosed as a solvable problem.

---

## Overall Assessment

| Category | Verdict |
|---|---|
| Model architecture | ✅ Excellent — progressive, LORO-validated throughout |
| Feature engineering | ✅ Strong — neighbourhood rings, edge distance, round features |
| Execution discipline | ⚠️ One critical failure (R16) out of 22 rounds |
| Strategic understanding | ⚠️ Leaderboard formula understood too late; round timing cost us the win |
| Early rounds | ❌ Missed R1–R4, R2 catastrophic — significant weighted points left early |
| Research velocity | ✅ 10 model versions in 22 rounds with clear gains each time |
| Peak performance | ✅ 84.9 raw (official best) — rank 23 of 100+ teams, 8.5 pts from first |
| Agent instruction | ❌ Too few formalised guardrails, checklists, and separation of config vs. log |
| GCP / infrastructure | ❌ Used only for polling loop — GPU training, parallel LORO and Monte Carlo simulation never attempted |
| Conceptual clarity | ⚠️ Stochastic gap, LORO bias, leaderboard formula — all known but not operationalised |

### In one sentence

> We had the model quality to win — rank 23 with a higher best raw than the first-place team. The gap was entirely strategic: U-Net six rounds late, and leaderboard formula exploitation understood six rounds late.

---

## Should We Have Run Daily Post-Mortems During the Competition?

**Yes — and the R16 catastrophe is the proof.**

A structured end-of-day review was not about reflection. It was about forcing the discipline checks that the agent and the team would otherwise skip under time pressure.

### What it would have caught

| Day | What a review would have flagged |
|---|---|
| Day 1 | Leaderboard formula not yet fully internalised — forced explicit calculation of future round weights |
| Day 2 | R2 baseline bug pattern — in-sample-only validation as a systemic risk |
| Day 3 | Bayesian overlay change in queue, not LORO-validated — **would have blocked the R16 deployment** |
| Day 4 | U-Net LORO superiority already confirmed; no explicit plan to prototype earlier |

A rule compliance check — *"was this LORO-validated before deployment?"* — is one question. It takes ten seconds. It did not exist.

### Why we didn't do it

The competition moved fast. KNOWLEDGE.md was a running log, but it was append-only and human-oriented. There was no structured checkpoint forcing us to:
- Compare today's live scores against LORO expectations
- Verify that CURRENT_CONFIG.md (which didn't exist) matched what was actually submitted
- Recalculate which future rounds had the highest weight and adjust risk appetite accordingly
- Produce a forced-priority list of maximum 3 items for tomorrow

Without that rhythm, we optimised at the level of individual model decisions — not at the competition level.

### What the prompt should have looked like

The prompt has been created at [.github/prompts/daily-review.prompt.md](.github/prompts/daily-review.prompt.md). It covers six structured steps:

1. **Today's rounds** — score vs. LORO expectation per round
2. **Rule compliance check** — explicit yes/no on LORO-validation for every change deployed today; auto-appends violations to KNOWLEDGE.md
3. **KNOWLEDGE.md update** — forced append, minimum one entry per round
4. **CURRENT_CONFIG.md sync** — always reflects the exact configuration in production; flags open/unvalidated experiments
5. **Strategy pulse check** — current best weighted score, remaining round weights, explicit mode declaration (conservative vs. experiment-ok)
6. **Tomorrow's plan** — maximum 3 priorities, each tagged `[VALIDATE]` / `[IMPLEMENT]` / `[EXPERIMENT]` / `[CONSERVATIVE]` / `[BLOCKED]`

The forced-priority list (max 3 items) is deliberate. Under time pressure, an unbounded backlog means the highest-leverage items are not always addressed first.

### This should have been ready before day 1

This prompt requires no knowledge of the specific competition — only the general structure of: LORO-validated models, a config sheet, a knowledge log, and a leaderboard formula. Everything in the prompt could have been written the week before the competition started, based solely on reading the API documentation and scoring formula.

**The `CURRENT_CONFIG.md` template, `DEPLOYMENT_RULES.md`, and the daily review prompt should all be in the repository before the first round opens.**

---

## Top 10: What We Should Have Done Differently

Ordered by estimated impact on the final leaderboard score.

---

**1. Understand the leaderboard formula before round 1 — and act on it**

The leaderboard takes only your *single best weighted round*. We knew this in theory but didn't internalise it until Round 17. The correct implication: rounds 1–9 are for experimentation and model building, rounds 16+ are for deploying your best validated model and not touching it. We had this backwards for most of the competition.

**Estimated impact**: the entire gap to first (8.5 weighted points) traces back to this failure.

---

**2. Prototype U-Net during rounds 8–12, not round 15**

U-Net (V10) arrived at Round 15 with a confirmed LORO superiority of +0.78 over GBM+MLP. The infrastructure was not exotic — 40×40 input grids, D4 symmetry TTA, entropy-weighted KL loss. It could have been built during rounds 8–12 when weights were low and a failed experiment cost almost nothing. Instead we spent that period iterating on features for a model architecture we were going to replace anyway.

**Estimated impact**: +15–25 weighted points from earlier deployment on high-weight rounds.

---

**3. Run a daily structured review — `daily-review.prompt.md` — every evening**

No checkpoint existed to flag unvalidated changes, verify production settings, or recalculate upcoming round weights. A ten-second compliance check would have blocked the R16 mistake. The prompt now exists at [.github/prompts/daily-review.prompt.md](.github/prompts/daily-review.prompt.md) — see *Should We Have Run Daily Post-Mortems?* for the full analysis.

**Estimated impact**: would have blocked the R16 mistake and likely 1–2 other sub-optimal decisions.

---

**4. Create `DEPLOYMENT_RULES.md` and `CURRENT_CONFIG.md` before round 1**

Rules in KNOWLEDGE.md were prose — an agent cannot enforce prose, and neither can time-pressured humans. `DEPLOYMENT_RULES.md` needs machine-readable absolute constraints; `CURRENT_CONFIG.md` needs to always reflect what is actually in production. See *Agent Use* for the full specification of both documents.

**Estimated impact**: structural prevention of R16-class mistakes.

---

**5. Validate the first submission before going live — never skip a sanity check**

Round 2 scored **3.02** because the naive empirical model replaced every prior with a single observation, producing 0/1 probabilities. Printing the output tensor — checking that values are between 0 and 1, that they sum to 1 per cell, that distributions look sensible — takes 30 seconds. The opening-strategy document described Bayesian updating correctly. The implementation simply wasn't verified before the first submission.

**Estimated impact**: R2 was a discarded early round, so direct leaderboard cost was low — but the precedent of skipping sanity checks persisted.

---

**6. Never deploy an unvalidated change to a high-weight round**

The R16 overlay change was in-sample only, broke a documented rule, and cost momentum on the heaviest rounds to date. No mechanism existed to enforce the rule — not KNOWLEDGE.md prose, not a checklist, not an agent guardrail. See *Strategy: Where the strategy failed* (item 1) for the full account.

**Estimated impact**: momentum loss and an unvalidated-deployment precedent at the worst possible time.

---

**7. Allocate the 5 leftover queries as repeats, not new viewports, from round 6**

R6–R13: the spare 5 queries were used on offset viewports (new positions), marginally increasing coverage. Better strategy: repeat the standard 9-viewport grid on the most dynamically uncertain seed. Repeated observations provide independent stochastic samples for the Bayesian overlay, directly improving calibration of transition rates on volatile rounds. Fixed in R14. Eight rounds late.

**Estimated impact**: −0.3 to −0.8 raw points per round × 8 rounds.

---

**8. Give the Copilot agent explicit behavioural constraints, not just context**

`copilot-instructions.md` described context and conventions, not rules. The agent had no instruction to cite LORO-delta before proposing a change, flag rule violations, or push back on unvalidated live deployments. It did exactly what it was asked. See *Agent Use* for the five specific changes needed.

**Estimated impact**: structural — would have added friction to bad decisions at the point of implementation.

---

**9. Build a hierarchical Bayesian model across seeds within a round**

All 5 seeds in a given round share the same hidden parameters — winter severity, expansion rate, raid frequency. We used per-round transition rates as model features, but never pooled information across seeds in a principled way. A hierarchical model could have inferred tighter posterior estimates of the round's hidden parameters from all 5 seeds combined, producing more accurate per-cell predictions on volatile and unusual rounds (R12, R7, R18).

**Estimated impact**: primarily relevant for stability-surge rounds (R12: 58.89). Estimated +2–5 raw points on those rounds.

---

**10. Plan for out-of-distribution rounds before they happen**

R12 (settlement stability surge, S→S = 0.595) and R7/R18 (partial stability with observation overestimation) were our chronic weaknesses. We never developed a detection mechanism — a signal that would flag at the start of a round that the observed transition rate diverged significantly from the training distribution, and switch to a more conservative prior. These rounds were predictable in type (the simulator always had high-variance parameters) even if the specific values weren't. A fallback strategy for OOD rounds should have been designed early.

**Estimated impact**: R12 alone cost ~15 raw points vs. a stable round. A fallback wouldn't solve it fully but could recover 5–8 raw points.

---

## Recommendations for the Next Competition

1. **`CURRENT_CONFIG.md` before round 1** — operative settings, always in sync with code.
2. **`DEPLOYMENT_RULES.md` before round 1** — machine-readable guardrails; enforce with the agent.
3. **`daily-review.prompt.md` every evening** — compliance check, config sync, risk tier, 3-priority plan.
4. **Pre-submit checklist in `copilot-instructions.md`** — agent runs it as part of every submission.
5. **Internalise the leaderboard formula before round 1** — risk tiers printed and consulted every day.
6. **Prototype architecture leaps in rounds 2–4** — not round 15.
7. **LORO-delta required for every proposal** — not just in-sample.
8. **High-weight round protection**: rounds with weight >2.0 get validated settings only. No experiments.
9. **GCP capability audit on day 1** — GPU for model training, parallel LORO via Cloud Run Jobs, Monte Carlo simulation as batch jobs. Treat the GCP account as compute infrastructure, not just a deployment target.
10. **Operationalise the observation-GT asymmetry from day 1** — every cell observation is one stochastic sample. Frame query strategy around how many samples are needed to estimate the run-average probability, and set Bayesian overlay parameters accordingly.
11. **Print round weights before round 1 and declare risk modes explicitly** — the leaderboard formula is a decision tool, not background reading. Five minutes on day one eliminates sixteen rounds of implicit risk-taking.
