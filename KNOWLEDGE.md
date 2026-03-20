# Astar Island — Knowledge Base

Critical learnings accumulated during the competition. Copilot should append findings here as they are discovered.

---

## Simulation Rules (Observed)

- [Round 2] Settlements expand aggressively into empty/plains cells — 16.4% of initial empty cells became settlements
- [Round 2] Forests also get colonized by settlements — 19.1% of initial forest cells became settlements
- [Round 2] Settlements have ~38.8% chance of dying and reverting to empty (not ruin), only ~5.4% become ruins
- [Round 2] Initial settlements have ~38.1% survival rate (staying as settlement)
- [Round 2] 17.3% of initial settlements get reclaimed by forest
- [Round 2] Ports are relatively stable — 58.3% of initial ports remain ports (n=12, small sample)
- [Round 2] Mountains never change (100% stay mountain, confirmed)
- [Round 2] Forests moderately stable — 66.5% remain forest after 50 years
- [Round 2] 0% dead settlements observed in stats — all observed settlements were alive (may be because dead ones become ruins/empty)
- [Round 2] Settlement stats: mean pop=1.15 (max 3.93), mean food=0.68, mean wealth=0.02 (very low), mean defense=0.52

## Hidden Parameter Estimates

- [Round 2] Expansion appears aggressive — settlements spread into 16% of empty cells and 19% of forest cells
- [Round 2] Winter severity seems moderate — settlements die (~39% revert to empty) but food mean=0.68 suggests not extreme starvation
- [Round 2] Wealth near zero (mean 0.02) — trade may be minimal or wealth gets consumed quickly
- [Round 2] Raiding effects unclear — no dead settlements observed, but 5.4% ruin rate suggests some conflict

## Terrain Transition Patterns

- [Round 2] Full transition matrix (initial → final, n=9,681 observations):
  ```
  Empty→Empty:       76.8%    Empty→Settlement: 16.4%    Empty→Forest:  4.0%
  Settlement→Empty:  38.8%    Settlement→Sett:  38.1%    Settlement→Forest: 17.3%   Settlement→Ruin: 5.4%
  Port→Empty:        41.7%    Port→Port:        58.3%    (small sample n=12)
  Forest→Forest:     66.5%    Forest→Settlement:19.1%    Forest→Empty: 11.0%
  Mountain→Mountain: 100%
  ```
- [Round 2] Key insight: Empty cells are NOT safe — 23.2% of them change (mostly to settlement)
- [Round 2] Forest is the second most volatile initial type — 33.5% change
- [Round 2] Settlement is the MOST volatile — 61.9% of initial settlements don't survive as-is
- [Backtest R1] Oracle transition matrix from R1 ground truth:
  ```
  Empty→Empty:       82.2%    Empty→Settlement: 12.9%    Empty→Forest:  2.8%
  Settlement→Empty:  37.1%    Settlement→Sett:  41.0%    Settlement→Forest: 18.1%   Settlement→Ruin: 3.1%
  Port→Empty:        36.4%    Port→Sett:        12.0%    Port→Port: 31.9%   Port→Forest: 17.6%
  Forest→Forest:     74.5%    Forest→Settlement:15.9%    Forest→Empty: 7.0%
  Mountain→Mountain: 100%  (No initial Ruins in R1)
  ```
- [Backtest] Key differences R1 vs R2: R1 empty stayed 82% vs R2 77%; R1 forest stayed 75% vs R2 67%; Port much more volatile in R1 (32% stay vs 58%)
- [Backtest] Historical averaged transition matrix stored in model.py as HISTORICAL_TRANSITIONS
- [R1-R4] Updated all-rounds transition matrix (20 seeds, 32K cells):
  ```
  Empty→Empty:       86.7%    Empty→Settlement:  9.0%    Empty→Forest:  2.7%   Empty→Port: 0.7%  Empty→Ruin: 0.8%
  Settlement→Empty:  47.9%    Settlement→Sett:  27.0%    Settlement→Forest: 22.4%   Settlement→Ruin: 2.3%
  Port→Empty:        49.1%    Port→Sett:         7.6%    Port→Port: 18.2%   Port→Forest: 23.0%  Port→Ruin: 2.1%
  Forest→Forest:     80.1%    Forest→Settlement: 11.2%   Forest→Empty:  6.9%
  Mountain→Mountain: 100%     (No initial Ruins in any round)
  ```

## Query Strategy Insights

- [Round 2] 9-viewport grid (step 13, overlapping) covers full 40x40 map — uses 9 queries per seed
- [Round 2] 45 queries for full coverage of all 5 seeds, leaving 5 spare for repeat observations
- [Round 2] Each seed gets 304 cells with 2 observations (overlap zones) — the rest get 1
- [Round 2] ~120-130 high-entropy cells per seed (entropy > 0.5) — these are settlement/forest boundary areas
- [Round 2] Extra queries best spent on repeat observations of settlement-dense seeds (1 and 3 had 56 settlements each)
- [Round 2] Viewport positions used: (0,0),(13,0),(26,0),(0,13),(13,13),(26,13),(0,26),(13,26),(26,26)

## Scoring & Prediction Insights

- ~~[Round 2] Using probability floor 0.01 (default) — awaiting score to calibrate~~
- [Backtest R1] Probability floor sweep: 0.001 > 0.005 > 0.01. Floor 0.001 scores 74.3 vs 72.7 at 0.01 (+1.6 pts). Now using 0.001.
- [Backtest R1] Pure transition model (alpha=1.0) scores 72.7, beating all blends with initial prior (alpha=0.5 → 61.4, alpha=0.0 → 15.8)
- [Backtest R1] Spatial smoothing on oracle transitions HURTS (sigma=0.5 → 62.9). Keep smoothing only for interpolating observed → unobserved cells.
- [Backtest R1] Neighbor density boost (settlement proximity) is marginally helpful (74.4 at boost=3.0 vs 72.7 baseline with oracle)
- [Backtest R1] Historical averaged transition matrix (R1+R2) scores 74.4 on R1 ground truth with ZERO queries
- [Backtest R1] R2 transitions scored 74.3 on R1 ground truth — transitions generalize across rounds
- [Backtest R1] LOO cross-validation within R1 seeds: 72.0 average — same-round per-seed transitions are consistent
- [Backtest R1] R1 oracle transition matrix scored LOWER (72.2) than R2's (74.3) on R1 data — more spread/uncertainty in R2 matrix matches probabilistic ground truth better
- [Round 2] Cross-seed transition model should help significantly — 61.9% of settlements change, so initial-state prior alone is very wrong for settlement cells
- [Round 2] Final observed terrain distribution: ~60% empty, ~17% settlement, ~17% forest, ~2% ruin, ~1% port, ~2% mountain
- [Backtest R1] Ground truth class distribution: ~74% Empty, ~2% Settlement, ~21% Forest, ~2% Mountain (by argmax)
- [Backtest R1] Mean entropy across ground truth: 0.487-0.639 per seed, ~80-85% of cells are dynamic (entropy > 0.01)
- [Backtest R1] **Spatial conditional model** (GBR, 18 features): LOO avg 79.95, training avg 82.7. +5.6 pts over transition-only baseline.
- [Backtest R1] Features: initial class one-hot (6) + 3×3 neighbor fractions (6) + 5×5 outer ring fractions (6) = 18 features
- [Backtest R1] Spatial model trained on 5 seeds × 1600 cells = 8000 samples. Model saved to data/spatial_model.pkl.
- [Backtest R1] Spatial(70%) + Transitions(30%) blend = 82.8, slightly better than pure spatial (82.7)
- [Backtest R1] **Bayesian Dirichlet update is CRITICAL**: with 1 obs/cell, Bayesian=83.1 vs Naive=0.14 (catastrophic overconfidence). Old empirical_model was doing naive replacement.
- [Backtest R1] Bayesian prior_strength sweep: optimal around 15-20. Using 20. Score with 1 obs/cell: 83.7 (str=20).
- [Backtest R1] With 10 obs/cell + Bayesian(str=20): 85.0. Observations provide diminishing returns when prior is good.
- [Backtest R1] 98.4% of score loss comes from Empty (68.9%) and Forest (29.5%) cells. Perfecting top 400 worst cells → score 91.5.

## Per-Round Notes

### Round 1
- Missed entirely (completed before we got auth token)
- Round weight: 1.05
- Ground truth downloaded for all 5 seeds — used as backtesting validation set
- R1 oracle transition matrix learned (no Ruin→ data since no initial ruins existed)
- R1+R2 averaged transitions stored as HISTORICAL_TRANSITIONS in model.py

### Round 2 (Active — 2026-03-19)
- Round ID: 76909e29-f664-4b2f-b16b-61b7507277e9
- Round weight: 1.1025
- Status: 5/5 seeds submitted, 50/50 queries used
- Strategy: full grid coverage (9 viewports/seed) + 4 extra queries on seeds 1,3
- Model: initial prior → cross-seed transitions (60/40 blend) → empirical update → spatial smooth (σ=1.5) → neighbor inference → floor(0.01)
- **Post-round model update**: Changed to pure transition model (alpha=1.0), floor=0.001, historical fallback transitions. These changes would have improved score if applied before submission.
- Map: 40x40, 5 seeds, ~33-56 initial settlements per seed
- Score: **3.02** (rank 141) — catastrophically low due to naive empirical_model bug (fixed post-round with Bayesian updates)

### Round 3
- Round ID: f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb
- Missed — no submission
- Ground truth downloaded for all 5 seeds
- R3 scores much lower across all models (~42-58 range) — suggests very different hidden parameters

### Round 4
- Round ID: 8e839974-b13b-407b-a5e7-fc749d877195
- Missed — no submission
- Ground truth downloaded for all 5 seeds
- R4 scores very high (~89-91) — suggests a more predictable/stable round

### Multi-Round Retraining (R1-R4)
- [R1-R4] LORO cross-validation: spatial=66.52, baseline=69.94, Δ=-3.43 — spatial model doesn't generalize well across rounds
- [R1-R4] Per-round LORO: R1=69.3(s) vs 74.4(b), R2=63.4(s) vs 79.7(b), R3=43.7(s) vs 42.3(b), R4=89.7(s) vs 83.3(b)
- [R1-R4] Spatial model beats baseline only on R3 (+1.4) and R4 (+6.4); loses badly on R2 (-16.4)
- [R1-R4] Hidden parameters vary dramatically between rounds — R3 is very different, R4 is very predictable
- [R1-R4] Updated HISTORICAL_TRANSITIONS from all 4 rounds (32K cells, 20 seeds)
- [R1-R4] Updated transition matrix: Empty stays 86.7%, Settlement stays 27.0% (down from old 39.6%), Forest stays 80.1% (up from 70.5%), Port stays 18.2% (down from 45.1%)
- [R1-R4] No initial Ruin cells found in any of 20 seeds — Ruin row stays as prior [0.5, 0, 0, 0.5, 0, 0]
- [R1-R4] Final spatial model trained on all 32K cells, n_estimators=300, saved to data/spatial_model.pkl
- [R1-R4] Training set scores: R1=75.6, R2=75.3, R3=56.3, R4=91.2
