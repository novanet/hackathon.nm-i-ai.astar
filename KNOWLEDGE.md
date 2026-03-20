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

- **LEADERBOARD FORMULA**: `leaderboard_score = max(round_score × round_weight)` across all rounds. Only your SINGLE BEST weighted result matters. Round weights = `1.05^round_number`. Missing rounds doesn't penalize — you just miss opportunities for a higher score.
- **HOT STREAK**: Average of last 3 round scores is also tracked (separate from leaderboard). Unclear if this affects final ranking.
- **ROUND SCORE**: Average of 5 per-seed scores. Unsubmitted seeds = 0. Always submit all 5.
- **SCORE FORMULA**: `score = max(0, min(100, 100 × exp(-3 × weighted_kl)))` where weighted_kl = entropy-weighted average KL across dynamic cells only.
- **FLOOR**: ~~Docs recommend 0.01 but our backtest shows 0.001 is better (+1.6 pts).~~ Now using 0.0003 (optimal per LORO sweep). Never use 0.0.
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
- [R1-R5] **Observation strategy**: cell-level Bayesian updates HURT score when spatial model is strong. Switch to using observations only for round-level transition matrix calibration.
- [R1-R5] Distance-to-settlement features are the biggest spatial signal — empty cells near settlements behave very differently from isolated ones.
- [R1-R5] Pure spatial model (alpha=1.0) scores 81.7 on R5 vs 77.6 with 60/40 blend — transitions dilute spatial accuracy.
- [R1-R5] Alpha=0.85 is the best blend: marginal safety net from transitions (+0.7 total) without much spatial dilution.
- [R6] **Alpha is round-dependent**: 0.85 is good for "normal" rounds but catastrophic for high-activity rounds. Use observations to calibrate alpha per round.
- [R6] Proximity-conditioned transitions (near/far settlement) capture key spatial patterns without relying on pre-trained model.
- [R6] 10% cell-level obs blending consistently helps on top of any base model (biased proxy metric, but directionally correct).
- [R7] **Observation bias**: Individual simulation runs overestimate transition rates vs GT probability tensor. E.g., observed S→S=38% but GT S→S=60.5%. This is because the GT averages across many stochastic runs, smoothing out volatility. Our observation-calibrated transitions are systematically too volatile.
- [R7-R8] **Model is hypersensitive to round feature estimates**: Oracle test shows +8.74 pts on R7 from using GT features vs our estimates, yet the feature differences are tiny (S→S: 0.4026 vs 0.4216, only 4.5% error). The model learned to rely heavily on the S→S feature.
- [R7-R8] **Feature estimation errors (obs-available rounds)**: S→S std=0.014, E→E std=0.005, F→F std=0.007, E→S std=0.005. S→S has 3x higher variance than others.
- [R7-R8] **Cell-level observation blending still hurts**: Tested alpha=[0.3, 0.5, 0.7, 1.0, adaptive]. All alphas make scores WORSE on both R6 and R7. Single-run stochastic noise overwhelms the signal.
- [R7-R8] **Inference-time ensemble (feature perturbation)**: Sampling 20 predictions with perturbed round features and averaging shows +1.21 on R7, +0.07 on R2, but -0.59 on R6. Marginal, inconsistent.
- [R7-R8] **Training with noisy round features (augmentation)**: Shows +8.09 on R7 in train-set eval, but **LORO shows -0.32 avg** — no real generalization benefit. The improvements were overfitting artifacts.
- [R7-R8] **LORO CV still at 70.25** (orig) vs 69.93 (augmented). No improvement path found from feature noise approaches.
- [R7-R8] **R7 diagnosis**: Settlement prediction is the main problem. When GT=Settlement, model only assigns 0.289 probability (vs 0.375 on R6). Leaks mass to Empty (0.435) and Forest (0.242).
- [R7-R8] **Key insight**: More training data (new rounds) is the primary improvement lever. Model needs to have seen similar dynamics to generalize.
- [R8] **Temperature scaling**: LORO sweep across T=[0.9..1.5] shows T=1.15 optimal. LORO avg 73.96→75.50 (+1.54). Consistent improvement on 6/7 rounds (R3 regresses due to its extreme outlier dynamics). Applied in `build_prediction()` before floor.
  - Per-round gains: R1 +5.66, R2 +2.80, R4 -0.51, R5 +1.14, R6 +5.38, R7 +1.32; R3 -5.05
- [R8] **n_jobs=-1 causes Windows hangs**: `MultiOutputRegressor(n_jobs=-1)` triggers multiprocessing that stalls on Windows. Fixed to `n_jobs=1` in training + `load_spatial_model()` patches loaded pickles.
- [R8] **Heavy regularization (+0.97 LORO)**: LightGBM params changed from (n_est=1000, depth=6, leaves=31, reg_alpha/lambda=0.1) → (n_est=500, depth=4, leaves=15, min_child=50, subsample=0.7, colsample=0.6, reg_alpha/lambda=1.0). LORO 75.50→76.47. Dramatically fixes R3 overfitting (in-sample R3 jumped from 37.0→91.2).
- [R8] **Per-class temperature scaling (+0.34 LORO)**: E=1.15, S=0.80, P=0.80, R=0.80, F=1.15, M=1.0. Settlement/Port/Ruin need SHARPENING (T<1), not softening. Empty/Forest need smoothing. Applied after spatial prediction, before floor.
- [R8] **Floor sweep**: 0.0003 is optimal (+0.08 LORO over 0.001). ~~0.001 was previous best.~~
- [R8] **Combined LORO gain**: 75.50 → 76.88 (+1.38 total). heavy_reg +0.97, per-class temps +0.34, floor +0.08.
- [R8] **Meta-model REJECTED**: LOO bias correction for round features worsens ALL test rounds. Only 4 observed rounds with S→S std=0.014 — insufficient data for reliable correction. Skip.

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
- ~~[R1-R4] Updated HISTORICAL_TRANSITIONS from all 4 rounds (32K cells, 20 seeds)~~
- ~~[R1-R4] Updated transition matrix: Empty stays 86.7%, Settlement stays 27.0% (down from old 39.6%), Forest stays 80.1% (up from 70.5%), Port stays 18.2% (down from 45.1%)~~
- [R1-R4] No initial Ruin cells found in any of 20 seeds — Ruin row stays as prior [0.5, 0, 0, 0.5, 0, 0]
- ~~[R1-R4] Final spatial model trained on all 32K cells, n_estimators=300, saved to data/spatial_model.pkl~~
- ~~[R1-R4] Training set scores: R1=75.6, R2=75.3, R3=56.3, R4=91.2~~

### Multi-Round Retraining (R1-R5)
- [R1-R5] HISTORICAL_TRANSITIONS updated from all 5 rounds (40K cells, 25 seeds)
- [R1-R5] Transition matrix: Empty→Empty 86.5%, Set→Set 28.2%, For→For 79.4%, Port→Port 18.4%
- [R1-R5] Spatial model retrained with 22 features (was 18): added dist-to-settlement, dist-to-forest, dist-to-port, settlement-count-radius-5
- [R1-R5] LORO cross-val: spatial=64.8, baseline=67.5, Δ=-2.8 (spatial loses on avg, but that's driven by R3 outlier)
- [R1-R5] LORO per-round: R1=61.0(s)/64.4(b), R2=63.4(s)/66.6(b), R3=40.7(s)/57.8(b), R4=89.2(s)/87.0(b), R5=69.6(s)/61.9(b)
- [R1-R5] Spatial beats baseline on R4 (+2.2) and R5 (+7.7) — good for "normal" rounds
- [R1-R5] Training set scores: R1=83.1, R2=80.9, R3=61.4, R4=91.7, R5=81.7
- [R1-R5] **Cell-level Bayesian observation updates ABANDONED** — replaced with round-level transition calibration
- [R1-R5] Observation-calibrated transitions: use Bayesian blend of historical + observed transitions, applied globally (not per-cell)
- [R1-R5] Optimal spatial/transitions blend: alpha=0.85 spatial + 0.15 transitions. Pipeline total across all rounds = 463.9 (vs 463.2 spatial-only, 394.2 transitions-only)
- [R1-R5] R5 new pipeline score (backtest): 80.5 vs old 75.38 = +5.1 improvement
- [R1-R5] R2 is the only round where observations significantly help (83.4 with obs vs 80.9 spatial-only)

### Round 5
- Round ID: fd3c92ff-3178-4dc9-8d9b-acf389b3982b
- **Score: 75.38 (rank 47)** — first real submission with improved model
- Seeds: 72.4, 72.7, 76.5, 76.6, 78.6
- 45/50 queries used (9-viewport grid coverage per seed)
- ~~Spatial-only would have scored ~76.3 — observations actually hurt by ~1 pt~~
- Spatial-only (new 22-feature model): 81.7, observations with calibrated transitions: 80.5
- Transition-only baseline: ~62.6 — spatial model added +18 pts
- Ground truth downloaded for all 5 seeds
- **Key insight**: ~~Bayesian observation updates with PRIOR_STRENGTH=20 pull predictions slightly worse than pure spatial prior. Consider increasing prior_strength or skipping observations when spatial model is confident.~~ Switched to round-level transition calibration instead of cell-level Bayesian updates. Cell-level updates hurt because single observations override a strong spatial prior.
- [R5] Oracle transition matrix: Empty→Empty 85.4%, Set→Set 32.7%, For→For 76.9% — close to historical avg
- [R5] Mean entropy: 0.482, 78.7% dynamic cells

### Round 6
- Round ID: ae78003a-4efe-425a-881a-d16a39bca0ad
- Round weight: 1.34 (1.05^6)
- Map: 40x40, 5 seeds, 39-56 initial settlements per seed
- 50/50 queries used (45 grid + 5 repeat on center viewport)
- **Score: TBD** (round still active)
- **HIGH-ACTIVITY ROUND**: drastically different from historical averages
  - Empty→Empty: 69.5% (vs historical 86.5%) — massive expansion
  - Forest→Forest: 53.4% (vs historical 79.4%) — heavy colonization
  - Settlement→Settlement: 39.5% (vs historical 28.2%) — more stable
  - Empty→Settlement: 20.8% (vs historical 9.0%) — very aggressive expansion
  - Forest→Settlement: 26.9% (vs historical 11.2%)
- **Repeat observations**: only 42-49% of cells match between two independent simulations — extremely stochastic
- **Key finding: alpha=0.85 is WRONG for this round**. Spatial model trained on calmer rounds massively underpredicts expansion.
  - Alpha sweep: optimal alpha = 0.20-0.25 (not 0.85)
  - Per-class alpha: Empty=0.3, Settlement=0.1, Port=0.0, Forest=0.1, Mountain=0.0
- **Proximity matters**: Empty cells near settlements (dist≤3) have 26.4% chance of becoming settlement vs 14.2% for far cells
- **Best approach**: Hybrid model — proximity-conditioned transitions (near/far) + per-class spatial blend + 10% cell-level obs blending
  - Hybrid: avg_ll=-0.750, vs Per-class=-0.759, Global=-0.760, Proximity-only=-0.763, Pure-transitions=-0.774
- 8 submission passes: baseline → obs-calibrated → alpha=0.2 → alpha=0.25+obs → per-class alpha → proximity → hybrid (FINAL)
- **Takeaway**: When round dynamics differ from training data, reduce spatial model weight and rely more on round-specific calibrated transitions
- [R6] **Adaptive model deployed**: `build_prediction()` now auto-detects round activity from calibrated transitions. Activity threshold 0.10: high-activity → proximity-conditioned + per-class alpha; normal → alpha=0.85 spatial dominant.
- [R6] **Vectorized proximity computation**: O(H×W) Python loop replaced with numpy broadcasting — 17s→1.5s per seed. All 5 seeds in <8s.
- [R6] **Cloud Run deployment ready**: Code pushed, Dockerfile includes spatial_model.pkl. Deploy with `gcloud run deploy astar-solver --source . --region europe-north1 --allow-unauthenticated --memory 1Gi --timeout 300 --set-env-vars ASTAR_TOKEN=<token>`
- [R6] **Settlement stats integration**: Added `_extract_settlement_stats()` to compute mean_pop, mean_food, mean_wealth, mean_defense from observed settlements. Used as secondary signal in activity detection (25% weight). R6 stats: pop=1.24, food=0.58, defence=0.47 (vs R5 calm baseline: pop=1.06, food=0.71, def=0.31). Adaptive food-based alpha: when food<0.5, settlement alpha drops to 0.05 (starving settlements are more volatile). No regression on R1-R5 backtests.
- **Score: 77.9 (rank 32/186), weighted 104.4** — seeds: 78.1, 78.3, 75.9, 79.2, 78.0
- [R6] Ground truth downloaded for all 5 seeds

### Multi-Round Retraining (R1-R6)
- [R1-R6] HISTORICAL_TRANSITIONS updated from all 6 rounds (48K cells, 30 seeds)
- [R1-R6] Transition matrix: Empty→Empty 83.8%, Set→Set 30.5%, For→For 75.3%, Port→Port 19.7%
- [R1-R6] Spatial model retrained with 22 features on 48K samples, saved to data/spatial_model.pkl
- [R1-R6] Training set (in-sample) backtests: R1=83.9, R2=84.4, R3=53.1, R4=91.6, R5=82.2, R6=80.3
- [R1-R6] Deployed as Cloud Run revision astar-solver-00005-ksw

### Round 7
- Round ID: 36e581f1-73f8-453f-ab98-cbe3052b701b
- Round weight: 1.41 (1.05^7)
- Map: 40x40, 5 seeds, 44-60 initial settlements per seed (seed 4 had 60 — highest)
- 50/50 queries used (45 grid + 5 extra on high-settlement seeds)
- All 5 seeds submitted (status=accepted), score pending (round closes 14:48 UTC)
- **Bug encountered**: `run_r7.py` saved duplicate simulation files in raw format (without `request`/`response` wrapper), causing `KeyError: 'request'` in `load_simulations()`. Fixed by removing bad files and patching the script to not save duplicates (client.py already handles logging).
- First submission attempt failed (all 5 seeds); resubmitted after cleanup — all accepted
- Cloud Run redeployed as revision astar-solver-00006-hsg (URL: astar-solver-464650180745.europe-north1.run.app)
- Generic `run_next_round.py` created: auto-finds active round, diagnostics, extra query spending, no duplicate file saves
- **Score: 56.85 (weighted 80.0)** — seeds: 51.5, 59.6, 58.1, 57.0, 58.0
- **NORMAL round by our detector** (activity=0.073, below 0.10 threshold) but GT shows unusual parameters
- **Root cause of poor score**: Observation-based transitions overestimate volatility. Single sim runs show S→S=38.3%, but GT probability tensor shows S→S=60.5% (settlements survive 60% of runs, but in any single run they may die or live)
- R7 GT transitions vs observed:
  - S→S: **60.5% GT** vs 38.3% observed vs 30.5% historical — settlements far more stable
  - E→S: **6.1% GT** vs 11.3% observed vs 11.1% historical — much less expansion
  - E→E: **93.7% GT** vs 85.1% observed vs 83.8% historical — very calm
  - F→F: **83.7% GT** vs 74.5% observed vs 75.3% historical — forests stable
- **Key insight**: Individual stochastic observations systematically overestimate transition rates. The GT probability tensor reflects averaging across many runs, which is calmer than any single run.
- Ground truth downloaded for all 5 seeds

### Multi-Round Retraining (R1-R7)
- [R1-R7] HISTORICAL_TRANSITIONS updated from all 7 rounds (56K cells, 35 seeds)
- [R1-R7] Transition matrix: Empty→Empty 84.0%, Set→Set 32.4%, For→For 75.3%, Port→Port 19.7%
- [R1-R7] Spatial model retrained with 22 features on 56K samples, saved to data/spatial_model.pkl
- [R1-R7] Full-pipeline backtests: R1=83.5, R2=84.2, R3=49.0, R4=90.3, R5=83.5, R6=80.1, R7=64.0
- [R1-R7] R7 improved from 56.9 (official) to 64.0 (retrained backtest) — +7.1 pts from including R7 GT in training

### Model V3: Round-Conditioned Spatial + LightGBM (post-R7)
- **Architecture**: LightGBM (1000 trees, depth=6, lr=0.05) with 27 features (22 spatial + 5 round-level)
- **Round features**: E→E, S→S, F→F, E→S transition rates + settlement_density — computed from debiased observations at inference
- **Observation debiasing**: Shrinkage matrix (GT/obs ratio per transition) applied to observation-calibrated transitions. Main classes ~1.0, Port noisy (0.47-1.46)
- **Key finding**: Debiased observation features match GT within ±0.021 — debiasing works well
- **Pure spatial is optimal**: Alpha=1.0 (no transition blending), obs_w=0.0 (no cell-level observation blending). Added blending always hurts.
- **Backtests**: R1=75.3, R2=91.2, R3=37.0, R4=86.4, R5=89.0, R6=90.5, R7=71.1
  - vs old: R2+5.5, R5+5.5, R6+10.4, R7+7.1 — massive improvements on observed rounds
  - R1/R3 regressed (no observations → fallback to historical features) — irrelevant for future rounds
  - **R6 weighted = 121.3** — beats leader at 118.6!
- **LORO cross-val**: avg=70.3 (was ~65). Per-round: R2=83.1, R5=82.3, R6=77.3, R7=64.9
- **Overfitting gap**: ~8 pts between in-sample and LORO, driven by limited round diversity (7 rounds)
- **Round features have huge variance**: R3 E→E=0.989 (calm) vs R6 E→E=0.702 (chaos) vs R7 E→E=0.852 — this is the killer feature
- ~~[R6] Alpha is round-dependent~~ → now handled by round-conditioned features in spatial model
- ~~[R6] Proximity-conditioned transitions~~ → subsumed by spatial model with round features

### Round 8
- Round ID: c5cdf100-a876-4fb7-b5d8-757162c97989
- Round weight: 1.477 (1.05^8)
- Map: 40x40, 5 seeds; settlements per seed: 60, 37, 56, 38, 37; ports: 4, 3, 1, 2, 0
- 50/50 queries used (45 grid + 5 extra on settlement-dense seeds)
- All 5 seeds submitted (pass 1 + pass 2 with extra obs), status=accepted
- **VERY CALM round** (activity=0.00%): Empty→Empty +11% above historical, Settlement→Settlement -25.7% (but this is debiased)
- Transition diagnostics: Empty stable, Settlements dying more, Forest much more stable (+16.6% above hist)
- Model V3 used (LightGBM, 27 features, debiased round features)
- **Patched mid-round**: Resubmitted all 5 seeds with 20-sample inference-time ensemble (feature perturbation). Then resubmitted again with T=1.15 temperature scaling (LORO +1.54).
- **n_jobs bug**: `MultiOutputRegressor(n_jobs=-1)` causes Windows multiprocessing hangs during prediction. Fixed by setting `n_jobs=1` in training and patching `load_spatial_model()` to force it on loaded pickles.
- **Score: 73.68 (weighted 108.87, rank 81/214)** — NEW BEST weighted score (beats R6's 104.4)
- Seeds: 68.66, 80.27, 68.78, 75.87, 74.84
- Ground truth downloaded for all 5 seeds
- **R8 was a "CIVILIZATION COLLAPSE" round** — 0% settlement survival in GT argmax. ALL settlements died.
  - Probabilistic S→S = 6.7%, S→E = 61.5%, S→F = 30.7%
  - E→S = 2.0%, E→E = 94.8%, F→F = 90.1%
  - GT argmax: 0% settlement, 0% port, 0% ruin — pure empty/forest/mountain
  - But GT probability: residual 2-3% settlement probability (some sim runs kept settlements alive)
  - Mean entropy: 0.270 (lowest of all rounds — very confident predictions)
- **Matches R3 and R4 collapse pattern**: R3 (S→S=1.8%), R4 (S→S=23.5%), R8 (S→S=6.7%) — all "harsh" rounds
- **Round dynamics clustering**: 3/8 rounds (37.5%) are "collapse" (R3, R4, R8); 5/8 are "normal" (R1, R2, R5, R6, R7: S→S=33-42%)
- **Current optimized model REGRESSED on R8**: 71.51 avg vs 73.68 submitted (-2.17 pts worse)
  - All 5 seeds regressed: per-seed delta -1.41 to -3.16
  - Cause: Per-class settlement sharpening (T=0.80) pushes already-small settlement probs even lower, but GT wants ~2-3%
  - Current model: Sett=0.84-1.26%. GT: Sett=1.56-3.01%. Submitted: Sett=0.91-1.21%
  - **Per-class temps hurt on collapse rounds where settlement prob is small but nonzero**
- **Error analysis (R8 S0)**: 62.4% of KL loss from Empty cells, 32.0% from Forest, 4.9% from Settlement. Settlement has highest mean KL per cell (0.136) despite small count (n=56)

### Multi-Round Retraining (R1-R8)
- [R1-R8] HISTORICAL_TRANSITIONS updated from all 8 rounds (64K cells, 40 seeds)
- [R1-R8] Transition matrix: Empty→Empty 85.4%, Set→Set 29.3%, For→For 77.2%, Port→Port 17.3%
- [R1-R8] Spatial model retrained with 27 features (22 spatial + 5 round-level) on 64K samples
- [R1-R8] **LORO avg: 77.79** (was 75.40 with R1-R7 — +2.39 pts from adding R8 data!)
  - Per-round: R1=75.5, R2=83.4, R3=83.3, R4=86.9, R5=82.3, R6=77.3, R7=66.1, R8=67.6
  - R3 LORO: 57.4 → 83.3 (+25.9!) — R8 collapse data dramatically helps R3 prediction
  - R8 LORO: 67.6 (new) — decent for a collapse round given only R3/R4 as similar training
- [R1-R8] In-sample backtests (with observations): R1=80.3, R2=88.8, R4=85.0, R5=85.3, R6=87.2, R7=70.7, R8=93.3
  - R3=35.1 (no observation files — build_prediction falls back to bare model)
- [R1-R8] Model saved to data/spatial_model.pkl

### Leaderboard Analysis (post-R8)
- Our leaderboard score: 108.87 (rank 88/278)
- Top team: 140.30. Top 10: ~137-140. Top 50: ~118-120.
- Gap to top 50: ~10-12 pts weighted. Need R9+ to score ~77-80 raw to reach top 50 by R14.
- Each round's weight is 1.05^n, so later rounds carry more. R14 weight = 1.98.

### Critical Insights for R9+
- **Collapse detection is KEY**: 37.5% of rounds are collapse. Model needs to identify this from observations and respond appropriately.
  - When observed S→S ≈ 0, settlements are dying. Don't sharpen settlement predictions — they need to stay soft.
  - ~~Per-class temps of T=0.80 for settlements HURT on collapse rounds. Consider adaptive temps based on observed dynamics.~~ FIXED in V4.
- **Observation debiasing confirmed**: Debiased S→S correlates well with GT. But single-run observations still overestimate volatility by ~20%.
- **More training data = biggest lever**: R3 LORO jumped +25.9 from having R8 (similar dynamics). Each new round improves generalization significantly.
- **Model performs well on "normal" rounds**: R2=83.4, R5=82.3, R6=77.3 are solid. The weak points are collapse rounds (R8=67.6) and R7=66.1 (which had unusual settlement stability).
- ~~**Per-class temps should be adaptive**: Use T=0.80 for settlements only when the round is "normal" (S→S > 0.15). For collapse rounds, use T=1.15 (softer) for settlements.~~ IMPLEMENTED in V4.
- ~~**Priority improvements**: (1) Adaptive per-class temps based on observed dynamics, (2) Better collapse detection from early observations, (3) Consider submitting in multiple passes — first a safe prediction, then refine based on analysis~~ ALL DONE in V4.

### Model V4: Optimized Spatial + Adaptive Temps (post-R8)
- **Architecture**: LightGBM (500 trees, depth=4, leaves=15, min_child=50) with **30 features** (22 spatial + 8 round-level)
- **New round features** (was 5, now 8): E→E, S→S, F→F, E→S, settlement_density, **mean_food, mean_wealth, mean_defense** (from observed settlement stats)
- **Adaptive per-class temps**: Collapse detection via debiased S→S < 0.15:
  - Normal rounds: [1.20, 1.0, 1.0, 1.0, 1.20, 1.0] — only soften Empty/Water, NO sharpening of S/F/P
  - Collapse rounds: [1.15, 1.15, 1.15, 1.0, 1.15, 1.0] — soften everything so settlement residuals preserved
- **Key insight**: ~~Old PER_CLASS_TEMPS=[1.15, 0.80, 0.80, 0.80, 1.15, 1.0] sharpened S/F/P~~ The V4 spatial model already calibrates the right class distributions. Additional per-class sharpening was counterproductive. Empty/Water softening (T=1.20) is the only beneficial post-processing.
- **PROB_FLOOR**: 0.0002 (was 0.0003, sweep showed marginal improvement)
- **SHRINKAGE_MATRIX**: Updated from R1-R8 data (was R1-R7). Notable changes: Settlement→Port 1.27→1.94, Forest→Settlement 0.97→0.80
- **HISTORICAL_TRANSITIONS**: Updated from R1-R8 averages (was R1-R7)
- **Settlement stat fallback**: When no observations, derive proxies from transition matrix: mean_food=0.3+0.7×S→S, mean_wealth=E→S×0.3, mean_defense=1.0-S→E
- **LORO**: 78.05 (was 77.79 — +0.26 from settlement stat features)
- **Backtest (rounds with observations)**: R2=87.59 (+1.32), R5=84.69 (+4.34), R6=86.99 (+5.58), R7=67.13 (+3.31), R8=93.39 (+21.88). Average **+7.29 pts** on obs rounds.
- **Backtest (rounds without observations)**: R1=81.80 (-0.36), R3=36.10 (-47.24), R4=84.46 (-6.42). These regress but are irrelevant — live rounds always have observations.
- **Observation blending tested and REJECTED**: Blending spatial model with per-cell observation frequencies (Bayesian, MODEL_STRENGTH=3) made ALL rounds worse (-0.5 to -10 pts). Observations are single stochastic runs, not probability distributions. The spatial model already captures the distribution; blending with individual outcomes adds noise.
- **Temperature sweep**: Monotonic trend — softer is better for V4. Sharpening S/F/P (old T=0.80) actively hurts. E/W softening (T=1.20) adds +0.17 pts over T=1.15.
- **R9 weight**: 1.05^9 = 1.55. Raw ~90 needed to compete with leader (140.30). Realistic target: 80-85 raw → 124-132 weighted.

### Model V5: Entropy-Weighted Training + Calibration (post-R8)
- **Change 1: Entropy-weighted training** (ENTROPY_WEIGHT_POWER=0.25): Training samples weighted by GT entropy^0.25. High-entropy cells (settlement/forest boundaries) dominate the scoring function (88-99% of loss), but MSE training treats all cells equally. Weighting forces the model to focus on cells that matter for the score.
- **Change 2: Post-model calibration** (CALIBRATION_FACTORS=[1.03, 0.92, 0.92, 0.92, 1.0, 1.0]): Model systematically over-predicts Settlement (+13-54%), Port (+45-4600%), Ruin (+37-85%), and under-predicts Empty (-3-7%). Applied per-class multiplicative correction after spatial model, before temperature scaling. Bias is consistent across ALL 8 rounds.
- **LORO (with temps + calibration)**: 80.66 (was 79.78 baseline with temps, or 78.05 raw). Total improvement: **+0.88 vs V4** with temps.
  - Entropy weighting alone (power=0.25): +0.64 LORO
  - Calibration alone: +0.05-0.14 LORO (marginal)
  - Combined: +0.88 LORO (synergistic)
- **LORO raw (no temps, from train_spatial.py)**: 79.67 (was 78.05 in V4 — +1.62 from entropy weighting)
- **Entropy weight sweep**: power=0.15: +0.22, 0.25: +0.64, 0.35: +0.59, 0.50: +0.77, 0.75: +0.60, 1.0: +0.35, 2.0: -1.87. Sweet spot 0.25-0.50. Chose 0.25 for conservative, stable per-round distribution.
- **Calibration sweep**: S/P/R 0.92 + E 1.03 was optimal in LORO. S/P/R 0.95/0.90 also tested. Port-specific calibration (0.80) marginally better (+0.14).
- **Log-target training REJECTED**: Training on log(p) targets catastrophically failed (LORO -21.10 pts). MSE is better.
- **Key insight**: The scoring formula (entropy-weighted KL) means high-entropy cells contribute disproportionately. Training with entropy weights aligns the loss function with the scoring metric.
- **Key insight**: Entropy weighting is purely a **training-time** technique — it changes model weights, not inference behavior. No need to detect or classify cells at inference time. The model naturally produces better predictions for high-entropy cells because it was optimized for them.
- **Key insight**: The calibration bias is a property of the model architecture (LightGBM with regularization pushes predictions toward minority classes), not the data. It persists across all rounds and all training sets. Safe to apply uniformly at inference.
- **Key insight**: Calibration and entropy weighting are **synergistic** (+0.88 combined vs +0.64 + +0.07 individually). Entropy weighting makes the model better at predicting boundary cells, then calibration corrects the remaining systematic class-level bias.
