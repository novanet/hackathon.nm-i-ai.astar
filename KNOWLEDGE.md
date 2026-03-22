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
- [Round 18] All-or-nothing full-grid logic is a late-round trap: if budget < 45, spend the remainder on a seed-balanced partial grid before any repeat queries.

## Scoring & Prediction Insights

- **LEADERBOARD FORMULA**: `leaderboard_score = max(round_score × round_weight)` across all rounds. Only your SINGLE BEST weighted result matters. Round weights = `1.05^round_number`. Missing rounds doesn't penalize — you just miss opportunities for a higher score.
- **HOT STREAK**: Average of last 3 round scores is also tracked (separate from leaderboard). Unclear if this affects final ranking.
- **ROUND SCORE**: Average of 5 per-seed scores. Unsubmitted seeds = 0. Always submit all 5.
- **SCORE FORMULA**: `score = max(0, min(100, 100 × exp(-3 × weighted_kl)))` where weighted_kl = entropy-weighted average KL across dynamic cells only.
- [Round 10 prep] Blending a stochastic spatial simulator into the spatial model improved LORO average from 82.78 to 83.79 (+1.02) at 15% blend weight.
- [Round 10 prep] Simulator blend helps most on collapse rounds: R8 improved from 78.91 to 86.46 (+7.55) in LORO. It hurts R5/R7 slightly, so keep blend moderate (15%, not 20%+).
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
- [R7-R8] **Cell-level observation blending still hurts**: Tested alpha=[0.3, 0.5, 0.7, 1.0, adaptive]. All alphas make scores WORSE on both R6 and R7. Single-run stochastic noise overwhelms the signal. ~~**REVISED R12**: Adaptive Bayesian overlay (entropy-scaled prior_strength) helps when model is OOD: +1.41 on R12, +0.47 on R11. Key: scale prior_strength by model confidence per cell (min_ps=5 for uncertain, max_ps=100 for confident). Uniform blending still hurts, but adaptive does not regress.~~
- [V9] **Adaptive Bayesian overlay confirmed**: min_ps=5, max_ps=100, entropy-scaled. After adding this, sim blend becomes redundant (alpha=0 beats 0.15 everywhere). The overlay is CRITICAL — it's worth +0.04 to +1.41 across rounds with zero regression.
- [R16] ~~**Bayesian overlay was WAY too conservative**: Backtest on R2-R15 with GT shows (5,100) barely moves predictions (max 4% shift). Sweeping prior strengths: optimal is min_ps=0.5, max_ps=3.0. Avg score: 40.56 (current) -> 40.66 (0.5-3), +2.72 vs no overlay. "Empirical (0.5-5)" won 11/12 rounds vs no overlay. Going too aggressive (0.1-1) drops below current. Sweet spot: min_ps=0.3-0.5, max_ps=3-5.~~
- [R16] **OVERLAY BACKTEST WAS MISLEADING**: (0.5,3) won R2-R15 backtest (+2.72 avg) but CATASTROPHICALLY hurt on R16: scored 71.51 vs 87.00 with old (5,100) settings — **-15.5 point loss**. On volatile rounds (S→S=0.270), observations are misleading and aggressive overlay poisons predictions. REVERTED to (5,100). Lesson: in-sample backtest on observations is NOT reliable for overlay tuning.
- [R17 prep] **M5 ENTROPY BUCKET TEMPS: disable by setting all to 1.0 (+0.226 on R9-R16)**: The optimized per-class bucket temps were tuned on GBM-only baseline; after adding U-Net blend, they became counterproductive. USE_ENTROPY_TEMPS=True but temps=np.ones(6) is correct. WARNING: setting USE_ENTROPY_TEMPS=False falls back to a DIFFERENT legacy adaptive temp path that is worse — must keep it True.
- [R17 prep] **CALIBRATION FACTORS: keep at 1.0 (+0.070 on R9-R16)**: The Mountain calibration factor (0.95) was marginally helping on older rounds but slightly hurting on recent rounds. Combined with temps fix = +0.272 avg on R9-R16 validated.
- [R17 prep] **U-Net blend ratio LORO sweep (R1-R15, 16 rounds)**: U-Net weight: 0%=83.95, 20%=87.29, 30%=88.44, 40%=89.41, 50%=90.21, **60%=90.88** (best tested). Higher U-Net % wins on every single round. Did NOT test >60%. ~~Current model.py: 40% U-Net~~
- [R17 resubmit] **Quick blend sweep with saved models on R13-R16**: 30%=92.12, 40%=92.45, 50%=92.72, 60%=92.93, 70%=93.08, **80%=93.18**, **90%=93.23**, 100%=93.22. Monotonically increasing to 90%. R13 peaks at 50% (only round that prefers less U-Net). Updated UNET_BLEND_W to **0.80** (+0.73 over 0.40). Resubmitted R17.
- [R17 resubmit] **Bayesian overlay sweep with 80% U-Net (R13-R16)**: MIN_PS insensitive (1-20 all tie). MAX_PS=200 best (93.215), MAX_PS=100 at 93.181, delta only +0.034. MAX_PS=500 slightly worse. Overlay disabled = 93.147 (-0.034 from current). Updated MAX_PS to 200 for future rounds. Not worth resubmitting R17 for +0.03.
- [R17 prep] **Full LORO U-Net blend sweep (R1-R15) with extended weights**: avg scores 80%=92.06, 90%=92.37, 100%=92.52. Pure U-Net is best on this sweep, but gains above 80% are small and the curve is mostly flat on stable rounds (R8/R9/R10/R13/R15).
- [R17 prep] **Important caveat on the full LORO blend sweep**: GBM+MLP are retrained per fold, but the U-Net is a single global model reused across all folds, so higher U-Net weights are biased upward. Treat 100% as an optimistic upper bound, with 80-90% as the safer operating range.
- ~~[R17 prep] **Fast sweep on R14-R16 (with current models)**: Floor 0.0001 is optimal (no improvement at other values). Overlay (5,100) ties or beats all alternatives tested. Confirmed safe settings.~~
- [R12] **Oracle transitions are TERRIBLE (score 21)**: Even GT-perfect per-class transition matrices score only ~21 on R12. Spatial information is CRITICAL — the model must know WHICH cells of the same initial class will transition. Our spatial model at 67 captures massive spatial signal (67 >> 21). Leaders at ~88 have a better spatial model, not better transitions.
- [R12] **Dynamic calibration HURTS**: Scaling model predictions to match observed transition rates (-1 to -5 pts). Model predictions are already better calibrated than flat transition matrices because they incorporate spatial context.
- [R12] **Adaptive Bayesian observation overlay**: Entropy-adaptive prior_strength per cell. Uncertain cells (high entropy) trust observation (ps=5), confident cells (low entropy) trust model (ps=100). R9+0.04, R10+0.08, R11+0.47, R12+1.41. No regression on any round. Observations are stochastic (independent samples per API query). Implemented in build_prediction step 6b.
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

### Model V6: LGB+XGB Ensemble (post-R8)
- **Change**: Added XGBoost as second GBM with 70/30 LGB/XGB blend. Same hyperparameters (depth=4, lr=0.05, subsample=0.7, etc.).
- **LORO**: +0.37 vs V5

### Model V7: Triple-Blend with KL-Loss MLP (post-R9)
- **Core insight**: GBM models (LGB+XGB) are trained with MSE loss, but scoring uses entropy-weighted KL divergence. Loss function mismatch was the #1 untried improvement.
- **Change**: Added PyTorch MLP (128→64→6, softmax, BatchNorm, Dropout 0.1) trained with entropy-weighted KL divergence loss. Architecture directly optimizes the competition scoring metric.
- **Blend**: 60% GBM + 40% MLP (LORO-optimized)
- **LORO with post-processing**: 82.24 (was 80.66 for GBM-only — **+1.58 improvement**)
- **Per-round LORO** (60/40 blend vs GBM-only):
  - R1: 81.4 vs 80.8 (+0.6)
  - R2: 86.1 vs 86.8 (-0.7) — GBM wins slightly
  - R3: 83.4 vs 80.5 (+2.9) — MLP helps a lot
  - R4: 87.4 vs 85.2 (+2.2)
  - R5: 81.8 vs 81.7 (+0.1)
  - R6: 82.5 vs 83.3 (-0.8)
  - R7: 65.4 vs 66.5 (-1.1)
  - R8: 89.9 vs 80.3 (+9.6) — huge MLP win, likely collapse round where softmax output handles better
- **Weight sweep**: 0%→100% GBM tested. Sweet spot at 50-60% GBM. Pure MLP=79.69, pure GBM=80.66. 60/40 blend=82.24.
- **Key insight**: MLP and GBM have truly complementary strengths. GBM better on "normal" rounds (R1,R2,R5,R6), MLP better on unusual/collapse rounds (R3,R4,R8).
- **Key insight**: KL-loss softmax MLP naturally produces better-calibrated probability distributions. GBM outputs can go negative/above 1, requiring clipping. MLP softmax is always a valid distribution.
- **Training config**: Adam W (lr=1e-3, weight_decay=1e-4), cosine annealing, batch_size=4096, 200-300 epochs with patience=20 early stopping. 90/10 val split. Grad clip=1.0.
- **Files**: `train_mlp.py` (training), `data/mlp_model.pt` (weights), `model.py` updated with `TripleBlendPredictor`

### CRITICAL BUG FIX: Settlement Stats Train/Test Mismatch (discovered during R9)
- **Bug**: `compute_round_features()` overrode food/wealth/defense with real settlement observation stats at inference time, but the model was trained on GT-derived synthetic proxies (food=0.3+0.7×S→S, wealth=E→S×0.3, defense=1.0−S→E). This created a train/test domain shift.
- **Impact**: Catastrophic on some rounds — R5 went from 19.85 → 84.52 after fix (!), R7 from 44.39 → 67.92, R2 from 81.57 → 89.00
- **Root cause**: Real settlement food/wealth/defense stats measure actual settlement attributes (e.g., food=0.69), while training proxies measure transition-derived quantities (e.g., food=0.3+0.7×0.32=0.52). Completely different domains despite same feature name.
- **Fix**: Removed the settlement_stats override in `compute_round_features()`. Always use transition-derived proxies, matching what the model learned.
- **Full pipeline backtest (in-sample) with fix**: R1=82.8, R2=89.0, R3=41.3, R4=86.3, R5=84.5, R6=86.1, R7=67.9, R8=94.9. Avg=79.1
- **R8 weighted = 140.2** (in-sample) — matches leader's 140.3!
- **Key lesson**: When adding new features to the model, the training-time feature computation MUST use the same formula as inference-time. Settlement stats were only available at inference (from observations) but not at training (which used GT transitions), so the override created an undetectable mismatch.
- ~~[V4] Settlement stat fallback: When no observations, derive proxies from transition matrix~~ Now ALWAYS uses transition-derived proxies, even when settlement stats are available.
- **R9 resubmitted** with V7 + this fix. Score pending (round closes 20:47 UTC).

### Round 9
- Round ID: 2a341ace-0f57-4309-9b89-e59fe0f09179
- Round weight: 1.551 (1.05^9)
- NORMAL mode, 5 seeds
- 50/50 queries used
- Submitted with V7 triple-blend + settlement stats bug fix
- **Score: 90.59 (weighted 140.5, NEW #1 LEADERBOARD!)** — seeds pending
- Bug fix was worth ~17 pts on this round (R9 with old V6+bug = would have been ~73)
- R9 appears to be a "NORMAL" round with high predictability (raw 90.59 is our best ever)
- Ground truth downloaded for all 5 seeds

### Multi-Round Retraining (R1-R9)
- [R1-R9] HISTORICAL_TRANSITIONS updated from all 9 rounds (72K cells, 45 seeds)
- [R1-R9] **LORO avg: 82.32** (was 79.67 with R1-R8 — **+2.65 from adding R9 data!**)
  - Per-round: R1=84.0, R2=87.6, R3=82.9, R4=90.2, R5=81.9, R6=78.4, R7=68.1, R8=78.5, R9=89.2
  - R1 LORO: 77.5→84.0 (+6.5), R4: 87.3→90.2 (+2.9), R8: 75.7→78.5 (+2.8)
- [R1-R9] In-sample: R1=87.2, R2=91.0, R3=91.6, R4=93.0, R5=86.4, R6=88.7, R7=75.3, R8=95.3, R9=93.2
- [R1-R9] MLP trained to val_kl=0.0474 (was 0.0510 with R1-R8)

### Model V8: Edge Distance Feature + Re-tuned Calibration (post-R9)
- **New feature**: `edge_dist = min(y, x, map_h-1-y, map_w-1-x) / (min(map_h, map_w) / 2)` — normalized distance to nearest map edge. Cells at edge behave differently (less expansion, fewer neighbors).
- **Feature count**: 23 spatial + 8 round = **31 features** (was 30)
- **GBM LORO**: 82.78 (was 82.32 — **+0.46**)
- **MLP val_kl**: 0.0347 (was 0.0474 — **much better**)
- **Feature ablation results**:
  - edge_dist alone: +0.57 LORO — clear winner
  - mountain_dist alone: -0.01 LORO — useless, rejected
  - Both together: +0.22 — mountain_dist dilutes edge_dist
  - Bigger model (800 trees, depth 5): -0.68 — overfits, rejected
- **Re-tuned post-processing**: New temps [E=1.10, S=1.05, P=1.10, R=1.0, F=1.10, M=1.0]. Calibration [E=0.98, S=1.0, P=1.0, R=1.0, F=0.95, M=1.0] best for R9+ (tested 4 configs).
- **Key insight**: Greedy parameter sweeps (optimize one param at a time) can find suboptimal configs. Direct comparison of full config combos is more reliable for final tuning.
- **Key insight**: Re-sweep calibration/temps after every model change. Old optimal [E=0.98, F=0.95] was still best for the new 31-feature model, not the sweep-found [E=1.07, SPR=0.92].
- **Backtest**: R9=93.44 raw (weighted 145.0). Was 90.59 submitted (weighted 140.5). **+4.5 weighted pts improvement.**
- **Full backtest**: R1=83.9, R2=90.2, R3=45.0, R4=88.7, R5=87.3, R6=89.4, R7=73.7, R8=95.2, R9=93.4. Avg=83.0.
- **Gap to leader**: Leader=146.3 (raw ~94.3). Our R9 backtest=93.4 (weighted 145.0). Gap = 1.3 weighted, 0.9 raw.
- R10 weight = 1.05^10 = 1.629. If we score 93+ raw on R10, weighted = 151+ → new leader.

### Pre-R10 Optimization Audit
- **Blend ratio re-swept**: 50/50 GBM/MLP now optimal (was 60/40). MLP improved significantly with edge_dist feature (val_kl 0.0474→0.0347), justifying more MLP weight.
- **Calibration re-swept for 50/50**: E=1.0 (was 0.98) — blend shift already corrects Empty-class bias. F=0.95 still needed.
- **Temperature sweep confirmed**: Current per-class temps (E/F=1.10, S=1.05, P=1.10) maximize R9 (93.47). Uniform T=1.05 gives better avg (88.44) but worse R9 (93.20). Optimize for leaderboard = max weighted round.
- **Calibration order**: cal_then_temp vs temp_then_cal makes zero difference — mathematically expected since both are multiplicative.
- **Collapse detection verified**: R8 correctly detected (S→S=0.079 < 0.15 threshold). All other rounds classified normal.
- **R9 per-seed analysis**: Seeds 0,1 weakest (91.8, 92.5), seeds 2-4 strongest (94.0-94.6). Loss dominated by Empty (55-65%) and Forest (28-33%). Settlement loss only 1-3%. No actionable fix — variability inherent in round dynamics.
- **Grid coverage verified**: 9 viewports (15×15) achieve 100% coverage of 40×40 map with 19% overlap.
- **All 9 historical rounds**: 40×40, 5 seeds each. Model handles variable sizes via params.
- **Updated backtest**: R9=93.47 (+0.03), R7=73.96 (+0.29), avg=83.07 (+0.10). Weighted R9=145.0.

### Round 10
- Round ID: 75e625c3-60cb-4392-af3e-c86a98bde8c2
- Round weight: 1.6289 (1.05^10)
- Map: 40×40, 5 seeds. Seed settlements: 39, 52, 41, 56, 50. Ports: 3, 1, 0, 2, 3.
- Model: V8+ (50% GBM + 50% MLP, 31 features, edge_dist)
- 50/50 queries used. Grid 9 viewports/seed + 5 extra (one per seed, shifted viewports).
- Submitted pass 1 + pass 2 (with extra observations).
- **Observed transition diagnostics vs historical**: S→S is -23.8% (settlements dying more), E→S is -9.4% (less expansion), F→F is +19.0% (forest very stable), P→P is -10.7%. Detected as NORMAL mode.
- This appears to be a "quiet" round — low settlement activity, forest-dominant final state.
- **Score: 91.61 (weighted 149.22, rank 8/238)**
- Seeds: 90.40, 92.00, 91.60, 92.22, 91.83
- **R10 was a COLLAPSE round** (S→S=0.058 GT, similar to R3/R8): all settlements died
  - GT transitions: E→E=0.981, S→S=0.058, F→F=0.959, E→S=0.007
  - Was misdetected as NORMAL mode at submission (obs showed S→S=-23.8% vs hist, but collapse detection uses debiased S→S < 0.15)
- Despite NORMAL mode detection, model scored well (91.61) thanks to spatial model generalization from R3/R8 collapse training data
- Ground truth downloaded for all 5 seeds

### Multi-Round Retraining (R1-R10)
- [R1-R10] HISTORICAL_TRANSITIONS updated from all 10 rounds (80K cells, 50 seeds)
- [R1-R10] Transition matrix: Empty→Empty 86.3%, Set→Set 26.7%, For→For 78.6%, Port→Port 15.3%
- [R1-R10] SHRINKAGE_MATRIX updated from 7 obs rounds (R2, R5, R6, R7, R8, R9, R10)
- [R1-R10] Spatial model retrained with 31 features on 80K samples
- [R1-R10] **LORO avg: 84.26 (spatial), 84.98 (+sim blend, d=+0.71)**
  - Per-round LORO: R1=84.6, R2=89.5, R3=87.6, R4=91.1, R5=81.5, R6=81.2, R7=66.4, R8=86.7, R9=90.6, R10=90.7
  - R3 LORO: 82.9→87.6 (+4.7) — R10 collapse data further helps R3
  - R8 LORO: 78.5→86.7 (+8.2) — huge gain from two more collapse examples
  - R10 LORO: 90.7 (new, very strong for a collapse round)
- [R1-R10] In-sample backtests (with obs): R1=82.7, R2=90.4, R3=48.7, R4=90.6, R5=87.0, R6=89.3, R7=73.3, R8=95.7, R9=93.4, R10=93.4
  - R3=48.7 (no obs files — falls back to bare model)
- [R1-R10] MLP val_kl: 0.0326 (was 0.0347 with R1-R9)
- [R1-R10] Weighted leaderboard projections: R10=152.1 (best), R9=144.9, R8=141.3
- [R1-R10] **4 collapse rounds now in training** (R3, R4, R8, R10) — 40% of rounds. Model should generalize much better to future collapses.

### Round 11
- Round ID: 324fde07-1670-4202-b199-7aa92ecb40ee
- Round weight: 1.7103 (1.05^11)
- Map: 40×40, 5 seeds. Seed settlements: 33, 32, 56, 35, 34. Ports: 0, 1, 1, 1, 1.
- Model: V8+ (50% GBM + 50% MLP, 31 features, trained on R1-R10)
- 50/50 queries used. Grid 9 viewports/seed + 5 extra.
- Submitted pass 1 + pass 2 (with extra observations).
- **Observed transition diagnostics**: E→E: -15.5%, S→S: +23.4%, F→F: -18.3%, E→S: +14.9%, P→P: +26.5%. HIGH-ACTIVITY mode detected.
- This was a **boom/growth round** — settlements expanded aggressively
- **Score: 81.77 (weighted 139.85)**
- Seeds: 83.75, 81.01, 80.11, 82.66, 81.32
- GT transitions: E→E=0.721, S→S=0.495, F→F=0.602, E→S=0.222
- Highest S→S (49.5%) and E→S (22.2%) of any round — extreme settlement boom
- LORO prediction was 80.69 — actual 81.77 in line with expectation
- Ground truth downloaded for all 5 seeds

### Multi-Round Retraining (R1-R11)
- [R1-R11] HISTORICAL_TRANSITIONS updated from all 11 rounds (88K cells, 55 seeds)
- [R1-R11] Transition matrix: Empty→Empty 85.1%, Set→Set 28.5%, For→For 76.9%, Port→Port 15.9%
- [R1-R11] SHRINKAGE_MATRIX updated from 8 obs rounds (R2, R5, R6, R7, R8, R9, R10, R11)
- [R1-R11] **LORO avg: 83.99 (spatial), 84.64 (+sim blend, d=+0.65)**
  - Per-round LORO: R1=85.0, R2=89.5, R3=86.9, R4=91.0, R5=81.1, R6=82.7, R7=66.3, R8=86.7, R9=90.8, R10=90.4, R11=80.7
  - R11 LORO: 78.78→80.69 with sim (+1.91) — sim blend helps on high-activity
- [R1-R11] MLP val_kl: 0.03326 (was 0.0326 with R1-R10)
- [R1-R11] Best weighted score remains R10=149.22. R11=139.85 not a new best.
- [R1-R11] Competitive thresholds to beat 152.1 leader: R12=84.69, R13=80.66, R14=76.82

### Round 12
- Round ID: 795bfb1f-54bd-4f39-a526-9868b36f7ebd
- Round weight: 1.7959 (1.05^12)
- **Score: 58.89 → improved to 67.41 (backtest) with recency-weighted transitions + activity detector fix**
- GT S→S per seed: 0.572, 0.628, 0.554, 0.583, 0.636 (mean 0.5946) — extreme unprecedented high
- Calibrated S→S = 0.6005, Debiased S→S = 0.6104 (both very close to GT)
- **Key analysis**: Oracle transitions (using GT) only score ~21 — spatial info is critical. Our model at 67 captures major spatial signal.
- **Dynamic calibration (scaling model to match transitions) HURTS** — model predictions are better calibrated than flat transition matrices
- **Adaptive Bayesian observation overlay**: +1.41 on R12 (67.41→68.81), +0.47 on R11, +0.08 on R10, +0.04 on R9. NO regression. Implemented in build_prediction() with min_ps=5, max_ps=100.
- Observations are STOCHASTIC: each API query runs a fresh simulation. Overlapping viewports give independent samples. R12 S0: 70/473 multi-obs cells disagree.
- Full 40×40 map is observed via 9 viewports of 15×15 (with overlaps giving 1-4 obs per cell).
- ~~Remaining ~20pt gap to leaders unexplained — likely requires fundamentally better spatial model architecture.~~ Gap closed to ~7 pts after V9 (7×7 ring + Bayesian overlay). R13 scored 90.10 (weighted 169.90, leader 177.1).

### Multi-Round Retraining (R1-R12)
- Model retrained at 2026-03-21 07:15 (includes R12's S→S=0.59 in training data)
- LORO avg with adaptive overlay: R9=93.35, R10=93.19, R11=88.22, R12=68.81
- After V9 optimizations (7×7 ring, sim blend disabled): R9=93.54, R10=94.36, R11=88.47, R12=73.31

### Round 13
- Round ID: 7b4bda99-6165-4221-97cc-27880f5e6d95
- Round weight: 1.8856 (1.05^13)
- Map: 40×40, 5 seeds. Settlements: 43, 53, 54, 44, 56. Ports: 0, 3, 1, 0, 5.
- **S→S = 0.260 (NORMAL mode)** — similar to R9 (0.275) where we scored 93.31
- 50/50 queries used. Submitted with adaptive Bayesian overlay.
- Port collapse observed: Port→Empty +13-17%, Port→Port -6.6%
- Settlement declining: S→S about -5% below historical
- **Score: 90.10 (weighted 169.90, rank 32)** — seeds: 88.92, 89.81, 90.31, 91.45, 90.04
- Ground truth downloaded for all 5 seeds

### Model V9: 7×7 Ring Features + Sim Blend Disabled (post-R12 optimization)
- **Change 1: 7×7 outer ring features**: Added 6 features for class fractions in the 7×7 outer ring (cells at Chebyshev distance exactly 3). Total features: **29 spatial + 8 round = 37**.
  - Feature breakdown: one-hot(6) + 3×3 neighborhood(6) + 5×5 outer ring(6) + 7×7 outer ring(6) + dist features(5) + round features(8)
  - Backtest improvement: R12 +2.22 pts (70.24→72.46), R10 +0.76 (93.38→94.14), consistent across rounds
- **Change 2: Sim blend DISABLED (alpha=0)**: After adding adaptive Bayesian overlay (step 6b), sim blend (step 5a) became redundant. Tested alpha sweep [0, 0.05, 0.10, 0.15]: alpha=0 beats alpha=0.15 on ALL rounds. R9: +0.05, R10: +0.18, R11: +0.10, R12: +1.43.
- **9×9 ring REJECTED**: Tested outer ring at Chebyshev distance 4. LORO dropped 81.32→80.86, R12 LORO crashed 51.25→47.19. Overfitting on 40×40 map.
- **Blend ratio verified**: 50/50 GBM/MLP is optimal. Tested 30/70 through 70/30 — differences <±0.6 pts, inconsistent.
- **Calibration/temp verified**: Current per-class temps and calibration factors are near-optimal. Tested 9 combinations — best is within 0.1 pts.
- **Settlement attributes analyzed**: Wealth has strongest survival signal (Q1=21.4% → Q4=34.1% survival rate), but post-hoc adjustment HURTS because adaptive Bayesian overlay already captures the signal from observations.
- **LORO avg: 82.46** (R1-R13, 37 features, 104K samples)
  - Per-round: R13 LORO = 91.13
- **Backtest (R1-R12 train)**: R9=93.54, R10=94.36, R11=88.47, R12=73.31

### Multi-Round Retraining (R1-R13)
- [R1-R13] Training on 104,000 samples (13 rounds × 5 seeds × 1600 cells), 37 features
- [R1-R13] LORO avg: 82.46 (up from 81.32 with R1-R12)
- [R1-R13] R13 LORO: 91.13 (strong — normal round data helps)
- [R1-R13] MLP + LGB + XGB ensemble + adaptive Bayesian overlay saved to data/

### Leaderboard Context (post-R13)
- Our best weighted: R13 = 169.90. Leader = 177.1 (from R13, implying raw ~93.9).
- Gap: 7.2 weighted pts. To beat on R14 (weight=1.9799): need ≥89.45 raw.
- Our model backtests ~93 on normal rounds, so R14 should close the gap if normal.
- Leaderboard API showed 0.00 for all teams on 2026-03-21 09:33 UTC (possible reset/scoring period change).

### Round 14 (Active)
- Round ID: d0a2c894-2162-4d49-86cf-435b9013f3b8
- Round weight: 1.9799 (1.05^14)
- Map: 40×40, 5 seeds, closes 2026-03-21T11:59:55 UTC
- Mountain cluster at rows 3-8 NE and rows 23-29 center-left (5s in grid)
- Ports: at least 2 visible (grid value 2 at ~(35,2) and ~(37,3))
- 0/50 queries used — awaiting user permission to start
- Initial grid: 51 settlements, 2 ports, 335 forests, 21 mountains, 1000 plains, 191 ocean

### Session Findings (2026-03-21 pre-R14)

#### Simulator Verified Inert
- **Simulator has ZERO effect on live predictions**: `simulator.py` is NOT imported during `build_prediction()`. The sim blend code (model.py lines 866-879) is fully commented out. The `simulator` module is only loaded by `train_spatial.py` for LORO diagnostic printing (`.+sim=` scores) — it does NOT affect trained model weights or inference.
- Verified by running `build_prediction()` and confirming `'simulator' not in sys.modules`.

#### D4 Rotation Augmentation REJECTED
- **Tested D4 group augmentation** (4 rotations × 2 flips = 8× data) for GBM training.
- **LORO dropped from 82.46 → 82.19** with augmentation enabled.
- **Per-round regressions**: R9: 93.54→90.00 (-3.54), R10: 94.36→90.25 (-4.11), R11: 88.47→78.17 (-10.30), R12: 73.31→52.48 (-20.83).
- **Root cause**: Tree models (LGB 500 trees, depth 4) underfit with 8× training data (832K samples). Features (neighborhood fractions, distances, edge_dist) are already approximately rotation-invariant — augmentation adds redundant data that dilutes specific spatial patterns.
- **Conclusion**: Data augmentation does NOT help GBM/XGB spatial models here. Would need capacity increase (more trees, deeper) which risks overfitting. Code remains in `train_spatial.py` but `USE_AUGMENTATION = False`.

#### Extra Queries → Repeat Strategy
- **Changed `spend_extra_queries`** in `run_next_round.py`: instead of querying offset viewports at different positions, now repeats the standard 9-viewport grid on the most dynamic seed (sorted by settlement count descending).
- **Rationale**: Repeat observations give cells a 2nd independent stochastic sample, strengthening the adaptive Bayesian overlay (alpha += 1 per observation). The overlay is worth +0.04 to +1.41 per round — more observations compound this benefit.
- **Budget math**: 50 total - 45 grid = 5 remaining → repeats on top seed's first 5 viewports.

### Round 14 Results
- Round ID: d0a2c894-2162-4d49-86cf-435b9013f3b8
- Round weight: 1.9799 (1.05^14)
- **Score: 81.93 (weighted 162.2, rank 35)** — seeds: 82.69, 83.08, 81.69, 81.64, 80.57
- Diagnostics: S→S=0.510 (HIGH-RETENTION), E→S=+11.4%, F→F=-18.4%, Activity=14.09%
- BOOM round confirmed — settlement retention well above historical
- LORO prediction was 81.32 — actual 81.93, in line
- 50/50 queries used (45 grid + 5 repeat on most dynamic seed)
- GT downloaded for all 5 seeds

### Multi-Round Retraining (R1-R14)
- [R1-R14] Training on 112,000 samples (14 rounds × 5 seeds × 1600 cells), 37 features
- [R1-R14] **LORO avg: 82.54** (was 82.46 with R1-R13, +0.08 from R14 data)
  - Per-round: R1=85.18, R2=88.55, R3=88.47, R4=91.02, R5=82.23, R6=82.97, R7=69.83, R8=79.58, R9=89.42, R10=90.79, R11=78.34, R12=56.39, R13=91.42, R14=81.32
- [R1-R14] MLP val_kl: 0.03501 (epoch 109, early stop 129). Architecture: [256, 128, 64] with residual connections.
- [R1-R14] Learned debiasing REJECTED again (LOO MSE 0.000156 → 0.014984, 100× worse). Static SHRINKAGE_MATRIX remains optimal.
- [R1-R14] Deeper MLP [256,128,64] with residuals deployed — provides better feature extraction for triple-blend.

### Leaderboard Context (post-R14)
- Our best weighted: R13 = 169.90 (90.10 × 1.8856). R14 weighted = 162.2 (not a new best).
- To beat leader (177.1) on R15 (weight=2.0789): need ≥85.19 raw.
- R15 is now available (score=None, 0 queries).

### Model V10: U-Net Spatial Blend (post-R14)
- **Architecture**: U-Net (19 input channels, 6 output classes, base_channels=32, ~477K params) added as third model head alongside GBM+MLP.
  - Input channels: 6 terrain one-hot + 3 distance features (settlement/forest/port EDT) + settlement count (radius-5 conv) + edge distance + 8 round features = 19 channels on 40×40 grid
  - D4 test-time augmentation (TTA): 8 transforms (4 rotations × 2 flips), averaged
  - Loss: entropy-weighted KL divergence (matches competition scorer)
- **U-Net LORO (standalone)**: avg=82.34 base, **83.32 with TTA** (vs GBM+MLP LORO 82.54)
  - Per-round LORO+TTA: R1=87.51, R2=89.99, R3=85.15, R4=91.50, R5=86.47, R6=88.19, R7=76.18, R8=84.57, R9=92.69, R10=90.27, R11=78.58, R12=61.46, R13=92.22, R14=77.20
  - U-Net LORO beats GBM+MLP LORO on 9/14 rounds, loses on R7, R8, R10, R12, R14
- **CAUTION: In-sample vs LORO discrepancy**: Full-training in-sample scores show GBM+MLP=88.36 vs UNet+TTA=83.33 — U-Net loses on 13/14 rounds by 7-10 pts in-sample. LORO tells a different story (U-Net wins 9/14). This means U-Net generalizes better to unseen rounds but is weaker when the round is in training data. Blend ratios optimized on in-sample backtest may be overconfident.
- **Blend ratio sweep** (backtest with observations, in-sample):
  - 0% U-Net: 88.63 avg
  - 20% U-Net: 89.36
  - 30% U-Net: 89.49
  - **40% U-Net: 89.51** (BEST)
  - 50% U-Net: 89.44
  - 70% U-Net: 89.08
- **Optimal blend (in-sample)**: 60% GBM+MLP + 40% U-Net+TTA. +0.88 pts over GBM+MLP-only in backtest.
- **TODO**: LORO-based blend sweep needed to validate out-of-sample. In-sample optimal may differ from LORO optimal.
- **Pipeline**: In `build_prediction()` step 4b, after spatial_prior (GBM+MLP), blend with U-Net TTA predictions before calibration/temperature/overlay.
- **Key insight**: U-Net should NOT be used standalone — it's significantly weaker in-sample. Value is as an ensemble component only.
- **Key insight**: TTA is critical for U-Net (+0.98 LORO). The 40×40 grid is small enough that 8× forward passes add minimal latency (~1-2s per seed).
- **Risk note**: U-Net has only been LORO-validated and in-sample backtested, not yet validated on a live round.

### Round 15
- Round ID: cc5442dd-bc5d-418b-911b-7eb960cb0390
- Round weight: 2.0789 (1.05^15)
- Map: 40×40, 5 seeds. Settlements: 32, 40, 53, 31, 59. Ports: 1, 0, 3, 0, 1.
- Model: **V10** — first live use of U-Net blend (60% GBM+MLP + 40% U-Net+TTA)
- 50/50 queries used (45 grid + 5 repeat on seed 4, highest settlement count 59)
- Submitted pass 1 + pass 2 (with extra obs on seed 4)
- **Observed transition diagnostics**:
  - E→E: -8.4% (moderate expansion)
  - S→S: +3.8% above historical (0.348 vs 0.310) — above-average retention
  - E→S: +5.8% above historical — significant expansion
  - F→F: stable (no notable deviation)
  - P→Ruin: +7.1% — ports decaying
  - P→Settlement: -5.1%, P→Forest: -5.2% — ports losing to all classes
  - Activity: 8.44% → NORMAL mode
- **Prediction: ~85 ± 4 raw** (range 81-89). Favorable S→S (0.348) suggests a predictable round. To beat leader (177.1), need ≥85.19 raw. Weighted at 85 = 176.7. At 87 = 180.9 (new best!).
- **Score: 92.53 raw (weighted 192.37)** — BEST EVER by huge margin
  - Per-seed: 91.90, 93.13, 91.51, 92.47, 93.64 — all above 91!
  - Rank: 25 (leaderboard may be recalculating)
  - Previous best: R13 = 90.10 (169.90 weighted)
  - **Key takeaway**: V10 (U-Net blend + M5 bucket temps) is a massive improvement. High S→S (0.348) round was favorable, but the model also generalized well across all 5 seeds (tight spread: 91.5-93.6).

### Round 16
- Round ID: 8f664aed-8839-4c85-bed0-77a2cac7c6f5
- Round weight: 2.1829 (1.05^16)
- Map: 40×40, 5 seeds. Settlements: 58, 43, 50, 35, 37. Ports: 4, 1, 1, 0, 0.
- Model: **V10 + overlay fix** — same GBM+MLP/U-Net blend + two key changes:
  - PROB_FLOOR: 0.0001 → 0.0003 (LORO +0.066 avg)
  - Bayesian overlay: (5,100) → (0.5,3) — backtest showed +2.72 avg on R2-R15 GT
- 50/50 queries used (45 grid + 5 repeat on seed 0, highest settlement count 58)
- **Observed transition diagnostics**:
  - E→E: +7.6% above historical — less expansion than usual
  - E→S: -5.3% — settlements expanding less
  - S→S: 0.270 (vs 0.318 historical) — below-average retention
  - F→F: +12.7% — strong forest retention
  - S→F: +3.8% — settlements being reclaimed by forest
  - P→E: +19.0%, P→P: -11.2% — ports very volatile
  - Activity: 0.00% → NORMAL mode
- **Score: 71.51 raw (weighted 156.10)** — WORST since R8. Rank 171/272.
  - Per-seed: 67.75, 75.10, 74.24, 66.52, 73.94
  - **Root cause**: Aggressive overlay change (5,100)→(0.5,3) cost 15.5 points. Old settings would have scored 87.00.
  - On this volatile round (S→S=0.270), observations were misleading. Aggressive overlay pushed predictions toward noisy obs.
  - **REVERTED** overlay to (5,100) and floor to 0.0001 for R17+
  - Leaderboard still led by R15 = 192.37 weighted (unchanged)

### Round 17
- Round ID: 3eb0c25d-28fa-48ca-b8e1-fc249e3918e9
- Round weight: 2.2920 (1.05^17) — closing target was stale; live leader after close is 217.38 weighted
- Map: 40×40, 5 seeds. S→S=0.448 (very high retention), Activity=14.26%
- Model: **V10 + calibration/temps optimization** (models retrained on R1-R16):
  - CALIBRATION_FACTORS: [1,1,1,1,0.95,1] → [1,1,1,1,1,1] (Mountain calibration removed)
  - All ENTROPY_BUCKET_TEMPS: per-class optimized values → np.ones(6) (effectively disabled)
  - Combined improvement validated: +0.272 avg on R9-R16 (91.241 → 91.513)
  - Per-round validation (E=combined): R9=93.92, R10=94.60, R11=91.93, R12=81.84, R13=93.33, R14=91.43, R15=94.40, R16=90.65
  - Models freshly retrained on all R1-R16 data
- 50/50 queries used
- **Score: 84.7388 raw (weighted 194.22)** — new personal best weighted score, but not enough to catch the new leader
  - Rank: 96/283
  - Per-seed: 86.6977, 86.1768, 83.0849, 85.0461, 82.6886
  - Would have missed the current 217.38 leader by 23.15 weighted
  - R17 beat R15 on weighted score (194.22 vs 192.37), so the resubmission/revert recovered a new best, just not a podium-level one

### Scoring & Strategy Correction (post-R17)
- **Round weight growth does NOT help us catch the leader**: Weight = 1.05^n applies to ALL teams equally. If the leader also scores 85 raw on R20, their weighted = 85 × 2.653 = 225.5, pulling further ahead. Higher weights amplify the gap, not close it.
- **What actually matters**: Peak raw score relative to the leader's peak raw score. Leader's 217.38 weighted ÷ their round weight = their best raw score. We need to OUTSCORE them on raw in the same round (or a round where they underperform).
- **Our raw ceiling**: R9=90.6, R10=91.6, R13=90.1, R15=92.5. We can hit 90+ on favorable rounds. The leader likely hit ~93-95 raw on their best round. Gap is 1-3 raw points.
- **Priority**: Maximize peak raw score through clean execution + retrained models. No experiments mid-round (R16 overlay change cost 15.5 raw points).

### Multi-Round Retraining (R1-R17)

**GBM+MLP LORO (R1-R17, 85 seeds, 136K cells):**
| Round | Spatial | +Sim | Baseline |
|-------|---------|------|----------|
| R1 | 85.68 | 85.37 | 69.61 |
| R2 | 89.48 | 89.85 | 73.35 |
| R3 | 87.60 | 87.58 | 52.29 |
| R4 | 90.33 | 90.67 | 87.17 |
| R5 | 84.16 | 83.02 | 67.06 |
| R6 | 84.30 | 84.28 | 52.70 |
| R7 | 68.54 | 67.02 | 40.71 |
| R8 | 81.90 | 87.77 | 75.07 |
| R9 | 90.97 | 90.99 | 82.87 |
| R10 | 91.19 | 91.26 | 65.89 |
| R11 | 84.07 | 85.30 | 50.15 |
| R12 | 58.34 | 56.48 | 24.09 |
| R13 | 91.96 | 91.77 | 84.60 |
| R14 | 80.21 | 79.18 | 38.62 |
| R15 | 91.33 | 91.83 | 77.80 |
| R16 | 83.01 | 82.50 | 75.35 |
| R17 | 86.29 | 86.99 | 56.55 |
| **Avg** | **84.08** | **84.23** | **63.17** |

- Previous R1-R16 avg was spatial=83.31, +sim=83.62. Adding R17 improved avg by +0.77/+0.61.
- R17 LORO: spatial=86.29, +sim=86.99 (strong out-of-sample performance)

**U-Net LORO (R1-R17, 85 seeds, TTA):**
| Round | UNet | +TTA |
|-------|------|------|
| R1 | 80.18 | 81.60 |
| R2 | 84.60 | 85.70 |
| R3 | 87.91 | 88.29 |
| R4 | 91.57 | 92.13 |
| R5 | 83.47 | 83.95 |
| R6 | 82.74 | 83.65 |
| R7 | 71.71 | 73.36 |
| R8 | 83.16 | 83.50 |
| R9 | 92.81 | 93.27 |
| R10 | 90.92 | 91.35 |
| R11 | 82.32 | 83.69 |
| R12 | 48.71 | 51.75 |
| R13 | 92.14 | 92.70 |
| R14 | 76.06 | 76.78 |
| R15 | 91.07 | 92.04 |
| R16 | 84.96 | 86.31 |
| R17 | 82.31 | 83.71 |
| **Avg** | **82.74** | **83.75** |

- Models saved: `data/spatial_model.pkl`, `data/mlp_model.pt`, `data/unet_model.pt`
- All trained on R1-R17 (17 rounds, 85 seeds)

