# Plan 02 — Dramatic Score Improvement

*Created: 2026-03-20, based on Round 1 ground truth backtesting*

**Current baseline:** 74.4/100 with zero queries (historical transition matrix + floor 0.001)

---

## Where the Score Is Lost

| Initial Class | Cells | Score Loss Share | Insight |
|---|---|---|---|
| **Empty** | 1186 (74%) | **68.9%** | Our prediction says 14.7% → Settlement globally. True rate varies 0–50% by position. |
| **Forest** | 347 (22%) | **29.5%** | Similar: 17.5% → Settlement globally, but spatially concentrated near existing settlements. |
| Settlement | 29 (2%) | 1.4% | Already well-predicted by transition matrix. |
| Mountain | 37 (2%) | 0.0% | Perfect — always stays Mountain. |

**The entire game is predicting Empty and Forest cells.** Together they are 97% of cells and 98.4% of score loss.

### Diminishing returns curve (perfecting top-N worst cells):

| Top-N cells fixed | Score | Gain |
|---|---|---|
| 0 (baseline) | 73.4 | — |
| 50 | 79.9 | +6.5 |
| 100 | 83.1 | +9.7 |
| 200 | 87.0 | +13.6 |
| 400 | 91.5 | +18.1 |
| 800 | 96.3 | +22.9 |

**Fixing just the 400 worst cells gets us above 90.** These are empty/forest cells near settlement boundaries.

---

## Improvement Strategies (Ranked by Expected Impact)

### 1. Spatial Conditional Model — *predict per-cell, not per-class* ⭐⭐⭐

**Current:** Every empty cell gets the same prediction (14.7% Settlement, 79.5% Empty, ...).

**Better:** P(final | initial_class, neighbor_initial_classes, distance_to_nearest_settlement, ...)

**Why it matters:** An empty cell surrounded by 3 settlements should predict ~40% Settlement. An isolated empty cell should predict ~2% Settlement. Currently both get 14.7%.

**How:**
- Features per cell: initial class (one-hot), count of each initial class in 3×3 and 5×5 neighborhoods, distance to nearest settlement, distance to nearest coast/port, distance to nearest mountain
- Target: ground truth probability distribution (6-dim)
- Model: train a small neural net or gradient-boosted model on Round 1 data (5 seeds × 1600 cells = 8000 training samples)
- At prediction time: compute features from initial state, run model
- **Estimated gain: +8 to +15 points** (this IS the #1 lever)

### 2. Bayesian Observation Updates — *don't throw away the prior* ⭐⭐⭐

**Current:** If a cell is observed once as "Empty," the prediction snaps to [1,0,0,0,0,0]. This is a terrible estimate of a probability — it replaces a good prior with a single sample.

**Better:** Dirichlet-Multinomial Bayesian update:
```
prior:           α = transition_probs × prior_strength (e.g., α = [7.9, 1.5, 0.1, 0.1, 0.3, 0.0] for empty cell)
after 1 obs:     α' = α + observation_one_hot
posterior mean:  α' / sum(α')
```

With `prior_strength=10` and 1 observation of Empty:
- Before: `[0.795, 0.147, 0.012, 0.014, 0.034, 0.000]`
- Current code: `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]` ← bad!
- Bayesian: `[0.811, 0.134, 0.011, 0.013, 0.031, 0.000]` ← sensible shift

With 5 observations (4 Empty, 1 Settlement):
- Bayesian: `[0.792, 0.163, 0.008, 0.009, 0.023, 0.000]`

**Estimated gain: +3 to +7 points** (fixes the systematic overconfidence on observed cells)

### 3. Adaptive Query Targeting — *observe what matters* ⭐⭐

**Current:** 9 viewports per seed for full coverage (1 obs per cell, 2 in overlaps).

**Better, two-phase approach:**
1. **Phase 1 (2 queries/seed):** Place 2 viewports on settlement-dense regions to learn round-specific transitions fast
2. **Phase 2 (remaining 8 queries/seed):** Target the highest-uncertainty cells — settlement/forest boundaries where transition model disagrees with spatial model

Or even: **skip full coverage entirely.** The transition model scores 74 with zero observations. Use all 10 queries per seed for 5× repeat observations of the ~500 most uncertain cells. This gives us 5+ observations per cell in the zones that matter.

**Query allocation math:**
- 50 queries / 5 seeds = 10 per seed
- 10 queries × 225 cells = 2250 cell-observations
- Option A: 1600 cells × ~1.4 obs each (current) 
- Option B: 500 cells × ~4.5 obs each (targeted) ← better per-cell estimates where it matters

**Estimated gain: +2 to +5 points** (depends on accuracy of targeting)

### 4. Settlement Metadata for Survival Prediction ⭐⭐

**Unused data:** Each simulation response includes settlement metadata: population, food, wealth, defense, has_port, alive, owner_id.

**Insight:** We can correlate these with survival:
- High population → more likely to survive
- Low food → more likely to die
- Has port → more stable
- Multiple settlements with same owner_id clustered → colony protection

**How:** Collect settlement stats from multiple observations. Settlements that consistently appear across queries are more likely to exist in the final state. Settlement positions that appear in 8/10 queries → ~80% settlement probability.

**Estimated gain: +1 to +3 points** (helps specifically for settlement cells, which are only 1.4% of loss)

### 5. Round-Specific Transition Calibration ⭐⭐

**Current:** Uses fixed historical avg transition matrix until observations arrive, then swaps to round-specific.

**Better:** Bayesian blend of historical and round-specific transitions:
```
effective_transitions = (historical × prior_weight + round_obs × obs_weight) / (prior_weight + obs_weight)
```

After observing 500 cells, the round-specific matrix is fairly trustworthy. But for rare transitions (Port, Ruin), we should lean on historical data.

**Estimated gain: +1 to +3 points** (biggest gain in first few queries; useful for adaptive query strategy)

### 6. Cloud Run Deployment — *never miss a round* ⭐

Currently we submit manually or via local polling (auto_round.py). A deployed Cloud Run endpoint would:
- Respond to /solve within 60s
- Auto-submit using latest model
- Guarantee participation in every round

**Estimated gain: 0 points per round, ∞ points over competition** (we already missed Round 1)

### 7. Multi-Round Model Accumulation ⭐

After each round, download ground truth and:
- Update HISTORICAL_TRANSITIONS with R1+R2+R3+... data
- Retrain spatial conditional model on growing dataset
- Track which transitions vary between rounds (hidden param sensitivity)

**Estimated gain: compounding +0.5 to +2 per round**

---

## Implementation Priority

| # | Task | Impact | Effort | Do When |
|---|---|---|---|---|
| 1 | **Bayesian observation updates** | +3–7 | Small | NOW — 30min |
| 2 | **Spatial conditional model** | +8–15 | Medium | NOW — 2hr |
| 3 | **Adaptive query targeting** | +2–5 | Medium | After #1,#2 validated |
| 4 | **Settlement metadata model** | +1–3 | Small | After #2 |
| 5 | **Round-specific calibration** | +1–3 | Small | After #1 |
| 6 | **Cloud Run deploy** | ∞ | Medium | Before next round |
| 7 | **Multi-round accumulation** | +0.5–2/round | Small | After each round |

**Realistic target:** With #1 + #2 implemented, **82–89** on R1 backtest. With observations: **85–92+**.

---

## Quick Wins Checklist

- [ ] Fix `empirical_model()` to do Bayesian update instead of replacement
- [ ] Train sklearn model: features=(initial_class, neighbor_counts) → target=(ground_truth dist)
- [ ] Backtest spatial model against R1 ground truth
- [ ] Update `build_prediction()` pipeline to use spatial model as prior
- [ ] Implement two-phase query strategy in auto_round.py
- [ ] Deploy to Cloud Run
