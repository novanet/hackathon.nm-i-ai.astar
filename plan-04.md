# Plan 04 — Close the Gap: Structural Improvements

**Date**: 2026-03-21
**Current best**: R17 = 84.74 × 2.292 = **194.22** (rank 96/283 on the round)
**Leader**: **217.38** ("Laurbaerene")
**Gap**: **23.15 weighted points**
**Latest blend evidence**: full LORO blend sweep says 80%=92.06, 90%=92.37, 100%=92.52, but 100% is biased upward because U-Net was not retrained per fold
**Current operating range**: U-Net blend **0.80-0.90**; 0.90 is the most credible exploit move, 1.00 is not yet trustworthy
**Competition state**: R18 is active

## Round Weight Math

| Round | Weight | Raw needed for 217.38 | Raw needed for 220 |
|-------|--------|-----------------------|--------------------|
| R18   | 2.407  | 90.3                  | 91.4               |
| R19   | 2.527  | 86.0                  | 87.1               |
| R20   | 2.653  | 81.9                  | 82.9               |

R17 proved the current stack can still produce a new personal best, but not a winning one. The target is no longer "be solid"; it is "find one round that lands 90+ raw, or survive to a later round where mid-80s can win." We need to **preserve the 90+ ceiling** and stop spending time on tiny post-processing deltas.

---

## Post-R17 Directive

1. **Treat R18 as an execution round, not a research round.** The current stack is competitive enough to score in the mid/high 80s. Large architecture churn now is more likely to miss than to win.
2. **If making one exploit change, use 0.90 U-Net blend.** That is the strongest move supported by evidence. Do not jump to 1.00 unless a cleaner validation removes the fold-leakage caveat.
3. **Do not spend more time on temperatures, floors, or overlay sweeps.** Those knobs are effectively exhausted; the upside is now basis points.
4. **Use queries to reduce hidden-parameter uncertainty, not to chase full coverage perfection.** The remaining edge is better round identification, especially on settlement-retention and collapse risk.
5. **The only remaining structural bet worth making mid-competition is a learned gate between GBM+MLP and U-Net.** The problem is no longer whether U-Net helps; it is when pure U-Net overshoots.

---

## Baseline for Regression Testing

**Mandatory before merging ANY change**: Run full LORO CV on R1-R12 and compare per-round.

Current baseline (V8 + sim blend, R1-R11 training — retrain with R12 first):

| Round | LORO (spatial) | LORO (+sim) |
|-------|---------------|-------------|
| R1    | 85.0          | —           |
| R2    | 89.5          | —           |
| R3    | 86.9          | —           |
| R4    | 91.0          | —           |
| R5    | 81.1          | —           |
| R6    | 82.7          | —           |
| R7    | 66.3          | —           |
| R8    | 86.7          | —           |
| R9    | 90.8          | —           |
| R10   | 90.4          | —           |
| R11   | 80.7          | —           |
| **Avg** | **84.0**    | **84.6**    |

**Step 0 (before everything)**: Retrain on R1-R12, get updated baseline. Every subsequent item compares against this.

**Logging requirement**: Every experiment must print and save:
1. Per-round LORO scores (all 12+ rounds)
2. LORO average
3. Delta vs baseline per round
4. In-sample scores for sanity check
5. Wall-clock time

Use `backtest_pipeline.py` or equivalent — results to stdout AND a summary JSON.

---

## Improvements (Priority Order)

### 1. U-Net / CNN for Spatial Correlations

**What**: Replace or blend a convolutional model that takes the full 40×40 initial map as input and outputs 40×40×6 probabilities. Round features broadcast as extra channels.

**Why**: Current GBM+MLP treats each cell independently. GT has strong spatial correlations — settlements cluster, forest reclamation propagates. A CNN naturally models these output-space dependencies.

**Architecture**:
- Input: 40×40 × (6 one-hot + 5 distance features + 8 round features) = 40×40×19 channels
- Model: 3-layer U-Net encoder-decoder. Encoder: Conv(19→32, 3×3) → Conv(32→64, 3×3, stride 2) → Conv(64→128, 3×3, stride 2). Decoder mirrors with skip connections. Final: Conv(→6) + softmax.
- Loss: Entropy-weighted KL (same as MLP)
- Augmentation: 4 rotations × 2 flips = 8× data (terrain physics is rotationally symmetric)
- Training data: 60 seeds × 8 augmentations = 480 images
- Blend: ensemble with GBM+MLP, sweep blend ratio in LORO

**Impact**: +2–4 LORO points. This is the biggest untapped structural improvement. Cell-independent models fundamentally cannot capture settlement clustering and forest propagation patterns in the output.

**Build time**: ~2–3 hours (model definition, training loop, LORO integration, blend sweep)

**Regression danger**: LOW if blended. The U-Net is added as a third model in the ensemble — if it's bad, its weight goes to 0 in the blend sweep. Pure U-Net replacement would be HIGH risk.

**Acceptance criteria**:
- LORO avg improves by ≥0.5 pts when blended
- No single round regresses by >2 pts vs baseline
- If any round regresses >3 pts, cap U-Net weight for that round type

---

### 2. Better Round-Feature Estimation (Learned Debiasing)

**What**: Replace the static `SHRINKAGE_MATRIX` with a learned mapping: `(observed_transitions, settlement_stats) → GT_transitions`. Train on the 9+ observed rounds where we have both obs and GT.

**Why**: The current shrinkage matrix is a fixed ratio (GT/obs). But the debiasing needed varies by round type — collapse rounds need different correction than boom rounds. A learned model can capture this. Oracle test showed +8.7 pts on R7 from perfect features; current estimation has std=0.014 on S→S.

**Architecture**:
- Input: 6×6 observed transition matrix (flattened non-trivial entries, ~20 features) + 4 settlement stats
- Output: 8 round features (E→E, S→S, F→F, E→S, sett_density, food, wealth, defense)
- Model: Ridge regression or 2-layer MLP (tiny — only 9-12 training rounds)
- Regularization: Heavy (ridge alpha=10+, or dropout=0.3) due to tiny n
- Fallback: If learned model is worse than static shrinkage in LORO, keep static

**Impact**: +1–2 LORO points. The round-feature sensitivity analysis showed the model is hypersensitive to S→S — even 2% better estimation = +1 LORO point.

**Build time**: ~1 hour (data prep, train tiny model, integrate into `compute_round_features`, LORO comparison)

**Regression danger**: MEDIUM. If the learned model overfits to the 9 training rounds (LOO on 9 samples is noisy), it could produce worse features on unseen round types. Mitigate by comparing learned vs static debiasing in LORO — use whichever wins.

**Acceptance criteria**:
- LORO avg improves by ≥0.3 pts
- Learned feature MSE vs GT < static shrinkage MSE vs GT (measured in LOO on observed rounds)

---

### 3. Rotation/Flip Data Augmentation for All Models

**What**: Apply 4 rotations (0°, 90°, 180°, 270°) and 2 flips (none, horizontal) to training maps before feature extraction. 8× more training data for GBM, MLP, and U-Net.

**Why**: Terrain physics is rotationally symmetric (north/south/east/west are equivalent). Current models see each map only once. GBM benefits from more samples for rare transitions. MLP and CNN benefit from seeing patterns in all orientations.

**Implementation**:
- Rotate/flip the initial grid AND the GT tensor simultaneously
- Recompute `compute_cell_features()` on each augmented grid (distance features change with rotation!)
- Round features stay the same (rotation doesn't change round-level stats)
- For U-Net: augmentation is trivial (rotate both input and target image)
- For GBM/MLP: augmentation happens at feature level — 8× more rows in training matrix

**Impact**: +0.5–1.5 LORO points for GBM/MLP, more for CNN. Most impact on rounds with unusual spatial patterns (R3, R7, R11) where training data diversity matters.

**Build time**: ~45 min (augmentation utility, integrate into `build_training_data_multi`, LORO comparison)

**Regression danger**: LOW. More data with correct augmentation should only help or be neutral. Edge distance feature (`edge_dist`) needs to be recomputed correctly for rotated maps — verify this.

**Acceptance criteria**:
- LORO avg improves by ≥0.3 pts OR no regression while having more robust per-round scores (lower variance across rounds)

---

### 4. Query Strategy: More Repeats for Better Feature Estimation

**What**: Change from 9 unique viewports + 1 extra per seed to 6 unique viewports + 4 repeats of center viewport per seed. Total still 50 queries.

**Why**: Each viewport observation is one stochastic realization. Transition estimates from single observations have std=0.014 for S→S. Averaging 4 center viewport repeats reduces noise by √4 = 2×. Better transition estimates → better round features → better predictions. Coverage drops from 100% to ~85% but the spatial model already handles unobserved cells well.

**Trade-off analysis**:
- Lost: corners of the map not directly observed (~15% of cells)
- Gained: 4× averaged transitions on the center 15×15 area (225 cells × 4 observations each)
- The center viewport is the most representative (avoids edge effects)

**Impact**: +0.5–1.0 LORO points (indirectly, through better round features). Hard to backtest directly since LORO uses GT features.

**Build time**: ~30 min (modify `run_next_round.py` query logic)

**Regression danger**: LOW-MEDIUM. Lost coverage in corners could hurt if there are unusual patterns there (edge settlements, ports). `edge_dist` feature already captures edge behavior. Mitigate by checking if corner cells have higher error in historical backtests.

**Acceptance criteria**:
- Simulated feature estimation error (using subsampled observations from GT) reduces by ≥30%
- No round shows >1 pt regression in simulated coverage comparison

---

### 5. Per-Entropy-Bucket Calibration (Isotonic Regression)

**What**: Instead of global `CALIBRATION_FACTORS`, group cells by predicted entropy into 4-5 buckets and fit isotonic regression per class within each bucket.

**Why**: High-entropy cells (settlement/forest boundaries) dominate the score — they account for 88-99% of KL loss. These cells may need different calibration than low-entropy cells (pure empty/mountain). Current global calibration applies the same correction to all cells regardless of uncertainty.

**Implementation**:
- After spatial model prediction, compute predicted entropy per cell
- Bucket: [0, 0.2), [0.2, 0.5), [0.5, 1.0), [1.0, ∞)
- For each bucket: fit isotonic regression mapping predicted probability → calibrated probability (per class, using LORO validation predictions vs GT)
- Apply at inference time

**Impact**: +0.3–0.8 LORO points. Calibration currently adds +0.05-0.14 — better calibration targeting high-entropy cells could multiply this.

**Build time**: ~1 hour (compute predicted entropies in LORO, fit isotonic regressors, integrate into pipeline, sweep bucket boundaries)

**Regression danger**: MEDIUM. Isotonic regression with small validation sets can overfit, especially in extreme entropy buckets with few samples. Mitigate by using pooled LORO (all rounds) and minimum bucket size of 500 cells.

**Acceptance criteria**:
- LORO avg improves by ≥0.2 pts
- No round regresses by >1 pt

---

### 6. Synthetic Training Data from Simulator

**What**: Generate 50-100 synthetic "rounds" by running the stochastic simulator with varied parameters, producing synthetic GT probability tensors. Add to training data alongside real rounds.

**Why**: Only 12 real rounds exist, limiting round-feature diversity. The model struggles with novel round types. Synthetic data lets us explore the full parameter space: extreme collapses (S→S=0.01), extreme booms (E→S=0.30), moderate rounds, etc.

**Implementation**:
- Sample random SimParams from plausible ranges (calibrated from the 12 real rounds)
- For each: generate a random initial map (or use existing ones), run 200+ MC sims → GT tensor
- Compute GT round features from synthetic GT
- Add to training set with optional downweighting (e.g., 0.5× weight vs real data)
- Retrain everything

**Impact**: +1–3 LORO points, mainly on rare round types (R3, R7 — historically worst). The biggest single gain came from adding similar rounds to training: R3 jumped +26 when R8 collapse data was added.

**Build time**: ~2 hours (synthetic data generation pipeline, parameter sampling, integration into training, LORO with/without synthetic data)

**Regression danger**: MEDIUM. If synthetic data distribution doesn't match real data well, it could bias the model. Mitigate by:
- Downweight synthetic data (0.3-0.5×)
- Validate that LORO on real rounds doesn't regress
- Ablation: compare with/without synthetic data

**Acceptance criteria**:
- LORO avg on REAL rounds improves by ≥0.3 pts (synthetic data not included in test set)
- Worst-round LORO (R7) improves

---

### 7. Deeper/Wider MLP Architecture

**What**: Upgrade MLP from 128→64 to 256→128→64 with residual connections.

**Why**: Current MLP is very small. The 31 features have complex nonlinear interactions (round features × spatial features). A deeper network can model these better, especially the interaction between "how hot is this round" and "how close is the nearest settlement."

**Impact**: +0.3–0.7 LORO points.

**Build time**: ~45 min (update `KLDivMLP`, retrain, LORO comparison, sweep width/depth)

**Regression danger**: LOW. If deeper MLP overfits, the blend weight will be lower. Early stopping + dropout protect against catastrophic overfitting.

**Acceptance criteria**:
- MLP val_kl decreases by ≥5%
- LORO avg improves by ≥0.2 pts when blended

---

## Execution Order (Time-Boxed)

Phase A — **Foundation** (1 hour):
1. Retrain on R1-R12, establish baseline LORO numbers
2. Item 3 (rotation augmentation) — low risk, benefits everything downstream
3. Item 2 (learned debiasing) — quick to test

Phase B — **Big Bet** (2.5 hours):
4. Item 1 (U-Net) — the structural improvement. Build, train, blend, LORO.
5. Item 7 (deeper MLP) — quick to try alongside U-Net work

Phase C — **Polish** (1.5 hours):
6. Item 5 (per-entropy calibration)
7. Item 6 (synthetic data) — only if time remaining
8. Item 4 (query strategy) — apply to next live round

**Stop gate**: After Phase A, if LORO has regressed, diagnose before moving to Phase B. After Phase B, evaluate whether Phase C is worth it or if we should lock and focus on execution.

---

## What NOT to Do

- **Don't re-sweep temperatures/floors mid-plan**. These are fine-tuning that should happen AFTER structural improvements are locked in.
- **Don't change the query grid for the current round**. Only change query strategy for the NEXT round after validation.
- **Don't replace models — ensemble them**. Every new model is additive. If it's bad, its blend weight goes to 0.
- **Don't spend >30 min debugging an approach that isn't working**. Move on to the next item.
- **Don't forget to save predictions on every submit**. We need the artifacts for post-hoc analysis.

---

## Summary Table

| # | Improvement | Impact (LORO pts) | Build Time | Regression Risk | Depends On |
|---|------------|-------------------|------------|-----------------|------------|
| 0 | R12 retrain + baseline | — | 15 min | — | — |
| 1 | U-Net / CNN | +2–4 | 2–3 hr | Low (blended) | 0 |
| 2 | Learned debiasing | +1–2 | 1 hr | Medium | 0 |
| 3 | Rotation augmentation | +0.5–1.5 | 45 min | Low | 0 |
| 4 | Query strategy | +0.5–1.0 | 30 min | Low-Med | 2 |
| 5 | Isotonic calibration | +0.3–0.8 | 1 hr | Medium | 0 |
| 6 | Synthetic training data | +1–3 | 2 hr | Medium | 0, 3 |
| 7 | Deeper MLP | +0.3–0.7 | 45 min | Low | 0, 3 |
| **Total** | | **+5.6–13** | **~8 hr** | | |

Realistic expectation from here is not another broad +4-8 LORO leap. The winning path is narrower: one exploit move (likely 0.90 U-Net), disciplined live execution, and one sharper round-identification improvement that converts an 85-87 round into a 90+ round.
