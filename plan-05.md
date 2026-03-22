# Plan 05 — Final Push: Target the Real Bottlenecks

**Date**: 2026-03-21
**Current best**: Weighted ~170 (rank ~8)
**Leader**: ~177 ("Six Seven")
**Gap**: ~7 weighted points
**LORO avg (R1-R14, GBM+MLP)**: 88.36
**Training data**: 14 rounds, 70 seeds, 112K cells
**Competition ends**: March 22 ~15:00 CET (~24h remaining)

---

## Model Status

| Model | LORO Avg (R1-R14) | Notes |
|-------|-------------------|-------|
| GBM+MLP (V9) | **88.36** | Production. LGB+XGB 70/30 + KL-loss MLP, blended 50/50 |
| U-Net + TTA | 83.33 | 5 pts below GBM. Wins only R7 (+0.6). Not worth standalone use |

**Key insight**: Model capacity is NOT the bottleneck. The dataset is tiny (70 maps). Adding more architectures (transformers, GNNs, deeper CNNs) faces the same data scarcity wall. The U-Net experiment confirmed this — even with D4 augmentation (8× data), spatial correlations weren't enough to overcome the feature quality advantage of handcrafted GBM features.

---

## Where the Points Are

### Worst rounds drag the average

| Round | Score | Gap to avg | Impact |
|-------|-------|-----------|--------|
| R12 | 70.0 | -18.4 | Hardest round by far |
| R7 | 75.5 | -12.8 | Collapse-type round |
| R5 | 86.3 | -2.1 | Moderate |
| R14 | 87.2 | -1.2 | New round, less data |

Fixing R12 by even 5 pts and R7 by 3 pts = +0.57 LORO avg = significant at this stage.

### The real bottleneck: round feature estimation

Oracle test showed **+8.7 pts on R7** from perfect round features. Current static shrinkage has std=0.014 on S→S estimation. Even 2% better = +1 LORO pt. This is the single highest-leverage improvement available.

---

## Priority Stack (Execution Order)

### 1. Learned Debiasing — ~1 hour, expected +1-2 LORO pts

**What**: Replace static `SHRINKAGE_MATRIX` with a learned mapping: `(observed_transitions, settlement_stats) → GT_transitions`.

**Why**: Round feature estimation is the #1 error source. Oracle features give +8.7 pts on R7 alone. Static shrinkage applies the same correction regardless of round type — collapse rounds need different debiasing than boom rounds.

**How**:
- Input: 6×6 observed transition matrix (~20 non-trivial entries) + 4 settlement stats
- Output: 8 round features (E→E, S→S, F→F, E→S, sett_density, food, wealth, defense)
- Model: Ridge regression (heavy regularization, only 14 training samples)
- Validation: LOO on the 14 rounds, compare MSE vs static shrinkage
- Fallback: Keep static shrinkage if learned model is worse

**Acceptance**: LORO avg improves ≥0.3 pts. Learned MSE < static MSE on LOO.

---

### 2. Per-Entropy Calibration — ~1 hour, expected +0.3-0.8 LORO pts

**What**: Replace global `CALIBRATION_FACTORS` with isotonic regression per entropy bucket.

**Why**: High-entropy cells (settlement/forest boundaries) account for 88-99% of KL loss. These cells may need different calibration than low-entropy cells. Current global calibration doesn't distinguish.

**How**:
- After prediction, bucket cells by predicted entropy: [0, 0.2), [0.2, 0.5), [0.5, 1.0), [1.0, ∞)
- Fit isotonic regression per class per bucket using pooled LORO validation predictions
- Minimum 500 cells per bucket to avoid overfitting

**Acceptance**: LORO avg improves ≥0.2 pts. No round regresses >1 pt.

---

### 3. Synthetic Training Data — ~2 hours, expected +1-3 LORO pts (esp. R7, R12)

**What**: Generate 50-100 synthetic rounds by running the stochastic simulator with varied parameters. Add to training data.

**Why**: Only 14 real rounds, and extreme round types (R7 collapse, R12 ???) are underrepresented. The biggest historical gain came from adding similar-type rounds to training — R3 jumped +26 when R8 collapse data became available.

**How**:
- Sample SimParams from plausible ranges (calibrated from 14 real rounds)
- Generate random initial maps or reuse existing ones
- Run 200+ MC sims per synthetic round → GT tensor
- Downweight synthetic data 0.3-0.5× vs real data
- Retrain GBM+MLP with augmented dataset

**Acceptance**: LORO avg on REAL rounds improves ≥0.3 pts. R7 and R12 improve.

---

### 4. Query Strategy for Next Live Round — ~30 min

**What**: Spend 4 extra queries on center viewport repeats instead of 1.

**Why**: Averaging 4 observations reduces transition estimation noise by 2×. Better transitions → better round features → better predictions. Coverage drops from 100% to ~85% but spatial model handles unobserved cells well.

**Acceptance**: Simulated feature estimation error reduces ≥30%.

---

## What NOT to Do

- **Don't add more model architectures.** U-Net experiment proved model capacity isn't the bottleneck with 70 training maps. Transformers/GNNs would score worse.
- **Don't re-sweep temperatures/floors.** These are fine-tuning — do after structural improvements are locked.
- **Don't spend time on U-Net ensemble blending.** The blend sweep showed marginal gains at best. Time is better spent on feature quality.
- **Don't change the query grid for the current round.** Only for the next round after validation.

---

## Stop Gates

- After item 1 (learned debiasing): If LORO regressed, diagnose. If improved, lock and continue.
- After item 2 (calibration): Evaluate remaining time. If <8h to competition end, lock everything and focus on execution (query + submit for live rounds).
- After item 3 (synthetic data): Only if >12h remain. This is the riskiest bet.
