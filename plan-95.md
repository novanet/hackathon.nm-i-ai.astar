# Plan 95 — Breaking the 95 Raw Score Barrier

**Date**: 2026-03-22  
**Current peak raw**: 92.53 (R15)  
**Current LORO**: GBM+MLP=84.08, U-Net+TTA=83.75, Blend(90%)≈87–88  
**Target**: 95+ raw on a single round  
**Scoring math**: `score = 100 × exp(-3 × wKL)` → 95 requires `wKL ≤ 0.0171` (vs ~0.026 at 92.5 = **34% reduction**)

---

## Feasibility Verdict

**95 is hard but the leader is already there. We must close the gap.**

| Evidence | Score | What it means |
|----------|-------|---------------|
| Best live round (R15) | 92.53 | Current ceiling — **2.5 pts to close** |
| Best single seed (R15 s4) | 93.64 | Proves 93+ is reachable per-seed today |
| Best U-Net LORO (R9) | 93.27 | Best OOS with GT features — ceiling with perfect estimation |
| In-sample backtest (R8) | 95.7 | The dynamics allow 95+ — the model just needs to generalize |
| Leader's implied peak | ~94.8 | **Someone is already there.** We need to match them. |

The gap is 2.5 pts. Our in-sample backtests already hit 95.7. The U-Net LORO≈in-sample (underfitting). A bigger model closes this.

**The three things that move the ceiling:**
1. **Bigger U-Net + FiLM** — closes the generalization gap (largest single lever, +1-2 pts)
2. **Better training** — OneCycle + mixup + label smoothing squeeze more from the architecture (+0.5-1)
3. **Nail feature estimation** — repeat queries cut S→S noise from 0.014 → 0.007 (+0.3-0.7)

**Strategy**: Go all-in on U-Net architecture. The blend gate helps the floor, not the ceiling. On a favorable round with an upgraded U-Net, 95 is reachable.

---

## Diagnosis: Where the Points Are

### Score decomposition (R15, our best round at 92.53)

| Source | Share of KL loss | Implication |
|--------|-----------------|-------------|
| Empty cells | ~55-65% | Biggest class, moderate per-cell error |
| Forest cells | ~28-33% | Second biggest contributor |
| Settlement cells | ~1-5% | Small count, but highest per-cell KL |
| Port/Ruin/Mountain | <3% | Negligible |

The loss is dominated by Empty and Forest — the two most abundant high-entropy classes. Improving predictions on these by even 5-10% relative KL moves the needle more than perfecting rare classes.

### Model ceiling analysis

| Model | LORO Avg | Best single-round LORO | Notes |
|-------|----------|----------------------|-------|
| GBM+MLP | 84.08 | R4=90.33, R13=91.96 | Strong in-sample, weaker OOS |
| U-Net+TTA | 83.75 | R9=93.27, R13=92.70 | Better OOS generalization |
| Blend (80-90%) | ~87-88 | R15=92.53 (live) | Current production |
| **Oracle transitions** | ~21 | — | Transitions alone are useless |
| **In-sample backtest** | ~93+ | R8=95.7, R10=93.4 | Shows the ceiling per-round |

**Key insight**: Our in-sample backtests already hit 93-95 on favorable rounds. The gap from LORO (~84) to in-sample (~93) is **~9 pts of overfitting/generalization gap**. Closing even 2 pts of this gap with a better model yields 95 on a favorable round.

---

## The Three Paths to 95

### Path A: Better U-Net (highest ceiling, medium risk)
### Path B: Smarter blending (medium ceiling, low risk)  
### Path C: Better round-feature estimation (medium ceiling, low risk)

All three stack. Expected combined gain: +2-4 LORO pts → peak round 95-97.

---

## Path A: Bigger, Better U-Net

### A1. Scale Up the Architecture

**Current**: 2-level U-Net (40→20→10), channels 32→64→128, ~477K params  
**Proposed**: 3-level U-Net (40→20→10→5), channels 48→96→192→384, ~2.5M params

```
Encoder:   48(40×40) → 96(20×20) → 192(10×10) → 384(5×5)
Decoder:   384(5×5) → 192(10×10) → 96(20×20) → 48(40×40) → 6(40×40)
```

**Why this won't overfit**: 
- 17 rounds × 5 seeds × 8 D4 augmentations = 680 training images
- Current model (477K params) has LORO ≈ in-sample for U-Net (83.75 vs 83.33) — evidence of **underfitting**, not overfitting
- Dropout (0.1→0.15) + weight decay (1e-4→5e-4) provide regularization headroom
- Early stopping at patience=30 already prevents catastrophic overfitting

**Implementation**:
1. Add a third encoder/decoder level in `astar/unet.py`
2. Increase `base_channels` from 32 to 48
3. Increase dropout from 0.1 to 0.15
4. Increase weight decay from 1e-4 to 5e-4
5. Run LORO, compare per-round

**Acceptance**: LORO avg ≥ 84.5 (currently 83.75). No single round regresses >2 pts.

**Time**: ~2 hours (architecture change + full LORO training)

---

### A2. FiLM Conditioning for Round Features

**Current**: Round features are broadcast as 8 extra spatial channels (channels 11-18). The convolutions treat them like any other spatial feature, but they're spatially constant — wasteful channel capacity.

**Proposed**: Replace broadcast channels with **FiLM (Feature-wise Linear Modulation)**. Each encoder/decoder block gets an affine modulation from the round features:

```python
class FiLMBlock(nn.Module):
    def __init__(self, n_round_feats: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(n_round_feats, n_channels)
        self.beta = nn.Linear(n_round_feats, n_channels)
    
    def forward(self, x, round_feats):
        # x: (B, C, H, W), round_feats: (B, n_rf)
        gamma = self.gamma(round_feats).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(round_feats).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
```

Apply after each `DoubleConv` block. This lets the network learn "when S→S is low, strengthen the forest-reclamation filters" rather than hoping convolutions discover this from constant-channel inputs.

**Benefit**: Reduces input channels from 19 to 11 (removes 8 broadcast channels), frees capacity for spatial features, and gives the model a principled way to condition on round dynamics.

**Implementation**:
1. Add `FiLMBlock` class to `astar/unet.py`
2. Modify `UNet.__init__` to create FiLM layers for each conv block
3. Modify `forward()` to accept `round_feats` and apply modulation
4. Update `compute_unet_input()` to return 11 channels (remove broadcast)
5. Update training and inference to pass round features separately

**Acceptance**: LORO avg ≥ 84.0, or clear per-round wins on extreme rounds (R7, R12, R14).

**Time**: ~2 hours (architecture + plumbing changes + LORO)

---

### A3. Attention Gates on Skip Connections

**Current**: Skip connections concatenate encoder features with decoder features directly.

**Proposed**: Add lightweight channel attention (SE-style) at each skip connection:

```python
class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1)
        self.psi = nn.Conv2d(inter_ch, 1, 1)
    
    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = torch.sigmoid(self.psi(F.relu(g + s)))
        return skip * attn
```

This lets the decoder selectively attend to relevant encoder features — e.g., suppress mountain features when decoding settlement predictions near forest boundaries.

**Acceptance**: LORO avg ≥ 84.0 or wins on hard rounds.

**Time**: ~1 hour

---

### A4. Longer / Better Training

**Current**: 300 epochs, AdamW (lr=1e-3, wd=1e-4), cosine annealing, patience=30.

**Changes**:
- **Warmup**: 10-epoch linear warmup before cosine decay (prevents early gradient explosions with larger model)
- **OneCycleLR**: Replace cosine annealing with OneCycle (higher peak lr=3e-3, min 1e-5). OneCycle typically converges faster and to better minima.
- **Longer training**: 500 epochs with patience=50 (larger model needs more epochs)
- **Label smoothing**: Blend GT targets with uniform distribution at α=0.01 — prevents overconfident predictions, directly reduces KL on high-entropy cells
- **Mixup augmentation**: Blend pairs of training maps: `x_mix = λ*x_i + (1-λ)*x_j`, `y_mix = λ*y_i + (1-λ)*y_j` with λ~Beta(0.2, 0.2). Creates intermediate "virtual" rounds that interpolate between boom and collapse dynamics.

**Acceptance**: LORO avg ≥ 84.5.

**Time**: ~1 hour (changes are small, training takes longer)

---

## Path B: Smarter Blending

### B1. Per-Round Adaptive Blend Weight

**Current**: Fixed `UNET_BLEND_W = 0.90` for all rounds.

**Problem**: U-Net dominates on normal rounds (R9, R13, R15) but struggles on extreme rounds (R7 UNet=73.36 vs GBM=68.54, R12 UNet=51.75 vs GBM=58.34). On R12, 90% U-Net weight is catastrophic.

**Proposed**: Simple rule-based gate using observed round features:

```python
def adaptive_blend_weight(round_feats: np.ndarray) -> float:
    """Returns U-Net weight based on round characteristics."""
    ss = round_feats[1]  # S→S feature
    ee = round_feats[0]  # E→E feature
    
    # Extreme S→S (very high retention or very low) → reduce U-Net
    if ss > 0.45:    # Boom round (R12-like: S→S=0.59, R14: S→S=0.51)
        return 0.60
    elif ss < 0.10:  # Collapse round (R8: S→S=0.067, R3: S→S=0.018)
        return 0.70
    else:             # Normal round (R9, R13, R15: S→S=0.25-0.35)
        return 0.90
```

**Evidence from LORO** (which model wins per round):

| Round | S→S (GT) | U-Net LORO | GBM LORO | U-Net wins? |
|-------|----------|------------|----------|-------------|
| R3 | 0.018 | 88.29 | 87.60 | Yes (collapse) |
| R4 | 0.235 | 92.13 | 90.33 | Yes (normal) |
| R7 | 0.605 | 73.36 | 68.54 | Yes (boom) |
| R8 | 0.067 | 83.50 | 81.90 | Yes (collapse) |
| R9 | 0.275 | 93.27 | 90.97 | Yes (normal) |
| R10 | 0.058 | 91.35 | 91.19 | ~Tie (collapse) |
| R12 | 0.595 | 51.75 | 58.34 | **No (extreme boom)** |
| R13 | 0.260 | 92.70 | 91.96 | Yes (normal) |
| R14 | 0.510 | 76.78 | 80.21 | **No (boom)** |
| R15 | 0.348 | 92.04 | 91.33 | Yes (normal) |
| R16 | 0.270 | 86.31 | 83.01 | Yes (normal) |

**Pattern**: U-Net loses only when S→S > 0.45 (boom rounds: R12, R14). Simple threshold-based gating catches this cleanly.

**Implementation**: ~15 lines in `build_prediction()`.

**Expected gain**: +0.5-1.0 avg LORO, +5-10 on catastrophic rounds (R12, R14).

**CAVEAT — noise margin on estimated S→S**: At inference we use debiased estimated S→S, not GT. Estimation std≈0.014. R14's GT S→S=0.510 could easily be estimated as 0.44, flipping the gate. Mitigate with a **hysteresis band**: use thresholds 0.42 and 0.10 (not 0.45/0.10), or use a smooth sigmoid ramp instead of a hard threshold:

```python
def adaptive_blend_weight(round_feats: np.ndarray) -> float:
    ss = round_feats[1]
    # Smooth ramp: full UNet (0.90) when ss in [0.10, 0.40]
    # Reduced (0.60) when ss >> 0.45 or ss << 0.05
    high_gate = 1.0 / (1.0 + np.exp(20 * (ss - 0.42)))  # sigmoid drop at 0.42
    low_gate = 1.0 / (1.0 + np.exp(-20 * (ss - 0.08)))   # sigmoid drop at 0.08
    gate = high_gate * low_gate  # ~1 in normal, ~0 at extremes
    return 0.60 + 0.30 * gate    # ranges from 0.60 to 0.90
```

**Time**: 30 minutes.

---

### B2. Stacking / Meta-Learner (Stretch Goal)

Instead of a weighted average, train a tiny per-cell meta-model:

- **Input**: GBM prediction (6 classes) + U-Net prediction (6 classes) + round features (8) + cell entropy features (2) = 22 features
- **Output**: Final 6-class distribution
- **Model**: Logistic regression or 1-layer MLP with softmax
- **Training**: On LORO predictions (not in-sample!) to avoid leakage

This learns non-linear blending: "when U-Net says Settlement but GBM says Empty, trust GBM on boom rounds." Current linear blend can't do this.

**Risk**: Only 17 rounds of LORO data to train on. Must use nested CV.

**Expected gain**: +0.5-1.5 beyond adaptive gate.

**Time**: 2-3 hours.

---

## Path C: Better Round-Feature Estimation

### C1. Multi-Observation Aggregation

**Current**: 45 grid queries (9 viewports × 5 seeds) + 5 repeat queries on densest seed. Transition estimates from all observations are pooled.

**Problem**: S→S has std=0.014 from single observations. The model is hypersensitive to S→S (oracle +8.7 pts on R7).

**Proposed**: Restructure queries for better estimation:
- **7 unique viewports** (step=14, covers ~92% of 40×40) → 35 grid queries
- **3 repeat viewports** per seed on the center (high-info) region → 15 repeat queries
- **Total**: 50 queries, same budget

Each center cell gets ~4 independent observations. Transition estimate noise drops by √4 = 2×. S→S std → ~0.007.

The spatial model already handles unobserved corner cells well (edge_dist feature captures boundary effects).

**Expected gain**: +0.3-0.7 on rounds where feature estimation matters most (R7-type).

**Time**: 30 minutes.

---

### C2. Bayesian Transition Estimation

**Current**: Flat average of observed transitions across all viewport cells.

**Proposed**: Bayesian conjugate update with historical prior:

```python
prior_counts = HISTORICAL_TRANSITIONS * prior_strength  # e.g., strength=50
obs_counts = observed_transition_counts  # from viewport observations
posterior = (prior_counts + obs_counts) / (prior_strength + obs_counts.sum(axis=1, keepdims=True))
```

This shrinks noisy per-round observations toward the historical average, with adaptive strength based on observation count. Rounds with more observations get less shrinkage.

**Expected gain**: +0.2-0.5 (reduces feature estimation variance).

**Time**: 30 minutes.

---

## Execute Now (Ordered for Maximum Peak)

1. **Upgrade U-Net architecture** — 3 encoder levels (48→96→192→384), FiLM conditioning, attention gates. This is the highest-ceiling change and the only one that structurally raises the peak. Do this FIRST.
2. **Upgrade training regime** — OneCycle LR (peak 3e-3), 10-epoch warmup, mixup augmentation (λ~Beta(0.2,0.2)), label smoothing (α=0.01), 500 epochs patience=50. Squeezes more from the bigger architecture.
3. **Run full LORO + blend sweep** — Validates the new U-Net and finds the optimal blend weight. Also gives us the baseline delta.
4. **Restructure queries for tighter feature estimation** — 7 viewports + 3 center repeats per seed. Cuts S→S noise by 2×.
5. **Deploy + resubmit R15** — If LORO validates ≥2 pts gain, resubmit R15 immediately for guaranteed weighted-score improvement. Then deploy for the next live round.
6. **Adaptive blend gate** — Ship after U-Net is validated. Nice-to-have for floor, not the ceiling driver.

---

## Implementation Priority & Schedule

### Impact × Effort Matrix

| # | Task | Ceiling Impact | Effort | ROI | Dependency |
|---|------|---------------|--------|-----|------------|
| 1 | **U-Net: 3 levels + FiLM + attention** | **★★★★★** (+1.0-2.5) | 3h | **Highest** | None |
| 2 | **Training: OneCycle + mixup + label smooth** | **★★★** (+0.5-1.0) | 1h | **High** | After #1 |
| 3 | **LORO validation + blend sweep** | Validates #1+#2 | 1h | Mandatory | After #2 |
| 4 | **Query restructure (7 VP + 3 repeats)** | **★★** (+0.3-0.7) | 30m | High | None (parallel) |
| 5 | **Resubmit R15** | Guaranteed weighted pts | 15m | **Infinite** | After #3 validates ≥2pt |
| 6 | **Adaptive blend gate** | ★★ (+0.5 floor) | 30m | Medium | After #3 |
| 7 | **Bayesian transition estimation** | ★ (+0.2-0.5) | 30m | Medium | None |
| 8 | **Stacking meta-learner** | ★★★ (+0.5-1.5) | 2-3h | Low (risky) | After #3 |

### Timeline (starting now — March 22)

```
TODAY (Mar 22):
  [0h-3h]   #1  U-Net architecture overhaul (code changes + start training)
  [3h-4h]   #2  Training regime upgrades (apply to new architecture)
            #4  Query restructure (parallel — no dependency on #1)
  [4h-5h]   #3  Full LORO runs completing, blend sweep

  CHECKPOINT: Do we have ≥1 pt LORO gain? 
    YES → continue to deployment
    NO  → debug, try A1 without FiLM, bisect

  [5h-5.5h] #5  Resubmit R15 if ≥2 pt gain validated
  [5.5h-6h] #6  Adaptive blend gate

TOMORROW (Mar 23):
  Deploy upgraded stack for R19 (or whatever the next live round is)
  #7  Bayesian transition estimation (if time permits)
  #8  Stacking meta-learner (stretch — only if #1-#6 ship clean)
```

### Critical Path

```
#1 (U-Net) ──→ #2 (training) ──→ #3 (LORO) ──→ #5 (resubmit R15)
                                      │
                                      └──→ #6 (blend gate)
#4 (queries) ─── runs in parallel ───────────────────────────┘
```

Items #1+#2 are the **only changes that raise the ceiling**. Everything else is floor/reliability. Ship the U-Net first.

**Note: expected gains do NOT stack linearly.** The combined gain from all items is likely 60-70% of the sum of individual gains.

---

## Validation Protocol

Every change must be validated with **full LORO on R1-R17** before deployment.

```
Baseline (current):
  GBM+MLP LORO: 84.08
  U-Net+TTA LORO: 83.75
  Blend (90%) est LORO: ~87-88

After each change, report:
  1. Per-round LORO scores (all 17 rounds)
  2. LORO average
  3. Delta vs baseline per round
  4. Worst-round regression (must be < 3 pts)
  5. Best-round improvement
```

**Blend sweep after every U-Net change**: The optimal blend ratio may shift. Always sweep [50%, 60%, 70%, 80%, 90%, 100%] U-Net weight.

---

## Resubmission Strategy

We can resubmit any previous round. If the upgraded stack gains ≥2 raw pts on LORO, **resubmit R15** (our best round, ideal dynamics S→S=0.348) rather than only hoping the next live round is equally favorable.

- R15 scored 92.53 live. If the improved stack backtests ≥94.5 on R15 in LORO, resubmit.
- R15 weight = 2.0789. A 95 raw = 197.5 weighted. A 96 raw = 199.6 weighted.
- This is a **guaranteed improvement** if LORO validates, unlike waiting for a random favorable round.
- Also consider resubmitting R9 (weight 1.551, scored 90.59) or R13 (weight 1.886, scored 90.10) if the improved model backtests higher on those.

---

## Competition Timeline

Rounds advance approximately daily. Assuming the competition runs through ~R25:
- **~7-8 rounds remain** (R18-R25)
- Round weights continue 1.05^n: R20=2.65, R23=3.07, R25=3.39
- Each remaining round is a fresh chance. But **only ~50% of rounds have favorable dynamics** (S→S 0.25-0.35)
- Realistic shots at 95+: **3-4 favorable rounds** out of the remaining 7-8

**Implication**: Ship B1 + A1 before R19 if possible. The more rounds we have with the upgraded stack, the more likely we catch a favorable one.

---

## What NOT to Do

1. **No more overlay/temp/floor/calibration sweeps** — these are exhausted (basis-point territory)
2. **No D4 augmentation for GBM** — confirmed LORO regression (-0.27 avg, -20 on R12)
3. **No learned debiasing** — rejected twice (LOO MSE 100× worse than static shrinkage)
4. **No dynamic calibration** — model predictions are better calibrated than transition matrices
5. **No cell-level observation blending** — single stochastic runs poison the spatial model
6. **No mid-round experiments** — R16 overlay change cost 15.5 raw points. Deploy only validated configs.
7. **No changing blend weight mid-round without LORO validation** — the adaptive gate (B1) must be validated offline before deployment. Ad-hoc adjustments during a live round are how R16 happened.

---

## Target Scenarios

| Scenario | LORO Avg | Peak Raw (favorable round) |
|----------|----------|---------------------------|
| Current production | ~87-88 | 92.5 (R15) |
| **After A1+A2+A3+A4 (full U-Net overhaul)** | **~89-91** | **94-96** |
| + Resubmit R15 with upgraded stack | — | **94-96 on R15 (guaranteed)** |
| + blend gate + query improvements | ~90-91 | **95-97** |

**R15 was a 92.53 on S→S=0.348 — a near-ideal round.** With an improved U-Net, that same round backtests 94-96. Resubmitting R15 is the **fastest guaranteed path to a higher weighted score**.

**For a NEW round to break 95**: the upgraded U-Net just needs to land on a favorable round (S→S 0.25-0.35, ~50% of rounds). With ~7-8 rounds remaining, we get 3-4 shots. That's good odds if the model is ready.
