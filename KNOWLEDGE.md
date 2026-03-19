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

## Query Strategy Insights

- [Round 2] 9-viewport grid (step 13, overlapping) covers full 40x40 map — uses 9 queries per seed
- [Round 2] 45 queries for full coverage of all 5 seeds, leaving 5 spare for repeat observations
- [Round 2] Each seed gets 304 cells with 2 observations (overlap zones) — the rest get 1
- [Round 2] ~120-130 high-entropy cells per seed (entropy > 0.5) — these are settlement/forest boundary areas
- [Round 2] Extra queries best spent on repeat observations of settlement-dense seeds (1 and 3 had 56 settlements each)
- [Round 2] Viewport positions used: (0,0),(13,0),(26,0),(0,13),(13,13),(26,13),(0,26),(13,26),(26,26)

## Scoring & Prediction Insights

- [Round 2] Using probability floor 0.01 (default) — awaiting score to calibrate
- [Round 2] Cross-seed transition model should help significantly — 61.9% of settlements change, so initial-state prior alone is very wrong for settlement cells
- [Round 2] Final observed terrain distribution: ~60% empty, ~17% settlement, ~17% forest, ~2% ruin, ~1% port, ~2% mountain

## Per-Round Notes

### Round 1
- Missed entirely (completed before we got auth token)
- Round weight: 1.05

### Round 2 (Active — 2026-03-19)
- Round ID: 76909e29-f664-4b2f-b16b-61b7507277e9
- Round weight: 1.1025
- Status: 5/5 seeds submitted, 50/50 queries used
- Strategy: full grid coverage (9 viewports/seed) + 4 extra queries on seeds 1,3
- Model: initial prior → cross-seed transitions (60/40 blend) → empirical update → spatial smooth (σ=1.5) → neighbor inference → floor(0.01)
- Map: 40x40, 5 seeds, ~33-56 initial settlements per seed
- Score: (awaiting — round closes 23:47 UTC)
