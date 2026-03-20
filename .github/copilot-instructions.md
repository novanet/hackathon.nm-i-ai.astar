## Project Context

This is a hackathon project for NM i AI 2026 — Challenge 2: Astar Island. We observe a black-box Norse civilisation simulator through limited viewports and predict the final world state (W×H×6 probability tensor). Scoring is entropy-weighted KL divergence.

Key files:
- `astar/client.py` — API client for querying the simulator
- `astar/model.py` — Prediction model
- `astar/submit.py` — Submission logic
- `astar/replay.py` — Replay/analysis of observations
- `astar/viz.py` — Visualization
- `opening-strategy.md` — Prioritized strategy guide
- `KNOWLEDGE.md` — **Critical learnings accumulated during the competition**

## Knowledge File Protocol

`KNOWLEDGE.md` is the persistent knowledge base for this project. **Always read it at the start of a task** to build on previous learnings.

**When to append to KNOWLEDGE.md:**
- A simulation rule is confirmed through observation (e.g., "settlements never expand more than 2 cells per year")
- A hidden parameter estimate is calibrated from data (e.g., "round 3: winter severity ~0.7")
- A terrain transition probability is measured empirically
- A query strategy is tested and its effectiveness assessed
- A scoring insight is discovered (e.g., "floor of 0.02 outperformed 0.01")
- A round is completed — record parameter estimates and final score

**How to append:**
- Place the finding under the appropriate section heading in `KNOWLEDGE.md`
- Use a bullet point with the date/round context: `- [Round 3] Settlement expansion radius appears to be 2 cells`
- If a previous finding is contradicted, strike it through (~~old~~) and add the corrected version
- Keep entries concise — one line per finding when possible

**Sections:**
| Section | What goes here |
|---------|---------------|
| Simulation Rules | Confirmed mechanics: expansion, raids, trade, winter, forest reclamation |
| Hidden Parameter Estimates | Per-round calibrated values for hidden params |
| Terrain Transition Patterns | Observed class transition probabilities |
| Query Strategy Insights | Viewport placement, coverage trade-offs |
| Scoring & Prediction Insights | Probability floors, calibration, high-entropy handling |
| Per-Round Notes | Round-specific observations and final scores |

## Deployment

We do **not** expose a `/solve` endpoint — the organizers confirmed this is not required. Instead, we query the simulator and submit predictions directly via the API client. We use **Google Cloud Run** only as a compute environment for running our solver scripts.

- **Skill**: Use the `cloud-run-deploy` skill (`.github/skills/cloud-run-deploy/`) for setup and troubleshooting
- **Region**: Always `europe-north1` (closest to API servers)
- **GCP docs**: https://app.ainm.no/docs/google-cloud/setup

## Coding Conventions

- Python 3.11+, use type hints for function signatures
- numpy for tensor operations, avoid pandas unless needed for analysis
- Always enforce a probability floor (minimum 0.01) before submitting predictions
- Auth token comes from `ASTAR_TOKEN` environment variable — never hardcode it
