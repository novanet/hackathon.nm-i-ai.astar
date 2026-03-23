---
mode: 'agent'
description: 'End-of-day competition review. Run after the last round of the day closes.'
tools: ['read_file', 'replace_string_in_file', 'create_file']
---

Read the following files in full before proceeding:
- `KNOWLEDGE.md`
- `CURRENT_CONFIG.md` (create it from scratch if it does not exist — see step 4)
- `opening-strategy.md`

Then work through each section below in order. Update files as you go — do not just answer inline.

---

## 1. Today's rounds

List every round that closed today:

| Round | Raw score | Weight (1.05^n) | Weighted | One-sentence driver |
|---|---|---|---|---|

If a round scored more than 3 points below LORO expectation, mark it **UNDERPERFORM** and flag it for step 2.

---

## 2. Rule compliance check

For every model or parameter change deployed today, answer explicitly:

1. Was it LORO-validated before deployment? (yes / no / partial)
2. Did it touch overlay or floor parameters? If yes, was LORO-delta confirmed positive?
3. Was the round weight below 1.7 (experiment-safe) at deployment time?

**If any answer is "no" for items 1 or 2**: append a violation entry to KNOWLEDGE.md under the relevant round's Per-Round Notes section using this format:
```
- [VIOLATION, Round X] Changed <param> without LORO-validation. In-sample delta was +Y. Live result: Z.
```

This step is non-optional. If there were no changes today, write "No changes deployed today."

---

## 3. KNOWLEDGE.md update

Append all new confirmed findings to KNOWLEDGE.md under the correct section. Use this format:
```
- [Round X, Day Y] <one-line finding>
```

Minimum one entry per round that closed today. If nothing new was learned, write:
```
- [Round X, Day Y] No new rule confirmed. Score consistent with LORO expectation.
```

---

## 4. CURRENT_CONFIG.md sync

Update (or create) `CURRENT_CONFIG.md` so it reflects **exactly** the configuration submitted in the last round today.

Required fields:
```markdown
# CURRENT_CONFIG.md — last updated: [date, after Round X]

## Active model settings
UNET_BLEND_W = <value>      # LORO-validated: [citation]
MIN_PS       = <value>      # WARNING if < 5
MAX_PS       = <value>
PROB_FLOOR   = <value>

## Change locks (require LORO-revalidation before any change)
- UNET_BLEND_W  — reason: [cite sensitivity finding]
- MIN_PS/MAX_PS — reason: overlay catastrophe risk (see R16 warning in KNOWLEDGE.md)

## Open experiments (not yet LORO-validated)
- [None] or: <description, proposed change, current status>
```

If any open experiments exist, they must not be deployed to rounds with weight > 1.7 until promoted to "validated."

---

## 5. Strategy pulse check

Answer each question explicitly:

1. **Current best weighted score**: `raw × 1.05^n = ?` — which round?
2. **Remaining round weights**: list estimated weights for rounds not yet completed (use `1.05^n` where n = round number).
3. **Mode for tomorrow**: 
   - If tomorrow's round weight ≥ 2.0 → **CONSERVATIVE** (validated settings only, no experiments)
   - If tomorrow's round weight < 1.7 → **EXPERIMENT-OK** (LORO-validated changes allowed)
4. **One assumption to revisit**: Is there anything we believed at the start of today that the data has challenged?

---

## 6. Tomorrow's plan

Propose a maximum of **3 priorities** for tomorrow, in order of impact. Each must be tagged:

- `[VALIDATE]` — LORO-validate a specific pending change before it can be deployed
- `[IMPLEMENT]` — build a specific improvement; must state expected LORO gain
- `[EXPERIMENT]` — only allowed if tomorrow's round weight is < 1.7
- `[CONSERVATIVE]` — submit current validated settings, no changes (required if weight ≥ 2.0)
- `[BLOCKED]` — add this tag if a priority cannot start without completing a dependency

Format:
```
1. [TAG] Description — expected gain / dependency
2. [TAG] Description — expected gain / dependency
3. [TAG] Description — expected gain / dependency
```

Do not list more than 3. Forced prioritisation is the point.
