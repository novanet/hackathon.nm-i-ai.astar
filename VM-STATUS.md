# GCP VM Status — `astar-sweep`

## VM Details
- **Name**: `astar-sweep`
- **Machine**: `c2-standard-8` (8 vCPUs, 32 GB RAM)
- **Zone**: `europe-north1-a`
- **Project**: `ai-nm26osl-1710`
- **SSH user**: `devstar17101` (NEVER use `hbrot`)
- **Python**: 3.11.2 in `/home/devstar17101/venv`
- **Code**: `/home/devstar17101/app/`

## Quick Status Check

```bash
gcloud compute ssh devstar17101@astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710 --command="tail -30 ~/app/big_unet_output.log"
```

## Fetching Results

```bash
python fetch_vm_results.py
```

This checks if the comparison is still running and prints the full log. When finished, it also copies the log to `comparison_vm_results.txt` locally.

## Manual SSH

```bash
gcloud compute ssh devstar17101@astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710
```

## Current Job: U-Net Architecture Comparison (`run_comparison.py`)

**Started**: 2026-03-22 ~02:00 UTC  
**Script**: `run_comparison.py` (LORO on 7 representative rounds × 4 configs × 150 epochs)

### Configs Being Tested

| Config | Params | Description |
|--------|--------|-------------|
| V1 | 471,942 | Baseline 2-level, 32ch (current production) |
| V1_mix | 471,942 | V1 + mixup + label smoothing + OneCycleLR |
| V2_48_2l | ~1.05M | 2-level, 48ch (wider) |
| V2_3l_32 | ~1.88M | 3-level, 32ch (deeper) |

### Test Rounds
R1, R5, R9, R12, R15, R17, R18 (7 rounds, LORO leave-one-out)

### Results So Far

| Config | R1 | R5 | R9 | R12 | R15 | R17 | R18 | AVG |
|--------|-----|-----|-----|------|------|------|------|------|
| V1 | 85.02 | | 92.99 | | | | | — |
| V1_mix | | | | | | | | — |
| V2_48_2l | | | | | | | | — |
| V2_3l_32 | | | | | | | | — |

*Update this table by running `python fetch_vm_results.py`*

---

## Queued Job: Big U-Net LORO (`train_big_unet.py`)

**Queued**: 2026-03-22 ~02:22 UTC (starts after comparison finishes)  
**Script**: `train_big_unet.py` — full LORO on ALL rounds × 4 big configs × 300 epochs  
**Log**: `big_unet_output.log`

### Configs Being Tested

| Config | Params | Channels | Levels | FiLM | Attention | Mixup+LS+OneCycle |
|--------|--------|----------|--------|------|-----------|-------------------|
| V2_big | ~4.3M | 48 | 3 | No | No | No |
| V2_big_mix | ~4.3M | 48 | 3 | No | No | Yes |
| V2_film | ~4.3M | 48 | 3 | Yes | Yes | Yes |
| V2_huge | ~7.6M | 64 | 3 | Yes | Yes | Yes |

### Check progress

```bash
gcloud compute ssh devstar17101@astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710 --command="tail -30 ~/app/big_unet_output.log"
```

---

## VM Lifecycle

```bash
# Stop VM (to save costs when not in use)
gcloud compute instances stop astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710

# Start VM again
gcloud compute instances start astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710

# Delete VM (when done with all sweeps)
gcloud compute instances delete astar-sweep --zone=europe-north1-a --project=ai-nm26osl-1710
```
