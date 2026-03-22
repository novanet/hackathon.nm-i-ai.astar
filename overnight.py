"""
Overnight orchestrator — trimmed pipeline.

Priority order:
  1. Big U-Net LORO (4 flagship configs, ~3.5h) — find best architecture
  2. Download R19 GT (R19 long closed by now)
  3. Retrain production models on R1-R19 (~40min): GBM+MLP, V1, V2_big_mix
  --- bonus (if VM stays alive) ---
  4. Quick blend weight sweep with LORO winner (~1.5h)
  5. Re-run Big U-Net LORO on R1-R19 (~3.5h)

Usage on VM:
    nohup python -u overnight.py > overnight_output.log 2>&1 &
"""

import subprocess
import sys
import time
import json
import gc
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, ".")

DATA_DIR = Path("data")

R19_ID = "597e60cf-d1a1-4627-ac4d-2a61da68b6df"


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_all_rounds() -> dict:
    """Load all available round data. Returns {rnum: (rid, detail, gts)}."""
    from train_unet import ROUND_IDS, load_round_data
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds", flush=True)
    # Also load R19 if not in ROUND_IDS
    if 19 not in all_data:
        detail, gts = load_round_data(R19_ID)
        if gts:
            all_data[19] = (R19_ID, detail, gts)
            print(f"  R19: {len(gts)} seeds (manually added)", flush=True)
    return all_data


def download_r19_gt() -> bool:
    """Download R19 ground truth. Returns True if successful."""
    from astar.client import _request, get_round_detail

    rdir = DATA_DIR / f"round_{R19_ID}"
    rdir.mkdir(parents=True, exist_ok=True)

    existing = list(rdir.glob("ground_truth_s*.json"))
    if len(existing) >= 5:
        print(f"R19 GT already downloaded ({len(existing)} files)", flush=True)
        return True

    print("Downloading R19 ground truth...", flush=True)
    detail = get_round_detail(R19_ID)
    n_seeds = len(detail.get("initial_states", []))

    detail_path = rdir / f"round_detail_{R19_ID[:8]}.json"
    if not detail_path.exists():
        detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    ok = 0
    for seed_idx in range(n_seeds):
        gt_path = rdir / f"ground_truth_s{seed_idx}.json"
        if gt_path.exists():
            ok += 1
            continue
        try:
            analysis = _request("GET", f"/analysis/{R19_ID}/{seed_idx}")
            gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            ok += 1
            print(f"  Seed {seed_idx}: OK", flush=True)
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}", flush=True)

    print(f"R19 GT: {ok}/{n_seeds} seeds downloaded", flush=True)
    return ok == n_seeds


# ── Step 1: Big U-Net LORO ────────────────────────────────────────────────

def run_big_unet_loro():
    """Run 4-config flagship LORO via train_big_unet.py."""
    print("\n" + "=" * 70, flush=True)
    print("STEP 1: Big U-Net LORO (4 flagship configs)", flush=True)
    print("=" * 70, flush=True)

    result = subprocess.run(
        [sys.executable, "-u", "train_big_unet.py"],
        capture_output=False, text=True, timeout=86400,
    )
    if result.returncode == 0:
        print("Big U-Net LORO completed!", flush=True)
    else:
        print(f"Big U-Net LORO failed with code {result.returncode}", flush=True)


# ── Step 2: Download R19 GT ──────────────────────────────────────────────

def download_r19_gt_with_token(saved_token: str | None) -> bool:
    """Restore token briefly, download R19 GT, remove token again."""
    print("\n" + "=" * 70, flush=True)
    print("STEP 2: Downloading R19 ground truth", flush=True)
    print("=" * 70, flush=True)

    if saved_token:
        os.environ["ASTAR_TOKEN"] = saved_token
    ok = download_r19_gt()
    if "ASTAR_TOKEN" in os.environ:
        del os.environ["ASTAR_TOKEN"]
    print("ASTAR_TOKEN removed after R19 download", flush=True)
    return ok


# ── Step 3: Retrain production models on R1-R19 ──────────────────────────

def retrain_gbm_mlp():
    """Retrain GBM and MLP via train_spatial.py."""
    print("\n  Retraining GBM + MLP...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "train_spatial.py"],
            capture_output=False, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print("  GBM + MLP retrained successfully!", flush=True)
        else:
            print(f"  GBM + MLP training failed with code {result.returncode}", flush=True)
    except Exception as e:
        print(f"  GBM + MLP training error: {e}", flush=True)


def retrain_production_unet():
    """Retrain production V1 U-Net on all available rounds."""
    print("\n  Retraining production U-Net (V1)...", flush=True)

    import torch
    from astar.unet import UNet, N_INPUT_CHANNELS, NUM_CLASSES
    from train_unet import build_unet_dataset, train_unet_fold

    device = torch.device("cpu")
    all_data = _load_all_rounds()
    all_rounds = sorted(all_data.keys())
    print(f"  Training V1 on {len(all_data)} rounds: {all_rounds}", flush=True)

    imgs, tgts = build_unet_dataset(all_rounds, all_data, augment=True, film_mode=False)

    unet_config = dict(
        in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
        base_channels=32, dropout=0.1, n_levels=2,
        use_film=False, use_attention=False,
    )

    model = train_unet_fold(
        imgs, tgts, device,
        n_epochs=200, lr=1e-3, patience=30,
        verbose=True, unet_config=unet_config,
    )

    save_path = DATA_DIR / "unet_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Production V1 U-Net saved to {save_path}", flush=True)

    del model, imgs, tgts
    gc.collect()


def train_v2_big_mix_production():
    """Train V2_big_mix on ALL data as production model."""
    print("\n  Training V2_big_mix production model...", flush=True)

    import torch
    from astar.unet import UNet, N_INPUT_CHANNELS, NUM_CLASSES
    from train_unet import build_unet_dataset, train_unet_fold

    device = torch.device("cpu")
    all_data = _load_all_rounds()
    all_rounds = sorted(all_data.keys())
    print(f"  Training V2_big_mix on {len(all_data)} rounds", flush=True)

    imgs, tgts = build_unet_dataset(all_rounds, all_data, augment=True, film_mode=False)

    unet_config = dict(
        in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
        base_channels=48, dropout=0.15, n_levels=3,
        use_film=False, use_attention=False,
    )

    model = train_unet_fold(
        imgs, tgts, device,
        n_epochs=300, lr=1e-3, patience=40, verbose=True,
        unet_config=unet_config,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    )

    save_path = DATA_DIR / "unet_v2_big_mix.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  V2_big_mix production model saved to {save_path}", flush=True)

    del model, imgs, tgts
    gc.collect()


def retrain_all_production():
    """Step 3: Retrain all production models on R1-R19."""
    print("\n" + "=" * 70, flush=True)
    print("STEP 3: Retraining production models on R1-R19", flush=True)
    print("=" * 70, flush=True)

    retrain_gbm_mlp()
    retrain_production_unet()
    train_v2_big_mix_production()


# ── Step 4 (bonus): Quick blend weight sweep ──────────────────────────────

def run_blend_sweep():
    """Test LORO winner (V2_big_mix) at 4 blend weights × 7 rounds."""
    print("\n" + "=" * 70, flush=True)
    print("STEP 4 (BONUS): Quick blend weight sweep", flush=True)
    print("=" * 70, flush=True)

    import torch
    from astar.unet import UNet, N_INPUT_CHANNELS, NUM_CLASSES
    from astar.submit import score_prediction
    import astar.model as m
    from train_unet import (load_round_data, build_unet_dataset,
                            train_unet_fold, compute_gt_round_features,
                            score_unet_on_round)

    device = torch.device("cpu")
    all_data = _load_all_rounds()
    all_rounds = sorted(all_data.keys())

    TEST_ROUNDS = [r for r in [1, 5, 9, 12, 15, 17, 18] if r in all_data]
    BLEND_WEIGHTS = [0.80, 0.85, 0.90, 0.95]

    unet_config = dict(
        in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
        base_channels=48, dropout=0.15, n_levels=3,
        use_film=False, use_attention=False,
    )

    results = {}  # bw -> {rnum: score}

    for test_rnum in TEST_ROUNDS:
        train_rounds = [r for r in all_rounds if r != test_rnum]

        # Train fold U-Net (once per round, test all blend weights)
        imgs, tgts = build_unet_dataset(train_rounds, all_data, augment=True, film_mode=False)
        model = train_unet_fold(
            imgs, tgts, device, n_epochs=300, lr=1e-3,
            patience=40, verbose=False, unet_config=unet_config,
            use_mixup=True, label_smoothing=0.01, use_onecycle=True,
        )

        # Save fold model, swap into production slot
        fold_path = DATA_DIR / "_blend_fold_unet.pt"
        torch.save(model.state_dict(), fold_path)
        del model, imgs, tgts
        gc.collect()

        rid, detail, gts = all_data[test_rnum]
        n_seeds = len(gts)

        # Back up existing production model
        prod_path = DATA_DIR / "unet_model.pt"
        backup_path = DATA_DIR / "_unet_backup.pt"
        if prod_path.exists():
            prod_path.rename(backup_path)
        fold_path.rename(prod_path)

        for bw in BLEND_WEIGHTS:
            if bw not in results:
                results[bw] = {}

            old_blend = m.UNET_BLEND_WEIGHT
            m.UNET_BLEND_WEIGHT = bw

            scores = []
            for s in range(n_seeds):
                try:
                    pred = m.build_prediction(rid, detail, s)
                    gt = np.array(gts[s]["ground_truth"], dtype=np.float64)
                    scores.append(score_prediction(pred, gt))
                except Exception as e:
                    print(f"  ERROR bw={bw} R{test_rnum} s{s}: {e}", flush=True)

            m.UNET_BLEND_WEIGHT = old_blend

            if scores:
                avg = float(np.mean(scores))
                results[bw][test_rnum] = avg

        # Restore
        prod_path.rename(fold_path)
        fold_path.unlink(missing_ok=True)
        if backup_path.exists():
            backup_path.rename(prod_path)

        gc.collect()

    # Print results
    print(f"\n{'bw':>6}", end="", flush=True)
    for r in TEST_ROUNDS:
        print(f"  R{r:>2}", end="")
    print(f"  {'AVG':>6}")
    for bw in BLEND_WEIGHTS:
        print(f"{bw:>6.2f}", end="")
        for r in TEST_ROUNDS:
            v = results.get(bw, {}).get(r, float("nan"))
            print(f" {v:5.1f}", end="")
        avg = np.mean(list(results.get(bw, {}).values())) if results.get(bw) else 0
        print(f"  {avg:>6.2f}")

    best_bw = max(results, key=lambda bw: np.mean(list(results[bw].values())) if results[bw] else 0)
    best_avg = np.mean(list(results[best_bw].values()))
    print(f"\n*** BEST blend_weight={best_bw:.2f}  avg={best_avg:.2f} ***", flush=True)

    # Save
    results_path = DATA_DIR / "overnight_blend_sweep_results.json"
    results_path.write_text(json.dumps(
        {f"bw_{bw:.2f}": v for bw, v in results.items()}, indent=2
    ))
    print(f"Results saved to {results_path}", flush=True)


# ── Step 5 (bonus): Re-run LORO with R19 data ────────────────────────────

def rerun_loro_with_r19():
    """Re-run big U-Net LORO now that R19 is in the training data."""
    print("\n" + "=" * 70, flush=True)
    print("STEP 5 (BONUS): Re-run Big U-Net LORO with R1-R19", flush=True)
    print("=" * 70, flush=True)

    # train_big_unet.py uses ROUND_IDS from train_unet.py
    # R19 should already be in ROUND_IDS if train_unet.py was updated
    result = subprocess.run(
        [sys.executable, "-u", "train_big_unet.py"],
        capture_output=False, text=True, timeout=86400,
    )
    if result.returncode == 0:
        print("Re-run LORO with R19 completed!", flush=True)
    else:
        print(f"Re-run LORO failed with code {result.returncode}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Overnight orchestrator started at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # SAFETY: Remove API token so training steps cannot reach the API
    saved_token = os.environ.get("ASTAR_TOKEN")
    if "ASTAR_TOKEN" in os.environ:
        del os.environ["ASTAR_TOKEN"]
    print("ASTAR_TOKEN removed — training steps cannot reach the API", flush=True)

    # ── Step 1: Big U-Net LORO (~3.5h) ────────────────────────────────────
    run_big_unet_loro()

    # ── Step 2: Download R19 GT ───────────────────────────────────────────
    r19_ok = download_r19_gt_with_token(saved_token)

    # ── Step 3: Retrain production models on R1-R19 (~40min) ──────────────
    if r19_ok:
        retrain_all_production()
    else:
        print("R19 GT not available — skipping production retrain", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}", flush=True)
    print(f"Essential steps done! ({elapsed/3600:.1f} hours)", flush=True)
    print(f"{'=' * 70}\n", flush=True)

    # ── Bonus steps (if VM stays alive) ───────────────────────────────────
    print("Starting bonus steps...", flush=True)

    # Step 4: Quick blend weight sweep (~1.5h)
    try:
        run_blend_sweep()
    except Exception as e:
        print(f"Blend sweep failed: {e}", flush=True)

    # Step 5: Re-run LORO with R19 data (~3.5h)
    if r19_ok:
        try:
            rerun_loro_with_r19()
        except Exception as e:
            print(f"Re-run LORO failed: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}", flush=True)
    print(f"Overnight complete! Total time: {elapsed/3600:.1f} hours", flush=True)
    print(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
