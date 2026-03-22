"""
Big U-Net full LORO training — plan-95 flagship configs.

Tests the largest architectures from plan-95:
  1. V2_big:     3-level, 48ch, no FiLM, no attention, vanilla training
  2. V2_big_mix: 3-level, 48ch, no FiLM, no attention, mixup+LS+OneCycle
  3. V2_film:    3-level, 48ch, FiLM, attention gates, mixup+LS+OneCycle
  4. V2_huge:    3-level, 64ch, FiLM, attention gates, mixup+LS+OneCycle

Runs full LORO on all available rounds (R1-R18).

Usage:
    python train_big_unet.py           # All 4 configs
    python train_big_unet.py V2_big    # Single config
"""

import json
import sys
import time
import gc
import numpy as np
import torch

sys.path.insert(0, ".")  # ensure astar package is importable

from astar.unet import UNet, N_INPUT_CHANNELS, N_SPATIAL_CHANNELS, N_ROUND_CHANNELS, NUM_CLASSES
from astar.submit import score_prediction
from astar.model import apply_floor, PROB_FLOOR
from train_unet import (
    ROUND_IDS, load_round_data, compute_gt_round_features,
    build_unet_dataset, train_unet_fold, score_unet_on_round,
)

device = torch.device("cpu")

# ── Configs ────────────────────────────────────────────────────────────────

CONFIGS = {
    "V2_big": dict(
        unet_config=dict(
            in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
            base_channels=48, dropout=0.15, n_levels=3,
            use_film=False, use_attention=False,
        ),
        film_mode=False,
        n_epochs=300, lr=1e-3, patience=40,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False,
    ),
    "V2_big_mix": dict(
        unet_config=dict(
            in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
            base_channels=48, dropout=0.15, n_levels=3,
            use_film=False, use_attention=False,
        ),
        film_mode=False,
        n_epochs=300, lr=1e-3, patience=40,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    ),
    "V2_film": dict(
        unet_config=dict(
            in_channels=N_SPATIAL_CHANNELS, n_classes=NUM_CLASSES,
            base_channels=48, dropout=0.15, n_levels=3,
            use_film=True, use_attention=True,
            n_round_feats=N_ROUND_CHANNELS,
        ),
        film_mode=True,
        n_epochs=300, lr=1e-3, patience=40,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    ),
    "V2_huge": dict(
        unet_config=dict(
            in_channels=N_SPATIAL_CHANNELS, n_classes=NUM_CLASSES,
            base_channels=64, dropout=0.15, n_levels=3,
            use_film=True, use_attention=True,
            n_round_feats=N_ROUND_CHANNELS,
        ),
        film_mode=True,
        n_epochs=300, lr=1e-3, patience=40,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    ),
}


def main():
    # Parse optional config filter
    selected = None
    if len(sys.argv) > 1:
        selected = sys.argv[1:]
        for s in selected:
            if s not in CONFIGS:
                print(f"Unknown config '{s}'. Available: {list(CONFIGS.keys())}")
                sys.exit(1)

    # Load all rounds
    print("Loading all rounds...", flush=True)
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds", flush=True)
        else:
            print(f"  R{rnum}: NO DATA", flush=True)
    all_rounds = sorted(all_data.keys())
    print(f"Total: {len(all_data)} rounds\n", flush=True)

    all_results = {}

    configs_to_run = {k: v for k, v in CONFIGS.items() if selected is None or k in selected}

    for cname, cfg in configs_to_run.items():
        # Print param count
        test_model = UNet(**cfg["unet_config"])
        n_params = sum(p.numel() for p in test_model.parameters())
        del test_model
        print(f"\n{'='*70}", flush=True)
        print(f"=== {cname} ({n_params:,} params) ===", flush=True)
        print(f"    Config: base_ch={cfg['unet_config']['base_channels']}, "
              f"levels={cfg['unet_config']['n_levels']}, "
              f"film={cfg['unet_config'].get('use_film', False)}, "
              f"attn={cfg['unet_config'].get('use_attention', False)}", flush=True)
        print(f"    Training: epochs={cfg['n_epochs']}, mixup={cfg['use_mixup']}, "
              f"LS={cfg['label_smoothing']}, OneCycle={cfg['use_onecycle']}", flush=True)
        print(f"{'='*70}", flush=True)

        results = {}
        t0_config = time.time()

        for test_rnum in all_rounds:
            train_rounds = [r for r in all_rounds if r != test_rnum]
            t0_fold = time.time()

            if cfg["film_mode"]:
                imgs, rf, tgts = build_unet_dataset(
                    train_rounds, all_data, augment=True, film_mode=True)
            else:
                imgs, tgts = build_unet_dataset(
                    train_rounds, all_data, augment=True, film_mode=False)
                rf = None

            model = train_unet_fold(
                imgs, tgts, device,
                n_epochs=cfg["n_epochs"], lr=cfg["lr"],
                patience=cfg["patience"], verbose=False,
                round_feats_arr=rf, unet_config=cfg["unet_config"],
                use_mixup=cfg["use_mixup"],
                label_smoothing=cfg["label_smoothing"],
                use_onecycle=cfg["use_onecycle"],
            )

            _, det, gts = all_data[test_rnum]
            test_rf = compute_gt_round_features(det, gts)
            scores = score_unet_on_round(model, det, gts, test_rf, device, use_tta=True)
            avg = np.mean(scores)
            results[test_rnum] = avg
            elapsed_fold = time.time() - t0_fold
            print(f"  R{test_rnum}: {avg:.2f}  ({elapsed_fold:.0f}s)", flush=True)

            del model, imgs, tgts
            if rf is not None:
                del rf
            gc.collect()

        elapsed_config = time.time() - t0_config
        all_results[cname] = results
        avg_score = np.mean(list(results.values()))
        print(f"\n  >>> {cname} AVG: {avg_score:.2f}  (total {elapsed_config:.0f}s)", flush=True)

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n\n{'='*80}", flush=True)
    print("SUMMARY — Big U-Net LORO Results", flush=True)
    print(f"{'='*80}", flush=True)

    header = f"{'Config':>14}"
    for r in all_rounds:
        header += f"  R{r:>2}"
    header += f"  {'AVG':>6}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for cname in configs_to_run:
        row = f"{cname:>14}"
        for r in all_rounds:
            v = all_results.get(cname, {}).get(r, float("nan"))
            row += f" {v:5.1f}"
        avg = np.mean(list(all_results.get(cname, {}).values()))
        row += f"  {avg:>6.2f}"
        print(row, flush=True)

    # Reference V1 baseline for comparison
    print(f"\n(Reference V1 LORO avg from comparison: ~83.75)")
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
