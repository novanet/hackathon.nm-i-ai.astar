"""
Compare U-Net configurations via LORO to find the best architecture.

Configs tested:
  V1: Current (2-level, 32ch, 19-input, cosine)
  V2a: 3-level, 48ch (bigger capacity, same input scheme)
  V2b: 3-level, 48ch, FiLM conditioning
  V2c: 3-level, 48ch, FiLM + attention gates
  V2d: V2c + mixup + OneCycleLR

Each config does full LORO (train on N-1 rounds, test on held-out round).
"""
import sys
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from astar.unet import (
    UNet, N_INPUT_CHANNELS, N_SPATIAL_CHANNELS, N_ROUND_CHANNELS, NUM_CLASSES,
    compute_unet_input, d4_augment, entropy_weighted_kl_loss,
    predict_unet, predict_unet_with_tta,
)
from astar.submit import score_prediction
from astar.model import apply_floor
from train_unet import (
    ROUND_IDS, load_round_data, compute_gt_round_features,
    build_unet_dataset, train_unet_fold, score_unet_on_round,
)

CONFIGS = {
    "V1_baseline": {
        "unet_config": {
            "in_channels": N_INPUT_CHANNELS,
            "n_classes": NUM_CLASSES,
            "base_channels": 32,
            "dropout": 0.1,
            "n_levels": 2,
            "use_film": False,
            "use_attention": False,
        },
        "film_mode": False,
        "n_epochs": 300,
        "lr": 1e-3,
        "patience": 30,
        "use_mixup": False,
        "label_smoothing": 0.0,
        "use_onecycle": False,
    },
    "V1_onecycle_mixup": {
        "unet_config": {
            "in_channels": N_INPUT_CHANNELS,
            "n_classes": NUM_CLASSES,
            "base_channels": 32,
            "dropout": 0.1,
            "n_levels": 2,
            "use_film": False,
            "use_attention": False,
        },
        "film_mode": False,
        "n_epochs": 300,
        "lr": 1e-3,
        "patience": 30,
        "use_mixup": True,
        "label_smoothing": 0.01,
        "use_onecycle": True,
    },
    "V2_48ch_2lev": {
        "unet_config": {
            "in_channels": N_INPUT_CHANNELS,
            "n_classes": NUM_CLASSES,
            "base_channels": 48,
            "dropout": 0.1,
            "n_levels": 2,
            "use_film": False,
            "use_attention": False,
        },
        "film_mode": False,
        "n_epochs": 300,
        "lr": 1e-3,
        "patience": 30,
        "use_mixup": False,
        "label_smoothing": 0.0,
        "use_onecycle": False,
    },
    "V2_3lev_32ch": {
        "unet_config": {
            "in_channels": N_INPUT_CHANNELS,
            "n_classes": NUM_CLASSES,
            "base_channels": 32,
            "dropout": 0.1,
            "n_levels": 3,
            "use_film": False,
            "use_attention": False,
        },
        "film_mode": False,
        "n_epochs": 300,
        "lr": 1e-3,
        "patience": 30,
        "use_mixup": False,
        "label_smoothing": 0.0,
        "use_onecycle": False,
    },
    "V2_FiLM_48_3lev": {
        "unet_config": {
            "in_channels": N_SPATIAL_CHANNELS,
            "n_classes": NUM_CLASSES,
            "base_channels": 48,
            "dropout": 0.1,
            "n_levels": 3,
            "use_film": True,
            "n_round_feats": N_ROUND_CHANNELS,
            "use_attention": True,
        },
        "film_mode": True,
        "n_epochs": 400,
        "lr": 1e-3,
        "patience": 40,
        "use_mixup": True,
        "label_smoothing": 0.01,
        "use_onecycle": True,
    },
}


def run_loro_for_config(config_name: str, config: dict, all_data: dict,
                        device: torch.device) -> dict[int, float]:
    """Run full LORO for a single config. Returns {round_num: tta_score}."""
    all_rounds = sorted(all_data.keys())
    results = {}
    film_mode = config["film_mode"]

    for test_rnum in all_rounds:
        train_rounds = [r for r in all_rounds if r != test_rnum]

        # Build data
        if film_mode:
            train_images, train_rf, train_targets = build_unet_dataset(
                train_rounds, all_data, augment=True, film_mode=True)
        else:
            train_images, train_targets = build_unet_dataset(
                train_rounds, all_data, augment=True, film_mode=False)
            train_rf = None

        # Train
        model = train_unet_fold(
            train_images, train_targets, device,
            n_epochs=config["n_epochs"],
            lr=config["lr"],
            patience=config["patience"],
            verbose=False,
            round_feats_arr=train_rf,
            unet_config=config["unet_config"],
            use_mixup=config["use_mixup"],
            label_smoothing=config["label_smoothing"],
            use_onecycle=config["use_onecycle"],
        )

        # Score with TTA
        _, test_detail, test_gts = all_data[test_rnum]
        test_rf = compute_gt_round_features(test_detail, test_gts)
        scores = score_unet_on_round(model, test_detail, test_gts,
                                     test_rf, device, use_tta=True)
        avg = np.mean(scores)
        results[test_rnum] = avg
        print(f"    R{test_rnum}: {avg:.2f}")

        # Free GPU memory
        del model, train_images, train_targets
        if train_rf is not None:
            del train_rf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def main():
    # Parse args
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    # Filter to valid configs
    selected = [c for c in selected if c in CONFIGS]
    if not selected:
        print(f"Available configs: {list(CONFIGS.keys())}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n=== LOADING DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")
    print(f"  Total: {sum(len(g) for _,_,g in all_data.values())} seeds")

    # Run each config
    all_results = {}
    for cname in selected:
        config = CONFIGS[cname]
        # Print param count
        test_model = UNet(**config["unet_config"])
        n_params = sum(p.numel() for p in test_model.parameters())
        del test_model
        print(f"\n=== {cname} ({n_params:,} params) ===")
        t0 = time.time()
        results = run_loro_for_config(cname, config, all_data, device)
        elapsed = time.time() - t0
        all_results[cname] = results
        avg = np.mean(list(results.values()))
        print(f"  LORO avg: {avg:.2f}  [{elapsed:.0f}s]")

    # Summary
    all_rounds = sorted(all_data.keys())
    print(f"\n{'='*60}")
    print(f"=== COMPARISON SUMMARY ===")
    header = f"{'Round':>8}"
    for cname in selected:
        header += f" {cname:>16}"
    print(header)
    for rnum in all_rounds:
        row = f"  R{rnum:<5d}"
        for cname in selected:
            if rnum in all_results.get(cname, {}):
                row += f" {all_results[cname][rnum]:>16.2f}"
            else:
                row += f" {'---':>16}"
        print(row)
    row = f"{'AVG':>8}"
    for cname in selected:
        if cname in all_results:
            avg = np.mean(list(all_results[cname].values()))
            row += f" {avg:>16.2f}"
    print(row)


if __name__ == "__main__":
    main()
