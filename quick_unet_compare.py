"""
Quick A/B test of U-Net configs on a 5-round subset with reduced epochs.
Purpose: Get directional signal fast (~10min per config on CPU).
"""
import sys, time
import numpy as np
import torch

sys.path.insert(0, ".")
from astar.unet import (
    UNet, N_INPUT_CHANNELS, N_SPATIAL_CHANNELS, N_ROUND_CHANNELS, NUM_CLASSES,
    entropy_weighted_kl_loss,
)
from astar.submit import score_prediction
from astar.model import apply_floor
from train_unet import (
    ROUND_IDS, load_round_data, compute_gt_round_features,
    build_unet_dataset, train_unet_fold, score_unet_on_round,
)

# Use 5 representative rounds for quick testing (mix of easy/hard)
TEST_ROUNDS = [1, 5, 9, 13, 17]

CONFIGS = {
    "V1": dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=100, lr=1e-3, patience=20,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False,
    ),
    "V1_mix": dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=100, lr=1e-3, patience=20,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    ),
    "V2_48_2lev": dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=48, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=100, lr=1e-3, patience=20,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False,
    ),
    "V2_3lev_32": dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=3,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=100, lr=1e-3, patience=20,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False,
    ),
    "V2_FiLM": dict(
        unet_config=dict(in_channels=N_SPATIAL_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=48, dropout=0.1, n_levels=3,
                         use_film=True, n_round_feats=N_ROUND_CHANNELS, use_attention=True),
        film_mode=True, n_epochs=100, lr=1e-3, patience=20,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True,
    ),
}


def quick_loro(config: dict, all_data: dict, test_rounds: list[int],
               device: torch.device) -> dict[int, float]:
    all_rounds = sorted(all_data.keys())
    results = {}
    film_mode = config["film_mode"]

    for test_rnum in test_rounds:
        if test_rnum not in all_data:
            continue
        train_rounds = [r for r in all_rounds if r != test_rnum]

        if film_mode:
            imgs, rf, tgts = build_unet_dataset(train_rounds, all_data, augment=True, film_mode=True)
        else:
            imgs, tgts = build_unet_dataset(train_rounds, all_data, augment=True, film_mode=False)
            rf = None

        model = train_unet_fold(
            imgs, tgts, device,
            n_epochs=config["n_epochs"], lr=config["lr"],
            patience=config["patience"], verbose=False,
            round_feats_arr=rf, unet_config=config["unet_config"],
            use_mixup=config["use_mixup"], label_smoothing=config["label_smoothing"],
            use_onecycle=config["use_onecycle"],
        )

        _, det, gts = all_data[test_rnum]
        test_rf = compute_gt_round_features(det, gts)
        scores = score_unet_on_round(model, det, gts, test_rf, device, use_tta=True)
        avg = np.mean(scores)
        results[test_rnum] = avg
        print(f"    R{test_rnum}: {avg:.2f}")

        del model, imgs, tgts
        if rf is not None:
            del rf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def main():
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(CONFIGS.keys())
    selected = [c for c in selected if c in CONFIGS]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== LOADING ALL ROUND DATA ===")
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    all_results = {}
    for cname in selected:
        cfg = CONFIGS[cname]
        test_model = UNet(**cfg["unet_config"])
        n_params = sum(p.numel() for p in test_model.parameters())
        del test_model
        print(f"\n=== {cname} ({n_params:,} params) ===")
        t0 = time.time()
        results = quick_loro(cfg, all_data, TEST_ROUNDS, device)
        elapsed = time.time() - t0
        all_results[cname] = results
        avg = np.mean(list(results.values()))
        print(f"  5-fold avg: {avg:.2f}  [{elapsed:.0f}s]")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Config':>16}", end="")
    for r in TEST_ROUNDS:
        print(f" {'R'+str(r):>8}", end="")
    print(f" {'AVG':>8}")
    for cname in selected:
        print(f"{cname:>16}", end="")
        for r in TEST_ROUNDS:
            v = all_results.get(cname, {}).get(r, float('nan'))
            print(f" {v:>8.2f}", end="")
        avg = np.mean(list(all_results.get(cname, {}).values()))
        print(f" {avg:>8.2f}")


if __name__ == "__main__":
    main()
