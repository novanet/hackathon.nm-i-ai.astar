"""
Train a U-Net for Astar Island spatial prediction.

Uses D4 augmentation (8× data) and entropy-weighted KL loss.
LORO (Leave-One-Round-Out) cross-validation on R1-R13.

Usage:
    python train_unet.py              # Full LORO CV + final model training
    python train_unet.py --fast       # Quick test with fewer epochs
"""

import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from astar.unet import (
    UNet, N_INPUT_CHANNELS, compute_unet_input,
    d4_augment, entropy_weighted_kl_loss,
    predict_unet, predict_unet_with_tta,
)
from astar.model import apply_floor, PROB_FLOOR
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

DATA_DIR = Path("data")
UNET_MODEL_PATH = DATA_DIR / "unet_model.pt"

# All completed rounds with ground truth
ROUND_IDS = {
    1: "71451d74-be9f-471f-aacd-a41f3b68a9cd",
    2: "76909e29-f664-4b2f-b16b-61b7507277e9",
    3: "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb",
    4: "8e839974-b13b-407b-a5e7-fc749d877195",
    5: "fd3c92ff-3178-4dc9-8d9b-acf389b3982b",
    6: "ae78003a-4efe-425a-881a-d16a39bca0ad",
    7: "36e581f1-73f8-453f-ab98-cbe3052b701b",
    8: "c5cdf100-a876-4fb7-b5d8-757162c97989",
    9: "2a341ace-0f57-4309-9b89-e59fe0f09179",
    10: "75e625c3-60cb-4392-af3e-c86a98bde8c2",
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
    16: "8f664aed-8839-4c85-bed0-77a2cac7c6f5",
    17: "3eb0c25d-28fa-48ca-b8e1-fc249e3918e9",
    18: "b0f9d1bf-4b71-4e6e-816c-19c718d29056",
}


# ── Data Loading (reused patterns from train_spatial.py) ───────────────────


def load_round_data(round_id: str):
    """Load round detail and ground truths for a completed round."""
    rdir = DATA_DIR / f"round_{round_id}"
    detail_files = sorted(rdir.glob("round_detail_*.json"))
    if not detail_files:
        return None, []
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    gts = []
    for s in range(len(detail.get("initial_states", []))):
        gt_path = rdir / f"ground_truth_s{s}.json"
        if not gt_path.exists():
            break
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        gts.append(np.array(gt["ground_truth"], dtype=np.float64))
    return detail, gts


def compute_gt_round_features(detail: dict, gts: list[np.ndarray]) -> np.ndarray:
    """Compute round-level features from GT transition matrix.
    Returns 8 features matching compute_round_features() in model.py:
    E→E, S→S, F→F, E→S, settlement_density, mean_food, mean_wealth, mean_defense.
    Settlement stats are transition-derived proxies (NEVER real settlement attributes).
    """
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = 0
    total = 0
    for s in range(len(gts)):
        init_grid = states[s]["grid"]
        for y in range(map_h):
            for x in range(map_w):
                init_cls = TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                gt_counts[init_cls] += gts[s][y, x]
                if init_cls == 1:
                    n_sett += 1
                total += 1
    row_sums = np.maximum(gt_counts.sum(axis=1, keepdims=True), 1.0)
    gt_trans = gt_counts / row_sums
    sett_density = n_sett / max(total, 1)
    ss_rate = gt_trans[1, 1]
    return np.array([
        gt_trans[0, 0],          # E→E
        gt_trans[1, 1],          # S→S
        gt_trans[4, 4],          # F→F
        gt_trans[0, 1],          # E→S
        sett_density,
        0.3 + 0.7 * ss_rate,    # food proxy
        gt_trans[0, 1] * 0.3,   # wealth proxy
        1.0 - gt_trans[1, 0],   # defense proxy
    ], dtype=np.float64)


# ── Dataset Construction ───────────────────────────────────────────────────


def build_unet_dataset(round_nums: list[int], all_data: dict,
                       augment: bool = True,
                       film_mode: bool = False) -> tuple:
    """
    Build image/target arrays for given round numbers.

    Args:
        round_nums: list of round numbers to include
        all_data: dict mapping round_num → (round_id, detail, gts)
        augment: if True, apply D4 augmentation (8× data)
        film_mode: if True, return separate round_features array

    Returns (non-FiLM):
        images: (N, 19, H, W) float32
        targets: (N, H, W, 6) float64
    Returns (FiLM):
        images: (N, 11, H, W) float32
        round_feats: (N, 8) float32
        targets: (N, H, W, 6) float64
    """
    images, targets = [], []
    round_feats_list = [] if film_mode else None
    for rnum in round_nums:
        if rnum not in all_data:
            continue
        _, detail, gts = all_data[rnum]
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            if film_mode:
                spatial_img, rf = compute_unet_input(init_grid, map_w, map_h, round_feats, film_mode=True)
                if augment:
                    for aug_img, aug_gt, aug_rf in d4_augment(spatial_img, gt, rf):
                        images.append(aug_img)
                        targets.append(aug_gt)
                        round_feats_list.append(aug_rf)
                else:
                    images.append(spatial_img)
                    targets.append(gt)
                    round_feats_list.append(rf)
            else:
                img = compute_unet_input(init_grid, map_w, map_h, round_feats)
                if augment:
                    for aug_img, aug_gt in d4_augment(img, gt):
                        images.append(aug_img)
                        targets.append(aug_gt)
                else:
                    images.append(img)
                    targets.append(gt)
    if film_mode:
        return (np.stack(images).astype(np.float32),
                np.stack(round_feats_list).astype(np.float32),
                np.stack(targets).astype(np.float64))
    return np.stack(images).astype(np.float32), np.stack(targets).astype(np.float64)


# ── Training ───────────────────────────────────────────────────────────────


def train_unet_fold(images: np.ndarray, targets: np.ndarray,
                    device: torch.device,
                    n_epochs: int = 300, lr: float = 1e-3,
                    batch_size: int = 16, patience: int = 30,
                    verbose: bool = True,
                    # V2 options
                    round_feats_arr: np.ndarray | None = None,
                    unet_config: dict | None = None,
                    use_mixup: bool = False,
                    label_smoothing: float = 0.0,
                    use_onecycle: bool = False) -> UNet:
    """
    Train U-Net on given data with early stopping on a 90/10 val split.

    Args:
        images:  (N, C, H, W) float32 — 19ch (V1) or 11ch (V2 FiLM)
        targets: (N, H, W, 6) float64
        round_feats_arr: (N, 8) float32 — only for FiLM mode
        unet_config: dict with UNet constructor kwargs (V2 params)
        use_mixup: enable mixup data augmentation
        label_smoothing: label smoothing alpha (0 = disabled)
        use_onecycle: use OneCycleLR instead of CosineAnnealingLR

    Returns:
        Trained UNet model (best val loss checkpoint).
    """
    n = len(images)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    # Move to tensors
    train_imgs = torch.tensor(images[train_idx], dtype=torch.float32, device=device)
    train_gts = torch.tensor(targets[train_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    val_imgs = torch.tensor(images[val_idx], dtype=torch.float32, device=device)
    val_gts = torch.tensor(targets[val_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2)

    film_mode = round_feats_arr is not None
    train_rf = val_rf = None
    if film_mode:
        train_rf = torch.tensor(round_feats_arr[train_idx], dtype=torch.float32, device=device)
        val_rf = torch.tensor(round_feats_arr[val_idx], dtype=torch.float32, device=device)

    # Label smoothing: blend targets toward uniform
    if label_smoothing > 0:
        n_classes = train_gts.shape[1]
        uniform = 1.0 / n_classes
        train_gts = (1 - label_smoothing) * train_gts + label_smoothing * uniform
        val_gts = (1 - label_smoothing) * val_gts + label_smoothing * uniform

    # Build model
    if unet_config:
        model = UNet(**unet_config).to(device)
    else:
        model = UNet(in_channels=images.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_steps_per_epoch = (len(train_imgs) + batch_size - 1) // batch_size
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 3, epochs=n_epochs,
            steps_per_epoch=n_steps_per_epoch, pct_start=0.05,
            anneal_strategy='cos')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        perm_t = torch.randperm(len(train_imgs), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_imgs), batch_size):
            idx = perm_t[i:i + batch_size]
            batch_imgs = train_imgs[idx]
            batch_gts = train_gts[idx]
            batch_rf = train_rf[idx] if film_mode else None

            # Mixup augmentation
            if use_mixup and len(idx) > 1:
                lam = np.random.beta(0.2, 0.2)
                shuffle_idx = torch.randperm(len(batch_imgs), device=device)
                batch_imgs = lam * batch_imgs + (1 - lam) * batch_imgs[shuffle_idx]
                batch_gts = lam * batch_gts + (1 - lam) * batch_gts[shuffle_idx]
                if film_mode:
                    batch_rf = lam * batch_rf + (1 - lam) * batch_rf[shuffle_idx]

            log_pred = model(batch_imgs, batch_rf)
            loss = entropy_weighted_kl_loss(log_pred, batch_gts)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if use_onecycle:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if not use_onecycle:
            scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(val_imgs), batch_size):
                vb_imgs = val_imgs[i:i + batch_size]
                vb_rf = val_rf[i:i + batch_size] if film_mode else None
                vl = entropy_weighted_kl_loss(
                    model(vb_imgs, vb_rf),
                    val_gts[i:i + batch_size]
                ).item()
                val_losses.append(vl * len(vb_imgs))
            val_loss = sum(val_losses) / len(val_imgs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch % 20 == 0 or wait == 0):
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch:3d}: train={epoch_loss / max(n_batches, 1):.5f}, "
                  f"val={val_loss:.5f}, best={best_val_loss:.5f}, "
                  f"lr={cur_lr:.1e}")

        if wait >= patience:
            if verbose:
                print(f"    Early stop at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ── Scoring ────────────────────────────────────────────────────────────────


def score_unet_on_round(model: UNet, detail: dict, gts: list[np.ndarray],
                        round_feats: np.ndarray, device: torch.device,
                        use_tta: bool = False) -> list[float]:
    """Score a trained U-Net on one round's ground truths.
    Automatically detects FiLM mode from model."""
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    scores = []
    for s, gt in enumerate(gts):
        init_grid = detail["initial_states"][s]["grid"]
        if use_tta:
            pred = predict_unet_with_tta(model, init_grid, map_w, map_h,
                                         round_feats, device)
        else:
            pred = predict_unet(model, init_grid, map_w, map_h,
                                round_feats, device)
        pred = apply_floor(pred)
        scores.append(score_prediction(pred, gt))
    return scores


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    fast_mode = "--fast" in sys.argv
    n_epochs = 80 if fast_mode else 300
    patience = 15 if fast_mode else 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if fast_mode:
        print("*** FAST MODE: reduced epochs for quick testing ***")

    # ── Load all data ──
    print("\n=== LOADING ALL ROUND DATA ===")
    all_data: dict[int, tuple] = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            map_w = detail.get("map_width", 40)
            map_h = detail.get("map_height", 40)
            print(f"  R{rnum}: {len(gts)} seeds, {map_w}×{map_h}")
        else:
            print(f"  R{rnum}: no data")

    total_seeds = sum(len(gts) for _, _, gts in all_data.values())
    print(f"  Total: {total_seeds} seeds ({total_seeds * 8} with D4 augmentation)")

    # ── Precompute all UNet inputs ──
    print("\n=== PRECOMPUTING U-NET INPUTS ===")
    t0 = time.time()
    # Just a quick timing test
    sample_rnum = next(iter(all_data))
    _, sample_detail, sample_gts = all_data[sample_rnum]
    sample_rf = compute_gt_round_features(sample_detail, sample_gts)
    _ = compute_unet_input(
        sample_detail["initial_states"][0]["grid"],
        sample_detail.get("map_width", 40),
        sample_detail.get("map_height", 40),
        sample_rf,
    )
    print(f"  Single input computation: {(time.time() - t0) * 1000:.1f}ms")

    # ── LORO Cross-Validation ──
    print(f"\n=== LEAVE-ONE-ROUND-OUT CROSS-VALIDATION (epochs={n_epochs}) ===")
    loro_results = {}
    loro_tta_results = {}
    all_rounds = sorted(all_data.keys())
    total_start = time.time()

    for test_rnum in all_rounds:
        fold_start = time.time()
        train_rounds = [r for r in all_rounds if r != test_rnum]
        print(f"\n  --- Test R{test_rnum} (train on {len(train_rounds)} rounds) ---")

        # Build training data with D4 augmentation
        train_images, train_targets = build_unet_dataset(train_rounds, all_data, augment=True)
        print(f"    Training data: {train_images.shape[0]} images "
              f"({len(train_rounds)} rounds × 5 seeds × 8 aug)")

        # Train
        model = train_unet_fold(
            train_images, train_targets, device,
            n_epochs=n_epochs, patience=patience, verbose=True,
        )

        # Score on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        # Without TTA
        scores = score_unet_on_round(model, test_detail, test_gts,
                                     test_round_feats, device, use_tta=False)
        avg = np.mean(scores)
        loro_results[test_rnum] = avg

        # With TTA
        scores_tta = score_unet_on_round(model, test_detail, test_gts,
                                         test_round_feats, device, use_tta=True)
        avg_tta = np.mean(scores_tta)
        loro_tta_results[test_rnum] = avg_tta

        fold_time = time.time() - fold_start
        print(f"    R{test_rnum}: unet={avg:.2f}, +tta={avg_tta:.2f} "
              f"(d={avg_tta - avg:+.2f})  "
              f"seeds: {[f'{s:.1f}' for s in scores]}  [{fold_time:.0f}s]")

    # ── LORO Summary ──
    loro_avg = np.mean(list(loro_results.values()))
    loro_tta_avg = np.mean(list(loro_tta_results.values()))
    total_time = time.time() - total_start

    print(f"\n=== LORO SUMMARY ===")
    print(f"{'Round':>8} {'UNet':>8} {'+ TTA':>8}")
    print(f"{'-----':>8} {'-----':>8} {'-----':>8}")
    for rnum in all_rounds:
        print(f"  R{rnum:<5d} {loro_results[rnum]:>8.2f} {loro_tta_results[rnum]:>8.2f}")
    print(f"{'Avg':>8} {loro_avg:>8.2f} {loro_tta_avg:>8.2f}")
    print(f"\n  Wall-clock time: {total_time:.0f}s ({total_time / 60:.1f}min)")

    # ── Train Final Model on All Data ──
    print(f"\n=== TRAINING FINAL U-NET (all {len(all_rounds)} rounds) ===")
    final_images, final_targets = build_unet_dataset(all_rounds, all_data, augment=True)
    print(f"  Training data: {final_images.shape[0]} images "
          f"({len(all_rounds)} rounds × 5 seeds × 8 aug)")

    final_model = train_unet_fold(
        final_images, final_targets, device,
        n_epochs=n_epochs, patience=patience, verbose=True,
    )

    # Save
    torch.save({
        "state_dict": final_model.state_dict(),
        "in_channels": N_INPUT_CHANNELS,
        "n_classes": NUM_CLASSES,
        "base_channels": 32,
        "dropout": 0.1,
    }, UNET_MODEL_PATH)
    print(f"  Saved to {UNET_MODEL_PATH}")

    # ── In-Sample Scores ──
    print(f"\n=== IN-SAMPLE SCORES (sanity check) ===")
    for rnum in all_rounds:
        rid, detail, gts = all_data[rnum]
        round_feats = compute_gt_round_features(detail, gts)
        scores = score_unet_on_round(final_model, detail, gts, round_feats,
                                     device, use_tta=False)
        scores_tta = score_unet_on_round(final_model, detail, gts, round_feats,
                                         device, use_tta=True)
        print(f"  R{rnum}: unet={np.mean(scores):.2f}, +tta={np.mean(scores_tta):.2f}  "
              f"seeds: {[f'{s:.1f}' for s in scores]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
