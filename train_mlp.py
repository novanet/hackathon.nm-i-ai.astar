"""
Train a KL-loss MLP that directly optimizes the competition scoring metric.

Architecture: 30 features → 128 → 64 → 6 (softmax)
Loss: entropy-weighted KL divergence (matches the official scorer)

Compares against the existing GBM ensemble via LORO cross-validation.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from astar.model import (
    compute_cell_features, apply_floor, _apply_transition_matrix,
    HISTORICAL_TRANSITIONS, PROB_FLOOR,
    CALIBRATION_FACTORS, PER_CLASS_TEMPS,
)
from astar.submit import score_prediction
from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

DATA_DIR = Path("data")

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
}

ENTROPY_WEIGHT_POWER = 0.25


def load_round_data(round_id: str):
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
    states = detail.get("initial_states", [])
    map_w = detail.get("map_width", 40)
    map_h = detail.get("map_height", 40)
    gt_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    n_sett = total = 0
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
        gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4], gt_trans[0, 1],
        sett_density,
        0.3 + 0.7 * ss_rate,
        gt_trans[0, 1] * 0.3,
        1.0 - gt_trans[1, 0],
    ], dtype=np.float64)


def build_training_data(round_ids: dict[int, str]):
    """Build X, Y, entropy_weights from rounds."""
    X_parts, Y_parts, W_parts = [], [], []
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            continue
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for s_idx, gt in enumerate(gts):
            init_grid = detail["initial_states"][s_idx]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
            X_parts.append(feat.reshape(-1, feat.shape[-1]))
            Y_parts.append(gt.reshape(-1, gt.shape[-1]))
            p = np.clip(gt, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p), axis=-1).flatten()
            W_parts.append(np.power(entropy + 0.01, ENTROPY_WEIGHT_POWER))
    return np.vstack(X_parts), np.vstack(Y_parts), np.concatenate(W_parts)


# ── MLP Definition ──

class KLDivMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 6,
                 hidden: list[int] = [256, 128, 64]):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden[0]),
            nn.Dropout(0.1),
        )
        # Residual blocks for deeper layers
        self.res_blocks = nn.ModuleList()
        self.res_projs = nn.ModuleList()  # projections for dimension changes
        prev = hidden[0]
        for h in hidden[1:]:
            block = nn.Sequential(
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.1),
            )
            self.res_blocks.append(block)
            # Linear projection for residual when dimensions differ
            self.res_projs.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            prev = h
        self.head = nn.Linear(prev, n_classes)

    def forward(self, x):
        h = self.input_proj(x)
        for block, proj in zip(self.res_blocks, self.res_projs):
            h = block(h) + proj(h)
        logits = self.head(h)
        return torch.log_softmax(logits, dim=-1)


def entropy_weighted_kl_loss(log_pred: torch.Tensor, target: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
    """
    Entropy-weighted KL divergence: exactly mirrors the competition scorer.
    KL(target || pred) = sum_c target_c * log(target_c / pred_c)
                       = sum_c target_c * (log(target_c) - log(pred_c))
    """
    eps = 1e-10
    target_clamped = target.clamp(min=eps)
    # KL per sample: sum over classes
    kl_per_sample = (target_clamped * (target_clamped.log() - log_pred)).sum(dim=-1)
    # Weighted mean
    return (weights * kl_per_sample).sum() / weights.sum().clamp(min=eps)


def train_mlp(X_train: np.ndarray, Y_train: np.ndarray, W_train: np.ndarray,
              n_epochs: int = 200, lr: float = 1e-3, batch_size: int = 4096,
              patience: int = 20, verbose: bool = True) -> KLDivMLP:
    """Train MLP with entropy-weighted KL loss. Uses 10% validation split for early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X_train)
    n_feat = X_train.shape[1]

    # Shuffle and split 90/10
    idx = np.random.RandomState(42).permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_t = torch.tensor(X_train[train_idx], dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train[train_idx], dtype=torch.float32, device=device)
    W_t = torch.tensor(W_train[train_idx], dtype=torch.float32, device=device)
    X_v = torch.tensor(X_train[val_idx], dtype=torch.float32, device=device)
    Y_v = torch.tensor(Y_train[val_idx], dtype=torch.float32, device=device)
    W_v = torch.tensor(W_train[val_idx], dtype=torch.float32, device=device)

    model = KLDivMLP(n_feat).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(X_t), device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(X_t), batch_size):
            batch_idx = perm[i:i+batch_size]
            log_pred = model(X_t[batch_idx])
            loss = entropy_weighted_kl_loss(log_pred, Y_t[batch_idx], W_t[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_log_pred = model(X_v)
            val_loss = entropy_weighted_kl_loss(val_log_pred, Y_v, W_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and (epoch % 20 == 0 or wait == 0):
            print(f"  Epoch {epoch:3d}: train_kl={epoch_loss/n_batches:.5f}, val_kl={val_loss:.5f}, best={best_val_loss:.5f}, lr={scheduler.get_last_lr()[0]:.1e}")

        if wait >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_with_mlp(model: KLDivMLP, features: np.ndarray,
                     map_h: int = 40, map_w: int = 40) -> np.ndarray:
    """Run MLP inference, return (H, W, 6) probability tensor."""
    device = next(model.parameters()).device
    with torch.no_grad():
        X = torch.tensor(features.reshape(-1, features.shape[-1]),
                         dtype=torch.float32, device=device)
        log_pred = model(X)
        pred = torch.exp(log_pred).cpu().numpy()
    pred = pred.reshape(map_h, map_w, NUM_CLASSES)
    pred = np.maximum(pred, 1e-10)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred


def score_with_postprocessing(pred: np.ndarray, gt: np.ndarray,
                              round_feats: np.ndarray,
                              round_detail: dict, seed_idx: int,
                              apply_calib: bool = True,
                              apply_temps: bool = True) -> float:
    """Score prediction with optional post-processing (calibration + temps + floor)."""
    from astar.model import per_class_temperature_scale

    if apply_calib:
        pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
        pred = pred / pred.sum(axis=-1, keepdims=True)

    if apply_temps:
        ss_rate = round_feats[1]
        if ss_rate < 0.15:
            temps = np.array([1.15, 1.15, 1.15, 1.0, 1.15, 1.0])
        else:
            temps = PER_CLASS_TEMPS
        pred = per_class_temperature_scale(pred, round_detail, seed_idx, temps=temps)

    pred = apply_floor(pred)
    return score_prediction(pred, gt)


def main():
    print("=== KL-LOSS MLP TRAINING ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load all data
    all_data = {}
    for rnum, rid in sorted(ROUND_IDS.items()):
        detail, gts = load_round_data(rid)
        if gts:
            all_data[rnum] = (rid, detail, gts)
            print(f"  R{rnum}: {len(gts)} seeds")

    # ── LORO Cross-Validation ──
    print("\n=== LEAVE-ONE-ROUND-OUT (MLP) ===")
    loro_mlp = {}
    loro_mlp_pp = {}  # with post-processing

    for test_rnum in sorted(all_data.keys()):
        train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
        X_train, Y_train, W_train = build_training_data(train_ids)

        print(f"\n  Test R{test_rnum}: training on {X_train.shape[0]} samples...")
        mlp = train_mlp(X_train, Y_train, W_train, verbose=False)

        # Evaluate on held-out round
        test_rid, test_detail, test_gts = all_data[test_rnum]
        map_w = test_detail.get("map_width", 40)
        map_h = test_detail.get("map_height", 40)
        test_round_feats = compute_gt_round_features(test_detail, test_gts)

        raw_scores = []
        pp_scores = []
        for s, gt in enumerate(test_gts):
            init_grid = test_detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=test_round_feats)
            pred = predict_with_mlp(mlp, feat, map_h, map_w)

            # Raw MLP score (just floor)
            raw = score_prediction(apply_floor(pred), gt)
            raw_scores.append(raw)

            # With post-processing (calibration + temps)
            pp = score_with_postprocessing(pred, gt, test_round_feats,
                                           test_detail, s)
            pp_scores.append(pp)

        raw_avg = np.mean(raw_scores)
        pp_avg = np.mean(pp_scores)
        loro_mlp[test_rnum] = raw_avg
        loro_mlp_pp[test_rnum] = pp_avg
        print(f"  R{test_rnum}: raw={raw_avg:.2f}, +pp={pp_avg:.2f}  seeds: {[f'{s:.1f}' for s in pp_scores]}")

    mlp_avg = np.mean(list(loro_mlp.values()))
    mlp_pp_avg = np.mean(list(loro_mlp_pp.values()))
    print(f"\n  LORO MLP raw:  {mlp_avg:.2f}")
    print(f"  LORO MLP +pp:  {mlp_pp_avg:.2f}")
    print(f"  (Compare: GBM LORO ~79.67 raw, ~80.66 +pp)")

    # ── Test ensemble: GBM + MLP blend ──
    print("\n=== TESTING GBM+MLP ENSEMBLE ===")
    from astar.model import load_spatial_model
    gbm = load_spatial_model()
    if gbm is not None:
        for blend_w in [0.3, 0.5, 0.7]:
            print(f"\n  Blend: {1-blend_w:.0%} MLP + {blend_w:.0%} GBM")
            # Need to retrain MLP on all data for ensemble test (or reuse LORO MLPs)
            # For now, train on all data
            X_all, Y_all, W_all = build_training_data(ROUND_IDS)
            mlp_full = train_mlp(X_all, Y_all, W_all, verbose=False)

            blend_scores = {}
            for test_rnum, (test_rid, test_detail, test_gts) in all_data.items():
                map_w = test_detail.get("map_width", 40)
                map_h = test_detail.get("map_height", 40)
                test_round_feats = compute_gt_round_features(test_detail, test_gts)
                scores = []
                for s, gt in enumerate(test_gts):
                    init_grid = test_detail["initial_states"][s]["grid"]
                    feat = compute_cell_features(init_grid, map_w, map_h,
                                                 round_features=test_round_feats)
                    # MLP prediction
                    mlp_pred = predict_with_mlp(mlp_full, feat, map_h, map_w)
                    # GBM prediction
                    flat_feat = feat.reshape(-1, feat.shape[-1])
                    gbm_pred = gbm.predict(flat_feat).reshape(map_h, map_w, NUM_CLASSES)
                    gbm_pred = np.maximum(gbm_pred, 1e-10)
                    gbm_pred = gbm_pred / gbm_pred.sum(axis=-1, keepdims=True)
                    # Blend
                    blended = blend_w * gbm_pred + (1 - blend_w) * mlp_pred
                    blended = blended / blended.sum(axis=-1, keepdims=True)
                    sc = score_with_postprocessing(blended, gt, test_round_feats,
                                                   test_detail, s)
                    scores.append(sc)
                blend_scores[test_rnum] = np.mean(scores)
            avg = np.mean(list(blend_scores.values()))
            print(f"    In-sample avg: {avg:.2f}  per-round: {blend_scores}")
    else:
        print("  GBM not loaded, skipping ensemble test")

    # ── Train final MLP on all data ──
    print("\n=== TRAINING FINAL MLP ON ALL DATA ===")
    X_all, Y_all, W_all = build_training_data(ROUND_IDS)
    print(f"  {X_all.shape[0]} samples, {X_all.shape[1]} features")
    final_mlp = train_mlp(X_all, Y_all, W_all, n_epochs=300, verbose=True)

    # Save
    mlp_path = DATA_DIR / "mlp_model.pt"
    torch.save({
        "state_dict": final_mlp.state_dict(),
        "n_features": X_all.shape[1],
        "n_classes": NUM_CLASSES,
        "hidden": [128, 64],
    }, mlp_path)
    print(f"  Saved to {mlp_path}")

    # In-sample scores
    print("\n=== IN-SAMPLE SCORES (MLP) ===")
    for rnum, (rid, detail, gts) in sorted(all_data.items()):
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        scores = []
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h,
                                         round_features=round_feats)
            pred = predict_with_mlp(final_mlp, feat, map_h, map_w)
            sc = score_with_postprocessing(pred, gt, round_feats, detail, s)
            scores.append(sc)
        print(f"  R{rnum}: avg={np.mean(scores):.2f}  seeds: {[f'{s:.1f}' for s in scores]}")


if __name__ == "__main__":
    main()
