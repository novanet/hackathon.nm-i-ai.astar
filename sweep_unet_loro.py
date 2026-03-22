"""LORO sweep of U-Net blend weight in the full GBM+MLP+UNet pipeline.
Trains fresh GBM + MLP per fold (like train_spatial.py LORO), then blends with U-Net.
This gives TRUE out-of-sample scores since the test round is held out from GBM+MLP training.
U-Net was trained on all 14 rounds so it's NOT held out — this biases U-Net favorably.
"""
import json
import sys
import warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from astar.model import (compute_cell_features, apply_floor, PROB_FLOOR,
                         entropy_bucket_temperature_scale, CALIBRATION_FACTORS)
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
    11: "324fde07-1670-4202-b199-7aa92ecb40ee",
    12: "795bfb1f-54bd-4f39-a526-9868b36f7ebd",
    13: "7b4bda99-6165-4221-97cc-27880f5e6d95",
    14: "d0a2c894-2162-4d49-86cf-435b9013f3b8",
    15: "cc5442dd-bc5d-418b-911b-7eb960cb0390",
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


def compute_gt_round_features(detail, gts):
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
        gt_trans[0, 0], gt_trans[1, 1], gt_trans[4, 4], gt_trans[0, 1],
        sett_density,
        0.3 + 0.7 * ss_rate,
        gt_trans[0, 1] * 0.3,
        1.0 - gt_trans[1, 0],
    ], dtype=np.float64)


def build_training_data(round_ids):
    all_X, all_Y, all_W = [], [], []
    for rnum, rid in sorted(round_ids.items()):
        detail, gts = load_round_data(rid)
        if not gts:
            continue
        map_w = detail.get("map_width", 40)
        map_h = detail.get("map_height", 40)
        round_feats = compute_gt_round_features(detail, gts)
        for s, gt in enumerate(gts):
            init_grid = detail["initial_states"][s]["grid"]
            feat = compute_cell_features(init_grid, map_w, map_h, round_features=round_feats)
            flat_feat = feat.reshape(-1, feat.shape[-1])
            flat_gt = gt.reshape(-1, NUM_CLASSES)
            all_X.append(flat_feat)
            all_Y.append(flat_gt)
            # Entropy weights
            ent = -np.sum(flat_gt * np.log(np.maximum(flat_gt, 1e-10)), axis=1)
            all_W.append(np.maximum(ent, 0.01) ** ENTROPY_WEIGHT_POWER)
    return np.vstack(all_X), np.vstack(all_Y), np.concatenate(all_W)


# Load U-Net once (trained on all rounds — not held out)
from astar.unet import UNet, predict_unet_with_tta
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_path = DATA_DIR / "unet_model.pt"
unet_ckpt = torch.load(unet_path, map_location=device, weights_only=True)
unet_model = UNet(
    in_channels=unet_ckpt["in_channels"],
    n_classes=unet_ckpt["n_classes"],
    base_channels=unet_ckpt.get("base_channels", 32),
    dropout=unet_ckpt.get("dropout", 0.1),
).to(device)
unet_model.load_state_dict(unet_ckpt["state_dict"])
unet_model.eval()
print("[INFO] U-Net loaded (NOTE: trained on all 14 rounds, biases LORO favorably)")

# Load all round data
print("Loading data...")
all_data = {}
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if gts:
        all_data[rnum] = (rid, detail, gts)

UNET_WEIGHTS = [0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90, 1.00]

# Results: {weight: {rnum: score}}
results = {w: {} for w in UNET_WEIGHTS}

print(f"\n=== LORO BLEND SWEEP ({len(all_data)} rounds) ===", flush=True)
for test_rnum in sorted(all_data.keys()):
    print(f"  Training fold {test_rnum}...", end="", flush=True)
    # Train GBM on all other rounds
    train_ids = {r: ROUND_IDS[r] for r in all_data if r != test_rnum}
    X_train, Y_train, W_train = build_training_data(train_ids)

    lgb_model = MultiOutputRegressor(
        lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=50, subsample=0.7,
            colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=1.0,
            verbosity=-1,
        ), n_jobs=1)
    lgb_model.fit(X_train, Y_train, sample_weight=W_train)

    xgb_model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=1.0, reg_lambda=1.0, verbosity=0,
        ), n_jobs=1)
    xgb_model.fit(X_train, Y_train, sample_weight=W_train)

    # MLP
    from train_mlp import KLDivMLP
    n_feat = X_train.shape[1]
    mlp = KLDivMLP(n_features=n_feat, n_classes=NUM_CLASSES, hidden=[256, 128, 64]).to(device)
    mlp_X = torch.tensor(X_train, dtype=torch.float32, device=device)
    mlp_Y = torch.tensor(Y_train, dtype=torch.float32, device=device)
    mlp_W = torch.tensor(W_train, dtype=torch.float32, device=device)
    # Quick MLP training (fewer epochs for speed)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    n = len(mlp_X)
    idx = torch.randperm(n)
    val_n = max(1, n // 10)
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    best_val, patience, best_state = 1e9, 0, None
    for epoch in range(150):
        mlp.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(perm), 4096):
            batch = perm[i:i+4096]
            log_pred = mlp(mlp_X[batch])
            target = torch.clamp(mlp_Y[batch], min=1e-8)
            kl = torch.sum(target * (torch.log(target) - log_pred), dim=-1)
            loss = (kl * mlp_W[batch]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        mlp.eval()
        with torch.no_grad():
            vlog = mlp(mlp_X[val_idx])
            vt = torch.clamp(mlp_Y[val_idx], min=1e-8)
            vkl = torch.sum(vt * (torch.log(vt) - vlog), dim=-1).mean().item()
        if vkl < best_val:
            best_val = vkl
            patience = 0
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        else:
            patience += 1
            if patience >= 20:
                break
    if best_state:
        mlp.load_state_dict(best_state)
    mlp.eval()

    # Test on held-out round
    test_rid, test_detail, test_gts = all_data[test_rnum]
    map_w = test_detail.get("map_width", 40)
    map_h = test_detail.get("map_height", 40)
    test_round_feats = compute_gt_round_features(test_detail, test_gts)

    scores_by_weight = {w: [] for w in UNET_WEIGHTS}

    for s, gt in enumerate(test_gts):
        init_grid = test_detail["initial_states"][s]["grid"]
        feat = compute_cell_features(init_grid, map_w, map_h, round_features=test_round_feats)
        flat_feat = feat.reshape(-1, feat.shape[-1])

        # GBM blend
        lgb_pred = lgb_model.predict(flat_feat)
        xgb_pred = xgb_model.predict(flat_feat)
        gbm_pred = 0.7 * lgb_pred + 0.3 * xgb_pred

        # MLP
        with torch.no_grad():
            mlp_pred = torch.exp(mlp(torch.tensor(flat_feat, dtype=torch.float32, device=device))).cpu().numpy()

        # GBM+MLP blend (50/50)
        base_pred = 0.5 * gbm_pred + 0.5 * mlp_pred
        base_pred = base_pred.reshape(map_h, map_w, NUM_CLASSES)
        base_pred = np.maximum(base_pred, 1e-10)
        base_pred = base_pred / base_pred.sum(axis=-1, keepdims=True)

        # U-Net prediction
        unet_pred = predict_unet_with_tta(unet_model, init_grid, map_w, map_h, test_round_feats)
        unet_pred = np.maximum(unet_pred, 1e-10)
        unet_pred = unet_pred / unet_pred.sum(axis=-1, keepdims=True)

        for w in UNET_WEIGHTS:
            if w == 0:
                pred = base_pred.copy()
            else:
                pred = (1 - w) * base_pred + w * unet_pred
                pred = pred / pred.sum(axis=-1, keepdims=True)
            # Apply calibration + M5 bucket temps (match live pipeline)
            pred = pred * CALIBRATION_FACTORS[np.newaxis, np.newaxis, :]
            pred = pred / pred.sum(axis=-1, keepdims=True)
            pred = entropy_bucket_temperature_scale(pred)
            pred = apply_floor(pred)
            scores_by_weight[w].append(score_prediction(pred, gt))

    line = f" done -> R{test_rnum:2d}:"
    for w in UNET_WEIGHTS:
        avg = np.mean(scores_by_weight[w])
        results[w][test_rnum] = avg
        line += f"  {w:.0%}={avg:.1f}"
    print(line)

print("\n=== LORO SUMMARY ===")
print(f"{'Weight':>8}", end="")
for w in UNET_WEIGHTS:
    print(f"  {w:.0%}", end="")
print()

for rnum in sorted(all_data.keys()):
    print(f"  R{rnum:2d}  ", end="")
    for w in UNET_WEIGHTS:
        print(f" {results[w].get(rnum, 0):5.1f}", end="")
    print()

print(f"\n{'AVG':>6}  ", end="")
best_w = None
best_avg = -1
for w in UNET_WEIGHTS:
    avg = np.mean(list(results[w].values()))
    if avg > best_avg:
        best_avg = avg
        best_w = w
    marker = " *" if w == best_w else ""
    print(f" {avg:5.2f}{marker}", end="")
print()
print(f"\nBest LORO weight: {best_w:.0%} U-Net (avg={best_avg:.2f})")
