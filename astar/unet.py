"""
U-Net model for Astar Island spatial prediction.

Input: 40×40 grid with 19 channels:
  - 6 one-hot terrain channels
  - 5 distance features (settlement, forest, port, sett_count_r5, edge)
  - 8 round features (E→E, S→S, F→F, E→S, sett_density, food, wealth, defense)

Output: 40×40×6 probability tensor.
Loss: entropy-weighted KL divergence (matches competition scorer).

Does NOT modify model.py — designed for standalone training and later ensemble blending.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve

from astar.replay import TERRAIN_TO_CLASS, NUM_CLASSES

N_SPATIAL_CHANNELS = 11  # 6 one-hot + 5 distance
N_ROUND_CHANNELS = 8
N_INPUT_CHANNELS = N_SPATIAL_CHANNELS + N_ROUND_CHANNELS  # 19


# ── Model ──────────────────────────────────────────────────────────────────


class DoubleConv(nn.Module):
    """Two 3×3 convolutions with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: affine transform conditioned on round features."""

    def __init__(self, n_round_feats: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(n_round_feats, n_channels)
        self.beta = nn.Linear(n_round_feats, n_channels)
        # Initialize to identity transform
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, round_feats: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), round_feats: (B, n_rf)
        gamma = self.gamma(round_feats).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(round_feats).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class AttentionGate(nn.Module):
    """Lightweight attention gate for skip connections."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.psi(g + s)
        return skip * attn


class UNet(nn.Module):
    """
    U-Net for 40×40 terrain prediction.

    V2: 3 encoder levels (40→20→10→5), FiLM conditioning, attention gates.
    V1 compatibility: base_channels=32, n_levels=2, use_film=False, use_attention=False.
    """

    def __init__(self, in_channels: int = N_INPUT_CHANNELS, n_classes: int = NUM_CLASSES,
                 base_channels: int = 32, dropout: float = 0.1,
                 n_levels: int = 2, use_film: bool = False,
                 n_round_feats: int = N_ROUND_CHANNELS, use_attention: bool = False):
        super().__init__()
        self.n_levels = n_levels
        self.use_film = use_film
        self.use_attention = use_attention

        # If using FiLM, spatial input excludes broadcast round features
        spatial_in = N_SPATIAL_CHANNELS if use_film else in_channels

        # Build channel list: base_channels * [1, 2, 4, 8, ...] up to n_levels+1
        channels = [base_channels * (2 ** i) for i in range(n_levels + 1)]
        # channels[0]=enc1, channels[1]=enc2, ..., channels[-1]=bottleneck

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        enc_in = spatial_in
        for i in range(n_levels):
            drop = dropout if i > 0 else 0.0
            self.encoders.append(DoubleConv(enc_in, channels[i], dropout=drop))
            self.pools.append(nn.MaxPool2d(2))
            enc_in = channels[i]

        # Bottleneck
        self.bottleneck = DoubleConv(channels[n_levels - 1], channels[n_levels], dropout=dropout)

        # FiLM blocks (one per encoder + bottleneck)
        if use_film:
            self.film_blocks = nn.ModuleList()
            for i in range(n_levels):
                self.film_blocks.append(FiLMBlock(n_round_feats, channels[i]))
            self.film_blocks.append(FiLMBlock(n_round_feats, channels[n_levels]))  # bottleneck

        # Decoder (reverse order)
        self.ups = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(n_levels - 1, -1, -1):
            self.ups.append(nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2))
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(channels[i], channels[i], channels[i] // 2 or 1))
            else:
                self.attention_gates.append(nn.Identity())
            drop = dropout if i > 0 else 0.0
            self.decoders.append(DoubleConv(channels[i] + channels[i], channels[i], dropout=drop))

        # Head
        self.head = nn.Conv2d(channels[0], n_classes, 1)

    def forward(self, x: torch.Tensor, round_feats: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass. Returns log-softmax probabilities (N, n_classes, H, W)."""
        # Encoder path
        skips = []
        h = x
        for i in range(self.n_levels):
            h = self.encoders[i](h)
            if self.use_film and round_feats is not None:
                h = self.film_blocks[i](h, round_feats)
            skips.append(h)
            h = self.pools[i](h)

        # Bottleneck
        h = self.bottleneck(h)
        if self.use_film and round_feats is not None:
            h = self.film_blocks[self.n_levels](h, round_feats)

        # Decoder path
        for i in range(self.n_levels):
            h = self.ups[i](h)
            skip = skips[self.n_levels - 1 - i]
            if self.use_attention:
                skip = self.attention_gates[i](h, skip)
            h = self.decoders[i](torch.cat([h, skip], dim=1))

        logits = self.head(h)
        return F.log_softmax(logits, dim=1)


# ── Input Feature Computation ──────────────────────────────────────────────


def compute_unet_input(init_grid: list[list[int]], map_w: int, map_h: int,
                       round_features: np.ndarray,
                       film_mode: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute input image for the U-Net.

    If film_mode=False (default/V1): returns (19, H, W) with broadcast round features.
    If film_mode=True (V2+FiLM): returns ((11, H, W), (8,)) tuple — spatial channels
    and round features separately. Round features are passed via FiLM conditioning.

    Spatial Channels (0-10):
      0-5:  one-hot terrain class
      6:    distance to nearest settlement (normalized)
      7:    distance to nearest forest (normalized)
      8:    distance to nearest port (normalized)
      9:    settlement count within radius 5 (normalized)
      10:   distance to nearest map edge (normalized)

    Round Channels (11-18, only in non-FiLM mode):
      11-18: round features broadcast spatially
    """
    cls_grid = np.array([[TERRAIN_TO_CLASS.get(init_grid[y][x], 0)
                          for x in range(map_w)] for y in range(map_h)])

    # Always compute spatial channels (11)
    n_spatial = N_SPATIAL_CHANNELS
    n_ch = n_spatial if film_mode else (n_spatial + len(round_features))
    img = np.zeros((n_ch, map_h, map_w), dtype=np.float32)

    # One-hot encoding (channels 0-5)
    for c in range(NUM_CLASSES):
        img[c] = (cls_grid == c).astype(np.float32)

    # Distance features (channels 6-8)
    max_dist = float(map_w + map_h)
    sett_mask = (cls_grid == 1) | (cls_grid == 2)
    forest_mask = cls_grid == 4
    port_mask = cls_grid == 2

    img[6] = (distance_transform_edt(~sett_mask) if sett_mask.any()
              else np.full((map_h, map_w), max_dist, dtype=np.float32)) / max_dist
    img[7] = (distance_transform_edt(~forest_mask) if forest_mask.any()
              else np.full((map_h, map_w), max_dist, dtype=np.float32)) / max_dist
    img[8] = (distance_transform_edt(~port_mask) if port_mask.any()
              else np.full((map_h, map_w), max_dist, dtype=np.float32)) / max_dist

    # Settlement count within radius 5 (Chebyshev distance ≤ 5 → 11×11 window)
    kernel = np.ones((11, 11), dtype=np.float32)
    sett_count = fftconvolve(sett_mask.astype(np.float32), kernel, mode='same')
    max_count = max(float(sett_count.max()), 1.0)
    img[9] = sett_count / max_count

    # Edge distance (channel 10)
    ys = np.arange(map_h, dtype=np.float32).reshape(-1, 1)
    xs = np.arange(map_w, dtype=np.float32).reshape(1, -1)
    edge_dist = np.minimum(np.minimum(ys, xs),
                           np.minimum(map_h - 1 - ys, map_w - 1 - xs))
    img[10] = edge_dist / (min(map_h, map_w) / 2)

    # Round features broadcast (channels 11-18) — only in non-FiLM mode
    if not film_mode:
        for i, val in enumerate(round_features):
            img[N_SPATIAL_CHANNELS + i] = val
        return img
    else:
        return img, round_features.astype(np.float32)


# ── D4 Augmentation ───────────────────────────────────────────────────────


def d4_augment(img: np.ndarray, gt: np.ndarray, round_feats: np.ndarray | None = None):
    """
    Generate D4 group augmentations (4 rotations × 2 flips = 8 variants).

    For a square grid, all input features (one-hot, distance, edge_dist,
    sett_count, round_features) are exactly D4-equivariant because:
    - The grid is square → boundary conditions are symmetric
    - EDT uses Euclidean metric → rotation-invariant
    - Convolution kernel (ones) is D4-invariant
    - Zero-padding boundaries are D4-symmetric

    Args:
        img: (C, H, W) input feature image
        gt:  (H, W, 6) ground truth probabilities
        round_feats: optional (8,) round features for FiLM mode

    Yields:
        (augmented_img, augmented_gt) tuples (8 total), or
        (augmented_img, augmented_gt, round_feats) tuples if round_feats provided.
    """
    for k in range(4):
        ri = np.rot90(img, k=k, axes=(1, 2)).copy()
        rg = np.rot90(gt, k=k, axes=(0, 1)).copy()
        if round_feats is not None:
            yield ri, rg, round_feats
            yield np.flip(ri, axis=2).copy(), np.flip(rg, axis=1).copy(), round_feats
        else:
            yield ri, rg
            yield np.flip(ri, axis=2).copy(), np.flip(rg, axis=1).copy()


# ── Loss Function ──────────────────────────────────────────────────────────


def entropy_weighted_kl_loss(log_pred: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    """
    Entropy-weighted KL divergence matching the competition scorer.

    Args:
        log_pred: (N, 6, H, W) log-probabilities from the model
        target:   (N, 6, H, W) ground truth probabilities

    Returns:
        Scalar loss (weighted mean across all pixels and batch).
    """
    eps = 1e-10
    target_c = target.clamp(min=eps)

    # KL per pixel: sum over classes → (N, H, W)
    kl = (target_c * (target_c.log() - log_pred)).sum(dim=1)

    # Entropy per pixel as weight → (N, H, W)
    entropy = -(target_c * target_c.log()).sum(dim=1)

    total_entropy = entropy.sum()
    if total_entropy < eps:
        return kl.mean()

    return (entropy * kl).sum() / total_entropy


# ── Inference ──────────────────────────────────────────────────────────────


def predict_unet(model: UNet, init_grid: list[list[int]],
                 map_w: int, map_h: int,
                 round_features: np.ndarray,
                 device: torch.device | None = None) -> np.ndarray:
    """
    Run U-Net inference on a single seed.

    Returns: (H, W, 6) probability tensor (numpy).
    """
    if device is None:
        device = next(model.parameters()).device

    use_film = getattr(model, 'use_film', False)
    if use_film:
        spatial_img, rf = compute_unet_input(init_grid, map_w, map_h, round_features, film_mode=True)
        img_t = torch.tensor(spatial_img, dtype=torch.float32, device=device).unsqueeze(0)
        rf_t = torch.tensor(rf, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        img = compute_unet_input(init_grid, map_w, map_h, round_features)
        img_t = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
        rf_t = None

    model.eval()
    with torch.no_grad():
        log_pred = model(img_t, rf_t)  # (1, 6, H, W)
        pred = torch.exp(log_pred).squeeze(0)  # (6, H, W)

    # Convert to (H, W, 6) numpy
    pred_np = pred.permute(1, 2, 0).cpu().numpy()
    pred_np = np.maximum(pred_np, 1e-10)
    pred_np = pred_np / pred_np.sum(axis=-1, keepdims=True)
    return pred_np


def predict_unet_with_tta(model: UNet, init_grid: list[list[int]],
                          map_w: int, map_h: int,
                          round_features: np.ndarray,
                          device: torch.device | None = None) -> np.ndarray:
    """
    Test-Time Augmentation: average predictions over all 8 D4 transforms.

    Returns: (H, W, 6) probability tensor (numpy).
    """
    if device is None:
        device = next(model.parameters()).device

    use_film = getattr(model, 'use_film', False)
    if use_film:
        spatial_img, rf = compute_unet_input(init_grid, map_w, map_h, round_features, film_mode=True)
        rf_t = torch.tensor(rf, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        spatial_img = compute_unet_input(init_grid, map_w, map_h, round_features)
        rf_t = None

    # Dummy GT for d4_augment (not used for prediction)
    dummy_gt = np.zeros((map_h, map_w, NUM_CLASSES), dtype=np.float32)

    preds = []
    model.eval()
    with torch.no_grad():
        for k, (aug_img, _) in enumerate(d4_augment(spatial_img, dummy_gt)):
            img_t = torch.tensor(aug_img, dtype=torch.float32, device=device).unsqueeze(0)
            log_pred = model(img_t, rf_t)
            pred = torch.exp(log_pred).squeeze(0)  # (6, H, W)
            pred_np = pred.permute(1, 2, 0).cpu().numpy()  # (H, W, 6)

            # Invert the D4 transform
            rot_k = k // 2
            is_flipped = k % 2 == 1
            if is_flipped:
                pred_np = np.flip(pred_np, axis=1).copy()
            if rot_k > 0:
                pred_np = np.rot90(pred_np, k=-rot_k, axes=(0, 1)).copy()

            preds.append(pred_np)

    avg_pred = np.mean(preds, axis=0)
    avg_pred = np.maximum(avg_pred, 1e-10)
    avg_pred = avg_pred / avg_pred.sum(axis=-1, keepdims=True)
    return avg_pred
