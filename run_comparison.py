"""Quick LORO comparison of U-Net configs on representative rounds."""
import sys, time, gc
sys.path.insert(0, '.')
import numpy as np
import torch
from astar.unet import UNet, N_INPUT_CHANNELS, N_SPATIAL_CHANNELS, N_ROUND_CHANNELS, NUM_CLASSES
from astar.submit import score_prediction
from astar.model import apply_floor
from train_unet import (ROUND_IDS, load_round_data, compute_gt_round_features,
                        build_unet_dataset, train_unet_fold, score_unet_on_round)

device = torch.device('cpu')
print(f'Device: {device}', flush=True)

# Load ALL rounds including R18
print('Loading all rounds...', flush=True)
all_data = {}
for rnum, rid in sorted(ROUND_IDS.items()):
    detail, gts = load_round_data(rid)
    if gts:
        all_data[rnum] = (rid, detail, gts)
        print(f'  R{rnum}: {len(gts)} seeds', flush=True)
    else:
        print(f'  R{rnum}: NO DATA', flush=True)
print(f'Total: {len(all_data)} rounds', flush=True)

# Test on representative rounds
TEST_ROUNDS = [1, 5, 9, 12, 15, 17]
if 18 in all_data:
    TEST_ROUNDS.append(18)

CONFIGS = {
    'V1': dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=150, lr=1e-3, patience=25,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False),
    'V1_mix': dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=150, lr=1e-3, patience=25,
        use_mixup=True, label_smoothing=0.01, use_onecycle=True),
    'V2_48_2l': dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=48, dropout=0.1, n_levels=2,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=150, lr=1e-3, patience=25,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False),
    'V2_3l_32': dict(
        unet_config=dict(in_channels=N_INPUT_CHANNELS, n_classes=NUM_CLASSES,
                         base_channels=32, dropout=0.1, n_levels=3,
                         use_film=False, use_attention=False),
        film_mode=False, n_epochs=150, lr=1e-3, patience=25,
        use_mixup=False, label_smoothing=0.0, use_onecycle=False),
}

all_rounds = sorted(all_data.keys())
all_results = {}

for cname, cfg in CONFIGS.items():
    test_model = UNet(**cfg['unet_config'])
    n_params = sum(p.numel() for p in test_model.parameters())
    del test_model
    print(f'\n=== {cname} ({n_params:,} params) ===', flush=True)
    results = {}
    t0 = time.time()

    for test_rnum in TEST_ROUNDS:
        if test_rnum not in all_data:
            continue
        train_rounds = [r for r in all_rounds if r != test_rnum]

        if cfg['film_mode']:
            imgs, rf, tgts = build_unet_dataset(train_rounds, all_data, augment=True, film_mode=True)
        else:
            imgs, tgts = build_unet_dataset(train_rounds, all_data, augment=True, film_mode=False)
            rf = None

        model = train_unet_fold(
            imgs, tgts, device, n_epochs=cfg['n_epochs'], lr=cfg['lr'],
            patience=cfg['patience'], verbose=False,
            round_feats_arr=rf, unet_config=cfg['unet_config'],
            use_mixup=cfg.get('use_mixup', False),
            label_smoothing=cfg.get('label_smoothing', 0.0),
            use_onecycle=cfg.get('use_onecycle', False))

        _, det, gts = all_data[test_rnum]
        test_rf = compute_gt_round_features(det, gts)
        scores = score_unet_on_round(model, det, gts, test_rf, device, use_tta=True)
        avg = np.mean(scores)
        results[test_rnum] = avg
        print(f'  R{test_rnum}: {avg:.2f}', flush=True)

        del model, imgs, tgts
        if rf is not None:
            del rf
        gc.collect()

    elapsed = time.time() - t0
    all_results[cname] = results
    avg_score = np.mean(list(results.values()))
    print(f'  Subset avg: {avg_score:.2f}  [{elapsed:.0f}s]', flush=True)

# Summary table
sep = '=' * 70
print(f'\n{sep}', flush=True)
header = f'{"Config":>12}'
for r in TEST_ROUNDS:
    if r in all_data:
        header += f'  R{r:>2}'
header += '     AVG'
print(header)
for cname in CONFIGS:
    line = f'{cname:>12}'
    for r in TEST_ROUNDS:
        v = all_results.get(cname, {}).get(r, float('nan'))
        line += f' {v:5.1f}'
    avg_score = np.mean(list(all_results.get(cname, {}).values()))
    line += f'  {avg_score:5.1f}'
    print(line)
print('Done!', flush=True)
