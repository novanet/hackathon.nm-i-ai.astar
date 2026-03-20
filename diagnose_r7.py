"""Diagnose R7 underperformance: compare prediction vs ground truth per-class."""
import json, numpy as np, warnings
from pathlib import Path
from astar.model import build_prediction, CLASS_NAMES
from astar.submit import score_prediction

warnings.filterwarnings('ignore')

R7_ID = '36e581f1-73f8-453f-ab98-cbe3052b701b'
R6_ID = 'ae78003a-4efe-425a-881a-d16a39bca0ad'

def analyze_round(round_id: str, label: str):
    data_dir = Path('data') / f'round_{round_id}'
    detail_files = sorted(data_dir.glob('round_detail_*.json'))
    detail = json.loads(detail_files[-1].read_text(encoding='utf-8'))
    n_seeds = len(detail.get('initial_states', []))
    map_w = detail.get('map_width', 40)
    map_h = detail.get('map_height', 40)

    print(f'\n{"="*60}')
    print(f'{label}: round {round_id[:8]}... ({n_seeds} seeds, {map_w}x{map_h})')
    print(f'{"="*60}')

    all_scores = []
    class_kl_sums = np.zeros(6)
    class_counts = np.zeros(6)
    class_pred_means = np.zeros(6)
    class_gt_means = np.zeros(6)

    # Per-class confusion: what did we predict vs what actually happened?
    # For each GT class, what was our average predicted probability for that class?
    confidence_when_correct = [[] for _ in range(6)]
    # For each GT class, what did we predict instead? (average predicted distribution)
    avg_pred_given_gt = np.zeros((6, 6))
    gt_class_counts = np.zeros(6)

    for seed_idx in range(n_seeds):
        gt_path = data_dir / f'ground_truth_s{seed_idx}.json'
        if not gt_path.exists():
            print(f'  Seed {seed_idx}: no GT')
            continue

        analysis = json.loads(gt_path.read_text(encoding='utf-8'))
        gt = np.array(analysis['ground_truth'], dtype=np.float64)
        pred = build_prediction(round_id, detail, seed_idx, map_w, map_h)
        score = score_prediction(pred, gt)
        all_scores.append(score)

        # Per-cell analysis
        gt_classes = np.argmax(gt, axis=-1)  # H x W
        for c in range(6):
            mask = (gt_classes == c)
            n = mask.sum()
            if n > 0:
                class_counts[c] += n
                gt_class_counts[c] += n
                # Our predicted prob for the correct class
                pred_correct = pred[mask, c]
                confidence_when_correct[c].extend(pred_correct.tolist())
                # Average predicted distribution when GT is class c
                avg_pred_given_gt[c] += pred[mask].sum(axis=0)

                # KL per cell for this class
                gt_probs = gt[mask]
                pred_probs = pred[mask]
                kl_cells = np.sum(gt_probs * np.log(gt_probs / np.clip(pred_probs, 1e-10, None)), axis=-1)
                class_kl_sums[c] += kl_cells.sum()

        print(f'  Seed {seed_idx}: score={score:.2f}')

    if not all_scores:
        return

    avg = np.mean(all_scores)
    print(f'\n  Average score: {avg:.2f}')

    # Normalize
    for c in range(6):
        if gt_class_counts[c] > 0:
            avg_pred_given_gt[c] /= gt_class_counts[c]

    print(f'\n  Per-class analysis:')
    print(f'  {"Class":<12} {"GT count":>8} {"Avg KL":>8} {"Conf":>6} {"Pred dist when GT=class":>40}')
    for c in range(6):
        if class_counts[c] > 0:
            avg_kl = class_kl_sums[c] / class_counts[c]
            avg_conf = np.mean(confidence_when_correct[c]) if confidence_when_correct[c] else 0
            dist_str = ' '.join(f'{avg_pred_given_gt[c, j]:.3f}' for j in range(6))
            print(f'  {CLASS_NAMES[c]:<12} {int(class_counts[c]):>8} {avg_kl:>8.4f} {avg_conf:>6.3f}  [{dist_str}]')

    # Which class contributes most to total KL?
    total_kl = class_kl_sums.sum()
    print(f'\n  KL contribution by class:')
    for c in range(6):
        pct = 100 * class_kl_sums[c] / total_kl if total_kl > 0 else 0
        print(f'    {CLASS_NAMES[c]:<12}: {pct:5.1f}%  (total KL={class_kl_sums[c]:.2f})')

    # Initial state composition
    init_class_counts = np.zeros(6)
    for seed_idx in range(n_seeds):
        state = detail['initial_states'][seed_idx]
        # Parse initial grid - settlements
        for s in state.get('settlements', []):
            if s.get('alive'):
                init_class_counts[1] += 1  # Settlement
        for p in state.get('ports', []):
            init_class_counts[2] += 1  # Port
        for r_item in state.get('ruins', []):
            init_class_counts[3] += 1  # Ruin
    print(f'\n  Initial state: {int(init_class_counts[1])} settlements, {int(init_class_counts[2])} ports, {int(init_class_counts[3])} ruins (across all seeds)')

    return avg

print('\nComparing R6 (good: 77.9) vs R7 (bad: 56.9)')
print('Model: V3 LightGBM with round features')

r6_score = analyze_round(R6_ID, 'R6 (good)')
r7_score = analyze_round(R7_ID, 'R7 (bad)')

if r6_score and r7_score:
    print(f'\n\nDelta: R6={r6_score:.2f} vs R7={r7_score:.2f} (diff={r6_score - r7_score:.2f})')
