"""R8 post-round analysis: download GT, compare predictions, extract learnings."""
import json
import os
import numpy as np
from pathlib import Path
from scipy.special import rel_entr

from astar.model import build_prediction, PROB_FLOOR, PER_CLASS_TEMPS, TEMPERATURE
from astar.submit import score_prediction

R8_ID = "c5cdf100-a876-4fb7-b5d8-757162c97989"
DATA_DIR = Path("data") / f"round_{R8_ID}"
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
NC = len(CLASS_NAMES)
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

HAS_TOKEN = bool(os.environ.get("ASTAR_TOKEN"))

def api_request(method: str, path: str):
    if not HAS_TOKEN:
        return None
    from astar.client import _request
    return _request(method, path)

def api_get_analysis(round_id: str, seed_index: int):
    if not HAS_TOKEN:
        return None
    from astar.client import get_analysis
    return get_analysis(round_id, seed_index)


def get_cls_grid(initial_states, seed_idx):
    """Convert raw terrain grid to 0-5 class grid."""
    raw = np.array(initial_states[seed_idx]["grid"])
    out = np.zeros_like(raw)
    for k, v in TERRAIN_TO_CLASS.items():
        out[raw == k] = v
    return out


def main():
    print("=== CONFIGURATION ===")
    print(f"PROB_FLOOR: {PROB_FLOOR}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print(f"PER_CLASS_TEMPS: {PER_CLASS_TEMPS}")
    print()

    # 1. Official score + status
    print("=== R8 OFFICIAL SCORE ===")
    my = api_request("GET", "/my-rounds")
    r8 = None
    for r in my:
        if r.get("round_number") == 8 or r.get("round_id") == R8_ID:
            r8 = r
            break
    if r8:
        print(f"  Status: {r8.get('status')}")
        print(f"  Round score: {r8.get('round_score')}")
        print(f"  Seed scores: {r8.get('seed_scores')}")
        print(f"  Rank: {r8.get('rank')}/{r8.get('total_teams')}")
        rs = r8.get("round_score") or 0
        weighted = rs * 1.05**8
        print(f"  Weighted (1.05^8): {weighted:.2f}")
        print(f"  Queries used: {r8.get('queries_used')}/{r8.get('query_budget')}")
    else:
        print("  R8 not found in my-rounds!")
        for r in my:
            print(f"  round {r.get('round_number')}: {str(r.get('round_id',''))[:12]}... score={r.get('round_score')}")
    print()

    # 2. Leaderboard
    print("=== LEADERBOARD ===")
    lb = api_request("GET", "/leaderboard")
    for i, e in enumerate(lb[:15]):
        marker = " <-- US" if "novanet" in str(e).lower() else ""
        print(f"  {i+1:2d}. {e.get('team_name', '?'):30s} {e.get('score', 0):7.2f}{marker}")
    for i, e in enumerate(lb):
        if "novanet" in str(e).lower():
            print(f"  Our rank: {i+1}/{len(lb)}, score: {e.get('score', 0):.2f}")
            break
    print()

    # 3. Download GT and backtest
    print("=== DOWNLOADING R8 GROUND TRUTH ===")
    detail_files = sorted(DATA_DIR.glob("round_detail_*.json"))
    detail = json.loads(detail_files[len(detail_files) - 1].read_text(encoding="utf-8"))
    n_seeds = len(detail.get("initial_states", []))
    print(f"  Map size: {detail.get('map_width')}x{detail.get('map_height')}")
    print(f"  Seeds: {n_seeds}")
    print(f"  Sim years: {detail.get('simulation_years')}")
    print()

    backtest_scores = []
    all_pred_probs = []
    all_gt_probs = []
    initial_states = detail["initial_states"]

    for seed_idx in range(n_seeds):
        try:
            gt_path = DATA_DIR / f"ground_truth_s{seed_idx}.json"
            if gt_path.exists():
                analysis = json.loads(gt_path.read_text(encoding="utf-8"))
            else:
                analysis = api_get_analysis(R8_ID, seed_idx)
                gt_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

            gt = np.array(analysis["ground_truth"], dtype=np.float64)
            pred = build_prediction(R8_ID, detail, seed_idx)
            sc = score_prediction(pred, gt)
            backtest_scores.append(sc)
            all_pred_probs.append(pred)
            all_gt_probs.append(gt)
            print(f"  Seed {seed_idx}: backtest={sc:.2f}, gt_shape={gt.shape}")
        except Exception as e:
            print(f"  Seed {seed_idx}: ERROR - {e}")

    if backtest_scores:
        avg = np.mean(backtest_scores)
        weighted = avg * 1.05**8
        print(f"  Backtest avg: {avg:.2f} (weighted: {weighted:.2f})")
    print()

    if not all_gt_probs:
        print("No GT data available, stopping.")
        return

    all_pred = np.stack(all_pred_probs)  # (5, H, W, 6)
    all_gt = np.stack(all_gt_probs)  # (5, H, W, 6)

    # 4. Per-class analysis
    print("=== PER-CLASS ANALYSIS (R8) ===")
    gt_argmax = np.argmax(all_gt, axis=3)  # (5, H, W)
    pred_argmax = np.argmax(all_pred, axis=3)

    for c in range(NC):
        gt_frac = (gt_argmax == c).mean()
        pred_frac = (pred_argmax == c).mean()
        mask = gt_argmax == c
        if mask.any():
            gt_prob_when_true = all_gt[mask][:, c].mean()
            pred_prob_when_true = all_pred[mask][:, c].mean()
        else:
            gt_prob_when_true = pred_prob_when_true = 0
        print(
            f"  {CLASS_NAMES[c]:12s}: GT%={gt_frac*100:5.1f}%  Pred%={pred_frac*100:5.1f}%  "
            f"GT_conf={gt_prob_when_true:.3f}  Pred_conf={pred_prob_when_true:.3f}"
        )
    print()

    # 5. Transition analysis (seed 0)
    print("=== R8 TRANSITION MATRIX (initial -> GT argmax, seed 0) ===")
    initial = get_cls_grid(initial_states, 0)
    gt_am = gt_argmax[0]
    H, W = initial.shape

    trans = np.zeros((NC, NC))
    for i in range(H):
        for j in range(W):
            src = initial[i, j]
            dst = gt_am[i, j]
            trans[src, dst] += 1

    header = "            " + "".join(f"{CLASS_NAMES[c]:>10s}" for c in range(NC))
    print(header)
    for src in range(NC):
        row_total = trans[src].sum()
        if row_total > 0:
            pcts = trans[src] / row_total * 100
            row_str = f"  {CLASS_NAMES[src]:>10s}  " + "".join(
                f"{pcts[c]:9.1f}%" for c in range(NC)
            )
            print(f"{row_str}  (n={int(row_total)})")
        else:
            print(f"  {CLASS_NAMES[src]:>10s}  (no initial cells)")
    print()

    # 5b. Probabilistic transition matrix (average GT probs by initial class)
    print("=== R8 PROBABILISTIC TRANSITIONS (avg GT probs by initial class, all seeds) ===")
    for src in range(NC):
        probs = []
        for s in range(n_seeds):
            init_s = get_cls_grid(initial_states, s)
            mask = init_s == src
            if mask.any():
                probs.append(all_gt[s][mask].mean(axis=0))
        if probs:
            avg_p = np.mean(probs, axis=0)
            row_str = f"  {CLASS_NAMES[src]:>10s} -> " + "  ".join(
                f"{CLASS_NAMES[c]}={avg_p[c]:.3f}" for c in range(NC)
            )
            print(row_str)
    print()

    # 6. Per-seed initial state composition
    print("=== R8 INITIAL STATE COMPOSITION ===")
    for seed_idx in range(n_seeds):
        initial_s = get_cls_grid(initial_states, seed_idx)
        counts = np.bincount(initial_s.flatten(), minlength=NC)
        total = counts.sum()
        print(
            f"  Seed {seed_idx}: "
            + "  ".join(f"{CLASS_NAMES[c]}={counts[c]/total*100:.1f}%" for c in range(NC))
        )
    print()

    # 7. Error analysis
    print("=== WORST CELLS ANALYSIS (R8 Seed 0) ===")
    gt0 = all_gt[0]
    pred0 = all_pred[0]
    initial0 = get_cls_grid(initial_states, 0)
    H, W, C = gt0.shape

    kl_per_cell = np.zeros((H, W))
    entropy_per_cell = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            p = gt0[i, j]
            q = np.clip(pred0[i, j], 1e-10, 1)
            kl = rel_entr(p, q).sum()
            ent = -(p * np.log(p + 1e-10)).sum()
            kl_per_cell[i, j] = kl
            entropy_per_cell[i, j] = ent

    flat_idx = np.argsort(kl_per_cell.flatten())[::-1][:10]
    print("  Top-10 worst cells (by KL):")
    for rank, fi in enumerate(flat_idx):
        r, c = fi // W, fi % W
        init_cls = CLASS_NAMES[initial0[r, c]]
        gt_cls = CLASS_NAMES[gt0[r, c].argmax()]
        pred_cls = CLASS_NAMES[pred0[r, c].argmax()]
        print(
            f"    #{rank+1}: ({r},{c}) init={init_cls} gt={gt_cls} pred={pred_cls} "
            f"KL={kl_per_cell[r,c]:.4f} ent={entropy_per_cell[r,c]:.3f}"
        )

    print()
    print("  KL contribution by initial class:")
    for cls_idx in range(NC):
        mask = initial0 == cls_idx
        if mask.any():
            total_kl = kl_per_cell[mask].sum()
            mean_kl = kl_per_cell[mask].mean()
            n = mask.sum()
            pct = total_kl / kl_per_cell.sum() * 100
            print(
                f"    {CLASS_NAMES[cls_idx]:12s}: total_KL={total_kl:.4f}  mean_KL={mean_kl:.6f}  n={n}  ({pct:.1f}% of loss)"
            )
    print()

    # 8. Entropy analysis
    print("=== ENTROPY ANALYSIS (R8) ===")
    for seed_idx in range(n_seeds):
        gt_s = all_gt[seed_idx]
        ent = -(gt_s * np.log(gt_s + 1e-10)).sum(axis=2)
        dynamic_mask = ent > 0.01
        print(
            f"  Seed {seed_idx}: mean_ent={ent.mean():.3f}  "
            f"dynamic_cells={dynamic_mask.sum()}/{ent.size} ({dynamic_mask.mean()*100:.1f}%)"
        )
    print()

    # 9. Compare official vs current model
    print("=== SCORE COMPARISON: CURRENT MODEL vs OFFICIAL ===")
    if r8:
        official = r8.get("round_score", 0) or 0
        current = np.mean(backtest_scores) if backtest_scores else 0
        diff = current - official
        label = "IMPROVED" if diff > 0 else "REGRESSED" if diff < 0 else "SAME"
        print(f"  Official score:  {official:.2f}")
        print(f"  Current model:   {current:.2f}")
        print(f"  Difference:      {diff:+.2f} ({label})")
        print("  Note: Official = submitted model; Current = model with today's optimizations.")
    print()

    # 10. Compare with LORO predictions
    print("=== R8 ROUND FEATURES (observation-derived vs GT) ===")
    # Extract round-level features from GT
    for seed_idx in range(n_seeds):
        init_s = get_cls_grid(initial_states, seed_idx)
        gt_s = all_gt[seed_idx]
        gt_am = np.argmax(gt_s, axis=2)

        # Compute observed transition rates
        ee_n = (init_s == 0).sum()
        ss_n = (init_s == 1).sum()
        ff_n = (init_s == 4).sum()

        if ee_n > 0:
            ee = (gt_am[init_s == 0] == 0).mean()
        else:
            ee = 0
        if ss_n > 0:
            ss = (gt_am[init_s == 1] == 1).mean()
        else:
            ss = 0
        if ff_n > 0:
            ff = (gt_am[init_s == 4] == 4).mean()
        else:
            ff = 0
        if ee_n > 0:
            es = (gt_am[init_s == 0] == 1).mean()
        else:
            es = 0

        total = init_s.size
        sett_density = ss_n / total

        if seed_idx == 0:
            print(
                f"  Seed {seed_idx} GT features: E->E={ee:.4f} S->S={ss:.4f} "
                f"F->F={ff:.4f} E->S={es:.4f} sett_density={sett_density:.4f}"
            )

    print()

    # 11. All rounds summary
    print("=== ALL ROUND SCORES ===")
    for r in sorted(my, key=lambda x: x.get("round_number", 0)):
        rn = r.get("round_number", 0)
        rs = r.get("round_score", 0) or 0
        w = rs * 1.05**rn if rs else 0
        st = r.get("status", "?")
        print(f"  R{rn}: score={rs:6.2f}  weighted={w:7.2f}  status={st}")


if __name__ == "__main__":
    main()
