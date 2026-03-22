"""Compare R14 initial grid to all historical rounds."""
import json, numpy as np
from pathlib import Path
from scipy import ndimage

ROUND_IDS = {
    1: '71451d74-be9f-471f-aacd-a41f3b68a9cd',
    2: '76909e29-f664-4b2f-b16b-61b7507277e9',
    3: 'f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb',
    4: '8e839974-b13b-407b-a5e7-fc749d877195',
    5: 'fd3c92ff-3178-4dc9-8d9b-acf389b3982b',
    6: 'ae78003a-4efe-425a-881a-d16a39bca0ad',
    7: '36e581f1-73f8-453f-ab98-cbe3052b701b',
    8: 'c5cdf100-a876-4fb7-b5d8-757162c97989',
    9: '2a341ace-0f57-4309-9b89-e59fe0f09179',
    10: '75e625c3-60cb-4392-af3e-c86a98bde8c2',
    11: '324fde07-1670-4202-b199-7aa92ecb40ee',
    12: '795bfb1f-54bd-4f39-a526-9868b36f7ebd',
    13: '7b4bda99-6165-4221-97cc-27880f5e6d95',
    14: 'd0a2c894-2162-4d49-86cf-435b9013f3b8',
}

GT_SS = {1: 0.410, 2: 0.381, 3: 0.018, 4: 0.235, 5: 0.327, 6: 0.395, 7: 0.605, 8: 0.067, 9: 0.275, 10: 0.058, 11: 0.495, 12: 0.595, 13: 0.260}
LORO = {1: 84.93, 2: 89.55, 3: 88.67, 4: 90.90, 5: 82.64, 6: 83.08, 7: 69.65, 8: 79.64, 9: 89.74, 10: 90.39, 11: 78.55, 12: 53.04, 13: 91.13}

print("=== INITIAL GRID COMPARISON: ALL ROUNDS ===")
header = f"{'Rnd':>3} {'AvgSett':>7} {'AvgFor':>7} {'AvgPort':>7} {'AvgMtn':>7} {'SettClusters':>12} {'Density':>7} {'GT_SS':>6} {'LORO':>6} {'Type':>8}"
print(header)
print("-" * len(header))

for rn, rid in sorted(ROUND_IDS.items()):
    data_dir = Path("data") / f"round_{rid}"
    detail_files = sorted(data_dir.glob("round_detail_*.json"))
    if not detail_files:
        continue
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    if not states:
        continue

    setts, forests, ports, mtns, clusters_list, densities = [], [], [], [], [], []
    for st in states:
        g = np.array(st["grid"])
        h, w = g.shape
        ns = int(np.sum(g == 1))
        setts.append(ns)
        forests.append(int(np.sum(g == 4)))
        ports.append(int(np.sum(g == 2)))
        mtns.append(int(np.sum(g == 5)))
        densities.append(ns / (h * w))
        _, nc = ndimage.label(g == 1)
        clusters_list.append(nc)

    gt_ss = GT_SS.get(rn)
    loro = LORO.get(rn)
    rtype = ""
    if gt_ss is not None:
        if gt_ss < 0.10:
            rtype = "COLLAPSE"
        elif gt_ss > 0.40:
            rtype = "BOOM"
        else:
            rtype = "NORMAL"

    gt_str = f"{gt_ss:.3f}" if gt_ss else "  ?  "
    loro_str = f"{loro:.1f}" if loro else "  ?  "
    marker = " <-- R14" if rn == 14 else ""

    # isolation: clusters/settlements ratio
    iso = np.mean([c / max(s, 1) for c, s in zip(clusters_list, setts)])

    print(f"{rn:>3} {np.mean(setts):>7.1f} {np.mean(forests):>7.1f} {np.mean(ports):>7.1f} {np.mean(mtns):>7.1f} {np.mean(clusters_list):>8.1f}({iso:.2f}) {np.mean(densities):>7.4f} {gt_str:>6} {loro_str:>6} {rtype:>8}{marker}")

print()
print("=== SETTLEMENT ISOLATION RATIO (1.0 = all isolated, <1.0 = some clustered) ===")
for rn, rid in sorted(ROUND_IDS.items()):
    data_dir = Path("data") / f"round_{rid}"
    detail_files = sorted(data_dir.glob("round_detail_*.json"))
    if not detail_files:
        continue
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    if not states:
        continue

    isolation_ratios = []
    for st in states:
        g = np.array(st["grid"])
        ns = int(np.sum(g == 1))
        _, nc = ndimage.label(g == 1)
        if ns > 0:
            isolation_ratios.append(nc / ns)

    avg_iso = np.mean(isolation_ratios)
    gt_ss = GT_SS.get(rn)
    gt_str = f"{gt_ss:.3f}" if gt_ss else "?"
    loro_val = LORO.get(rn)
    loro_str = f"{loro_val:.1f}" if loro_val else "?"
    print(f"  R{rn}: isolation={avg_iso:.3f} | GT S->S={gt_str} | LORO={loro_str}")

print()

# Most similar to R14 by grid composition
print("=== SIMILARITY TO R14 (Euclidean distance in [sett, forest, port, mtn] space) ===")
r14_dir = Path("data") / f"round_{ROUND_IDS[14]}"
r14_detail = json.loads(sorted(r14_dir.glob("round_detail_*.json"))[-1].read_text(encoding="utf-8"))
r14_stats = []
for st in r14_detail["initial_states"]:
    g = np.array(st["grid"])
    r14_stats.append([np.sum(g == 1), np.sum(g == 4), np.sum(g == 2), np.sum(g == 5)])
r14_vec = np.mean(r14_stats, axis=0)

dists = []
for rn, rid in sorted(ROUND_IDS.items()):
    if rn == 14:
        continue
    data_dir = Path("data") / f"round_{rid}"
    detail_files = sorted(data_dir.glob("round_detail_*.json"))
    if not detail_files:
        continue
    detail = json.loads(detail_files[-1].read_text(encoding="utf-8"))
    states = detail.get("initial_states", [])
    if not states:
        continue
    stats = []
    for st in states:
        g = np.array(st["grid"])
        stats.append([np.sum(g == 1), np.sum(g == 4), np.sum(g == 2), np.sum(g == 5)])
    vec = np.mean(stats, axis=0)
    dist = np.linalg.norm(r14_vec - vec)
    dists.append((rn, dist, vec))

dists.sort(key=lambda x: x[1])
print(f"  R14 vector: sett={r14_vec[0]:.0f}, for={r14_vec[1]:.0f}, port={r14_vec[2]:.0f}, mtn={r14_vec[3]:.0f}")
for rn, d, v in dists:
    gt_ss = GT_SS.get(rn)
    gt_str = f"{gt_ss:.3f}" if gt_ss else "?"
    loro_str = f"{LORO.get(rn, 0):.1f}"
    print(f"  R{rn}: dist={d:.1f} (sett={v[0]:.0f}, for={v[1]:.0f}, port={v[2]:.0f}, mtn={v[3]:.0f}) | GT S->S={gt_str} | LORO={loro_str}")
