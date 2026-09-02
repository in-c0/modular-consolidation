#!/usr/bin/env python3
"""Choose a CAMS-v1 interference setting using the criterion predeclared in
docs/BENCHMARK-POLICY.md (committed before this script was written).

    K1  A1 single-adapter retention <= 0.95                (no ceiling)
    K2  A1 single-adapter retention >= 0.35                (no floor)
    K3  retention(C-OID) - retention(A1) >= 0.15           (modularity has headroom)
    K4  retention(C-OID) - retention(A2) >= 0.10           (capacity with a bad router
                                                            is NOT enough)
    K5  max_K retention(A3_K) - retention(A3_Kmax) >= 0.05 (over-allocation must COST
                                                            something -- Amendment A)

K4 is the condition that makes the regime able to distinguish consolidation from capacity.
EXP-000's separated-region stream failed it, which is why capacity explained everything
there.

The three arms referenced -- single adapter, fixed bank with random routing, and the oracle
task-ID upper bound -- none of which spawns, merges, retires or reinstates. No candidate
consolidation policy influences admissibility.

Among admissible settings the LEAST interfering is chosen, so difficulty is not escalated
until consolidation starts to look good.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from modular_consolidation import metrics, policies  # noqa: E402
from modular_consolidation.toy import StreamConfig, make_stream  # noqa: E402

K1_CEILING = 0.95
K2_FLOOR = 0.35
K3_MIN_HEADROOM = 0.15
K4_MIN_ROUTING_GAP = 0.10
K5_MIN_OVERALLOC_COST = 0.05
K5_CAPS = (1, 2, 4, 8, 16, 24)


def probe(cfg: StreamConfig, seed: int, bank_k: int) -> dict:
    stream = make_stream(cfg)
    runs = {
        "A1": policies.ArmConfig("A1", routing="random", cap=1, seed=seed),
        "A2": policies.ArmConfig("A2", routing="random", cap=bank_k, seed=seed),
        "OID": policies.ArmConfig("C-OID", routing="oracle", cap=cfg.k_star,
                                  task_free=False, seed=seed),
    }
    out = {}
    for name, arm in runs.items():
        out[name] = metrics.retention_matrix_stats(
            policies.run_arm(stream, arm).R)["retention"]
    # K5: does over-allocation cost anything? Swept with A3 (fixed bank, learned routing),
    # which does not spawn, merge, retire, reinstate or compress.
    curve = {}
    for K in K5_CAPS:
        curve[K] = metrics.retention_matrix_stats(policies.run_arm(
            stream, policies.ArmConfig(f"A3_K{K}", routing="learned", cap=K,
                                       seed=seed)).R)["retention"]
    out["curve"] = curve
    out["overalloc_cost"] = max(curve.values()) - curve[max(K5_CAPS)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902])
    ap.add_argument("--interference", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--region-scale", type=float, default=0.7)
    ap.add_argument("--n-context", type=int, default=6)
    ap.add_argument("--context-strength", type=float, default=1.6)
    ap.add_argument("--bank-k", type=int, default=6)
    ap.add_argument("--out", type=str, default="results/calibration-v1")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'interf':>8}{'A1':>8}{'A2':>8}{'OID':>8}{'K3 gap':>9}{'K4 gap':>9}"
          f"{'K5 cost':>9}{'K1':>5}{'K2':>5}{'K3':>5}{'K4':>5}{'K5':>5}{'admit':>8}")
    print("-" * 91)
    for f in args.interference:
        cfgs = [StreamConfig(region_scale=args.region_scale, interference=f,
                             n_context=args.n_context,
                             context_strength=args.context_strength, seed=s)
                for s in args.seeds]
        per = [probe(c, s, args.bank_k) for c, s in zip(cfgs, args.seeds)]
        a1 = float(np.mean([p["A1"] for p in per]))
        a2 = float(np.mean([p["A2"] for p in per]))
        oid = float(np.mean([p["OID"] for p in per]))
        k1 = a1 <= K1_CEILING
        k2 = a1 >= K2_FLOOR
        k3 = (oid - a1) >= K3_MIN_HEADROOM
        k4 = (oid - a2) >= K4_MIN_ROUTING_GAP
        cost = float(np.mean([p["overalloc_cost"] for p in per]))
        k5 = cost >= K5_MIN_OVERALLOC_COST
        admit = all((k1, k2, k3, k4, k5))
        row = {"interference": f, "A1": a1, "A2": a2, "OID": oid,
               "k3_gap": oid - a1, "k4_gap": oid - a2, "k5_overalloc_cost": cost,
               "capacity_curve": {str(K): float(np.mean([p["curve"][K] for p in per]))
                                  for K in K5_CAPS},
               "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "admissible": admit,
               "seeds": args.seeds}
        rows.append(row)
        tick = lambda b: "ok" if b else "--"  # noqa: E731
        print(f"{f:>8.2f}{a1:>8.3f}{a2:>8.3f}{oid:>8.3f}{oid - a1:>9.3f}{oid - a2:>9.3f}"
              f"{cost:>9.3f}{tick(k1):>5}{tick(k2):>5}{tick(k3):>5}{tick(k4):>5}"
              f"{tick(k5):>5}{str(admit):>8}")

    ok = [r for r in rows if r["admissible"]]
    chosen = min(ok, key=lambda r: r["interference"]) if ok else None
    payload = {
        "criterion": {"K1_ceiling": K1_CEILING, "K2_floor": K2_FLOOR,
                      "K3_min_headroom": K3_MIN_HEADROOM,
                      "K4_min_routing_gap": K4_MIN_ROUTING_GAP,
                      "K5_min_overalloc_cost": K5_MIN_OVERALLOC_COST,
                      "K5_caps": list(K5_CAPS)},
        "fixed": {"region_scale": args.region_scale, "n_context": args.n_context,
                  "context_strength": args.context_strength, "bank_k": args.bank_k},
        "rows": rows,
        "chosen_interference": chosen["interference"] if chosen else None,
    }
    (out / "calibration.json").write_text(json.dumps(payload, indent=2))

    if chosen is None:
        print("\nNo admissible interference setting. CAMS-v1 cannot yet distinguish "
              "consolidation from capacity at these settings; the generator or the context "
              "signal needs revision. Reported as a negative instrument result, not tuned "
              "away by relaxing the criterion.")
    else:
        print(f"\nchosen interference = {chosen['interference']} "
              f"(least interfering admissible; K3 {chosen['k3_gap']:.3f}, "
              f"K4 {chosen['k4_gap']:.3f}, K5 {chosen['k5_overalloc_cost']:.3f})")
    print(f"wrote {out/'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
