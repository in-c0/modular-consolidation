#!/usr/bin/env python3
"""Calibrate CAMS-v0 difficulty using a METHOD-INDEPENDENT criterion.

The first toy run showed the default stream is at ceiling: a single adapter reached the
same retention as every modular arm, so the benchmark could not discriminate. Difficulty
must therefore be calibrated -- but calibrating it by "which modular method wins" would
be exactly the tuning this track forbids.

The criterion used here references only two arms, neither of which is a candidate method:

* ``A1_single_adapter``  -- the no-modularity floor;
* ``C-OID`` oracle-task-ID routing over a bank of ``K*`` -- the routing upper bound.

A stream configuration is **admissible** when

    oracle_retention - single_adapter_retention >= MIN_HEADROOM

and no arm is at the accuracy ceiling. That is, the benchmark must leave room for
modularity to matter, without saying which modular policy should capture it.

Calibration is run on development seeds only. Confirmatory seeds are disjoint.
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

MIN_HEADROOM = 0.15
CEILING = 0.95


def probe(cfg: StreamConfig, seed: int) -> dict:
    stream = make_stream(cfg)
    single = policies.run_arm(stream, policies.ArmConfig(
        "A1_single_adapter", routing="none", cap=1, seed=seed))
    oracle = policies.run_arm(stream, policies.ArmConfig(
        "C-OID", routing="oracle", cap=cfg.k_star, task_free=False, seed=seed))
    s = metrics.retention_matrix_stats(single.R)
    o = metrics.retention_matrix_stats(oracle.R)
    return {
        "single_retention": s["retention"],
        "single_plasticity": s["plasticity"],
        "oracle_retention": o["retention"],
        "oracle_plasticity": o["plasticity"],
        "headroom": o["retention"] - s["retention"],
        "at_ceiling": s["retention"] > CEILING,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[900, 901, 902],
                    help="development seeds; must stay disjoint from confirmatory seeds")
    ap.add_argument("--region-scales", type=float, nargs="+",
                    default=[2.2, 1.6, 1.2, 0.9, 0.7, 0.5, 0.35])
    ap.add_argument("--out", type=str, default="results/calibration")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"{'region_scale':>13}{'single_ret':>12}{'oracle_ret':>12}{'headroom':>11}"
          f"{'ceiling':>9}{'admissible':>12}")
    print("-" * 69)
    for rs in args.region_scales:
        per_seed = [probe(StreamConfig(region_scale=rs, seed=s), s) for s in args.seeds]
        agg = {k: float(np.mean([p[k] for p in per_seed]))
               for k in ("single_retention", "oracle_retention", "headroom")}
        ceil = any(p["at_ceiling"] for p in per_seed)
        admissible = (agg["headroom"] >= MIN_HEADROOM) and not ceil
        agg.update({"region_scale": rs, "at_ceiling": ceil, "admissible": admissible,
                    "seeds": args.seeds})
        rows.append(agg)
        print(f"{rs:>13.2f}{agg['single_retention']:>12.3f}{agg['oracle_retention']:>12.3f}"
              f"{agg['headroom']:>11.3f}{str(ceil):>9}{str(admissible):>12}")

    ok = [r for r in rows if r["admissible"]]
    chosen = max(ok, key=lambda r: r["region_scale"]) if ok else None
    payload = {"criterion": {"min_headroom": MIN_HEADROOM, "ceiling": CEILING},
               "rows": rows,
               "chosen_region_scale": chosen["region_scale"] if chosen else None}
    (out / "calibration.json").write_text(json.dumps(payload, indent=2))

    if chosen is None:
        print("\nNo admissible configuration found. CAMS-v0 cannot currently support the "
              "question at these settings; widen the sweep or revise the generator.")
    else:
        print(f"\nchosen region_scale = {chosen['region_scale']} "
              f"(least difficult admissible setting; headroom {chosen['headroom']:.3f})")
    print(f"wrote {out/'calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
