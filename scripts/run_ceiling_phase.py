#!/usr/bin/env python3
"""EXP-003 — predeclared binding-capacity pressure phase diagram.

Design is frozen in experiments/EXP-003-CEILING-PHASE-PREREG.md. This runner has no
CLI knobs for the scientific grid: changing K*, ratios, seeds, recurrence or stream density
requires a new dated amendment before another run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modular_consolidation import metrics, policies  # noqa: E402
from modular_consolidation.toy import StreamConfig, make_stream  # noqa: E402
from run_ceiling import ARMS, K6_MIN_CEILING_COST, run_one  # noqa: E402


K_STARS = (6, 12, 24)
CEILING_RATIOS = ((1, 6), (1, 3), (1, 2), (2, 3), (5, 6))
DEV_SEEDS = tuple(range(900, 908))
SEGMENTS_PER_SKILL = 3
REGION_SCALE = 0.7
RECUR_PROB = 0.30
NEAR_DUP_PROB = 0.30


def ceiling_for(k_star: int, ratio: tuple[int, int]) -> int:
    num, den = ratio
    assert (k_star * num) % den == 0, (k_star, ratio)
    return (k_star * num) // den


def stream_for(k_star: int, seed: int):
    return make_stream(
        StreamConfig(
            k_star=k_star,
            n_segments=SEGMENTS_PER_SKILL * k_star,
            recur_prob=RECUR_PROB,
            near_dup_prob=NEAR_DUP_PROB,
            region_scale=REGION_SCALE,
            seed=seed,
        )
    )


def stream_diagnostic(stream) -> dict:
    counts = Counter(s.kind for s in stream.segments)
    exposures = np.bincount(
        [s.skill for s in stream.segments],
        minlength=stream.k_star,
    )
    observed = int(np.count_nonzero(exposures))
    return {
        "k_star": int(stream.k_star),
        "seed": int(stream.cfg.seed),
        "n_segments": int(stream.n_segments),
        "kind_counts": {k: int(v) for k, v in sorted(counts.items())},
        "distinct_skills_observed": observed,
        "all_skills_introduced": observed == stream.k_star,
        "exposures_per_skill": [int(x) for x in exposures],
        "exposure_mean": float(np.mean(exposures)),
        "exposure_std": float(np.std(exposures)),
    }


def paired(rows: list[dict], arm_a: str, arm_b: str, key: str) -> dict:
    aa = {int(r["seed"]): float(r[key]) for r in rows if r["arm"] == arm_a}
    bb = {int(r["seed"]): float(r[key]) for r in rows if r["arm"] == arm_b}
    seeds = sorted(set(aa) & set(bb))
    if seeds != list(DEV_SEEDS):
        raise RuntimeError(
            f"paired seed mismatch for {arm_a} vs {arm_b}: {seeds}"
        )
    ci = metrics.paired_bootstrap_ci(
        [aa[s] for s in seeds],
        [bb[s] for s in seeds],
        n_boot=20000,
        seed=1,
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": key,
        "mean_diff": float(ci["mean_diff"]),
        "ci_low": float(ci["ci_low"]),
        "ci_high": float(ci["ci_high"]),
        "excludes_zero": bool(ci["excludes_zero"]),
    }


def capacity_diagnostic(cell_rows: list[dict]) -> dict:
    per_seed = {}
    all_equal = True
    for seed in DEV_SEEDS:
        rs = [r for r in cell_rows if int(r["seed"]) == seed]
        params = {int(r["param_total"]) for r in rs}
        storage = {int(r["storage_total"]) for r in rs}
        k_final = {int(r["k_final"]) for r in rs}
        equal = len(params) == len(storage) == len(k_final) == 1
        all_equal = all_equal and equal
        per_seed[str(seed)] = {
            "equal": equal,
            "param_total": sorted(params),
            "storage_total": sorted(storage),
            "k_final": sorted(k_final),
        }
    return {"all_seeds_equal": all_equal, "per_seed": per_seed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/ceiling_phase")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Generate exactly one stream per (K*, seed); reuse it across every pressure cell and arm.
    streams = {
        (k_star, seed): stream_for(k_star, seed)
        for k_star in K_STARS
        for seed in DEV_SEEDS
    }
    stream_diags = [
        stream_diagnostic(streams[(k_star, seed)])
        for k_star in K_STARS
        for seed in DEV_SEEDS
    ]

    # Cache the unbounded learned-router reference once per (K*, seed).
    unbounded_ret = {}
    for k_star in K_STARS:
        for seed in DEV_SEEDS:
            st = streams[(k_star, seed)]
            res = policies.run_arm(
                st,
                policies.ArmConfig(
                    "A3_unbounded",
                    routing="learned",
                    cap=None,
                    seed=seed,
                ),
            )
            unbounded_ret[(k_star, seed)] = float(
                metrics.retention_matrix_stats(res.R)["retention"]
            )

    rows: list[dict] = []
    cells: list[dict] = []

    for k_star in K_STARS:
        for num, den in CEILING_RATIOS:
            ratio = (num, den)
            ceiling = ceiling_for(k_star, ratio)
            cell_rows: list[dict] = []
            capped_ret = []

            for seed in DEV_SEEDS:
                st = streams[(k_star, seed)]
                cap_res = policies.run_arm(
                    st,
                    policies.ArmConfig(
                        "A3_ceiling",
                        routing="learned",
                        cap=ceiling,
                        seed=seed,
                    ),
                )
                capped_ret.append(
                    float(metrics.retention_matrix_stats(cap_res.R)["retention"])
                )

                for name, on_full in ARMS:
                    row = run_one(st, name, on_full, ceiling, seed)
                    row.update(
                        {
                            "k_star": k_star,
                            "ceiling": ceiling,
                            "ratio_num": num,
                            "ratio_den": den,
                            "ceiling_ratio": num / den,
                            "n_segments": SEGMENTS_PER_SKILL * k_star,
                        }
                    )
                    rows.append(row)
                    cell_rows.append(row)

            ub = [unbounded_ret[(k_star, seed)] for seed in DEV_SEEDS]
            ceiling_cost = float(np.mean(ub) - np.mean(capped_ret))
            comparisons = {
                "merge_vs_deny_retention": paired(
                    cell_rows, "B-MERGE", "B-DENY", "retention"
                ),
                "merge_vs_deny_plasticity": paired(
                    cell_rows, "B-MERGE", "B-DENY", "plasticity"
                ),
                "merge_vs_evict_retention": paired(
                    cell_rows, "B-MERGE", "B-EVICT-LRU", "retention"
                ),
                "merge_vs_evict_plasticity": paired(
                    cell_rows, "B-MERGE", "B-EVICT-LRU", "plasticity"
                ),
            }
            cells.append(
                {
                    "k_star": k_star,
                    "ceiling": ceiling,
                    "ratio_num": num,
                    "ratio_den": den,
                    "ceiling_ratio": num / den,
                    "k6": {
                        "k6a": ceiling < k_star,
                        "k6b": ceiling_cost >= K6_MIN_CEILING_COST,
                        "ceiling_cost": ceiling_cost,
                        "threshold": K6_MIN_CEILING_COST,
                        "unbounded_retention_mean": float(np.mean(ub)),
                        "capped_retention_mean": float(np.mean(capped_ret)),
                        "note": "diagnostic only; no EXP-003 cell is excluded by K6b",
                    },
                    "capacity_equal": capacity_diagnostic(cell_rows),
                    "comparisons": comparisons,
                }
            )

    payload = {
        "status": "DEVELOPMENT_SIMULATOR",
        "preregistration": "experiments/EXP-003-CEILING-PHASE-PREREG.md",
        "design": {
            "k_stars": list(K_STARS),
            "ceiling_ratios": [
                {"num": n, "den": d, "value": n / d}
                for n, d in CEILING_RATIOS
            ],
            "seeds": list(DEV_SEEDS),
            "segments_per_skill": SEGMENTS_PER_SKILL,
            "region_scale": REGION_SCALE,
            "recur_prob": RECUR_PROB,
            "near_dup_prob": NEAR_DUP_PROB,
            "arms": [name for name, _ in ARMS],
            "k6_threshold": K6_MIN_CEILING_COST,
        },
        "stream_diagnostics": stream_diags,
        "cells": cells,
        "rows": rows,
    }
    target = out / "rows.json"
    target.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        "K*  ceil  ratio   merge-deny ret [95% CI]"
        "        merge-deny plast [95% CI]      K6cost"
    )
    print("-" * 104)
    for cell in cells:
        r = cell["comparisons"]["merge_vs_deny_retention"]
        p = cell["comparisons"]["merge_vs_deny_plasticity"]
        print(
            f"{cell['k_star']:>2}  {cell['ceiling']:>4}  "
            f"{cell['ratio_num']}/{cell['ratio_den']:<3}  "
            f"{r['mean_diff']:+.4f} [{r['ci_low']:+.4f},{r['ci_high']:+.4f}]   "
            f"{p['mean_diff']:+.4f} [{p['ci_low']:+.4f},{p['ci_high']:+.4f}]   "
            f"{cell['k6']['ceiling_cost']:+.4f}"
        )
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
