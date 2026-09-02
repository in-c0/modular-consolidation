#!/usr/bin/env python3
"""Run the CAMS-v0 toy simulator over the arm lattice and its derived controls.

This is a **development simulator**. Its outputs are not evidence for any claim about
modular consolidation in real models; it exists to check that the protocol, the budget
accounting and the metrics behave, and to find design faults cheaply. See
experiments/EXP-100-PREREG-DRAFT.md for what would count as evidence.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from modular_consolidation import metrics, policies  # noqa: E402
from modular_consolidation.toy import StreamConfig, make_stream, stream_summary  # noqa: E402


def score(res: policies.ArmResult, stream, reference_acc: float) -> dict:
    beh = metrics.retention_matrix_stats(res.R)
    led = res.ledger
    alloc = metrics.allocation_stats(res.k_final, res.k_peak, stream.k_star)
    reuse = metrics.reuse_stats(res.events, stream.n_segments, res.traffic_share)
    spawn = metrics.spawn_precision_recall(
        [c for c in res.spawn_chunks],
        [o * (stream.cfg.n_train_per_segment // stream.cfg.chunk)
         for o in stream.novel_onsets],
        tolerance=stream.cfg.n_train_per_segment // stream.cfg.chunk,
    )
    row = {
        "arm": res.arm,
        **beh,
        **alloc,
        **reuse,
        **spawn,
        "cap": res.config.get("cap"),
        "param_total": led["param_total"],
        "param_peak": led["param_peak"],
        "param_active_mean": led["param_active_mean"],
        "storage_total": led["storage_total"],
        "cold_bytes": led["cold_bytes"],
        "decision_flops": led["decision_flops"],
        "total_flops": led["total_algorithmic_flops"],
        "ppap": metrics.ppap(beh["avg_acc"], reference_acc, led["param_added"]),
        "retention_per_byte": metrics.retention_per_byte(beh["retention"],
                                                         led["storage_total"]),
        "flags": res.flags,
    }
    if res.route_probs.size:
        row["routing_entropy"] = metrics.routing_entropy(res.route_probs)
    row["specialisation_nmi"] = metrics.specialisation_nmi(res.assignments, res.truth)
    row["module_purity"] = metrics.module_purity(res.assignments, res.truth)
    if res.merges:
        row["n_merges"] = len(res.merges)
        row["mean_decision_loss"] = float(np.mean([m["decision_loss"] for m in res.merges]))
        row["mean_mechanism_loss"] = float(np.mean([m["mechanism_loss"] for m in res.merges]))
        row["mean_total_merge_loss"] = float(np.mean([m["total_merge_loss"] for m in res.merges]))
        judged = [m for m in res.merges if m["same_skill"] is not None]
        if judged:
            row["merge_precision"] = float(np.mean([m["same_skill"] for m in judged]))
    return row


def run_seed(seed: int, cfg: StreamConfig, bank_k: int) -> tuple[list[dict], dict]:
    stream = make_stream(cfg)
    results: dict[str, policies.ArmResult] = {}
    for arm in policies.primary_arms(seed=seed, bank_k=bank_k):
        results[arm.name] = policies.run_arm(stream, arm)
    for ctrl in policies.derive_controls(results, seed=seed):
        results[ctrl.name] = policies.run_arm(stream, ctrl)

    ref = metrics.retention_matrix_stats(results["A1_single_adapter"].R)["avg_acc"]
    rows = [score(r, stream, ref) for r in results.values()]
    return rows, stream_summary(stream)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--k-star", type=int, default=6)
    ap.add_argument("--segments", type=int, default=18)
    ap.add_argument("--bank-k", type=int, default=6)
    ap.add_argument("--region-scale", type=float, default=0.7,
                    help="difficulty; 0.7 chosen by scripts/calibrate_stream.py "
                         "on development seeds 900-902")
    ap.add_argument("--out", type=str, default="results/toy")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for seed in args.seeds:
        cfg = StreamConfig(k_star=args.k_star, n_segments=args.segments,
                           region_scale=args.region_scale, seed=seed)
        rows, summary = run_seed(seed, cfg, args.bank_k)
        for r in rows:
            r["seed"] = seed
        all_rows.extend(rows)
        (out / f"stream_seed{seed}.json").write_text(json.dumps(summary, indent=2))

    (out / "rows.json").write_text(json.dumps(all_rows, indent=2))

    arms = sorted({r["arm"] for r in all_rows})
    hdr = ["arm", "ret", "plast", "forget", "k_fin", "k*err", "P_tot", "P_act",
           "storage", "dec_FLOP", "reuse", "nmi"]
    print(f"{hdr[0]:<32}" + "".join(f"{h:>10}" for h in hdr[1:]))
    print("-" * (32 + 10 * (len(hdr) - 1)))
    for a in arms:
        rs = [r for r in all_rows if r["arm"] == a]
        m = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
        print(f"{a:<32}"
              f"{m('retention'):>10.3f}{m('plasticity'):>10.3f}{m('forgetting'):>10.3f}"
              f"{m('k_final'):>10.1f}{m('allocation_error'):>10.2f}"
              f"{m('param_total'):>10.0f}{m('param_active_mean'):>10.0f}"
              f"{m('storage_total'):>10.0f}{m('decision_flops'):>10.2e}"
              f"{m('reuse_rate'):>10.2f}{m('specialisation_nmi'):>10.2f}")

    merged = [r for r in all_rows if "mean_total_merge_loss" in r]
    if merged:
        print("\nmerge-loss decomposition (mean over seeds)")
        for a in sorted({r["arm"] for r in merged}):
            rs = [r for r in merged if r["arm"] == a]
            print(f"  {a:<30} decision={np.mean([r['mean_decision_loss'] for r in rs]):+.4f}"
                  f"  mechanism={np.mean([r['mean_mechanism_loss'] for r in rs]):+.4f}"
                  f"  total={np.mean([r['mean_total_merge_loss'] for r in rs]):+.4f}"
                  f"  precision={np.mean([r.get('merge_precision', float('nan')) for r in rs]):.2f}")

    flagged = [(r["arm"], r["seed"], r["flags"]) for r in all_rows if r["flags"]]
    if flagged:
        print("\nvalidity flags:")
        for a, s, f in flagged:
            print(f"  {a} seed={s}: {','.join(f)}")

    print(f"\nwrote {out/'rows.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
