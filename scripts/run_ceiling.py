#!/usr/bin/env python3
"""EXP-002 — the binding-ceiling regime (see docs/BENCHMARK-POLICY.md Amendment B).

Under a hard ceiling smaller than the number of latent skills, a policy that wants a new
module must free a slot. Every arm here ends each step at the SAME live-module count, the
same parameter count and the same storage. Capacity is equal by construction, so any
difference between them is attributable to the slot decision and to nothing else.

    B-DENY         refuse to spawn; keep using an existing module
    B-EVICT-LRU    delete the least recently used module, then spawn
    B-EVICT-RAND   delete a uniformly random module, then spawn   (eviction criterion control)
    B-MERGE        pool the two most functionally similar modules, then spawn
    B-MERGE-RAND   pool two uniformly random modules, then spawn  (merge criterion control)

K6 admissibility is checked first and reported whether or not it passes.
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

K6_MIN_CEILING_COST = 0.05

ARMS = [
    ("B-DENY", "deny"),
    ("B-EVICT-LRU", "evict_lru"),
    ("B-EVICT-RAND", "evict_random"),
    ("B-MERGE", "merge_best"),
    ("B-MERGE-RAND", "merge_random"),
]


def merge_event_records(res, **context) -> list[dict]:
    """One machine-readable record per merge event.

    Closes the D5 requirement that per-event merge loss and recovery be reportable across
    the whole phase diagram: the aggregate row fields alone discard individual events.

    This is a reporting projection of ``res.merges``. It selects nothing, re-runs nothing
    and derives recovery with exactly the same functions the aggregate row uses, so the
    aggregates remain reconstructible from these records.
    """
    out: list[dict] = []
    for idx, m in enumerate(res.merges):
        trace = list(m.get("recovery_trace", []))
        accs = [pt["acc"] for pt in trace]
        loss = m["total_merge_loss"]
        if accs:
            rec = metrics.recovery(loss, m["acc_after"], accs[-1])
            rt = metrics.recovery_time(loss, m["acc_after"], accs)
            censored = rt is None
        else:
            rec, rt, censored = None, None, None
        out.append({
            **context,
            "arm": res.arm,
            "event_index": idx,
            "chunk": m["chunk"],
            "segment_index": m.get("seen_at_merge"),
            "trigger": m.get("trigger", "opportunistic"),
            "pair": list(m["pair"]),
            "same_skill": m["same_skill"],
            # C1-C6, predeclared in experiments/CANDIDATE-DIVERSITY-PREDICTIONS-PREREG.md
            "n_live": m.get("n_live"),
            "n_candidate_pairs": m.get("n_candidate_pairs"),
            "best_score": m.get("best_score"),
            "second_best_score": m.get("second_best_score"),
            "score_margin": m.get("score_margin"),
            "score_mean": m.get("score_mean"),
            "score_std": m.get("score_std"),
            "acc_before": m["acc_no_merge"],
            "acc_no_merge": m["acc_no_merge"],
            "acc_exact_merge": m["acc_exact_merge"],
            "acc_operator_merge": m["acc_operator_merge"],
            "acc_after": m["acc_after"],
            "decision_loss": m["decision_loss"],
            "mechanism_loss": m["mechanism_loss"],
            "total_merge_loss": loss,
            "recovery_trace": trace,
            "recovery": rec,
            "recovery_time": rt,
            "recovery_censored": censored,
        })
    return out


def run_one_detailed(stream, name, on_full, ceiling, seed):
    """``run_one`` plus its per-event merge records, from the SAME single simulation run.

    Returns ``(row, merge_events)``. The row is byte-identical to ``run_one``'s.
    """
    cfg = policies.ArmConfig(name, routing="learned", cap=ceiling, on_full=on_full,
                             seed=seed)
    res = policies.run_arm(stream, cfg)
    return _score_row(res, name, seed), merge_event_records(res)


def run_one(stream, name, on_full, ceiling, seed):
    return run_one_detailed(stream, name, on_full, ceiling, seed)[0]


def _score_row(res, name, seed):
    beh = metrics.retention_matrix_stats(res.R)
    led = res.ledger
    row = {
        "arm": name, "seed": seed, **beh,
        "k_final": res.k_final, "k_peak": res.k_peak,
        "param_total": led["param_total"], "param_peak": led["param_peak"],
        "storage_total": led["storage_total"], "cold_bytes": led["cold_bytes"],
        "decision_flops": led["decision_flops"],
        "total_flops": led["total_algorithmic_flops"],
        "n_spawn": len([e for e in res.events if e["op"] == "spawn"]),
        "n_merge": len(res.merges),
        "n_evict": len(res.evictions),
        "specialisation_nmi": metrics.specialisation_nmi(res.assignments, res.truth),
    }
    if res.merges:
        row["mean_merge_loss"] = float(np.mean([m["total_merge_loss"] for m in res.merges]))
        row["mean_decision_loss"] = float(np.mean([m["decision_loss"] for m in res.merges]))
        row["mean_mechanism_loss"] = float(np.mean([m["mechanism_loss"] for m in res.merges]))
        judged = [m for m in res.merges if m["same_skill"] is not None]
        if judged:
            row["merge_precision"] = float(np.mean([m["same_skill"] for m in judged]))
        recs, times = [], []
        for m in res.merges:
            trace = [t["acc"] for t in m.get("recovery_trace", [])]
            if not trace:
                continue
            loss = m["total_merge_loss"]
            recs.append(metrics.recovery(loss, m["acc_after"], trace[-1]))
            rt = metrics.recovery_time(loss, m["acc_after"], trace)
            times.append(rt)
        if recs:
            row["merge_recovery"] = float(np.mean(recs))
            done = [t for t in times if t is not None]
            row["merge_recovery_time"] = float(np.mean(done)) if done else None
            row["merge_recovery_censored"] = sum(1 for t in times if t is None) / len(times)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k-star", type=int, default=6)
    ap.add_argument("--ceiling", type=int, default=3)
    ap.add_argument("--segments", type=int, default=18)
    ap.add_argument("--region-scale", type=float, default=0.7)
    ap.add_argument("--out", type=str, default="results/ceiling")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def stream_for(seed):
        return make_stream(StreamConfig(k_star=args.k_star, n_segments=args.segments,
                                        region_scale=args.region_scale, seed=seed))

    # --- K6 admissibility -------------------------------------------------
    unb, cap = [], []
    for seed in args.seeds:
        st = stream_for(seed)
        unb.append(metrics.retention_matrix_stats(policies.run_arm(
            st, policies.ArmConfig("A3_unbounded", routing="learned", cap=None,
                                   seed=seed)).R)["retention"])
        cap.append(metrics.retention_matrix_stats(policies.run_arm(
            st, policies.ArmConfig("A3_ceiling", routing="learned", cap=args.ceiling,
                                   seed=seed)).R)["retention"])
    k6a = args.ceiling < args.k_star
    ceiling_cost = float(np.mean(unb) - np.mean(cap))
    k6b = ceiling_cost >= K6_MIN_CEILING_COST
    print(f"K6a ceiling ({args.ceiling}) < K* ({args.k_star}): {k6a}")
    print(f"K6b ceiling cost {ceiling_cost:.3f} >= {K6_MIN_CEILING_COST}: {k6b}"
          f"   [unbounded {np.mean(unb):.3f} vs ceiling {np.mean(cap):.3f}]")
    admissible = k6a and k6b
    print(f"admissible: {admissible}\n")

    rows = []
    for seed in args.seeds:
        st = stream_for(seed)
        for name, on_full in ARMS:
            rows.append(run_one(st, name, on_full, args.ceiling, seed))

    payload = {"k6": {"k6a": k6a, "k6b": k6b, "ceiling_cost": ceiling_cost,
                      "admissible": admissible, "threshold": K6_MIN_CEILING_COST},
               "config": vars(args), "rows": rows}
    (out / "rows.json").write_text(json.dumps(payload, indent=2))

    hdr = ["arm", "ret", "plast", "forget", "P_tot", "storage", "spawn", "merge",
           "evict", "mergeloss", "recov", "prec"]
    print(f"{hdr[0]:<15}" + "".join(f"{h:>10}" for h in hdr[1:]))
    print("-" * (15 + 10 * (len(hdr) - 1)))
    agg = {}
    for name, _ in ARMS:
        rs = [r for r in rows if r["arm"] == name]
        g = lambda k, d=float("nan"): float(np.mean([r.get(k, d) for r in rs]))  # noqa: E731
        agg[name] = {k: g(k) for k in ("retention", "plasticity", "forgetting")}
        print(f"{name:<15}{g('retention'):>10.3f}{g('plasticity'):>10.3f}"
              f"{g('forgetting'):>10.3f}{g('param_total'):>10.0f}"
              f"{g('storage_total'):>10.0f}{g('n_spawn'):>10.1f}{g('n_merge'):>10.1f}"
              f"{g('n_evict'):>10.1f}{g('mean_merge_loss'):>10.4f}"
              f"{g('merge_recovery'):>10.3f}{g('merge_precision'):>10.2f}")

    def paired(a, b, key="retention"):
        ra = [r[key] for r in rows if r["arm"] == a]
        rb = [r[key] for r in rows if r["arm"] == b]
        ci = metrics.paired_bootstrap_ci(ra, rb, n_boot=20000, seed=1)
        star = "*" if ci["excludes_zero"] else " "
        print(f"  {a:<14} - {b:<14} {ci['mean_diff']:+.4f}  "
              f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] {star}")
        return ci

    print("\npaired comparisons (retention, paired bootstrap 95% CI):")
    cis = {
        "merge_vs_deny": paired("B-MERGE", "B-DENY"),
        "merge_vs_evict": paired("B-MERGE", "B-EVICT-LRU"),
        "merge_vs_randmerge": paired("B-MERGE", "B-MERGE-RAND"),
        "evictlru_vs_evictrand": paired("B-EVICT-LRU", "B-EVICT-RAND"),
        "evict_vs_deny": paired("B-EVICT-LRU", "B-DENY"),
    }
    print("\npaired comparisons (plasticity):")
    cis_p = {
        "merge_vs_deny": paired("B-MERGE", "B-DENY", "plasticity"),
        "merge_vs_randmerge": paired("B-MERGE", "B-MERGE-RAND", "plasticity"),
        "evict_vs_deny": paired("B-EVICT-LRU", "B-DENY", "plasticity"),
    }
    print("\npaired comparisons (forgetting, lower is better):")
    cis_f = {
        "merge_vs_deny": paired("B-MERGE", "B-DENY", "forgetting"),
        "evict_vs_deny": paired("B-EVICT-LRU", "B-DENY", "forgetting"),
    }
    payload["paired"] = cis
    payload["paired_plasticity"] = cis_p
    payload["paired_forgetting"] = cis_f
    (out / "rows.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out/'rows.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
