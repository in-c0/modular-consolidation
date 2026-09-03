#!/usr/bin/env python3
"""Read-only analysis of an EXP-003 payload, in the predeclared inspection order.

Order is fixed so that no favourable cell can be located before validity is established:

    1. validity / construction
    2. primary D5 contrasts (retention and plasticity, never collapsed)
    3. explanatory outcomes, including event-level merge diagnostics
    4. the owner-approved interpretation rule

This script computes nothing that was not already preregistered and mutates no results.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modular_consolidation import metrics  # noqa: E402

ARM_NAMES = ["B-DENY", "B-EVICT-LRU", "B-EVICT-RAND", "B-MERGE", "B-MERGE-RAND"]
PRIMARY_METRICS = ("retention", "plasticity")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# ------------------------------------------------------------ 1. validity

def validity(payload: dict) -> tuple[bool, list[str], dict]:
    problems: list[str] = []
    notes: dict = {}
    design = payload["design"]
    seeds = design["seeds"]
    cells = payload["cells"]
    rows = payload["rows"]

    expected_cells = len(design["k_stars"]) * len(design["ceiling_ratios"])
    if len(cells) != expected_cells:
        problems.append(f"expected {expected_cells} cells, found {len(cells)}")

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["k_star"], r["ratio_num"], r["ratio_den"])].append(r)

    for cell in cells:
        key = (cell["k_star"], cell["ratio_num"], cell["ratio_den"])
        cr = by_cell.get(key, [])
        if len(cr) != len(seeds) * len(ARM_NAMES):
            problems.append(f"cell {key}: {len(cr)} rows, expected {len(seeds)*len(ARM_NAMES)}")
        for arm in ARM_NAMES:
            got = sorted(int(r["seed"]) for r in cr if r["arm"] == arm)
            if got != sorted(seeds):
                problems.append(f"cell {key} arm {arm}: seeds {got}")
        # capacity equality, as recorded by the runner
        if not cell["capacity_equal"]["all_seeds_equal"]:
            bad = [s for s, v in cell["capacity_equal"]["per_seed"].items() if not v["equal"]]
            problems.append(f"cell {key}: capacity not equal across arms for seeds {bad}")
        # paired comparisons must rest on identical seed sets
        for name, comp in cell["comparisons"].items():
            a = sorted(int(r["seed"]) for r in cr if r["arm"] == comp["arm_a"])
            b = sorted(int(r["seed"]) for r in cr if r["arm"] == comp["arm_b"])
            if a != b or a != sorted(seeds):
                problems.append(f"cell {key} {name}: unpaired seed sets {a} vs {b}")
            for k in ("mean_diff", "ci_low", "ci_high"):
                if not finite(comp[k]):
                    problems.append(f"cell {key} {name}: non-finite {k}")

    for r in rows:
        for k in ("retention", "plasticity", "forgetting", "param_total", "storage_total"):
            if not finite(r.get(k)):
                problems.append(f"row {r['arm']}/{r['seed']}/K*{r['k_star']} bad {k}={r.get(k)}")

    diags = payload.get("stream_diagnostics", [])
    expected_diags = len(design["k_stars"]) * len(seeds)
    if len(diags) != expected_diags:
        problems.append(f"expected {expected_diags} stream diagnostics, found {len(diags)}")
    incomplete = [(d["k_star"], d["seed"], d["distinct_skills_observed"])
                  for d in diags if not d["all_skills_introduced"]]
    notes["streams_missing_skills"] = incomplete

    # K6 must be diagnostic only: every cell present regardless of k6b
    k6_failing = [(c["k_star"], f"{c['ratio_num']}/{c['ratio_den']}")
                  for c in cells if not c["k6"]["k6b"]]
    notes["cells_below_k6b"] = k6_failing
    notes["cells_present"] = len(cells)
    if len(cells) != expected_cells:
        problems.append("a cell is missing; K6b must never exclude a cell")

    ev = payload.get("merge_events")
    if ev is None:
        problems.append("merge_events missing: D5 per-event reporting absent")
    else:
        n_from_rows = sum(int(r.get("n_merge", 0)) for r in rows)
        if len(ev) != n_from_rows:
            problems.append(f"merge_events {len(ev)} != sum(n_merge) {n_from_rows}")
        notes["merge_events"] = len(ev)

    return (not problems), problems, notes


# ------------------------------------------------- 2. primary contrasts

def cell_key(c):
    return (c["k_star"], c["ratio_num"] / c["ratio_den"])


def sig(comp) -> str:
    if comp["ci_low"] > 0:
        return "+"
    if comp["ci_high"] < 0:
        return "-"
    return "."


def dominates(ret: dict, pla: dict) -> bool:
    """Strict frontier expansion, matching the preregistration's exact wording.

    The prereg states: "A mean-level tradeoff (one primary dimension up, the other down) is
    **not** called frontier expansion." So expansion requires a non-negative *mean*
    difference on BOTH primary axes plus a statistically significant gain on at least one.
    Requiring merely "no significant loss" would admit exactly the mean-level trade the
    prereg rules out, so the strict form drives the verdict.
    """
    no_mean_loss = ret["mean_diff"] >= 0 and pla["mean_diff"] >= 0
    significant_gain = ret["ci_low"] > 0 or pla["ci_low"] > 0
    return no_mean_loss and significant_gain


def dominates_loose(ret: dict, pla: dict) -> bool:
    """Permissive variant: significant gain on one axis, no *significant* loss on the other.

    Reported as a diagnostic only. It is NOT used for the verdict, because it would count a
    mean-level tradeoff as expansion.
    """
    gain = ret["ci_low"] > 0 or pla["ci_low"] > 0
    loss = ret["ci_high"] < 0 or pla["ci_high"] < 0
    return gain and not loss


def dominated_by(ret: dict, pla: dict) -> bool:
    """Is B-MERGE dominated by this baseline? (baseline better on both primary axes)."""
    return ret["ci_high"] < 0 and pla["ci_high"] < 0


def primary_table(payload: dict) -> list[dict]:
    out = []
    for c in sorted(payload["cells"], key=cell_key):
        cm = c["comparisons"]
        row = {
            "k_star": c["k_star"],
            "ceiling": c["ceiling"],
            "ratio": f"{c['ratio_num']}/{c['ratio_den']}",
            "ratio_value": c["ratio_num"] / c["ratio_den"],
            "md_ret": cm["merge_vs_deny_retention"],
            "md_pla": cm["merge_vs_deny_plasticity"],
            "me_ret": cm["merge_vs_evict_retention"],
            "me_pla": cm["merge_vs_evict_plasticity"],
            "k6_cost": c["k6"]["ceiling_cost"],
            "k6b": c["k6"]["k6b"],
        }
        row["dominates_deny"] = dominates(row["md_ret"], row["md_pla"])
        row["dominates_evict"] = dominates(row["me_ret"], row["me_pla"])
        row["pareto_extending"] = row["dominates_deny"] and row["dominates_evict"]
        row["pareto_extending_loose"] = (dominates_loose(row["md_ret"], row["md_pla"])
                                         and dominates_loose(row["me_ret"], row["me_pla"]))
        # Third reading: standard Pareto-frontier extension against the BASELINE SET
        # {deny, evict}. Merge extends that frontier if it is dominated by neither baseline
        # and strictly dominates at least one. Reported because the strict reading above is
        # unsatisfiable by construction against B-EVICT-LRU, which maximises plasticity by
        # always installing a fresh module -- nothing can beat it on that axis.
        row["dominated_by_deny"] = dominated_by(row["md_ret"], row["md_pla"])
        row["dominated_by_evict"] = dominated_by(row["me_ret"], row["me_pla"])
        row["baseline_frontier_extension"] = (
            (row["dominates_deny"] or row["dominates_evict"])
            and not row["dominated_by_deny"]
            and not row["dominated_by_evict"]
        )
        out.append(row)
    return out


# --------------------------------------------------------- 3. explanatory

def arm_means(payload: dict, key: str) -> dict:
    agg = defaultdict(list)
    for r in payload["rows"]:
        v = r.get(key)
        if finite(v):
            agg[(r["k_star"], r["ratio_num"] / r["ratio_den"], r["arm"])].append(float(v))
    return {k: float(np.mean(v)) for k, v in agg.items()}


def event_summary(payload: dict) -> dict:
    ev = payload.get("merge_events", [])
    by = defaultdict(list)
    for e in ev:
        by[(e["k_star"], e["ceiling_ratio"], e["arm"])].append(e)
    out = {}
    for k, es in by.items():
        traced = [e for e in es if e["recovery"] is not None]
        judged = [e["same_skill"] for e in es if e["same_skill"] is not None]
        times = [e["recovery_time"] for e in traced if e["recovery_time"] is not None]
        out[k] = {
            "n_events": len(es),
            "decision_loss": float(np.mean([e["decision_loss"] for e in es])),
            "mechanism_loss": float(np.mean([e["mechanism_loss"] for e in es])),
            "total_merge_loss": float(np.mean([e["total_merge_loss"] for e in es])),
            "precision": float(np.mean(judged)) if judged else float("nan"),
            "recovery": float(np.mean([e["recovery"] for e in traced])) if traced else float("nan"),
            "recovery_time": float(np.mean(times)) if times else float("nan"),
            "censored": (sum(1 for e in traced if e["recovery_censored"]) / len(traced))
            if traced else float("nan"),
        }
    return out


# ------------------------------------------------------ 4. interpretation

def interpret(table: list[dict]) -> dict:
    ext = [r for r in table if r["pareto_extending"]]
    ext_loose = [r for r in table if r["pareto_extending_loose"]]
    ext_set = [r for r in table if r["baseline_frontier_extension"]]
    by_k = defaultdict(list)
    for r in ext:
        by_k[r["k_star"]].append(r["ratio_value"])

    ratios_sorted = sorted({r["ratio_value"] for r in table})
    adjacent = []
    for k, vals in by_k.items():
        idx = sorted(ratios_sorted.index(v) for v in vals)
        for a, b in zip(idx, idx[1:]):
            if b - a == 1:
                adjacent.append((k, ratios_sorted[a], ratios_sorted[b]))

    replicated_ratio = [v for v in {r["ratio_value"] for r in ext}
                        if len({r["k_star"] for r in ext if r["ratio_value"] == v}) >= 2]

    reproducible = bool(adjacent) or bool(replicated_ratio)

    by_k_set = defaultdict(list)
    for r in ext_set:
        by_k_set[r["k_star"]].append(r["ratio_value"])
    adj_set = []
    for k, vals in by_k_set.items():
        idx = sorted(ratios_sorted.index(v) for v in vals)
        for a, b in zip(idx, idx[1:]):
            if b - a == 1:
                adj_set.append((k, ratios_sorted[a], ratios_sorted[b]))
    repl_set = [v for v in {r["ratio_value"] for r in ext_set}
                if len({r["k_star"] for r in ext_set if r["ratio_value"] == v}) >= 2]
    reproducible_set = bool(adj_set) and bool(repl_set)
    return {
        "rule": "strict: non-negative mean on BOTH primary axes + significant gain on >=1",
        "extending_cells": [(r["k_star"], r["ratio"]) for r in ext],
        "extending_cells_loose_diagnostic": [(r["k_star"], r["ratio"]) for r in ext_loose],
        "adjacent_runs": adjacent,
        "ratios_replicated_across_k": sorted(replicated_ratio),
        "reproducible_region": reproducible,
        "baseline_set_extension_cells": [(r["k_star"], r["ratio"]) for r in ext_set],
        "baseline_set_adjacent_runs": adj_set,
        "baseline_set_ratios_replicated_across_k": sorted(repl_set),
        "baseline_set_reproducible_region": reproducible_set,
        "note_strict_rule_vs_evict": (
            "B-EVICT-LRU maximises plasticity by construction, so 'dominate evict on both "
            "axes' cannot be satisfied by any arm. The strict verdict is therefore driven "
            "by an unsatisfiable clause; both readings are reported and the choice between "
            "them is an owner decision."),
        "verdict": (
            "ARCHITECTURE_SIGNAL: reproducible Pareto-extending region"
            if reproducible else
            ("EXPLORATORY: isolated extending cell(s), not replicated"
             if ext else
             "OPERATING_POINT: no Pareto-extending cell; methods-paper result stands")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="results/ceiling_phase/rows.json")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.payload).read_text())

    print("=" * 100)
    print("1. VALIDITY / CONSTRUCTION")
    print("=" * 100)
    ok, problems, notes = validity(payload)
    print(f"cells present            : {notes.get('cells_present')}")
    print(f"merge event records      : {notes.get('merge_events')}")
    print(f"cells below K6b (kept)   : {notes.get('cells_below_k6b')}")
    miss = notes.get("streams_missing_skills") or []
    print(f"streams missing skills   : {miss if miss else 'none'}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        print("\nSTOPPING: mechanical problem found; interpretation withheld.")
        return 2
    print("status                   : VALID")

    print()
    print("=" * 100)
    print("2. PRIMARY D5 CONTRASTS  (retention and plasticity reported separately)")
    print("=" * 100)
    table = primary_table(payload)
    print(f"{'K*':>3} {'ceil':>5} {'ratio':>6} | {'MERGE-DENY ret':>22} {'MERGE-DENY pla':>22}"
          f" | {'MERGE-EVICT ret':>22} {'MERGE-EVICT pla':>22} | frontier")
    print("-" * 130)
    last_k = None
    for r in table:
        if last_k is not None and r["k_star"] != last_k:
            print("-" * 130)
        last_k = r["k_star"]
        def f(c):
            return f"{c['mean_diff']:+.4f}[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]{sig(c)}"
        tag = ("PARETO-EXT" if r["pareto_extending"] else
               ("dom-deny" if r["dominates_deny"] else
                ("dom-evict" if r["dominates_evict"] else "")))
        print(f"{r['k_star']:>3} {r['ceiling']:>5} {r['ratio']:>6} | {f(r['md_ret']):>22}"
              f" {f(r['md_pla']):>22} | {f(r['me_ret']):>22} {f(r['me_pla']):>22} | {tag}")
    print("\n'+' CI excludes zero favouring B-MERGE, '-' favouring the baseline, '.' spans zero")

    print()
    print("=" * 100)
    print("3. EXPLANATORY OUTCOMES")
    print("=" * 100)
    forg = arm_means(payload, "forgetting")
    print("\nforgetting by arm:")
    print(f"{'K*':>3} {'ratio':>6} | " + " ".join(f"{a:>13}" for a in ARM_NAMES))
    for r in table:
        vals = [forg.get((r["k_star"], r["ratio_value"], a), float("nan")) for a in ARM_NAMES]
        print(f"{r['k_star']:>3} {r['ratio']:>6} | " + " ".join(f"{v:>13.4f}" for v in vals))

    ev = event_summary(payload)
    print("\nevent-level merge diagnostics (B-MERGE vs B-MERGE-RAND):")
    print(f"{'K*':>3} {'ratio':>6} {'arm':>13} {'n':>5} {'decision':>10} {'mechanism':>10}"
          f" {'total':>10} {'prec':>6} {'recov':>7} {'rtime':>7} {'cens':>6}")
    for r in table:
        for arm in ("B-MERGE", "B-MERGE-RAND"):
            e = ev.get((r["k_star"], r["ratio_value"], arm))
            if not e:
                continue
            print(f"{r['k_star']:>3} {r['ratio']:>6} {arm:>13} {e['n_events']:>5}"
                  f" {e['decision_loss']:>10.4f} {e['mechanism_loss']:>10.4f}"
                  f" {e['total_merge_loss']:>10.4f} {e['precision']:>6.2f}"
                  f" {e['recovery']:>7.3f} {e['recovery_time']:>7.2f} {e['censored']:>6.2f}")

    print("\npressure vs absolute K* separation (MERGE-DENY retention mean_diff):")
    print(f"{'ratio':>6} | " + " ".join(f"{'K*='+str(k):>12}" for k in sorted({r['k_star'] for r in table})))
    for rv in sorted({r["ratio_value"] for r in table}):
        cells = {r["k_star"]: r["md_ret"]["mean_diff"] for r in table if r["ratio_value"] == rv}
        lbl = next(r["ratio"] for r in table if r["ratio_value"] == rv)
        print(f"{lbl:>6} | " + " ".join(f"{cells.get(k, float('nan')):>12.4f}"
                                        for k in sorted(cells)))

    print()
    print("=" * 100)
    print("4. INTERPRETATION (owner-approved rule, applied mechanically)")
    print("=" * 100)
    verdict = interpret(table)
    for k, v in verdict.items():
        print(f"{k:>28}: {v}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(
            {"validity": {"ok": ok, "problems": problems, "notes": notes},
             "primary": [{kk: vv for kk, vv in r.items()} for r in table],
             "events": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in ev.items()},
             "interpretation": verdict}, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
