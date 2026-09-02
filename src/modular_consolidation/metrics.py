"""Metric implementations. Definitions live in docs/METRICS.md; this file must not
diverge from that document without an amendment note.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------- behaviour


def retention_matrix_stats(R: np.ndarray) -> dict[str, float]:
    """``R[t, i]`` = accuracy on segment ``i`` measured after finishing segment ``t``.

    Only the lower triangle (``i <= t``) is defined; entries above it are ignored.
    """
    R = np.asarray(R, dtype=float)
    S = R.shape[0]
    final = R[S - 1]
    diag = np.array([R[i, i] for i in range(S)])

    seen_final = final[:S]
    avg_acc = float(np.mean(seen_final))
    retention = float(np.mean(final[: S - 1])) if S > 1 else float(final[0])
    plasticity = float(np.mean(diag))

    forgetting = []
    bwt = []
    auc = []
    for i in range(S - 1):
        col = R[i:S, i]
        forgetting.append(float(np.max(col) - col[-1]))
        bwt.append(float(col[-1] - col[0]))
        auc.append(float(np.mean(col)))

    return {
        "avg_acc": avg_acc,
        "retention": retention,
        "plasticity": plasticity,
        "forgetting": float(np.mean(forgetting)) if forgetting else 0.0,
        "bwt": float(np.mean(bwt)) if bwt else 0.0,
        "retention_auc": float(np.mean(auc)) if auc else float(diag[0]),
    }


# ---------------------------------------------------------------- efficiency


def ppap(avg_acc: float, avg_acc_reference: float, param_added: int) -> float:
    """Performance per added parameter, relative to a non-adaptive reference."""
    return (avg_acc - avg_acc_reference) / max(param_added, 1)


def ppac(avg_acc: float, param_active_mean: float) -> float:
    return avg_acc / max(param_active_mean, 1.0)


def retention_per_byte(retention: float, storage_total: int) -> float:
    return retention / max(storage_total, 1)


def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices on the Pareto front of (maximise x, minimise y) pairs."""
    idx = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            (xj >= xi and yj <= yi) and (xj > xi or yj < yi)
            for j, (xj, yj) in enumerate(points)
            if j != i
        )
        if not dominated:
            idx.append(i)
    return idx


# ------------------------------------------------------- routing behaviour


def routing_entropy(route_probs: np.ndarray) -> float:
    """Mean entropy (nats) of the per-input routing distribution."""
    p = np.clip(np.asarray(route_probs, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(np.mean(-(p * np.log(p)).sum(axis=1)))


def specialisation_nmi(assignments: Sequence[int], ground_truth: Sequence[int]) -> float:
    """``I(module; skill) / H(skill)``. 1.0 = module identity determines skill."""
    a = np.asarray(assignments)
    g = np.asarray(ground_truth)
    if a.size == 0:
        return 0.0
    au, ai = np.unique(a, return_inverse=True)
    gu, gi = np.unique(g, return_inverse=True)
    joint = np.zeros((au.size, gu.size))
    np.add.at(joint, (ai, gi), 1.0)
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pg = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = joint * np.log(joint / (pa @ pg))
    mi = float(np.nansum(np.where(joint > 0, term, 0.0)))
    hg = float(-np.sum(pg * np.log(np.clip(pg, 1e-12, 1.0))))
    return mi / hg if hg > 1e-12 else 0.0


def module_purity(assignments: Sequence[int], ground_truth: Sequence[int]) -> float:
    a = np.asarray(assignments)
    g = np.asarray(ground_truth)
    if a.size == 0:
        return 0.0
    purities = []
    for m in np.unique(a):
        sub = g[a == m]
        _, counts = np.unique(sub, return_counts=True)
        purities.append(counts.max() / sub.size)
    return float(np.mean(purities))


# ------------------------------------------------------------- allocation


def allocation_stats(k_final: int, k_peak: int, k_star: int) -> dict[str, float]:
    return {
        "k_final": int(k_final),
        "k_peak": int(k_peak),
        "k_star": int(k_star),
        "over_allocation": max(0.0, (k_final - k_star) / k_star),
        "under_allocation": max(0.0, (k_star - k_final) / k_star),
        "allocation_error": abs(k_final - k_star) / k_star,
    }


def spawn_precision_recall(spawn_times: Sequence[int],
                           novel_onsets: Sequence[int],
                           tolerance: int) -> dict[str, float]:
    """A spawn counts as correct if it lands within ``tolerance`` of a true novel onset."""
    spawns = list(spawn_times)
    onsets = list(novel_onsets)
    if not spawns and not onsets:
        return {"spawn_precision": 1.0, "spawn_recall": 1.0}
    matched_onsets: set[int] = set()
    hits = 0
    for s in spawns:
        cand = [o for o in onsets if abs(o - s) <= tolerance and o not in matched_onsets]
        if cand:
            best = min(cand, key=lambda o: abs(o - s))
            matched_onsets.add(best)
            hits += 1
    precision = hits / len(spawns) if spawns else 0.0
    recall = len(matched_onsets) / len(onsets) if onsets else 1.0
    return {"spawn_precision": float(precision), "spawn_recall": float(recall)}


# -------------------------------------------------- consolidation dynamics


def merge_loss(acc_before: float, acc_after: float) -> float:
    """Positive means the merge hurt."""
    return acc_before - acc_after


def decompose_merge(acc_no_merge: float, acc_exact_merge: float,
                    acc_operator_merge: float) -> dict[str, float]:
    """Split the damage done by one merge event into decision and mechanism parts.

    ``decision_loss``  -- cost of choosing to merge these modules at all, measured with an
                          ideal merge operator, so it cannot be blamed on the operator.
    ``mechanism_loss`` -- extra cost incurred by the practical operator over the ideal one.
    """
    return {
        "decision_loss": acc_no_merge - acc_exact_merge,
        "mechanism_loss": acc_exact_merge - acc_operator_merge,
        "total_merge_loss": acc_no_merge - acc_operator_merge,
    }


def recovery(loss: float, acc_after: float, acc_later: float) -> float:
    """Fraction of a merge's damage recovered by later learning. Clipped to [0, 1]."""
    if loss <= 1e-12:
        return 1.0
    return float(min(1.0, max(0.0, (acc_later - acc_after) / loss)))


def recovery_time(loss: float, acc_after: float, trace: Sequence[float],
                  frac: float = 0.9) -> int | None:
    """Steps until ``frac`` of the loss is recovered; ``None`` if never (censored)."""
    if loss <= 1e-12:
        return 0
    target = acc_after + frac * loss
    for k, a in enumerate(trace, start=1):
        if a >= target:
            return k
    return None


# ------------------------------------------------------------ reuse metrics


def reuse_stats(events: Sequence[dict], n_segments: int,
                traffic_share: dict[int, float] | None = None,
                zombie_threshold: float = 0.01) -> dict[str, float]:
    spawns = [e for e in events if e["op"] == "spawn"]
    retires = [e for e in events if e["op"] == "retire"]
    reinstates = [e for e in events if e["op"] == "reinstate"]
    served_without_spawn = max(0, n_segments - len(spawns))
    stats = {
        "n_spawn": len(spawns),
        "n_merge": len([e for e in events if e["op"] == "merge"]),
        "n_retire": len(retires),
        "n_reinstate": len(reinstates),
        "reuse_rate": served_without_spawn / max(n_segments, 1),
        "reinstatement_rate": len(reinstates) / max(len(retires), 1),
    }
    if traffic_share is not None and traffic_share:
        zombies = sum(1 for v in traffic_share.values() if v < zombie_threshold)
        stats["zombie_rate"] = zombies / len(traffic_share)
    return stats


# ------------------------------------------------------------- statistics


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap CI for ``mean(a) - mean(b)``. Seeds are paired across arms."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap requires equal-length paired samples")
    d = a - b
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_pairs": int(d.size),
    }
