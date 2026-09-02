"""A generic module bank supporting the full lifecycle: spawn, specialise, merge,
retire, reinstate.

The learner inside each module is ridge regression over frozen random features, held as
sufficient statistics ``A = PhiT Phi``, ``b = PhiT Y``, ``n``. That choice is deliberate
and load-bearing:

* updates are closed-form, so no optimiser hyperparameter can be tuned to make a
  favoured arm win;
* merging the statistics (``A_i + A_j``) yields *exactly* the model that fitting the
  union of both modules' data would produce. That gives an **exact-merge reference**,
  which is what makes the merge-loss decomposition in docs/METRICS.md possible without
  counterfactual retraining.

Nothing here is specific to the toy stream; a module is any object with sufficient
statistics of this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .budget import BYTES_PER_PARAM, BudgetLedger


@dataclass
class Module:
    """One expert: a linear head plus an input-density signature used for routing."""

    mid: int
    dim: int
    n_out: int
    ridge: float = 1e-2

    A: np.ndarray = field(default=None, repr=False)      # dim x dim
    b: np.ndarray = field(default=None, repr=False)      # dim x n_out
    n: float = 0.0

    # input density signature (diagonal Gaussian over features)
    s1: np.ndarray = field(default=None, repr=False)     # sum of features
    s2: np.ndarray = field(default=None, repr=False)     # sum of squared features

    born_at: int = 0
    last_used_at: int = -1
    usage: int = 0
    retired_at: int | None = None
    reinstated_at: list[int] = field(default_factory=list)
    merged_from: tuple[int, ...] = ()
    provenance: list[str] = field(default_factory=list)

    # running statistics of this module's log-density on its OWN data, so that
    # novelty can be judged in module-relative units rather than by a raw threshold
    ld_n: float = 0.0
    ld_mean: float = 0.0
    ld_m2: float = 0.0

    _w: np.ndarray | None = field(default=None, repr=False)
    _dirty: bool = True

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = np.zeros((self.dim, self.dim))
        if self.b is None:
            self.b = np.zeros((self.dim, self.n_out))
        if self.s1 is None:
            self.s1 = np.zeros(self.dim)
        if self.s2 is None:
            self.s2 = np.zeros(self.dim)

    # -- capacity ---------------------------------------------------------
    @property
    def deployed_params(self) -> int:
        """Parameters that must exist to make a prediction."""
        return self.dim * self.n_out

    @property
    def state_bytes(self) -> int:
        """Consolidation state (sufficient statistics + density signature)."""
        return (self.dim * self.dim + self.dim * self.n_out + 2 * self.dim) * BYTES_PER_PARAM

    @property
    def cold_bytes(self) -> int:
        """Cost of holding this module in cold storage while retired."""
        return self.deployed_params * BYTES_PER_PARAM + self.state_bytes

    # -- learning ---------------------------------------------------------
    def observe(self, phi: np.ndarray, y_onehot: np.ndarray) -> None:
        self.A += phi.T @ phi
        self.b += phi.T @ y_onehot
        self.s1 += phi.sum(axis=0)
        self.s2 += (phi ** 2).sum(axis=0)
        self.n += phi.shape[0]
        self.usage += phi.shape[0]
        self._dirty = True

    @property
    def w(self) -> np.ndarray:
        if self._dirty or self._w is None:
            reg = self.ridge * max(self.n, 1.0)
            self._w = np.linalg.solve(self.A + reg * np.eye(self.dim), self.b)
            self._dirty = False
        return self._w

    def predict(self, phi: np.ndarray) -> np.ndarray:
        return phi @ self.w

    def record_self_score(self, score: float) -> None:
        """Welford update of the module's own log-density distribution."""
        self.ld_n += 1.0
        delta = score - self.ld_mean
        self.ld_mean += delta / self.ld_n
        self.ld_m2 += delta * (score - self.ld_mean)

    @property
    def ld_std(self) -> float:
        if self.ld_n < 2:
            return 1.0
        return float(max(np.sqrt(self.ld_m2 / (self.ld_n - 1)), 1e-6))

    def novelty_z(self, score: float) -> float:
        """How many own-data standard deviations below typical this score is."""
        if self.ld_n < 2:
            return 0.0
        return float((score - self.ld_mean) / self.ld_std)

    # -- routing signature -------------------------------------------------
    def log_density(self, phi: np.ndarray) -> np.ndarray:
        """Diagonal-Gaussian log density of features under this module's history.

        Label-free by construction: routing at inference must not see targets.
        """
        if self.n < 2:
            return np.full(phi.shape[0], -1e9)
        mu = self.s1 / self.n
        var = np.maximum(self.s2 / self.n - mu ** 2, 1e-4)
        z = (phi - mu) ** 2 / var
        return -0.5 * (z + np.log(var)).sum(axis=1)

    # -- lifecycle ---------------------------------------------------------
    def clone_stats(self) -> dict:
        return {"A": self.A.copy(), "b": self.b.copy(), "n": self.n,
                "s1": self.s1.copy(), "s2": self.s2.copy()}


def merge_exact(a: Module, b: Module, mid: int, t: int) -> Module:
    """Reference merge: summing sufficient statistics fits the union of both datasets.

    This is the *ideal* merge. It is not free of error -- if the two modules should not
    have been merged, the joint fit is still worse than keeping them apart. That residual
    is ``decision_loss``.
    """
    m = Module(mid=mid, dim=a.dim, n_out=a.n_out, ridge=a.ridge, born_at=t)
    m.A = a.A + b.A
    m.b = a.b + b.b
    m.n = a.n + b.n
    m.s1 = a.s1 + b.s1
    m.s2 = a.s2 + b.s2
    m.usage = a.usage + b.usage
    m.merged_from = tuple(sorted(set(a.merged_from + b.merged_from + (a.mid, b.mid))))
    m.provenance = a.provenance + b.provenance + [f"merge_exact({a.mid},{b.mid})@{t}"]
    m._dirty = True
    return m


def merge_operator(a: Module, b: Module, mid: int, t: int) -> Module:
    """Practical merge: sample-weighted parameter averaging, then re-derive statistics
    consistent with the averaged solution so learning can continue.

    Deliberately lossy. The gap to ``merge_exact`` is ``mechanism_loss``.
    """
    m = Module(mid=mid, dim=a.dim, n_out=a.n_out, ridge=a.ridge, born_at=t)
    wa, wb = a.w, b.w
    total = max(a.n + b.n, 1.0)
    w_avg = (a.n * wa + b.n * wb) / total
    m.A = a.A + b.A
    m.n = a.n + b.n
    reg = m.ridge * max(m.n, 1.0)
    m.b = (m.A + reg * np.eye(m.dim)) @ w_avg
    m.s1 = a.s1 + b.s1
    m.s2 = a.s2 + b.s2
    m.usage = a.usage + b.usage
    m.merged_from = tuple(sorted(set(a.merged_from + b.merged_from + (a.mid, b.mid))))
    m.provenance = a.provenance + b.provenance + [f"merge_operator({a.mid},{b.mid})@{t}"]
    m._dirty = True
    return m


class ModuleBank:
    """Live modules plus a cold store, with every lifecycle event charged to a ledger."""

    def __init__(self, dim: int, n_out: int, ledger: BudgetLedger, ridge: float = 1e-2):
        self.dim = dim
        self.n_out = n_out
        self.ridge = ridge
        self.ledger = ledger
        self.live: dict[int, Module] = {}
        self.cold: dict[int, Module] = {}
        self._next_mid = 0
        self.events: list[dict] = []

    # -- introspection -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.live)

    @property
    def live_ids(self) -> list[int]:
        return sorted(self.live)

    def all_modules(self) -> Iterable[Module]:
        yield from self.live.values()
        yield from self.cold.values()

    # -- lifecycle ---------------------------------------------------------
    def spawn(self, t: int, reason: str = "novelty") -> Module:
        m = Module(mid=self._next_mid, dim=self.dim, n_out=self.n_out,
                   ridge=self.ridge, born_at=t)
        m.provenance.append(f"spawn({reason})@{t}")
        self._next_mid += 1
        self.live[m.mid] = m
        self.ledger.add_params(m.deployed_params)
        self.ledger.set_live_modules(len(self.live))
        self._sync_state_bytes()
        self.events.append({"t": t, "op": "spawn", "mid": m.mid, "reason": reason})
        return m

    def merge(self, t: int, i: int, j: int, operator: str = "operator",
              reason: str = "affinity") -> Module:
        a, b = self.live[i], self.live[j]
        fn = merge_operator if operator == "operator" else merge_exact
        m = fn(a, b, self._next_mid, t)
        self._next_mid += 1
        del self.live[i]
        del self.live[j]
        self.live[m.mid] = m
        self.ledger.remove_params(a.deployed_params + b.deployed_params)
        self.ledger.add_params(m.deployed_params)
        self.ledger.set_live_modules(len(self.live))
        self.ledger.spend_consolidation(self.dim)
        self._sync_state_bytes()
        self.events.append({"t": t, "op": "merge", "mid": m.mid, "sources": [i, j],
                            "operator": operator, "reason": reason})
        return m

    def retire(self, t: int, mid: int, reason: str = "idle") -> None:
        """Remove from routing and from the active-parameter budget.

        The module is NOT deleted: it keeps costing cold-storage bytes. Retirement that
        does not cost anything would be an accounting error, not an algorithm.
        """
        m = self.live.pop(mid)
        m.retired_at = t
        m.provenance.append(f"retire({reason})@{t}")
        self.cold[mid] = m
        self.ledger.remove_params(m.deployed_params)
        self.ledger.set_live_modules(len(self.live))
        self.ledger.add_cold_bytes(m.cold_bytes)
        self._sync_state_bytes()
        self.events.append({"t": t, "op": "retire", "mid": mid, "reason": reason})

    def reinstate(self, t: int, mid: int, reason: str = "recognised") -> Module:
        m = self.cold.pop(mid)
        m.retired_at = None
        m.reinstated_at.append(t)
        m.provenance.append(f"reinstate({reason})@{t}")
        self.live[mid] = m
        self.ledger.remove_cold_bytes(m.cold_bytes)
        self.ledger.add_params(m.deployed_params)
        self.ledger.set_live_modules(len(self.live))
        self._sync_state_bytes()
        self.events.append({"t": t, "op": "reinstate", "mid": mid, "reason": reason})
        return m

    # -- routing -----------------------------------------------------------
    def score_live(self, phi: np.ndarray) -> tuple[list[int], np.ndarray]:
        """Label-free routing scores over live modules. Charges decision compute."""
        ids = self.live_ids
        if not ids:
            return [], np.zeros((phi.shape[0], 0))
        self.ledger.spend_decision(len(ids), self.dim)
        scores = np.stack([self.live[i].log_density(phi) for i in ids], axis=1)
        return ids, scores

    def score_cold(self, phi: np.ndarray) -> tuple[list[int], np.ndarray]:
        ids = sorted(self.cold)
        if not ids:
            return [], np.zeros((phi.shape[0], 0))
        self.ledger.spend_decision(len(ids), self.dim)
        scores = np.stack([self.cold[i].log_density(phi) for i in ids], axis=1)
        return ids, scores

    def _sync_state_bytes(self) -> None:
        self.ledger.set_state_bytes(sum(m.state_bytes for m in self.live.values()))

    def manifest(self) -> dict:
        return {
            "live": len(self.live),
            "cold": len(self.cold),
            "total_created": self._next_mid,
            "events": list(self.events),
            "provenance": {m.mid: m.provenance for m in self.all_modules()},
        }
