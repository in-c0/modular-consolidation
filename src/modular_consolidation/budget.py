"""Resource accounting.

Capacity growth is a first-class metric in this track, so it is accounted for by a ledger
that a policy cannot bypass. Retired modules keep costing cold-storage bytes; routing
decisions keep costing decision FLOPs. Both are common places where modular methods
quietly get a free lunch.

FLOP figures here are an explicit *algorithmic cost model*, not measured hardware FLOPs.
The model is declared in ``FLOP_MODEL`` so that it can be criticised rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

BYTES_PER_PARAM = 4  # float32

FLOP_MODEL = {
    "predict_per_param": 2.0,        # multiply-accumulate per active parameter
    "density_eval_per_dim": 3.0,     # per live module, per input, per feature dim
    "stat_update_per_dim2": 1.0,     # rank-1 update of a d x d sufficient statistic
    "solve_per_dim3": 0.33,          # Cholesky-ish solve
    "merge_per_dim2": 2.0,
}


class BudgetBreach(RuntimeError):
    """Raised when a declared ceiling is exceeded. Never caught inside a policy."""


@dataclass
class BudgetCeiling:
    """Hard limits declared before a run. ``None`` means unconstrained (and is reported)."""

    param_total: int | None = None
    storage_bytes: int | None = None
    total_flops: float | None = None
    live_modules: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetLedger:
    """Running account of everything a modular policy spends."""

    ceiling: BudgetCeiling = field(default_factory=BudgetCeiling)

    param_total: int = 0
    param_peak: int = 0
    param_base: int = 0
    live_modules: int = 0
    live_modules_peak: int = 0

    state_bytes: int = 0
    cold_bytes: int = 0
    replay_bytes: int = 0

    train_flops: float = 0.0
    infer_flops: float = 0.0
    decision_flops: float = 0.0
    consolidation_flops: float = 0.0

    param_writes: int = 0
    active_param_samples: list[int] = field(default_factory=list)

    def set_base(self, n_params: int) -> None:
        self.param_base = int(n_params)

    # -- capacity ---------------------------------------------------------
    def add_params(self, n: int) -> None:
        self.param_total += int(n)
        self.param_peak = max(self.param_peak, self.param_total)
        self._check()

    def remove_params(self, n: int) -> None:
        self.param_total -= int(n)
        if self.param_total < 0:
            raise BudgetBreach("param_total went negative; accounting bug")

    def set_live_modules(self, n: int) -> None:
        self.live_modules = int(n)
        self.live_modules_peak = max(self.live_modules_peak, self.live_modules)
        self._check()

    def record_active_params(self, n: int) -> None:
        """Parameters actually touched by one prediction."""
        self.active_param_samples.append(int(n))

    # -- storage ----------------------------------------------------------
    def add_cold_bytes(self, n: int) -> None:
        self.cold_bytes += int(n)
        self._check()

    def remove_cold_bytes(self, n: int) -> None:
        self.cold_bytes = max(0, self.cold_bytes - int(n))

    def set_state_bytes(self, n: int) -> None:
        self.state_bytes = int(n)
        self._check()

    # -- compute ----------------------------------------------------------
    def spend_predict(self, active_params: int, n: int = 1) -> None:
        self.infer_flops += FLOP_MODEL["predict_per_param"] * active_params * n
        self.record_active_params(active_params)

    def spend_decision(self, live_modules: int, dim: int) -> None:
        """Router / novelty-detector cost. Grows with live module count by design."""
        self.decision_flops += FLOP_MODEL["density_eval_per_dim"] * live_modules * dim

    def spend_train(self, dim: int, n_out: int, writes: int) -> None:
        self.train_flops += FLOP_MODEL["stat_update_per_dim2"] * dim * dim
        self.train_flops += FLOP_MODEL["predict_per_param"] * dim * n_out
        self.param_writes += int(writes)

    def spend_solve(self, dim: int) -> None:
        self.train_flops += FLOP_MODEL["solve_per_dim3"] * dim ** 3

    def spend_consolidation(self, dim: int, n_ops: int = 1) -> None:
        self.consolidation_flops += FLOP_MODEL["merge_per_dim2"] * dim * dim * n_ops

    # -- derived ----------------------------------------------------------
    @property
    def total_flops(self) -> float:
        return self.train_flops + self.infer_flops + self.decision_flops + self.consolidation_flops

    @property
    def param_added(self) -> int:
        return self.param_total - self.param_base

    @property
    def param_active_mean(self) -> float:
        if not self.active_param_samples:
            return 0.0
        return sum(self.active_param_samples) / len(self.active_param_samples)

    @property
    def storage_total(self) -> int:
        return (
            self.param_total * BYTES_PER_PARAM
            + self.state_bytes
            + self.cold_bytes
            + self.replay_bytes
        )

    def _check(self) -> None:
        c = self.ceiling
        if c.param_total is not None and self.param_total > c.param_total:
            raise BudgetBreach(f"param_total {self.param_total} > ceiling {c.param_total}")
        if c.live_modules is not None and self.live_modules > c.live_modules:
            raise BudgetBreach(f"live_modules {self.live_modules} > ceiling {c.live_modules}")
        if c.storage_bytes is not None and self.storage_total > c.storage_bytes:
            raise BudgetBreach(f"storage {self.storage_total} > ceiling {c.storage_bytes}")

    def check_flops(self) -> None:
        c = self.ceiling
        if c.total_flops is not None and self.total_flops > c.total_flops:
            raise BudgetBreach(f"total_flops {self.total_flops:.3e} > ceiling {c.total_flops:.3e}")

    def manifest(self) -> dict[str, Any]:
        return {
            "param_total": self.param_total,
            "param_peak": self.param_peak,
            "param_base": self.param_base,
            "param_added": self.param_added,
            "param_active_mean": self.param_active_mean,
            "live_modules": self.live_modules,
            "live_modules_peak": self.live_modules_peak,
            "state_bytes": self.state_bytes,
            "cold_bytes": self.cold_bytes,
            "replay_bytes": self.replay_bytes,
            "storage_total": self.storage_total,
            "train_flops": self.train_flops,
            "infer_flops": self.infer_flops,
            "decision_flops": self.decision_flops,
            "consolidation_flops": self.consolidation_flops,
            "total_algorithmic_flops": self.total_flops,
            "param_writes": self.param_writes,
            "flop_model": dict(FLOP_MODEL),
            "ceiling": self.ceiling.as_dict(),
        }
