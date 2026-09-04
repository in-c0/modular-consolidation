"""M6 — reporting-only instrumentation and derived-control arithmetic for NORACL.

Design contract, from `experiments/M6-NORACL-NATIVE-REANALYSIS-PREREG.md`:

* the official implementation runs **unmodified** at a pinned revision;
* every wrapper here calls the original function and returns its result
  **unchanged**, recording inputs and outputs on the way through;
* no growth decision, insertion width, initialization or metric is computed
  here — derived flags such as `ed_fired` are reconstructed **in analysis** from
  recorded inputs, so the instrumentation cannot drift away from the method.

Nothing in this module imports JAX or the official package at import time, so the
repository's own test suite runs without them.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Sequence

#: The exact official revision this instrumentation is written against.
NORACL_PIN = {
    "repo": "https://github.com/karthik-charan/NORACL",
    "sha": "aa0014c8478b18e70420d3ac451d4e4472ff7040",
    "branch": "main",
    "paper": "arXiv:2604.27031v1",
}

#: Authoritative identifier sets, copied verbatim from the pinned source.
#: `noracl/core/growth.py::GROWTH_TRIGGERS` and `noracl/core/init.py::INIT_STRATEGIES`.
GROWTH_TRIGGERS = ("ed_fisher", "ed_only", "fisher_only", "loss_plateau", "fixed_pertask")
INIT_STRATEGIES = ("qr_init", "he_normal", "xavier", "random", "nullspace", "zero", "vp_zfo")

BYTES_PER_PARAM = 4  # float32


# --------------------------------------------------------------- accounting

def mlp_param_count(n_in: int, widths: Sequence[int], n_out: int) -> int:
    """Weight-matrix parameter count for NORACL's MLP (no bias terms).

    Verified against the shipped published measurements: `bsmnist_2l_noracl_s0`
    reports 9 576 params at widths (12, 12) and 19 804 at widths (24, 38), with
    ``n_in=784``, ``n_out=2``.
    """
    dims = [int(n_in), *[int(w) for w in widths], int(n_out)]
    return sum(a * b for a, b in zip(dims, dims[1:]))


def derive_c_match_hidden_dim(target_params: int, n_in: int, n_layers: int,
                              n_out: int, max_width: int = 4096) -> int:
    """`C-MATCH`: the static width whose parameter count is closest to a target.

    Frozen in the preregistration: choose the integer ``hidden_dim`` minimising
    ``|params(static, hidden_dim) - target_params|``, ties broken to the smaller
    value. Derived from the target arm's own realised capacity; **never** from
    test performance.
    """
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    best_h, best_err = 1, None
    for h in range(1, max_width + 1):
        err = abs(mlp_param_count(n_in, [h] * n_layers, n_out) - target_params)
        if best_err is None or err < best_err:  # strict <: ties keep the smaller h
            best_h, best_err = h, err
    return best_h


def derive_t_count_k_fixed(initial_widths: Sequence[int], final_widths: Sequence[int],
                           n_tasks: int) -> int:
    """`T-COUNT`: the fixed per-task step matching NORACL's realised growth amount.

    Frozen in the preregistration: ``k_fixed = round(G / (T * L))`` clipped to
    ``>= 1``, where ``G`` is total neurons added, ``T = n_tasks - 1`` the number
    of tasks after the first, and ``L`` the number of growable hidden layers.

    Matches *how much* was grown while discarding *when* and *where*.
    """
    if len(initial_widths) != len(final_widths):
        raise ValueError("width vectors must have equal length")
    if n_tasks < 2:
        raise ValueError("need at least two tasks for a per-task growth step")
    grown = sum(int(f) - int(i) for i, f in zip(initial_widths, final_widths))
    layers = len(initial_widths)
    return max(1, int(round(grown / ((n_tasks - 1) * layers))))


def parameter_time_integral(width_trajectory: Iterable[tuple[int, Sequence[int]]],
                            n_in: int, n_out: int) -> int:
    """Capacity *exposure*: the sum of parameter counts over training epochs.

    A static model holds its full capacity from the first epoch, so equal final
    width does not mean equal capacity exposure. Each element is
    ``(n_epochs_at_this_width, widths)``.
    """
    return sum(int(n) * mlp_param_count(n_in, w, n_out) for n, w in width_trajectory)


# ------------------------------------------------------------ event records

@dataclass
class GrowthEvent:
    """One growth-decision record. Every field is observed, never recomputed."""

    seed: int | None = None
    arm: str | None = None
    task: int | None = None
    epoch: int | None = None
    trigger_mode: str | None = None
    init_mode: str | None = None
    # per-layer ED inputs, as passed to the official growth step
    phi_curr: list[float] = field(default_factory=list)
    phi_0: list[float] = field(default_factory=list)
    gamma: float | None = None
    widths_before: list[int] = field(default_factory=list)
    widths_after: list[int] = field(default_factory=list)
    params_before: int | None = None
    params_after: int | None = None
    fisher_tau: list[float] = field(default_factory=list)
    fisher_sat_percentile: float | None = None
    k_fixed: int | None = None
    plateau_stat: float | None = None
    grew: bool | None = None
    # measured around the insertion; validity check, not a scientific result
    function_preservation_error: float | None = None
    pre_growth_eval: float | None = None
    post_growth_eval: float | None = None
    recovery_trace: list[dict] = field(default_factory=list)
    bytes_delta: int | None = None
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


class GrowthEventRecorder:
    """Collects events and writes them as JSON Lines.

    Append-only and side-effect free with respect to the method under study.
    """

    def __init__(self, path: str | pathlib.Path | None = None, arm: str | None = None,
                 seed: int | None = None):
        self.path = pathlib.Path(path) if path else None
        self.arm = arm
        self.seed = seed
        self.events: list[GrowthEvent] = []
        self.epoch_metrics: list[dict] = []
        self._ctx: dict = {}

    # -- context the wrappers cannot infer on their own -------------------
    def set_context(self, **kwargs) -> None:
        self._ctx.update({k: v for k, v in kwargs.items() if v is not None})

    def record(self, event: GrowthEvent) -> GrowthEvent:
        if event.seed is None:
            event.seed = self.seed
        if event.arm is None:
            event.arm = self.arm
        for k, v in self._ctx.items():
            if getattr(event, k, "__missing__") in (None, [], {}):
                setattr(event, k, v)
        self.events.append(event)
        return event

    def record_epoch(self, **row) -> None:
        self.epoch_metrics.append({"arm": self.arm, "seed": self.seed, **row})

    # -- serialization ----------------------------------------------------
    def to_jsonl(self) -> str:
        return "".join(json.dumps(e.as_dict(), sort_keys=True) + "\n" for e in self.events)

    def flush(self) -> pathlib.Path | None:
        if self.path is None:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.to_jsonl())
        return self.path

    @staticmethod
    def load_jsonl(path: str | pathlib.Path) -> list[dict]:
        text = pathlib.Path(path).read_text()
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


# ---------------------------------------------- derived-in-analysis flags

def ed_fired(event: dict | GrowthEvent) -> bool | None:
    """Reconstruct the ED condition from recorded inputs.

    Mirrors `noracl/training/loop.py`: fires when ``phi_curr[l] > gamma*phi_0[l]``
    for any growable hidden layer, i.e. all but the last entry. Computed in
    analysis so the hot path stays untouched.
    """
    d = event.as_dict() if isinstance(event, GrowthEvent) else event
    pc, p0, g = d.get("phi_curr") or [], d.get("phi_0") or [], d.get("gamma")
    if not pc or not p0 or g is None or len(pc) != len(p0):
        return None
    return any(pc[l] > g * p0[l] for l in range(max(len(pc) - 1, 0)))


def width_delta(event: dict | GrowthEvent) -> list[int] | None:
    d = event.as_dict() if isinstance(event, GrowthEvent) else event
    b, a = d.get("widths_before") or [], d.get("widths_after") or []
    if not b or not a or len(b) != len(a):
        return None
    return [int(x) - int(y) for x, y in zip(a, b)]


# ------------------------------------------------------------ wrappers

def _shapes_to_widths(params: Sequence[Any]) -> list[int]:
    """Hidden-layer widths from a NORACL parameter list of weight matrices."""
    out = []
    for p in params[:-1]:
        shape = getattr(p, "shape", None)
        if shape is None or len(shape) < 2:
            return []
        out.append(int(shape[1]))
    return out


@dataclass
class Instrumentation:
    """Reporting-only wrappers around the official NORACL entry points.

    Each wrapper calls the original callable and returns its result unchanged.
    The class never decides anything.

    Patching note: `noracl/training/loop.py` does ``from noracl.core.ed import
    trigger_act``, so the *consumer module's* namespace must be patched, not the
    defining module's. ``install`` therefore takes the module object whose
    attributes the running code actually reads.
    """

    recorder: GrowthEventRecorder
    _originals: dict = field(default_factory=dict, repr=False)
    _target: Any = field(default=None, repr=False)
    _pending: GrowthEvent | None = field(default=None, repr=False)

    # -- wrappers ---------------------------------------------------------
    def wrap_neurogenesis_step(self, original: Callable) -> Callable:
        def wrapped(params, fisher, fisher_t, params_prev, phi_0, phi_curr, M, gamma,
                    init_fn, tau_l, fisher_sat_percentile, *args, **kwargs):
            before = _shapes_to_widths(params)
            result = original(params, fisher, fisher_t, params_prev, phi_0, phi_curr, M,
                              gamma, init_fn, tau_l, fisher_sat_percentile, *args, **kwargs)
            new_params = result[0] if isinstance(result, tuple) else result
            after = _shapes_to_widths(new_params)
            ev = GrowthEvent(
                phi_curr=[float(x) for x in phi_curr],
                phi_0=[float(x) for x in phi_0],
                gamma=float(gamma),
                widths_before=before,
                widths_after=after,
                fisher_tau=[float(t) for t in (tau_l or [])],
                fisher_sat_percentile=(None if fisher_sat_percentile is None
                                       else float(fisher_sat_percentile)),
                trigger_mode=kwargs.get("trigger"),
                k_fixed=kwargs.get("k_fixed"),
                grew=(before != after),
            )
            self.recorder.record(ev)
            self._pending = ev
            return result
        return wrapped

    def wrap_trigger_act(self, original: Callable) -> Callable:
        def wrapped(mat, *args, **kwargs):
            value = original(mat, *args, **kwargs)
            self.recorder.record_epoch(kind="trigger_act", value=float(value))
            return value
        return wrapped

    def wrap_resolve_init_fn(self, original: Callable) -> Callable:
        def wrapped(name, *args, **kwargs):
            fn = original(name, *args, **kwargs)
            self.recorder.set_context(init_mode=name)
            return fn
        return wrapped

    # -- install / uninstall ---------------------------------------------
    def install(self, target_module: Any, *, names: dict[str, str] | None = None) -> None:
        """Patch the consumer module's attribute bindings. Idempotent per name."""
        names = names or {
            "neurogenesis_step": "wrap_neurogenesis_step",
            "trigger_act": "wrap_trigger_act",
            "resolve_init_fn": "wrap_resolve_init_fn",
        }
        self._target = target_module
        for attr, wrapper_name in names.items():
            original = getattr(target_module, attr, None)
            if original is None or attr in self._originals:
                continue
            self._originals[attr] = original
            setattr(target_module, attr, getattr(self, wrapper_name)(original))

    def uninstall(self) -> None:
        for attr, original in self._originals.items():
            setattr(self._target, attr, original)
        self._originals.clear()

    def __enter__(self) -> "Instrumentation":
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()
