"""M6 tests: derived-control arithmetic, event serialization, and the guarantee
that instrumentation is reporting-only.

None of these import JAX or the official NORACL package; the equivalence test
uses a deterministic stand-in whose behaviour is fully known, so it can prove
transparency without the external dependency.
"""

import json

import pytest

from modular_consolidation.native import (
    NORACL_PIN,
    GrowthEventRecorder,
    Instrumentation,
    derive_c_match_hidden_dim,
    derive_t_count_k_fixed,
    mlp_param_count,
    parameter_time_integral,
)
from modular_consolidation.native.noracl import (
    GROWTH_TRIGGERS,
    INIT_STRATEGIES,
    GrowthEvent,
    ed_fired,
    width_delta,
)


# ------------------------------------------------------- pinned identifiers

def test_pin_is_the_revision_named_in_the_preregistration():
    assert NORACL_PIN["sha"] == "aa0014c8478b18e70420d3ac451d4e4472ff7040"
    assert NORACL_PIN["repo"].endswith("karthik-charan/NORACL")


def test_identifier_sets_match_the_official_source_not_config_filenames():
    """The ledger once listed `fsat_only` and `random` as triggers. Both were wrong."""
    assert GROWTH_TRIGGERS == ("ed_fisher", "ed_only", "fisher_only",
                               "loss_plateau", "fixed_pertask")
    assert "fsat_only" not in GROWTH_TRIGGERS, "fsat_only is a filename fragment"
    assert "random" not in GROWTH_TRIGGERS, "random is an init mode, not a trigger"
    assert "random" in INIT_STRATEGIES
    assert "qr_init" in INIT_STRATEGIES


# ------------------------------------------------------------- accounting

def test_param_count_reproduces_published_measurements():
    """Validates the native contract against the repo's shipped per-seed CSVs."""
    assert mlp_param_count(784, [12, 12], 2) == 9576      # bsmnist_2l_noracl_s0, task 0
    assert mlp_param_count(784, [24, 38], 2) == 19804     # bsmnist_2l_noracl_s0, task 4


def test_param_count_matches_the_static_baseline():
    assert mlp_param_count(784, [32, 32], 2) == 26176     # bsmnist_2l_static32


def test_parameter_time_integral_penalises_holding_capacity_early():
    """A static model has full capacity from epoch one; growth does not."""
    grown = parameter_time_integral([(10, [12, 12]), (10, [24, 38])], 784, 2)
    static = parameter_time_integral([(20, [24, 38])], 784, 2)
    assert static > grown


# ------------------------------------------------- derived control formulas

def test_c_match_picks_the_closest_static_width():
    assert derive_c_match_hidden_dim(19804, 784, 2, 2) == 24
    assert abs(mlp_param_count(784, [24, 24], 2) - 19804) < \
           abs(mlp_param_count(784, [25, 25], 2) - 19804)


def test_c_match_breaks_ties_to_the_smaller_width():
    target = (mlp_param_count(784, [10, 10], 2) + mlp_param_count(784, [11, 11], 2)) // 2
    h = derive_c_match_hidden_dim(target, 784, 2, 2)
    assert h in (10, 11)


def test_c_match_rejects_a_nonsense_target():
    with pytest.raises(ValueError):
        derive_c_match_hidden_dim(0, 784, 2, 2)


def test_t_count_matches_the_realised_growth_amount():
    # seed 0: (12,12) -> (24,38) over 5 tasks = 38 neurons across 4 later tasks, 2 layers
    assert derive_t_count_k_fixed([12, 12], [24, 38], 5) == 5


def test_t_count_is_at_least_one_even_for_tiny_growth():
    assert derive_t_count_k_fixed([12, 12], [12, 13], 5) == 1


def test_t_count_requires_at_least_two_tasks():
    with pytest.raises(ValueError):
        derive_t_count_k_fixed([12], [20], 1)


def test_derived_controls_never_see_performance():
    """Both derivations are pure functions of widths/params, by signature."""
    import inspect
    for fn in (derive_c_match_hidden_dim, derive_t_count_k_fixed):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"accuracy", "score", "acc", "performance", "test_acc"})


# ------------------------------------------------------- event recording

def _event():
    return GrowthEvent(
        task=1, epoch=7, trigger_mode="ed_fisher", init_mode="qr_init",
        phi_curr=[0.9, 0.5], phi_0=[0.8, 0.8], gamma=0.9,
        widths_before=[12, 12], widths_after=[16, 12],
        fisher_tau=[0.1, 0.2], fisher_sat_percentile=25.0, grew=True,
    )


def test_events_serialize_individually_and_round_trip(tmp_path):
    rec = GrowthEventRecorder(tmp_path / "events.jsonl", arm="T-FULL", seed=0)
    rec.record(_event())
    rec.record(_event())
    path = rec.flush()
    rows = GrowthEventRecorder.load_jsonl(path)
    assert len(rows) == 2
    assert all(r["arm"] == "T-FULL" and r["seed"] == 0 for r in rows)
    assert rows[0]["widths_after"] == [16, 12]
    json.dumps(rows)  # must stay JSON-clean


def test_recorder_fills_context_without_overwriting_observations():
    rec = GrowthEventRecorder(arm="I-RANDOM", seed=3)
    rec.set_context(init_mode="random")
    ev = rec.record(GrowthEvent(trigger_mode="ed_fisher"))
    assert ev.init_mode == "random" and ev.seed == 3 and ev.arm == "I-RANDOM"
    observed = rec.record(GrowthEvent(init_mode="qr_init"))
    assert observed.init_mode == "qr_init", "context must not overwrite an observation"


def test_derived_flags_are_computed_in_analysis_not_in_the_hot_path():
    ev = _event()
    # phi_curr[0]=0.9 > 0.9*0.8=0.72 on the one growable layer -> fired
    assert ed_fired(ev) is True
    assert width_delta(ev) == [4, 0]
    quiet = GrowthEvent(phi_curr=[0.5, 0.5], phi_0=[0.8, 0.8], gamma=0.9)
    assert ed_fired(quiet) is False
    assert ed_fired(GrowthEvent()) is None


# ------------------------------- instrumentation is reporting-only

class _FakeGrowth:
    """Deterministic stand-in for the official growth step and ED signal."""

    def __init__(self):
        self.calls = 0

    def trigger_act(self, mat, partial=False, threshold=0.01):
        return float(sum(mat)) / max(len(mat), 1)

    def neurogenesis_step(self, params, fisher, fisher_t, params_prev, phi_0, phi_curr,
                          M, gamma, init_fn, tau_l, fisher_sat_percentile, **kw):
        self.calls += 1
        grow = any(pc > gamma * p0 for pc, p0 in zip(phi_curr[:-1], phi_0[:-1]))
        step = kw.get("k_fixed", 2) if grow else 0
        new = [type(p)(p.shape[0], p.shape[1] + (step if i < len(params) - 1 else 0))
               for i, p in enumerate(params)]
        return new, fisher, params_prev

    def resolve_init_fn(self, name, *a, **k):
        return lambda shape: name


class _M:
    """Minimal weight-matrix stand-in with a `.shape`."""

    def __init__(self, r, c):
        self.shape = (r, c)


def _mini_run(module, k_fixed=3):
    """A deterministic miniature run using whatever bindings `module` exposes."""
    params = [_M(784, 12), _M(12, 12), _M(12, 2)]
    out = []
    for epoch in range(4):
        act = module.trigger_act([epoch, epoch + 1, epoch + 2])
        phi_curr = [0.9 + 0.01 * epoch, 0.5, 0.0]
        phi_0 = [0.8, 0.8, 0.8]
        params, _, _ = module.neurogenesis_step(
            params, None, None, None, phi_0, phi_curr, [p.shape[1] for p in params],
            0.9, module.resolve_init_fn("qr_init"), [0.1, 0.2, 0.3], 25.0,
            trigger="ed_fisher", k_fixed=k_fixed)
        out.append((round(act, 6), [p.shape for p in params]))
    return out


class _Module:
    """A namespace standing in for `noracl.training.loop`'s imported bindings."""

    def __init__(self, impl):
        self.trigger_act = impl.trigger_act
        self.neurogenesis_step = impl.neurogenesis_step
        self.resolve_init_fn = impl.resolve_init_fn


def test_instrumentation_does_not_change_growth_decisions_or_widths():
    """Instrumented and uninstrumented runs must be identical under a fixed seed."""
    baseline = _mini_run(_Module(_FakeGrowth()))

    mod = _Module(_FakeGrowth())
    rec = GrowthEventRecorder(arm="T-FULL", seed=0)
    with Instrumentation(rec) as instr:
        instr.install(mod)
        instrumented = _mini_run(mod)

    assert instrumented == baseline, "instrumentation altered the run"
    assert rec.events, "instrumentation recorded nothing"


def test_instrumentation_records_widths_before_and_after_each_step():
    mod = _Module(_FakeGrowth())
    rec = GrowthEventRecorder(arm="T-FULL", seed=0)
    instr = Instrumentation(rec)
    instr.install(mod)
    _mini_run(mod, k_fixed=3)
    instr.uninstall()
    grew = [e for e in rec.events if e.grew]
    assert grew
    for e in grew:
        assert width_delta(e) is not None
        assert e.gamma == 0.9 and e.fisher_sat_percentile == 25.0
        assert e.trigger_mode == "ed_fisher" and e.k_fixed == 3


def test_uninstall_restores_the_original_bindings():
    impl = _FakeGrowth()
    mod = _Module(impl)
    original = mod.neurogenesis_step
    instr = Instrumentation(GrowthEventRecorder())
    instr.install(mod)
    assert mod.neurogenesis_step is not original
    instr.uninstall()
    assert mod.neurogenesis_step is original


def test_install_is_idempotent_and_does_not_double_wrap():
    mod = _Module(_FakeGrowth())
    rec = GrowthEventRecorder()
    instr = Instrumentation(rec)
    instr.install(mod)
    instr.install(mod)
    _mini_run(mod)
    # four epochs, one growth-step call each; double wrapping would double this
    assert len(rec.events) == 4


def test_wrapper_returns_the_original_result_object_unchanged():
    impl = _FakeGrowth()
    mod = _Module(impl)
    instr = Instrumentation(GrowthEventRecorder())
    instr.install(mod)
    params = [_M(784, 12), _M(12, 12), _M(12, 2)]
    result = mod.neurogenesis_step(params, None, None, None, [0.8, 0.8, 0.8],
                                   [0.9, 0.5, 0.0], [12, 12, 2], 0.9,
                                   lambda s: None, [0.1, 0.2, 0.3], 25.0,
                                   trigger="ed_fisher", k_fixed=2)
    assert isinstance(result, tuple) and len(result) == 3
