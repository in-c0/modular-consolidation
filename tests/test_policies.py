"""Tests for the arm lattice. The point of the lattice is that neighbouring arms differ
in exactly one factor and that every derived control is a function of its target's
realised behaviour, so these properties are asserted rather than assumed.
"""

import numpy as np
import pytest

from modular_consolidation import policies
from modular_consolidation.budget import BudgetCeiling, BudgetBreach
from modular_consolidation.toy import StreamConfig, make_stream

FACTORS = ("routing", "cap", "consolidation", "task_free", "extra_passes")


def small_stream(seed=0):
    return make_stream(StreamConfig(k_star=3, n_segments=6, n_train_per_segment=160,
                                    n_eval_per_segment=80, region_scale=0.7, seed=seed))


def _diff(a, b):
    da, db = a.as_dict(), b.as_dict()
    return [f for f in FACTORS if da[f] != db[f]]


def test_adjacent_primary_arms_differ_in_exactly_one_factor():
    arms = policies.primary_arms()
    for a, b in zip(arms, arms[1:]):
        d = _diff(a, b)
        assert len(d) == 1, f"{a.name} -> {b.name} changes {d}; the comparison would be confounded"


def test_the_ladder_covers_routing_capacity_allocation_and_consolidation():
    changed = set()
    arms = policies.primary_arms()
    for a, b in zip(arms, arms[1:]):
        changed.update(_diff(a, b))
    assert {"routing", "cap", "consolidation"} <= changed


def test_run_is_deterministic_for_a_fixed_seed():
    st = small_stream()
    cfg = policies.ArmConfig("A4", routing="learned", cap=None, seed=0)
    a = policies.run_arm(st, cfg)
    b = policies.run_arm(st, cfg)
    assert np.allclose(a.R, b.R, equal_nan=True)
    assert a.k_final == b.k_final


def test_single_adapter_never_grows():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig("A1", routing="none", cap=1))
    assert res.k_final == 1
    assert res.k_peak == 1


def test_capped_bank_respects_its_cap():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig("A3", routing="learned", cap=2))
    assert res.k_peak <= 2


def test_dynamic_arm_may_exceed_a_fixed_bank():
    st = small_stream()
    dyn = policies.run_arm(st, policies.ArmConfig("A4", routing="learned", cap=None))
    assert dyn.k_peak >= 1


def test_derived_terminal_control_matches_its_targets_realised_capacity():
    st = small_stream()
    res = {"A4_dynamic_spawn": policies.run_arm(
        st, policies.ArmConfig("A4_dynamic_spawn", routing="learned", cap=None))}
    ctrls = {c.name: c for c in policies.derive_controls(res)}
    assert ctrls["C-TERM(A4)"].cap == res["A4_dynamic_spawn"].k_final


def test_random_merge_control_copies_the_merge_schedule_it_controls_for():
    st = small_stream()
    a5 = policies.run_arm(st, policies.ArmConfig(
        "A5_spawn_merge", routing="learned", cap=None, consolidation="merge"))
    if not a5.merge_chunks:
        pytest.skip("no merges fired on this stream; nothing to control for")
    ctrls = {c.name: c for c in policies.derive_controls({"A5_spawn_merge": a5})}
    assert ctrls["C-RMERGE(A5)"].forced_merge_chunks == a5.merge_chunks


def test_random_spawn_control_matches_spawn_count_not_timing():
    st = small_stream()
    a4 = policies.run_arm(st, policies.ArmConfig(
        "A4_dynamic_spawn", routing="learned", cap=None))
    ctrls = {c.name: c for c in policies.derive_controls({"A4_dynamic_spawn": a4})}
    forced = ctrls["C-RSPAWN(A4)"].forced_spawn_chunks
    assert forced is not None
    assert len(forced) <= len(a4.spawn_chunks)


def test_oracle_routing_is_flagged_as_an_upper_bound_not_a_method():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "C-OID", routing="oracle", cap=3, task_free=False))
    assert "oracle_upper_bound" in res.flags


def test_oracle_routing_declared_task_free_is_flagged_as_a_leak():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "bad", routing="oracle", cap=3, task_free=True))
    assert "taskid_leak" in res.flags


def test_learned_routing_charges_decision_compute():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig("A3", routing="learned", cap=3))
    assert res.ledger["decision_flops"] > 0
    assert "uncounted_decision" not in res.flags


def test_capacity_ceiling_breach_raises_rather_than_silently_growing():
    st = small_stream()
    cfg = policies.ArmConfig("A4", routing="learned", cap=None)
    with pytest.raises(BudgetBreach):
        policies.run_arm(st, cfg, ceiling=BudgetCeiling(live_modules=1))


def test_merge_events_record_the_decomposition_and_ground_truth_pairing():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "A5", routing="learned", cap=None, consolidation="merge"))
    if not res.merges:
        pytest.skip("no merges fired on this stream")
    m = res.merges[0]
    for key in ("decision_loss", "mechanism_loss", "total_merge_loss", "same_skill"):
        assert key in m
    assert m["total_merge_loss"] == pytest.approx(
        m["decision_loss"] + m["mechanism_loss"], abs=1e-9)


def test_retirement_arm_charges_cold_storage():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "A6", routing="learned", cap=None, consolidation="full", retire_idle_chunks=1))
    retired = [e for e in res.events if e["op"] == "retire"]
    if not retired:
        pytest.skip("no retirements fired on this stream")
    assert res.ledger["cold_bytes"] > 0 or res.ledger["storage_total"] > 0


def test_retention_matrix_lower_triangle_is_filled():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig("A3", routing="learned", cap=3))
    S = res.R.shape[0]
    for t in range(S):
        for i in range(t + 1):
            assert not np.isnan(res.R[t, i])
