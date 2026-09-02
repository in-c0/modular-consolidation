"""Tests for the operations added after the first owner review: structured compression,
permanent pruning, binding-ceiling slot policies, soft routing and recovery tracing.
"""

import numpy as np
import pytest

from modular_consolidation import policies
from modular_consolidation.budget import BudgetLedger
from modular_consolidation.modules import Module, ModuleBank, merge_exact
from modular_consolidation.toy import StreamConfig, make_stream


def small_stream(seed=0, **kw):
    base = dict(k_star=3, n_segments=6, n_train_per_segment=160, n_eval_per_segment=80,
                region_scale=0.7, seed=seed)
    base.update(kw)
    return make_stream(StreamConfig(**base))


# ---------------------------------------------------------------- compression

def test_compression_reduces_deployed_parameters():
    m = Module(0, dim=16, n_out=4)
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(64, 16))
    y = np.eye(4)[rng.integers(0, 4, 64)]
    m.observe(phi, y)
    before = m.deployed_params
    freed = m.compress_to(8)
    assert freed > 0
    assert m.deployed_params == before - freed
    assert m.width == 8


def test_compression_is_a_commitment_not_revisited_by_later_learning():
    m = Module(0, dim=16, n_out=4)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(64, 16))
    y = np.eye(4)[rng.integers(0, 4, 64)]
    m.observe(phi, y)
    m.compress_to(6)
    kept = m.active_dims.copy()
    m.observe(rng.normal(size=(64, 16)), np.eye(4)[rng.integers(0, 4, 64)])
    assert np.array_equal(m.active_dims, kept)
    assert m.width == 6


def test_compressed_module_only_uses_retained_dimensions():
    m = Module(0, dim=16, n_out=4)
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(64, 16))
    m.observe(phi, np.eye(4)[rng.integers(0, 4, 64)])
    m.compress_to(5)
    dropped = np.setdiff1d(np.arange(16), m.active_dims)
    assert np.allclose(m.w_effective[dropped], 0.0)


def test_compression_cannot_grow_a_module():
    m = Module(0, dim=16, n_out=4)
    rng = np.random.default_rng(3)
    m.observe(rng.normal(size=(64, 16)), np.eye(4)[rng.integers(0, 4, 64)])
    m.compress_to(4)
    assert m.compress_to(12) == 0
    assert m.width == 4


def test_bank_compression_frees_budget():
    led = BudgetLedger()
    bank = ModuleBank(dim=16, n_out=4, ledger=led)
    m = bank.spawn(0)
    rng = np.random.default_rng(4)
    m.observe(rng.normal(size=(64, 16)), np.eye(4)[rng.integers(0, 4, 64)])
    before = led.param_total
    bank.compress(1, m.mid, 8)
    assert led.param_total < before
    assert led.consolidation_flops > 0


def test_merging_compressed_modules_keeps_the_union_of_dimensions():
    rng = np.random.default_rng(5)
    a, b = Module(0, 16, 4), Module(1, 16, 4)
    for m in (a, b):
        m.observe(rng.normal(size=(64, 16)), np.eye(4)[rng.integers(0, 4, 64)])
    a.compress_to(6)
    b.compress_to(6)
    merged = merge_exact(a, b, 2, 0)
    assert set(merged.active_dims) == set(a.active_dims) | set(b.active_dims)


# --------------------------------------------------------------------- prune

def test_prune_frees_capacity_and_leaves_no_cold_cost():
    led = BudgetLedger()
    bank = ModuleBank(dim=8, n_out=3, ledger=led)
    m = bank.spawn(0)
    bank.prune(1, m.mid)
    assert led.param_total == 0
    assert led.cold_bytes == 0, "a pruned module is gone; it must not be charged storage"
    assert m.mid in bank.pruned


def test_prune_and_retire_are_different_operations():
    led_p, led_r = BudgetLedger(), BudgetLedger()
    bp = ModuleBank(dim=8, n_out=3, ledger=led_p)
    br = ModuleBank(dim=8, n_out=3, ledger=led_r)
    bp.prune(1, bp.spawn(0).mid)
    br.retire(1, br.spawn(0).mid)
    assert led_p.cold_bytes == 0
    assert led_r.cold_bytes > 0


# ------------------------------------------------------- binding ceiling

@pytest.mark.parametrize("on_full", ["deny", "evict_lru", "evict_random",
                                     "merge_best", "merge_random"])
def test_every_slot_policy_respects_the_ceiling(on_full):
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "B", routing="learned", cap=2, on_full=on_full, seed=0))
    assert res.k_peak <= 2
    assert res.k_final <= 2


def test_slot_policies_end_at_identical_capacity():
    """The whole point of the regime: capacity is equal by construction."""
    st = small_stream()
    totals = set()
    for on_full in ("deny", "evict_lru", "merge_best", "merge_random"):
        res = policies.run_arm(st, policies.ArmConfig(
            "B", routing="learned", cap=2, on_full=on_full, seed=0))
        totals.add(res.ledger["param_total"])
    assert len(totals) == 1, f"arms differ in capacity: {totals}"


def test_deny_never_frees_a_slot():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "B-DENY", routing="learned", cap=2, on_full="deny", seed=0))
    assert not res.evictions
    assert not res.merges


def test_evict_records_evictions_and_merge_does_not():
    st = small_stream()
    ev = policies.run_arm(st, policies.ArmConfig(
        "B-EVICT", routing="learned", cap=2, on_full="evict_lru", seed=0))
    mg = policies.run_arm(st, policies.ArmConfig(
        "B-MERGE", routing="learned", cap=2, on_full="merge_best", seed=0))
    if ev.evictions:
        assert not ev.merges
    if mg.merges:
        assert not mg.evictions


def test_ceiling_merges_are_marked_as_ceiling_triggered():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "B-MERGE", routing="learned", cap=2, on_full="merge_best", seed=0))
    if not res.merges:
        pytest.skip("ceiling never bound on this stream")
    assert all(m.get("trigger") == "ceiling" for m in res.merges)


# ------------------------------------------------------------ soft routing

def test_soft_routing_activates_every_live_module():
    st = small_stream()
    hard = policies.run_arm(st, policies.ArmConfig("h", routing="learned", cap=4, seed=0))
    soft = policies.run_arm(st, policies.ArmConfig("s", routing="soft", cap=4, seed=0))
    assert soft.ledger["param_active_mean"] > hard.ledger["param_active_mean"]


def test_soft_routing_charges_decision_compute():
    st = small_stream()
    res = policies.run_arm(st, policies.ArmConfig("s", routing="soft", cap=3, seed=0))
    assert res.ledger["decision_flops"] > 0
    assert "uncounted_decision" not in res.flags


# --------------------------------------------------------- recovery tracing

def test_merge_events_carry_a_recovery_trace_on_a_frozen_probe():
    st = small_stream(n_segments=8)
    res = policies.run_arm(st, policies.ArmConfig(
        "B-MERGE", routing="learned", cap=2, on_full="merge_best", seed=0))
    if not res.merges:
        pytest.skip("no merges fired")
    traced = [m for m in res.merges if m.get("recovery_trace")]
    assert traced, "merges must record post-merge recovery probes"
    m = traced[0]
    assert all(pt["chunk"] > m["chunk"] for pt in m["recovery_trace"])


# ------------------------------------------------------------- interference

def test_interference_forces_skills_to_share_input_regions():
    sep = make_stream(StreamConfig(k_star=6, interference=0.0, seed=0))
    shared = make_stream(StreamConfig(k_star=6, interference=1.0, seed=0))
    assert sep.n_regions == 6
    assert shared.n_regions == 1
    assert len(set(shared.region_of_skill)) == 1


def test_default_stream_config_reproduces_cams_v0():
    """EXP-000 must stay reproducible; the interference knobs default to off."""
    cfg = StreamConfig()
    assert cfg.interference == 0.0
    assert cfg.n_context == 0
    s = make_stream(cfg)
    assert s.n_regions == cfg.k_star
    assert s.segments[0].X.shape[1] == cfg.d_in
