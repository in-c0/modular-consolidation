import numpy as np
import pytest

from modular_consolidation.budget import BudgetLedger
from modular_consolidation.modules import Module, ModuleBank, merge_exact, merge_operator


def _fit(dim=8, n_out=3, n=200, seed=0):
    rng = np.random.default_rng(seed)
    phi = rng.normal(size=(n, dim))
    W = rng.normal(size=(dim, n_out))
    y = np.zeros((n, n_out))
    y[np.arange(n), np.argmax(phi @ W, axis=1)] = 1.0
    return phi, y


def test_exact_merge_equals_fitting_the_union():
    """The property the whole merge-loss decomposition rests on."""
    dim, n_out = 8, 3
    pa, ya = _fit(dim, n_out, seed=1)
    pb, yb = _fit(dim, n_out, seed=2)

    a = Module(0, dim, n_out); a.observe(pa, ya)
    b = Module(1, dim, n_out); b.observe(pb, yb)
    merged = merge_exact(a, b, 2, t=0)

    joint = Module(3, dim, n_out)
    joint.observe(np.vstack([pa, pb]), np.vstack([ya, yb]))

    assert np.allclose(merged.w, joint.w, atol=1e-8)


def test_operator_merge_differs_from_exact_merge():
    """If it did not, mechanism_loss would be identically zero by construction."""
    dim, n_out = 8, 3
    pa, ya = _fit(dim, n_out, seed=1)
    pb, yb = _fit(dim, n_out, seed=5)
    a = Module(0, dim, n_out); a.observe(pa, ya)
    b = Module(1, dim, n_out); b.observe(pb, yb)
    assert not np.allclose(merge_exact(a, b, 2, 0).w, merge_operator(a, b, 3, 0).w)


def test_merge_records_provenance():
    dim, n_out = 6, 2
    pa, ya = _fit(dim, n_out, seed=1)
    a = Module(0, dim, n_out); a.observe(pa, ya)
    b = Module(1, dim, n_out); b.observe(pa, ya)
    m = merge_exact(a, b, 7, t=4)
    assert m.merged_from == (0, 1)
    assert any("merge_exact" in p for p in m.provenance)


def test_retire_moves_cost_from_active_params_to_cold_storage():
    led = BudgetLedger()
    bank = ModuleBank(dim=8, n_out=3, ledger=led)
    m = bank.spawn(0)
    active_before = led.param_total
    assert active_before == m.deployed_params
    bank.retire(1, m.mid)
    assert led.param_total == 0
    assert led.cold_bytes > 0, "retirement must not be free"


def test_reinstate_restores_the_module_and_its_cost():
    led = BudgetLedger()
    bank = ModuleBank(dim=8, n_out=3, ledger=led)
    m = bank.spawn(0)
    bank.retire(1, m.mid)
    bank.reinstate(2, m.mid)
    assert m.mid in bank.live
    assert led.cold_bytes == 0
    assert led.param_total == m.deployed_params
    assert m.reinstated_at == [2]


def test_routing_charges_decision_compute():
    led = BudgetLedger()
    bank = ModuleBank(dim=8, n_out=3, ledger=led)
    for _ in range(3):
        bank.spawn(0)
    phi, _ = _fit(8, 3, n=32)
    bank.score_live(phi)
    assert led.decision_flops > 0


def test_module_reports_novelty_in_own_units():
    dim, n_out = 8, 3
    phi, y = _fit(dim, n_out, seed=3)
    m = Module(0, dim, n_out)
    m.observe(phi, y)
    for _ in range(5):
        m.record_self_score(float(m.log_density(phi).mean()))
    typical = m.novelty_z(m.ld_mean)
    far = m.novelty_z(m.ld_mean - 10 * m.ld_std)
    assert typical == pytest.approx(0.0, abs=1e-6)
    assert far < -5
