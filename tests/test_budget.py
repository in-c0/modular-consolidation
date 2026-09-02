import pytest

from modular_consolidation.budget import BudgetCeiling, BudgetLedger, BudgetBreach


def test_param_ceiling_is_enforced():
    led = BudgetLedger(ceiling=BudgetCeiling(param_total=100))
    led.add_params(100)
    with pytest.raises(BudgetBreach):
        led.add_params(1)


def test_live_module_ceiling_is_enforced():
    led = BudgetLedger(ceiling=BudgetCeiling(live_modules=2))
    led.set_live_modules(2)
    with pytest.raises(BudgetBreach):
        led.set_live_modules(3)


def test_decision_flops_are_counted_separately_and_included_in_total():
    led = BudgetLedger()
    led.spend_decision(live_modules=8, dim=16)
    assert led.decision_flops > 0
    assert led.total_flops == pytest.approx(led.decision_flops)


def test_decision_cost_grows_with_live_modules():
    a, b = BudgetLedger(), BudgetLedger()
    a.spend_decision(2, 16)
    b.spend_decision(20, 16)
    assert b.decision_flops > a.decision_flops


def test_cold_storage_is_charged():
    led = BudgetLedger()
    before = led.storage_total
    led.add_cold_bytes(4096)
    assert led.storage_total == before + 4096
    led.remove_cold_bytes(4096)
    assert led.storage_total == before


def test_param_added_is_relative_to_declared_base():
    led = BudgetLedger()
    led.set_base(1000)
    led.add_params(1000)
    led.add_params(250)
    assert led.param_added == 250


def test_manifest_records_the_flop_model_so_it_can_be_criticised():
    m = BudgetLedger().manifest()
    assert "flop_model" in m and m["flop_model"]
