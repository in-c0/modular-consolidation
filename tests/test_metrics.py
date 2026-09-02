import numpy as np
import pytest

from modular_consolidation import metrics


def test_retention_matrix_stats_separates_plasticity_from_retention():
    # learns each segment well (diagonal high) but forgets everything (last row low)
    R = np.array([[0.9, np.nan, np.nan],
                  [0.5, 0.9, np.nan],
                  [0.3, 0.4, 0.9]])
    s = metrics.retention_matrix_stats(R)
    assert s["plasticity"] == pytest.approx(0.9)
    assert s["retention"] == pytest.approx(0.35)
    assert s["forgetting"] > 0.4
    assert s["bwt"] < 0


def test_no_forgetting_gives_zero_forgetting_and_zero_bwt():
    R = np.array([[0.8, np.nan], [0.8, 0.8]])
    s = metrics.retention_matrix_stats(R)
    assert s["forgetting"] == pytest.approx(0.0)
    assert s["bwt"] == pytest.approx(0.0)


def test_ppap_punishes_unbounded_growth():
    good = metrics.ppap(0.90, 0.70, param_added=1_000)
    bloated = metrics.ppap(0.91, 0.70, param_added=100_000)
    assert good > bloated, "a tiny gain bought with huge growth must score worse"


def test_pareto_front_keeps_only_undominated_points():
    # (accuracy up, params down)
    pts = [(0.9, 100), (0.8, 100), (0.95, 500), (0.9, 400)]
    front = set(metrics.pareto_front(pts))
    assert 0 in front and 2 in front
    assert 1 not in front and 3 not in front


def test_specialisation_nmi_is_one_for_perfect_specialisation():
    a = [0, 0, 1, 1, 2, 2]
    g = [5, 5, 6, 6, 7, 7]
    assert metrics.specialisation_nmi(a, g) == pytest.approx(1.0)


def test_specialisation_nmi_is_zero_when_routing_ignores_skill():
    a = [0, 1, 0, 1, 0, 1]
    g = [0, 0, 0, 0, 0, 0]
    assert metrics.specialisation_nmi(a, g) == pytest.approx(0.0)


def test_routing_entropy_is_zero_for_hard_routing():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert metrics.routing_entropy(p) < 1e-9  # clipped, not exactly zero


def test_merge_decomposition_splits_decision_from_mechanism():
    d = metrics.decompose_merge(acc_no_merge=0.90, acc_exact_merge=0.85,
                                acc_operator_merge=0.80)
    assert d["decision_loss"] == pytest.approx(0.05)
    assert d["mechanism_loss"] == pytest.approx(0.05)
    assert d["total_merge_loss"] == pytest.approx(0.10)


def test_a_perfect_operator_has_zero_mechanism_loss():
    d = metrics.decompose_merge(0.90, 0.85, 0.85)
    assert d["mechanism_loss"] == pytest.approx(0.0)
    assert d["decision_loss"] == pytest.approx(0.05)


def test_recovery_is_clipped_and_time_can_be_censored():
    assert metrics.recovery(loss=0.1, acc_after=0.7, acc_later=0.9) == pytest.approx(1.0)
    assert metrics.recovery(loss=0.1, acc_after=0.7, acc_later=0.75) == pytest.approx(0.5)
    assert metrics.recovery_time(0.1, 0.7, [0.71, 0.75]) is None
    assert metrics.recovery_time(0.1, 0.7, [0.71, 0.79, 0.80]) == 2


def test_allocation_error_detects_over_and_under_allocation():
    over = metrics.allocation_stats(k_final=12, k_peak=12, k_star=6)
    under = metrics.allocation_stats(k_final=3, k_peak=6, k_star=6)
    assert over["over_allocation"] == pytest.approx(1.0)
    assert over["under_allocation"] == 0.0
    assert under["under_allocation"] == pytest.approx(0.5)


def test_spawn_precision_recall_matches_onsets_within_tolerance():
    r = metrics.spawn_precision_recall(spawn_times=[10, 50], novel_onsets=[12, 90],
                                       tolerance=5)
    assert r["spawn_precision"] == pytest.approx(0.5)
    assert r["spawn_recall"] == pytest.approx(0.5)


def test_zombie_rate_flags_modules_with_no_traffic():
    stats = metrics.reuse_stats([{"op": "spawn"}], n_segments=4,
                                traffic_share={0: 0.99, 1: 0.001})
    assert stats["zombie_rate"] == pytest.approx(0.5)


def test_paired_bootstrap_requires_paired_samples():
    with pytest.raises(ValueError):
        metrics.paired_bootstrap_ci([1.0, 2.0], [1.0])


def test_paired_bootstrap_detects_a_consistent_difference():
    a = [0.80, 0.82, 0.79, 0.81, 0.83]
    b = [0.70, 0.72, 0.69, 0.71, 0.73]
    out = metrics.paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert out["excludes_zero"]
    assert out["mean_diff"] == pytest.approx(0.10, abs=1e-9)
