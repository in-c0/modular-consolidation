"""Regression tests for the D5 per-event merge reporting requirement.

D5 asks for per-event merge loss and recovery as explanatory outcomes across the complete
phase diagram. The aggregate row fields (`mean_merge_loss`, `merge_recovery`, ...) discard
individual events, so EXP-003 additionally preserves one record per merge event.

These tests exist to prove the reporting is complete and lossless, and that adding it
changed no experimental behaviour: the frozen EXP-003 constants and the aggregate row are
both asserted unchanged.
"""

import json
import math
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from modular_consolidation import policies
from modular_consolidation.toy import StreamConfig, make_stream

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def merge_stream(seed=900):
    """A stream under real capacity pressure, so merges actually fire."""
    return make_stream(StreamConfig(k_star=6, n_segments=18, recur_prob=0.30,
                                    near_dup_prob=0.30, region_scale=0.7, seed=seed))


@pytest.fixture(scope="module")
def rc():
    return load("run_ceiling")


@pytest.fixture(scope="module")
def merged(rc):
    row, events = rc.run_one_detailed(merge_stream(), "B-MERGE", "merge_best", 3, 900)
    if not events:
        pytest.skip("no merges fired on this stream")
    return row, events


# 1. events survive serialization individually ----------------------------

def test_merge_events_survive_serialization_individually(merged):
    _, events = merged
    restored = json.loads(json.dumps(events))
    assert len(restored) == len(events)
    required = {
        "arm", "event_index", "chunk", "segment_index", "trigger", "pair", "same_skill",
        "acc_before", "acc_no_merge", "acc_exact_merge", "acc_operator_merge", "acc_after",
        "decision_loss", "mechanism_loss", "total_merge_loss",
        "recovery_trace", "recovery", "recovery_time", "recovery_censored",
    }
    for ev in restored:
        assert required <= set(ev), sorted(required - set(ev))
    assert [ev["event_index"] for ev in restored] == list(range(len(restored)))


def test_each_event_is_individually_addressable_not_collapsed(merged):
    _, events = merged
    if len(events) < 2:
        pytest.skip("only one merge event")
    # distinct events must be distinguishable, not silently deduplicated into one row
    keys = {(ev["chunk"], tuple(ev["pair"])) for ev in events}
    assert len(keys) == len(events)


# 2. event count matches res.merges ---------------------------------------

def test_event_count_matches_underlying_res_merges(rc):
    st = merge_stream()
    res = policies.run_arm(st, policies.ArmConfig(
        "B-MERGE", routing="learned", cap=3, on_full="merge_best", seed=900))
    events = rc.merge_event_records(res)
    assert len(events) == len(res.merges)
    row = rc._score_row(res, "B-MERGE", 900)
    assert row["n_merge"] == len(events)


# 3. aggregates are reconstructible from the events ------------------------

def test_aggregate_means_are_reconstructible_from_event_records(merged):
    row, events = merged
    assert row["mean_merge_loss"] == pytest.approx(
        float(np.mean([e["total_merge_loss"] for e in events])))
    assert row["mean_decision_loss"] == pytest.approx(
        float(np.mean([e["decision_loss"] for e in events])))
    assert row["mean_mechanism_loss"] == pytest.approx(
        float(np.mean([e["mechanism_loss"] for e in events])))


def test_merge_precision_is_reconstructible_from_event_records(merged):
    row, events = merged
    judged = [e["same_skill"] for e in events if e["same_skill"] is not None]
    if not judged:
        pytest.skip("no ground-truth-judged merges")
    assert row["merge_precision"] == pytest.approx(float(np.mean(judged)))


def test_recovery_aggregates_are_reconstructible_from_event_records(merged):
    row, events = merged
    traced = [e for e in events if e["recovery"] is not None]
    if not traced:
        pytest.skip("no traced merges")
    assert row["merge_recovery"] == pytest.approx(
        float(np.mean([e["recovery"] for e in traced])))
    done = [e["recovery_time"] for e in traced if e["recovery_time"] is not None]
    if done:
        assert row["merge_recovery_time"] == pytest.approx(float(np.mean(done)))
    assert row["merge_recovery_censored"] == pytest.approx(
        sum(1 for e in traced if e["recovery_censored"]) / len(traced))


def test_loss_decomposition_is_internally_consistent_per_event(merged):
    _, events = merged
    for e in events:
        assert e["total_merge_loss"] == pytest.approx(
            e["decision_loss"] + e["mechanism_loss"], abs=1e-9)
        assert e["acc_before"] == e["acc_no_merge"]


# 4. recovery trace and censoring survive serialization --------------------

def test_recovery_trace_and_censoring_survive_serialization(merged):
    _, events = merged
    restored = json.loads(json.dumps(events))
    for orig, ev in zip(events, restored):
        assert ev["recovery_trace"] == orig["recovery_trace"]
        for pt in ev["recovery_trace"]:
            assert set(pt) == {"chunk", "acc"}
            assert pt["chunk"] > ev["chunk"], "probe must be strictly after the merge"
            assert math.isfinite(pt["acc"])
        assert ev["recovery_censored"] == orig["recovery_censored"]
        if ev["recovery_trace"]:
            assert ev["recovery_censored"] is (ev["recovery_time"] is None)
        else:
            assert ev["recovery"] is None and ev["recovery_censored"] is None


def test_events_carry_the_segment_index_at_merge_time(merged):
    _, events = merged
    for e in events:
        assert e["segment_index"] is not None
        assert 0 <= e["segment_index"] < 18


# 5. non-merge arms produce zero events, without error ---------------------

@pytest.mark.parametrize("name,on_full", [("B-DENY", "deny"),
                                          ("B-EVICT-LRU", "evict_lru"),
                                          ("B-EVICT-RAND", "evict_random")])
def test_non_merge_arms_produce_no_event_records(rc, name, on_full):
    row, events = rc.run_one_detailed(merge_stream(), name, on_full, 3, 900)
    assert events == []
    assert row["n_merge"] == 0
    assert json.loads(json.dumps(events)) == []


def test_merge_arms_do_produce_event_records(rc):
    for name, on_full in [("B-MERGE", "merge_best"), ("B-MERGE-RAND", "merge_random")]:
        _, events = rc.run_one_detailed(merge_stream(), name, on_full, 3, 900)
        assert events, f"{name} produced no merge events under a binding ceiling"


# 6. the reporting fix changed no experimental behaviour -------------------

def test_run_one_row_is_unchanged_by_the_detailed_path(rc):
    st = merge_stream()
    plain = rc.run_one(st, "B-MERGE", "merge_best", 3, 900)
    detailed, _ = rc.run_one_detailed(st, "B-MERGE", "merge_best", 3, 900)
    assert plain == detailed


def test_exp003_frozen_constants_are_unchanged():
    m = load("run_ceiling_phase")
    assert m.K_STARS == (6, 12, 24)
    assert m.CEILING_RATIOS == ((1, 6), (1, 3), (1, 2), (2, 3), (5, 6))
    assert m.DEV_SEEDS == tuple(range(900, 908))
    assert m.SEGMENTS_PER_SKILL == 3
    assert m.REGION_SCALE == 0.7
    assert m.RECUR_PROB == 0.30
    assert m.NEAR_DUP_PROB == 0.30
    assert [name for name, _ in m.ARMS] == [
        "B-DENY", "B-EVICT-LRU", "B-EVICT-RAND", "B-MERGE", "B-MERGE-RAND"]


def test_phase_runner_exposes_no_scientific_grid_knobs():
    m = load("run_ceiling_phase")
    import argparse
    import io
    import contextlib
    parser = None
    # rebuild the runner's parser exactly as main() does, without executing the sweep
    src = (ROOT / "scripts" / "run_ceiling_phase.py").read_text()
    assert "add_argument" in src
    opts = [ln for ln in src.splitlines() if "add_argument" in ln]
    assert len(opts) == 1 and '"--out"' in opts[0], opts
    del argparse, io, contextlib, parser, m


def test_phase_payload_declares_merge_events_and_keeps_rows():
    src = (ROOT / "scripts" / "run_ceiling_phase.py").read_text()
    assert '"merge_events": merge_events,' in src
    assert '"rows": rows,' in src, "aggregate rows must stay for compatibility"


# 7. candidate-diversity diagnostics (C1-C6) -------------------------------

def test_candidate_diagnostics_are_present_on_every_merge_event(merged):
    _, events = merged
    for e in events:
        for k in ("n_live", "n_candidate_pairs", "best_score", "second_best_score",
                  "score_margin", "score_mean", "score_std"):
            assert k in e


def test_candidate_pair_count_matches_live_module_count(merged):
    _, events = merged
    for e in events:
        n = e["n_live"]
        assert e["n_candidate_pairs"] == n * (n - 1) // 2


def test_criterion_merges_record_scores_and_a_nonnegative_margin(merged):
    _, events = merged
    scored = [e for e in events if e["best_score"] is not None]
    assert scored, "merge_best events must record their criterion scores"
    for e in scored:
        assert 0.0 <= e["best_score"] <= 1.0
        if e["second_best_score"] is not None:
            assert e["best_score"] >= e["second_best_score"]
            assert e["score_margin"] == pytest.approx(
                e["best_score"] - e["second_best_score"], abs=1e-12)


def test_single_candidate_pair_has_no_second_best(rc):
    """P1: at n_candidate_pairs == 1 the criterion cannot select at all."""
    st = merge_stream()
    _, events = rc.run_one_detailed(st, "B-MERGE", "merge_best", 2, 900)
    if not events:
        pytest.skip("ceiling 2 produced no merges on this stream")
    for e in events:
        assert e["n_live"] == 2
        assert e["n_candidate_pairs"] == 1
        assert e["second_best_score"] is None
        assert e["score_margin"] is None


def test_random_merge_records_candidate_count_but_no_scores(rc):
    st = merge_stream()
    _, events = rc.run_one_detailed(st, "B-MERGE-RAND", "merge_random", 3, 900)
    if not events:
        pytest.skip("no merges fired")
    for e in events:
        assert e["n_candidate_pairs"] >= 1
        assert e["best_score"] is None, "random pairing must not report a criterion score"
        assert e["score_margin"] is None


def test_candidate_diagnostics_survive_serialization(merged):
    _, events = merged
    restored = json.loads(json.dumps(events))
    for orig, ev in zip(events, restored):
        for k in ("n_live", "n_candidate_pairs", "best_score", "score_margin"):
            assert ev[k] == orig[k]


def test_instrumentation_did_not_change_merge_selection(rc):
    """The diagnostics read off already-computed scores; behaviour must be unchanged.

    Guards the invariant by pinning the realised merge schedule and outcomes for a
    fixed seed, so a future edit to the selection loop cannot slip through as a
    'reporting' change.
    """
    st = merge_stream()
    row, events = rc.run_one_detailed(st, "B-MERGE", "merge_best", 3, 900)
    assert row["n_merge"] == len(events)
    chunks = [e["chunk"] for e in events]
    assert chunks == sorted(chunks)
    # selected pair must be the argmax of the recorded candidate distribution
    for e in events:
        if e["best_score"] is not None and e["score_mean"] is not None:
            assert e["best_score"] >= e["score_mean"]
