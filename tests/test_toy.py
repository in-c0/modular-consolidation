import numpy as np
import pytest

from modular_consolidation.toy import StreamConfig, make_stream, onehot, stream_summary


def test_stream_has_ground_truth_structure():
    s = make_stream(StreamConfig(k_star=5, n_segments=14, seed=0))
    assert s.k_star == 5
    assert len({seg.skill for seg in s.segments}) <= 5
    assert all(0 <= seg.skill < 5 for seg in s.segments)


def test_near_duplicates_do_not_increase_k_star():
    s = make_stream(StreamConfig(k_star=4, n_segments=20, near_dup_prob=0.6, seed=1))
    dups = [seg for seg in s.segments if seg.kind == "near_dup"]
    assert dups, "config should produce near-duplicates"
    assert len({seg.skill for seg in s.segments}) <= 4


def test_novel_onsets_are_first_appearances():
    s = make_stream(StreamConfig(seed=2))
    seen = set()
    for seg in s.segments:
        if seg.is_novel_onset:
            assert seg.skill not in seen
        seen.add(seg.skill)
    assert s.novel_onsets == [seg.index for seg in s.segments if seg.is_novel_onset]


def test_stream_is_deterministic_given_a_seed():
    a = make_stream(StreamConfig(seed=7))
    b = make_stream(StreamConfig(seed=7))
    assert np.array_equal(a.segments[0].X, b.segments[0].X)
    assert [s.skill for s in a.segments] == [s.skill for s in b.segments]


def test_different_seeds_give_different_streams():
    a = make_stream(StreamConfig(seed=7))
    b = make_stream(StreamConfig(seed=8))
    assert not np.array_equal(a.segments[0].X, b.segments[0].X)


def test_labels_are_not_degenerate():
    s = make_stream(StreamConfig(seed=3))
    for seg in s.segments[:4]:
        assert len(np.unique(seg.y)) > 1, "a segment with one class cannot test anything"


def test_train_and_eval_come_from_the_same_segment_distribution():
    s = make_stream(StreamConfig(seed=4))
    seg = s.segments[0]
    assert seg.X.shape[1] == seg.Xe.shape[1]
    assert abs(seg.X.mean() - seg.Xe.mean()) < 0.3


def test_onehot_round_trips():
    y = np.array([0, 2, 1])
    assert np.array_equal(np.argmax(onehot(y, 3), axis=1), y)


def test_summary_records_config_for_provenance():
    s = make_stream(StreamConfig(seed=5))
    summ = stream_summary(s)
    assert summ["config"]["seed"] == 5
    assert summ["k_star"] == s.k_star
