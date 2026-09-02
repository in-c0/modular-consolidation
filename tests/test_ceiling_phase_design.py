from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_phase():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ceiling_phase.py"
    spec = spec_from_file_location("run_ceiling_phase", path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_exp003_grid_is_frozen_and_exact():
    m = load_phase()
    assert m.K_STARS == (6, 12, 24)
    assert m.CEILING_RATIOS == ((1, 6), (1, 3), (1, 2), (2, 3), (5, 6))
    assert m.DEV_SEEDS == tuple(range(900, 908))
    assert [[m.ceiling_for(k, r) for r in m.CEILING_RATIOS] for k in m.K_STARS] == [
        [1, 2, 3, 4, 5],
        [2, 4, 6, 8, 10],
        [4, 8, 12, 16, 20],
    ]


def test_exp003_stream_scaling_preserves_density_and_recurrence_config():
    m = load_phase()
    for k_star in m.K_STARS:
        stream = m.stream_for(k_star, 900)
        assert stream.cfg.n_segments == 3 * k_star
        assert stream.cfg.recur_prob == 0.30
        assert stream.cfg.near_dup_prob == 0.30
        assert stream.cfg.region_scale == 0.7
        diag = m.stream_diagnostic(stream)
        assert diag["n_segments"] == 3 * k_star
        # Total segment mass divided over K* latent skills is exactly three;
        # realised per-skill exposure dispersion is reported rather than resampled away.
        assert diag["exposure_mean"] == 3.0


def test_exp003_uses_all_five_predeclared_slot_policies():
    m = load_phase()
    assert [name for name, _ in m.ARMS] == [
        "B-DENY",
        "B-EVICT-LRU",
        "B-EVICT-RAND",
        "B-MERGE",
        "B-MERGE-RAND",
    ]
