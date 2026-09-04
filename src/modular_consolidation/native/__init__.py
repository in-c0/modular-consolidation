"""Layer-B native-fidelity support.

Nothing in this subpackage implements a published method. It provides
reporting-only instrumentation and derived-control arithmetic for running an
*official* implementation unmodified, at a pinned revision.
"""

from .noracl import (
    NORACL_PIN,
    GrowthEventRecorder,
    Instrumentation,
    mlp_param_count,
    derive_c_match_hidden_dim,
    derive_t_count_k_fixed,
    parameter_time_integral,
)

__all__ = [
    "NORACL_PIN",
    "GrowthEventRecorder",
    "Instrumentation",
    "mlp_param_count",
    "derive_c_match_hidden_dim",
    "derive_t_count_k_fixed",
    "parameter_time_integral",
]
