"""Compute profiling utilities for runtime, memory (RSS/VRAM), and energy (kWh) accounting."""

from tsdomain.compute.profiler import (
    DEFAULT_CPU_TDP_WATTS,
    profile_callable,
    summarize_profile_repeats,
)

__all__ = ["DEFAULT_CPU_TDP_WATTS", "profile_callable", "summarize_profile_repeats"]
