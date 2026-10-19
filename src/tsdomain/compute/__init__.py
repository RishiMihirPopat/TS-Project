"""Compute profiling utilities for runtime, memory (RSS/VRAM), and energy (kWh) accounting."""

from tsdomain.compute.profiler import ComputeProfileResult, profile_callable_repeats

__all__ = ["ComputeProfileResult", "profile_callable_repeats"]
