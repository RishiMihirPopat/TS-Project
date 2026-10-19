"""Zero-shot Time-Series Foundation Model (TSFM) wrappers and CLI runners."""

from tsdomain.models.base import (
    TSFMBaseAdapter,
    apply_common_scaling,
    invert_common_scaling,
)

__all__ = [
    "TSFMBaseAdapter",
    "apply_common_scaling",
    "invert_common_scaling",
]
