"""Shared input checks."""

from __future__ import annotations

import numbers


def check_count(value: object, name: str) -> int:
    """Return ``value`` as a non-negative ``int`` or raise."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def check_open_unit(value: object, name: str) -> float:
    """Return ``value`` as a ``float`` in the open interval (0, 1) or raise."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {value}")
    return value
