"""Paired comparison of two binary outcomes on the same episodes.

Two gates (or two configurations) are run on, or replayed over, the same
episodes, and each episode yields a binary outcome under both. The 2x2 paired
counts are

=================  ====================  =====================
                   second outcome = 1    second outcome = 0
=================  ====================  =====================
first outcome = 1  ``both``              ``first_only``
first outcome = 0  ``second_only``       ``neither``
=================  ====================  =====================

and the difference of paired proportions is
``(first_only - second_only) / n``.

Reference: R. G. Newcombe, "Improved confidence intervals for the difference
between binomial proportions based on paired data", Statistics in Medicine
17:2635-2650 (1998). In its notation ``e, f, g, h`` are ``both``,
``first_only``, ``second_only``, ``neither``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Tuple, Union

from scipy.stats import norm

from ._checks import check_count as _check_count
from ._checks import check_open_unit

__all__ = ["PairedInterval", "wilson_interval", "newcombe_paired_interval", "is_equivalent"]


def _wilson(x: int, n: int, z: float) -> Tuple[float, float]:
    p = x / n
    z2 = z * z
    center = 2.0 * x + z2
    half = z * math.sqrt(z2 + 4.0 * x * (1.0 - p))
    denom = 2.0 * (n + z2)
    lower = 0.0 if x == 0 else (center - half) / denom
    upper = 1.0 if x == n else (center + half) / denom
    return lower, upper


def wilson_interval(x: int, n: int, level: float = 0.95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a single proportion ``x / n``.

    The limits are the roots of ``|m - x/n| = z sqrt(m (1 - m) / n)`` with
    ``z`` the ``(1 + level) / 2`` normal quantile, without continuity
    correction.
    """
    x = _check_count(x, "x")
    n = _check_count(n, "n")
    if n == 0:
        raise ValueError("n must be at least 1")
    if x > n:
        raise ValueError(f"x must not exceed n, got x={x}, n={n}")
    z = float(norm.ppf(0.5 + check_open_unit(level, "level") / 2.0))
    return _wilson(x, n, z)


@dataclass(frozen=True)
class PairedInterval:
    """Confidence interval for a difference of paired proportions.

    Attributes
    ----------
    estimate:
        ``(first_only - second_only) / n``.
    lower, upper:
        Interval limits.
    level:
        Two-sided confidence level.
    phi:
        Correlation estimate used to combine the two score intervals.
    method:
        ``"newcombe10"`` or ``"newcombe8"``.
    """

    estimate: float
    lower: float
    upper: float
    level: float
    phi: float
    method: str

    @property
    def interval(self) -> Tuple[float, float]:
        """``(lower, upper)``."""
        return (self.lower, self.upper)


def newcombe_paired_interval(
    both: int,
    first_only: int,
    second_only: int,
    neither: int,
    level: float = 0.95,
    *,
    phi_correction: bool = True,
) -> PairedInterval:
    """Newcombe (1998) method 10 interval for a difference of paired proportions.

    The first and second marginal proportions ``p1 = (both + first_only) / n``
    and ``p2 = (both + second_only) / n`` each get a Wilson score interval
    ``[l1, u1]`` and ``[l2, u2]``. With ``theta = p1 - p2`` the interval is
    ``[theta - d, theta + e]``, where

    ``d = sqrt((p1 - l1)^2 - 2 phi (p1 - l1)(u2 - p2) + (u2 - p2)^2)``,
    ``e = sqrt((u1 - p1)^2 - 2 phi (u1 - p1)(p2 - l2) + (p2 - l2)^2)``,

    and ``phi = (both * neither - first_only * second_only) / sqrt(product of
    the four marginal totals)``, set to 0 when that product is 0. Method 10
    replaces the numerator of ``phi`` by ``max(both * neither - first_only *
    second_only - n / 2, 0)`` when it is positive. Set
    ``phi_correction=False`` for the uncorrected ``phi`` (method 8 of the same
    reference).

    Parameters
    ----------
    both, first_only, second_only, neither:
        Paired 2x2 counts (non-negative integers, total at least 1).
    level:
        Two-sided confidence level (default 0.95).
    phi_correction:
        Apply the continuity correction to ``phi`` (method 10, default).
    """
    e = _check_count(both, "both")
    f = _check_count(first_only, "first_only")
    g = _check_count(second_only, "second_only")
    h = _check_count(neither, "neither")
    level = check_open_unit(level, "level")
    if not isinstance(phi_correction, bool):
        raise TypeError(f"phi_correction must be a bool, got {phi_correction!r}")
    n = e + f + g + h
    if n == 0:
        raise ValueError("the paired counts must sum to at least 1")

    z = float(norm.ppf(0.5 + level / 2.0))
    p1 = (e + f) / n
    p2 = (e + g) / n
    l1, u1 = _wilson(e + f, n, z)
    l2, u2 = _wilson(e + g, n, z)

    margins = (e + f, g + h, e + g, f + h)
    if 0 in margins:
        phi = 0.0
    else:
        numerator = float(e * h - f * g)
        if phi_correction and numerator > 0:
            numerator = max(numerator - n / 2.0, 0.0)
        phi = numerator / math.sqrt(float(margins[0]) * margins[1] * margins[2] * margins[3])

    dl1, du1 = p1 - l1, u1 - p1
    dl2, du2 = p2 - l2, u2 - p2
    d = math.sqrt(max(dl1 * dl1 - 2.0 * phi * dl1 * du2 + du2 * du2, 0.0))
    eps = math.sqrt(max(du1 * du1 - 2.0 * phi * du1 * dl2 + dl2 * dl2, 0.0))
    theta = (f - g) / n
    return PairedInterval(
        estimate=theta,
        lower=max(theta - d, -1.0),
        upper=min(theta + eps, 1.0),
        level=level,
        phi=phi,
        method="newcombe10" if phi_correction else "newcombe8",
    )


def is_equivalent(
    interval: Union[PairedInterval, Tuple[float, float]], margin: float
) -> bool:
    """Equivalence check: does the interval lie inside ``(-margin, margin)``?

    Pass an interval computed at two-sided level ``1 - 2 * alpha`` (for
    example, ``newcombe_paired_interval(..., level=0.90)`` for ``alpha =
    0.05``). Containment in ``(-margin, margin)`` is then the two one-sided
    tests decision at level ``alpha``.
    """
    if isinstance(interval, PairedInterval):
        lower, upper = interval.lower, interval.upper
    else:
        try:
            lower, upper = interval
        except (TypeError, ValueError):
            raise TypeError("interval must be a PairedInterval or a (lower, upper) pair") from None
    for name, v in (("lower", lower), ("upper", upper), ("margin", margin)):
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or math.isnan(float(v)):
            raise TypeError(f"{name} must be a real number, got {v!r}")
    if lower > upper:
        raise ValueError(f"interval lower limit {lower} exceeds upper limit {upper}")
    if not margin > 0:
        raise ValueError(f"margin must be positive, got {margin}")
    return bool(-margin < lower and upper < margin)
