"""Certification of a harm rate with an exact binomial upper bound.

The certified quantity is the probability that an episode is harmed. An
episode whose harm is unknown is counted as harmed, so the certificate never
relies on outcomes the log does not record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from scipy.special import betaincinv

from ._checks import check_count as _check_count
from ._checks import check_open_unit
from .episode import Episode, _check_theta
from .stopping import StopReplay, _check_episodes

__all__ = [
    "Certificate",
    "LoggingComparison",
    "certify",
    "certify_replay",
    "clopper_pearson_upper",
    "compare_logging_schemes",
]


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial proportion.

    Returns the upper limit ``U`` with ``P(Bin(n, U) <= k) = alpha``, that is,
    the ``1 - alpha`` quantile of ``Beta(k + 1, n - k)``; ``U = 1`` when
    ``k = n``. The bound covers the true proportion with probability at least
    ``1 - alpha`` (Clopper and Pearson, 1934).

    Parameters
    ----------
    k:
        Number of harmed episodes, ``0 <= k <= n``.
    n:
        Number of episodes, ``n >= 1``.
    alpha:
        One-sided error level in ``(0, 1)``; the bound holds at level
        ``1 - alpha``. The default 0.05 gives a 95% bound.
    """
    k = _check_count(k, "k")
    n = _check_count(n, "n")
    alpha = check_open_unit(alpha, "alpha")
    if n == 0:
        raise ValueError("n must be at least 1")
    if k > n:
        raise ValueError(f"k must not exceed n, got k={k}, n={n}")
    if k == n:
        return 1.0
    return float(betaincinv(k + 1, n - k, 1.0 - alpha))


@dataclass(frozen=True)
class Certificate:
    """One-sided upper bound on P(harm).

    Attributes
    ----------
    n:
        Number of episodes.
    n_harmed:
        Counted harmed episodes, including ``n_unknown``.
    n_unknown:
        Episodes with unknown outcome, counted as harmed.
    alpha:
        One-sided error level.
    estimate:
        ``n_harmed / n``.
    upper:
        Exact Clopper-Pearson upper bound at level ``1 - alpha``.
    """

    n: int
    n_harmed: int
    n_unknown: int
    alpha: float
    estimate: float
    upper: float

    @property
    def level(self) -> float:
        """Confidence level ``1 - alpha``."""
        return 1.0 - self.alpha


def certify(n_harmed: int, n: int, *, n_unknown: int = 0, alpha: float = 0.05) -> Certificate:
    """Certify P(harm) from counts.

    Parameters
    ----------
    n_harmed:
        Episodes known to be harmed.
    n:
        Total number of episodes.
    n_unknown:
        Episodes with unknown outcome; they are counted as harmed.
    alpha:
        One-sided error level (default 0.05, a 95% bound).
    """
    n_harmed = _check_count(n_harmed, "n_harmed")
    n_unknown = _check_count(n_unknown, "n_unknown")
    n = _check_count(n, "n")
    k = n_harmed + n_unknown
    if n == 0:
        raise ValueError("n must be at least 1")
    if k > n:
        raise ValueError(
            f"n_harmed + n_unknown must not exceed n, got {n_harmed} + {n_unknown} > {n}"
        )
    return Certificate(
        n=n,
        n_harmed=k,
        n_unknown=n_unknown,
        alpha=check_open_unit(alpha, "alpha"),
        estimate=k / n,
        upper=clopper_pearson_upper(k, n, alpha),
    )


def certify_replay(replay: StopReplay, alpha: float = 0.05) -> Certificate:
    """Certify the replayed P(harm) of a stopping gate.

    Counts the episodes whose replayed harm is ``True`` or unknown, whatever
    ``unknown`` policy the replay used. The certificate covers the stated
    valuation (no harm after a stop); a declared post-stop harm probability
    ``p`` is not part of it.
    """
    if not isinstance(replay, StopReplay):
        raise TypeError(f"replay must be a StopReplay, got {type(replay).__name__}")
    return certify(replay.n_harm_known, replay.n, n_unknown=replay.n_unknown, alpha=alpha)


@dataclass(frozen=True)
class LoggingComparison:
    """Certificates for one stopping gate under two logging schemes.

    Attributes
    ----------
    per_step:
        Certificate when the log records harm at every step.
    episode_flag:
        Certificate when the log records only an episode-level harm flag.
    n_timing_unknown:
        Stopped episodes whose episode flag is ``True`` or unknown. With only
        the flag, harm in these episodes cannot be shown to have occurred
        after the stop, so each counts as harmed.
    n_extra:
        ``episode_flag.n_harmed - per_step.n_harmed``: stopped episodes the
        per-step log clears and the flag-only log cannot.
    """

    per_step: Certificate
    episode_flag: Certificate
    n_timing_unknown: int
    n_extra: int


def compare_logging_schemes(
    episodes: Sequence[Episode], theta: float, alpha: float = 0.05
) -> LoggingComparison:
    """Compare certificates from per-step harm and from an episode-level flag.

    The stopping gate with threshold ``theta`` is replayed under both schemes.
    On the agree branch both schemes give the logged outcome. On the stop
    branch, the per-step log yields the harm accrued before the stop; the
    flag-only log can clear a stopped episode only when its flag is
    ``False``, and counts it as harmed otherwise.
    """
    eps = _check_episodes(episodes)
    theta = _check_theta(theta)
    alpha = check_open_unit(alpha, "alpha")

    step_known = step_unknown = 0
    flag_known = flag_unknown = 0
    n_timing_unknown = 0
    for e in eps:
        sigma = e.first_exceedance(theta)
        flag = e.harm_flag
        step_harm = flag if sigma is None else e.harm_before(sigma)
        step_known += step_harm is True
        step_unknown += step_harm is None
        flag_known += flag is True
        flag_unknown += flag is None
        if sigma is not None and flag is not False:
            n_timing_unknown += 1

    n = len(eps)
    per_step = certify(step_known, n, n_unknown=step_unknown, alpha=alpha)
    episode_flag = certify(flag_known, n, n_unknown=flag_unknown, alpha=alpha)
    return LoggingComparison(
        per_step=per_step,
        episode_flag=episode_flag,
        n_timing_unknown=n_timing_unknown,
        n_extra=episode_flag.n_harmed - per_step.n_harmed,
    )

