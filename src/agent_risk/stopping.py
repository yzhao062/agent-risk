"""Replay of a stopping gate over pass-all logs.

A stopping gate with threshold ``theta`` stops an episode at the first step
``sigma`` whose score is strictly above ``theta``, before that step's action
executes. Replayed over a logged episode:

* if no score exceeds ``theta``, the episode **agrees** with the log and keeps
  its logged harm, failure, and cost;
* otherwise it **stops** at ``sigma``. Harm is the logged harm accrued at steps
  strictly before ``sigma``; harm at or after ``sigma`` is removed. Failure is
  set by the declared :class:`StopValuation` (default: a stopped episode
  counts as failed). Cost follows the cost-at-stop rule (default: steps up to
  and including the proposal at ``sigma``).

The log settles the agree branch and the pre-stop part of the stop branch.
What happens after a stop is never observed; :class:`StopValuation` declares
it.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

from .episode import Episode, HarmValue, _check_theta

__all__ = ["StopValuation", "EpisodeReplay", "StopReplay", "replay_stop"]

CostRule = Union[str, Callable[[Episode, int], float]]
_COST_RULES = ("through_stop", "before_stop")
_UNKNOWN_POLICIES = ("harm", "no_harm")


@dataclass(frozen=True)
class StopValuation:
    """Declared valuation of what happens after a stop.

    Parameters
    ----------
    failed:
        Failure status assigned to a stopped episode (default ``True``: an
        aborted task counts as failed).
    post_stop_harm:
        Declared probability ``p`` in ``[0, 1]`` that a stopped episode with
        no harm before the stop incurs harm after it (default 0: no harm after
        the stop). Harm accrued before the stop is always kept.
    """

    failed: bool = True
    post_stop_harm: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.failed, bool):
            raise TypeError(f"failed must be a bool, got {self.failed!r}")
        p = self.post_stop_harm
        if isinstance(p, bool) or not isinstance(p, numbers.Real):
            raise TypeError(f"post_stop_harm must be a real number, got {p!r}")
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"post_stop_harm must lie in [0, 1], got {p}")
        object.__setattr__(self, "post_stop_harm", p)


@dataclass(frozen=True)
class EpisodeReplay:
    """Replayed outcome of one episode under a stopping gate.

    Attributes
    ----------
    branch:
        ``"agree"`` if no score exceeds the threshold, else ``"stop"``.
    stop_step:
        Index ``sigma`` of the stopping step, or ``None`` on the agree branch.
    harm:
        Replayed harm under the stated valuation (no harm after a stop):
        ``True``, ``False``, or ``None`` if the log leaves it unknown.
    failed:
        Replayed failure: the logged value on the agree branch, the declared
        valuation on the stop branch.
    cost:
        Replayed cost.
    """

    branch: str
    stop_step: Optional[int]
    harm: HarmValue
    failed: bool
    cost: float


@dataclass(frozen=True)
class StopReplay:
    """Aggregate result of :func:`replay_stop`.

    Probabilities are plug-in estimates over the ``n`` logged episodes.

    Attributes
    ----------
    theta, valuation, cost_at_stop, unknown:
        The inputs that define the replay.
    episodes:
        Per-episode :class:`EpisodeReplay`, in input order.
    n, n_agree, n_stop:
        Episode counts.
    n_harm_known:
        Episodes whose replayed harm is known to be ``True``.
    n_unknown:
        Episodes whose replayed harm is unknown (``None``).
    n_harm:
        Episodes counted as harmed: ``n_harm_known`` plus ``n_unknown`` when
        ``unknown == "harm"``, else ``n_harm_known``.
    n_stop_clean:
        Stopped episodes with no harm before the stop (unknown pre-stop
        harm counts as clean only when ``unknown == "no_harm"``).
    agreement_mass, stop_mass:
        ``n_agree / n`` and ``n_stop / n``.
    stop_clean_share:
        ``n_stop_clean / n``: the share of all episodes that stop with no harm
        before the stop. The declared post-stop harm probability applies to
        this share.
    p_harm_stated:
        ``n_harm / n``: replayed P(harm) with no harm after a stop.
    p_harm:
        ``p_harm_stated + post_stop_harm * stop_clean_share``.
    p_failure:
        Replayed P(failure).
    mean_cost:
        Replayed mean cost per episode.
    """

    theta: float
    valuation: StopValuation
    cost_at_stop: CostRule
    unknown: str
    episodes: Tuple[EpisodeReplay, ...]
    n: int
    n_agree: int
    n_stop: int
    n_harm_known: int
    n_unknown: int
    n_harm: int
    n_stop_clean: int
    agreement_mass: float
    stop_mass: float
    stop_clean_share: float
    p_harm_stated: float
    p_harm: float
    p_failure: float
    mean_cost: float

    @property
    def branches(self) -> Tuple[str, ...]:
        """Per-episode branch labels (``"agree"`` or ``"stop"``)."""
        return tuple(e.branch for e in self.episodes)

    @property
    def stop_steps(self) -> Tuple[Optional[int], ...]:
        """Per-episode stopping step (``None`` on the agree branch)."""
        return tuple(e.stop_step for e in self.episodes)


def _stop_cost(episode: Episode, sigma: int, rule: CostRule) -> float:
    if callable(rule):
        value = rule(episode, sigma)
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"cost_at_stop callable must return a real number, got {value!r}")
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"cost_at_stop callable must return a finite non-negative value, got {value}")
        return value
    if rule == "through_stop":
        return float(math.fsum(episode.cost[: sigma + 1]))
    return float(math.fsum(episode.cost[:sigma]))


def _check_episodes(episodes: Sequence[Episode]) -> Tuple[Episode, ...]:
    if isinstance(episodes, Episode):
        raise TypeError("episodes must be a sequence of Episode objects, got a single Episode")
    eps = tuple(episodes)
    if not eps:
        raise ValueError("episodes is empty: at least one episode is required")
    for i, e in enumerate(eps):
        if not isinstance(e, Episode):
            raise TypeError(f"episodes[{i}] must be an Episode, got {type(e).__name__}")
    return eps


def replay_stop(
    episodes: Sequence[Episode],
    theta: float,
    *,
    valuation: Optional[StopValuation] = None,
    cost_at_stop: CostRule = "through_stop",
    unknown: str = "harm",
) -> StopReplay:
    """Replay a stopping gate with threshold ``theta`` over pass-all logs.

    Parameters
    ----------
    episodes:
        Logged episodes (at least one).
    theta:
        Threshold; the gate stops at the first step with score ``> theta``.
    valuation:
        Declared :class:`StopValuation` for stopped episodes. Defaults to
        ``StopValuation()`` (failed, no harm after the stop).
    cost_at_stop:
        ``"through_stop"`` (default) counts cost at steps ``0..sigma``,
        including the proposal at ``sigma``; ``"before_stop"`` counts steps
        ``0..sigma-1``; a callable ``f(episode, sigma) -> float`` sets a
        custom rule.
    unknown:
        How an unknown replayed harm enters the counts: ``"harm"`` (default,
        conservative) counts it as harmed, ``"no_harm"`` as unharmed. Unknown
        entries at or after the stopping step are removed by the replay and
        never matter.

    Returns
    -------
    StopReplay
    """
    eps = _check_episodes(episodes)
    theta = _check_theta(theta)
    if valuation is None:
        valuation = StopValuation()
    elif not isinstance(valuation, StopValuation):
        raise TypeError(f"valuation must be a StopValuation, got {type(valuation).__name__}")
    if not callable(cost_at_stop) and cost_at_stop not in _COST_RULES:
        raise ValueError(f"cost_at_stop must be one of {_COST_RULES} or a callable, got {cost_at_stop!r}")
    if unknown not in _UNKNOWN_POLICIES:
        raise ValueError(f"unknown must be one of {_UNKNOWN_POLICIES}, got {unknown!r}")

    rows = []
    for e in eps:
        sigma = e.first_exceedance(theta)
        if sigma is None:
            rows.append(EpisodeReplay("agree", None, e.harm_flag, e.failed, e.total_cost))
        else:
            rows.append(
                EpisodeReplay(
                    "stop",
                    sigma,
                    e.harm_before(sigma),
                    valuation.failed,
                    _stop_cost(e, sigma, cost_at_stop),
                )
            )

    n = len(rows)
    n_stop = sum(r.branch == "stop" for r in rows)
    n_agree = n - n_stop
    n_harm_known = sum(r.harm is True for r in rows)
    n_unknown = sum(r.harm is None for r in rows)
    n_harm = n_harm_known + (n_unknown if unknown == "harm" else 0)
    clean_values = (False,) if unknown == "harm" else (False, None)
    n_stop_clean = sum(r.branch == "stop" and r.harm in clean_values for r in rows)

    p_harm_stated = n_harm / n
    stop_clean_share = n_stop_clean / n
    return StopReplay(
        theta=theta,
        valuation=valuation,
        cost_at_stop=cost_at_stop,
        unknown=unknown,
        episodes=tuple(rows),
        n=n,
        n_agree=n_agree,
        n_stop=n_stop,
        n_harm_known=n_harm_known,
        n_unknown=n_unknown,
        n_harm=n_harm,
        n_stop_clean=n_stop_clean,
        agreement_mass=n_agree / n,
        stop_mass=n_stop / n,
        stop_clean_share=stop_clean_share,
        p_harm_stated=p_harm_stated,
        p_harm=p_harm_stated + valuation.post_stop_harm * stop_clean_share,
        p_failure=sum(r.failed for r in rows) / n,
        mean_cost=float(math.fsum(r.cost for r in rows)) / n,
    )
