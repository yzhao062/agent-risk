"""Bounds for a retry gate from pass-all logs.

A retry gate with threshold ``theta`` and retry cap ``c`` blocks an action
whose score is strictly above ``theta`` and lets the agent continue; after
``c`` blocks it stops. Replayed over a pass-all log:

* an episode whose scores never exceed ``theta`` **agrees** and keeps its
  logged harm;
* an episode with an exceedance reaches a first block at step ``tau``. The
  agent's continuation after a block is not in the log, so the episode's harm
  is only bounded: the lower end keeps the harm accrued before ``tau`` and
  assumes no further harm; the upper end assumes harm.

The cap ``c`` governs only the unlogged continuation, so for any ``c >= 1``
the plug-in outer interval does not depend on it. A cap of 0 is the stopping
gate; use :func:`agent_risk.replay_stop` for that case.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .episode import Episode, _check_theta
from .stopping import _check_episodes

__all__ = ["RetryBounds", "retry_harm_bounds"]


@dataclass(frozen=True)
class RetryBounds:
    """Plug-in outer interval for P(harm) under a retry gate.

    Attributes
    ----------
    theta, cap:
        The gate's threshold and retry cap.
    n, n_agree, n_continue:
        Episode counts. ``n_continue`` episodes reach a first block and
        continue off the log.
    n_harm_agree:
        Agree-branch episodes with logged harm.
    n_harm_before_block:
        Continuing episodes with harm before the first block.
    lower, upper:
        ``(n_harm_agree + n_harm_before_block) / n`` and
        ``(n_harm_agree + n_continue) / n``.
    branches:
        Per-episode ``"agree"`` or ``"continue"``.
    first_block:
        Per-episode index of the first block (``None`` on the agree branch).
    """

    theta: float
    cap: int
    n: int
    n_agree: int
    n_continue: int
    n_harm_agree: int
    n_harm_before_block: int
    lower: float
    upper: float
    branches: Tuple[str, ...]
    first_block: Tuple[Optional[int], ...]

    @property
    def interval(self) -> Tuple[float, float]:
        """``(lower, upper)``."""
        return (self.lower, self.upper)

    @property
    def width(self) -> float:
        """``upper - lower``, equal to the share of continuing episodes with
        no harm before the first block."""
        return self.upper - self.lower


def _known(value: Optional[bool], i: int, t: int) -> bool:
    if value is None:
        raise ValueError(
            f"retry_harm_bounds needs binary harm: episodes[{i}].harm[{t}] is None (unknown)"
        )
    return value


def retry_harm_bounds(episodes: Sequence[Episode], theta: float, cap: int = 1) -> RetryBounds:
    """Plug-in outer interval for P(harm) under a retry gate.

    Parameters
    ----------
    episodes:
        Logged pass-all episodes (at least one). Harm must be binary at every
        step the function reads: all steps of an agreeing episode and the
        steps before the first block of a continuing one.
    theta:
        Threshold; an action with score ``> theta`` is blocked.
    cap:
        Retry cap ``c >= 1``.

    Returns
    -------
    RetryBounds
    """
    eps = _check_episodes(episodes)
    theta = _check_theta(theta)
    if isinstance(cap, bool) or not isinstance(cap, numbers.Integral):
        raise TypeError(f"cap must be an integer, got {cap!r}")
    if cap < 1:
        raise ValueError(
            f"cap must be at least 1, got {cap}; a cap of 0 is the stopping gate (use replay_stop)"
        )

    branches = []
    first_block = []
    n_harm_agree = 0
    n_harm_before_block = 0
    for i, e in enumerate(eps):
        tau = e.first_exceedance(theta)
        first_block.append(tau)
        read_until = e.n_steps if tau is None else tau
        harmed = False
        for t in range(read_until):
            harmed = _known(e.harm[t], i, t) or harmed
        if tau is None:
            branches.append("agree")
            n_harm_agree += harmed
        else:
            branches.append("continue")
            n_harm_before_block += harmed

    n = len(eps)
    n_continue = branches.count("continue")
    return RetryBounds(
        theta=theta,
        cap=int(cap),
        n=n,
        n_agree=n - n_continue,
        n_continue=n_continue,
        n_harm_agree=n_harm_agree,
        n_harm_before_block=n_harm_before_block,
        lower=(n_harm_agree + n_harm_before_block) / n,
        upper=(n_harm_agree + n_continue) / n,
        branches=tuple(branches),
        first_block=tuple(first_block),
    )
