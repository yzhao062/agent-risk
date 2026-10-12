"""Logged episodes and loaders.

An episode is a sequence of steps logged under a baseline gate that never
stops (pass-all). Step ``t`` carries:

* ``scores[t]``: the monitor score in ``[0, 1]`` for the action proposed at
  step ``t``;
* ``harm[t]``: ``True`` if the action executed at step ``t`` violated policy,
  ``False`` if it did not, ``None`` if the outcome is unknown;
* ``cost[t]``: the non-negative cost of step ``t`` (for example, tokens).

The episode also records ``failed``: whether the task failed. Steps are
indexed from 0.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Episode", "load_episodes", "load_jsonl", "any_harm"]

HarmValue = Optional[bool]


def any_harm(values: Iterable[HarmValue]) -> HarmValue:
    """Three-valued OR over per-step harm indicators.

    Returns ``True`` if any entry is ``True``; otherwise ``None`` if any entry
    is unknown (``None``); otherwise ``False``. An empty sequence returns
    ``False`` (no executed step, no harm).
    """
    saw_unknown = False
    for v in values:
        if v is True:
            return True
        if v is None:
            saw_unknown = True
    return None if saw_unknown else False


def _as_float_tuple(values: Any, name: str) -> Tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be a sequence of numbers, got {type(values).__name__}")
    out: List[float] = []
    for t, v in enumerate(values):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise TypeError(f"{name}[{t}] must be a real number, got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"{name}[{t}] must be finite, got {v!r}")
        out.append(f)
    return tuple(out)


def _as_harm(v: Any, where: str) -> HarmValue:
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral) and int(v) in (0, 1):
        return bool(int(v))
    raise ValueError(
        f"{where} must be True, False, 0, 1, or None (unknown), got {v!r}"
    )


def _as_failed(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, numbers.Integral) and not isinstance(v, bool) and int(v) in (0, 1):
        return bool(int(v))
    raise ValueError(f"failed must be True, False, 0, or 1, got {v!r}")


@dataclass(frozen=True)
class Episode:
    """One logged episode under the pass-all baseline gate.

    Parameters
    ----------
    scores:
        Monitor score in ``[0, 1]`` for each step's proposed action.
    harm:
        Per-step harm indicator: ``True``, ``False``, or ``None`` (unknown).
        Integers 0 and 1 are accepted and converted to booleans.
    cost:
        Non-negative, finite cost of each step (for example, tokens).
    failed:
        Whether the task failed in the logged run.

    All three per-step sequences must have the same length. An episode may
    have zero steps. The sequences are stored as tuples.
    """

    scores: Sequence[float]
    harm: Sequence[HarmValue]
    cost: Sequence[float]
    failed: bool

    def __post_init__(self) -> None:
        scores = _as_float_tuple(self.scores, "scores")
        cost = _as_float_tuple(self.cost, "cost")
        if isinstance(self.harm, (str, bytes)) or not isinstance(self.harm, Iterable):
            raise TypeError(
                f"harm must be a sequence of True/False/None, got {type(self.harm).__name__}"
            )
        harm = tuple(_as_harm(v, f"harm[{t}]") for t, v in enumerate(self.harm))
        if not (len(scores) == len(harm) == len(cost)):
            raise ValueError(
                "scores, harm, and cost must have the same length, got "
                f"{len(scores)}, {len(harm)}, {len(cost)}"
            )
        for t, s in enumerate(scores):
            if not 0.0 <= s <= 1.0:
                raise ValueError(f"scores[{t}] must lie in [0, 1], got {s}")
        for t, c in enumerate(cost):
            if c < 0.0:
                raise ValueError(f"cost[{t}] must be non-negative, got {c}")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "harm", harm)
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "failed", _as_failed(self.failed))

    @property
    def n_steps(self) -> int:
        """Number of logged steps."""
        return len(self.scores)

    @property
    def harm_flag(self) -> HarmValue:
        """Episode-level harm flag: three-valued OR over all steps."""
        return any_harm(self.harm)

    @property
    def total_cost(self) -> float:
        """Sum of per-step cost over the whole logged episode."""
        return float(math.fsum(self.cost))

    def harm_before(self, step: int) -> HarmValue:
        """Harm accrued at steps strictly before ``step`` (three-valued)."""
        if not isinstance(step, numbers.Integral) or isinstance(step, bool):
            raise TypeError(f"step must be an integer, got {step!r}")
        if not 0 <= step <= self.n_steps:
            raise ValueError(f"step must lie in [0, {self.n_steps}], got {step}")
        return any_harm(self.harm[: int(step)])

    def first_exceedance(self, theta: float) -> Optional[int]:
        """Index of the first step whose score is strictly above ``theta``.

        Returns ``None`` if no score exceeds ``theta``.
        """
        theta = _check_theta(theta)
        for t, s in enumerate(self.scores):
            if s > theta:
                return t
        return None


def _check_theta(theta: Any) -> float:
    if isinstance(theta, (bool, np.bool_)) or not isinstance(theta, numbers.Real):
        raise TypeError(f"theta must be a real number, got {theta!r}")
    value = float(theta)
    if math.isnan(value):
        raise ValueError("theta must not be NaN")
    return value


def _episode_from_record(record: Mapping[str, Any], where: str) -> Episode:
    if not isinstance(record, Mapping):
        raise TypeError(f"{where}: expected a mapping, got {type(record).__name__}")
    if "failed" not in record:
        raise ValueError(f"{where}: missing key 'failed'")
    try:
        if "steps" in record:
            extra = {"scores", "harm", "cost"} & set(record)
            if extra:
                raise ValueError(
                    f"use either 'steps' or 'scores'/'harm'/'cost', not both (found {sorted(extra)})"
                )
            steps = record["steps"]
            if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
                raise TypeError("'steps' must be a list of step mappings")
            scores, harm, cost = [], [], []
            for t, step in enumerate(steps):
                if not isinstance(step, Mapping):
                    raise TypeError(f"steps[{t}] must be a mapping")
                for key in ("score", "harm", "cost"):
                    if key not in step:
                        raise ValueError(f"steps[{t}] is missing key {key!r}")
                scores.append(step["score"])
                harm.append(step["harm"])
                cost.append(step["cost"])
            return Episode(scores, harm, cost, record["failed"])
        missing = [k for k in ("scores", "harm", "cost") if k not in record]
        if missing:
            raise ValueError(f"missing key(s) {missing}")
        return Episode(record["scores"], record["harm"], record["cost"], record["failed"])
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{where}: {exc}") from None


def load_episodes(records: Iterable[Mapping[str, Any]]) -> List[Episode]:
    """Build episodes from a list of dicts.

    Each record is either columnar::

        {"scores": [...], "harm": [...], "cost": [...], "failed": false}

    or per-step::

        {"steps": [{"score": 0.1, "harm": false, "cost": 12}, ...], "failed": false}

    Use ``null``/``None`` for an unknown harm entry.
    """
    if isinstance(records, Mapping) or isinstance(records, (str, bytes)):
        raise TypeError("records must be an iterable of mappings, one per episode")
    return [_episode_from_record(r, f"record {i}") for i, r in enumerate(records)]


def load_jsonl(source: Union[str, "os.PathLike[str]", Iterable[str]]) -> List[Episode]:
    """Load episodes from JSON lines, one episode record per line.

    ``source`` is a file path, or an iterable of JSON strings (for example,
    an open file). Blank lines are skipped. The record format is the one
    accepted by :func:`load_episodes`.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "r", encoding="utf-8") as fh:
            return _load_lines(fh)
    return _load_lines(source)


def _load_lines(lines: Iterable[str]) -> List[Episode]:
    episodes: List[Episode] = []
    for lineno, line in enumerate(lines, start=1):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not isinstance(line, str):
            raise TypeError(f"line {lineno}: expected a JSON string, got {type(line).__name__}")
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {lineno}: invalid JSON ({exc.msg})") from None
        episodes.append(_episode_from_record(record, f"line {lineno}"))
    return episodes
