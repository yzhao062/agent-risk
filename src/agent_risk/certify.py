"""Statistical certificates for a violation rate with an exact binomial bound.

The certified quantity is the probability that an episode is a violation (for
example, harmed). A :class:`Certificate` holds a one-sided exact
Clopper-Pearson upper bound on that probability and, when a ``limit`` is set,
compares the bound with it. An episode whose outcome is unknown is counted as a
violation, so the certificate never relies on outcomes the log does not record.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import betaincinv

from ._checks import check_count as _check_count
from ._checks import check_open_unit
from .episode import Episode, _check_theta
from .retry import RetryBounds
from .stopping import StopReplay, _check_episodes

__all__ = [
    "Certificate",
    "LoggingComparison",
    "certify",
    "certify_replay",
    "certify_retry",
    "certify_violations",
    "clopper_pearson_upper",
    "compare_logging_schemes",
    "sample_size",
]

_DEFAULT_ALPHA = 0.05
_METHOD = "one-sided exact Clopper-Pearson"
_MAX_SAMPLE_SIZE = 10**12

CONDITIONS = (
    "The bound assumes that the episodes are independent draws from the task distribution, "
    "that their number was fixed before any outcome was seen and every drawn episode is counted, "
    "and that the deployment, the violation rule, the limit, and the confidence were chosen "
    "before the certification episodes were seen."
)

FAMILY_WARNING = (
    "Separate certificates each control their own error, but they do not control the chance "
    "that at least one incorrect certificate passes."
)


def _resolve_alpha(alpha: Optional[float], confidence: Optional[float]) -> float:
    """Return the one-sided error level from ``alpha`` or ``confidence``.

    ``None`` marks an argument that was not given. Passing both raises; passing
    neither gives 0.05. A ``confidence`` converts once, through a 12-digit
    decimal, so ``confidence=0.95`` gives exactly ``alpha=0.05``.
    """
    if alpha is not None and confidence is not None:
        raise TypeError("pass alpha or confidence, not both")
    if confidence is not None:
        c = check_open_unit(confidence, "confidence")
        resolved = float(f"{1 - c:.12g}")
        if not 0.0 < resolved < 1.0:
            raise ValueError(
                f"confidence {c} is too close to 0 or 1 to give an error level in (0, 1), got alpha={resolved}"
            )
        return resolved
    if alpha is None:
        return _DEFAULT_ALPHA
    return check_open_unit(alpha, "alpha")


def clopper_pearson_upper(
    k: int, n: int, alpha: Optional[float] = None, *, confidence: Optional[float] = None
) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial proportion.

    Returns the upper limit ``U`` with ``P(Bin(n, U) <= k) = alpha``, that is,
    the ``1 - alpha`` quantile of ``Beta(k + 1, n - k)``; ``U = 1`` when
    ``k = n``. The bound covers the true proportion with probability at least
    ``1 - alpha`` (Clopper and Pearson, 1934).

    Parameters
    ----------
    k:
        Number of violations, ``0 <= k <= n``.
    n:
        Number of episodes, ``n >= 1``.
    alpha:
        One-sided error level in ``(0, 1)``; the bound holds at level
        ``1 - alpha``. The default 0.05 gives a 95% bound.
    confidence:
        One-sided confidence level in ``(0, 1)``, equal to ``1 - alpha``. Pass
        ``alpha`` or ``confidence``, not both.

    Notes
    -----
    This bound is one-sided. A two-sided routine such as statsmodels
    ``proportion_confint(..., method="beta")`` at ``alpha=0.05`` returns the
    one-sided 97.5% bound as its upper end; the one-sided 95% bound needs
    ``alpha=0.10`` there.
    """
    k = _check_count(k, "k")
    n = _check_count(n, "n")
    alpha = _resolve_alpha(alpha, confidence)
    if n == 0:
        raise ValueError("n must be at least 1")
    if k > n:
        raise ValueError(f"k must not exceed n, got k={k}, n={n}")
    if k == n:
        return 1.0
    return float(betaincinv(k + 1, n - k, 1.0 - alpha))


def _reject_constant(name: str) -> Any:
    raise ValueError(f"JSON constant {name} is not allowed")


def _plain_scope_value(value: object, path: str) -> object:
    """Validate one scope value and return it as plain dicts, lists, and scalars."""
    kind = type(value)
    if value is None or kind is bool or kind is str or kind is int:
        return value
    if kind is float:
        if not math.isfinite(value):  # type: ignore[arg-type]
            raise ValueError(f"{path} must be finite, got {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        return [_plain_scope_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, Mapping):
        return _plain_scope_mapping(value, path)
    name = kind.__name__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"
    raise TypeError(
        f"{path} has unsupported type {name}; use None, bool, int, float, str, "
        "a list or tuple of these, or a mapping with string keys (convert NumPy scalars with .item())"
    )


def _plain_scope_mapping(mapping: Mapping, path: str) -> Dict[str, object]:
    plain: Dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise TypeError(f"{path} keys must be strings, got {key!r}")
        plain[key] = _plain_scope_value(item, f"{path}[{key!r}]")
    return plain


def _normalize_scope(scope: object) -> Optional[str]:
    """Return ``scope`` as a canonical JSON string, or ``None``.

    Accepts ``None``, a non-empty mapping with string keys and JSON values, or
    a JSON string that decodes to such a mapping (the stored form). Keys are
    sorted and separators are fixed, so equal scopes give equal strings.
    """
    if scope is None:
        return None
    if isinstance(scope, str):
        try:
            decoded = json.loads(scope, parse_constant=_reject_constant)
        except RecursionError:
            raise ValueError("scope string is nested too deeply") from None
        except ValueError as exc:
            raise ValueError(f"scope string is not valid JSON: {exc}") from None
        if not isinstance(decoded, dict):
            raise ValueError("scope string must decode to a JSON object")
        scope = decoded
    elif not isinstance(scope, Mapping):
        raise TypeError(f"scope must be a mapping, a JSON object string, or None, got {type(scope).__name__}")
    try:
        plain = _plain_scope_mapping(scope, "scope")  # type: ignore[arg-type]
    except RecursionError:
        raise ValueError("scope is nested too deeply or contains a cycle") from None
    if not plain:
        raise ValueError("scope must not be empty")
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    return float(value)


def _percent(confidence: float) -> str:
    """Format a confidence as a percent that never rounds up to 100%."""
    value = 100.0 * confidence
    for digits in range(6, 18):
        text = f"{value:.{digits}g}"
        if float(text) < 100.0:
            return f"{text}%"
    return f"{value!r}%"


def _bound_text(upper: float, limit: Optional[float]) -> str:
    """Format ``upper`` with the fewest digits (at least 4) that keep its order with ``limit``.

    The limit is printed exactly (its shortest round-trip form), so the printed
    bound must compare with it the way the stored bound does: a bound just
    above the limit must not print as equal to it.
    """
    for digits in range(4, 18):
        text = f"{upper:.{digits}g}"
        if limit is None:
            return text
        shown = float(text)
        if (shown <= limit) == (upper <= limit) and (shown == limit) == (upper == limit):
            return text
    return repr(upper)


def _code_span(text: str) -> str:
    """Wrap ``text`` in a Markdown code span that survives backticks inside it."""
    longest = run = 0
    for char in text:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


@dataclass(frozen=True)
class Certificate:
    """One-sided upper bound on the probability of a violation.

    Attributes
    ----------
    n:
        Number of episodes.
    n_harmed:
        Counted violations, including ``n_unknown``.
    n_unknown:
        Episodes with unknown outcome, counted as violations.
    alpha:
        One-sided error level.
    estimate:
        ``n_harmed / n``.
    upper:
        Exact Clopper-Pearson upper bound at level ``1 - alpha``.
    limit:
        Risk limit in ``(0, 1)`` fixed before the episodes were seen, or
        ``None`` when no limit is set.
    scope:
        Canonical JSON string of the configuration the certificate covers, or
        ``None``. A mapping or a JSON object string is accepted and normalized
        at construction.

    Notes
    -----
    With a ``limit``, the certificate passes when ``upper <= limit``. A pass
    states that the violation rate is at most ``limit`` at the stated
    confidence: whenever the rate exceeds ``limit``, the procedure passes with
    probability at most ``alpha``. A failure (``not_certified``) does not show
    that the rate exceeds ``limit``. The conditions behind the guarantee are
    stated in the ``conditions`` entry of :meth:`to_dict`.
    """

    n: int
    n_harmed: int
    n_unknown: int
    alpha: float
    estimate: float
    upper: float
    limit: Optional[float] = None
    scope: Optional[str] = None

    def __post_init__(self) -> None:
        if self.limit is not None:
            object.__setattr__(self, "limit", check_open_unit(self.limit, "limit"))
        object.__setattr__(self, "scope", _normalize_scope(self.scope))

    @property
    def level(self) -> float:
        """Confidence level ``1 - alpha``."""
        return 1.0 - self.alpha

    @property
    def confidence(self) -> float:
        """One-sided confidence level ``1 - alpha`` (alias of :attr:`level`)."""
        return self.level

    @property
    def n_violations(self) -> int:
        """Counted violations (alias of :attr:`n_harmed`)."""
        return self.n_harmed

    @property
    def passed(self) -> Optional[bool]:
        """``upper <= limit``, or ``None`` when no limit is set."""
        if self.limit is None:
            return None
        return self.upper <= self.limit

    @property
    def status(self) -> Optional[str]:
        """``"passed"`` or ``"not_certified"``, or ``None`` when no limit is set."""
        if self.limit is None:
            return None
        return "passed" if self.upper <= self.limit else "not_certified"

    def matches_scope(self, other: object) -> bool:
        """Whether ``other`` stores the same scope as this certificate.

        Parameters
        ----------
        other:
            A :class:`Certificate` or a mapping, normalized as ``scope`` is.

        Returns
        -------
        bool
            ``False`` when either side has no scope. A match shows only that
            two records store the same configuration, not that the
            deployment is unchanged.
        """
        if isinstance(other, Certificate):
            theirs = other.scope
        elif isinstance(other, Mapping):
            theirs = _normalize_scope(other)
        else:
            raise TypeError(f"other must be a Certificate or a mapping, got {type(other).__name__}")
        if self.scope is None or theirs is None:
            return False
        return self.scope == theirs

    def to_dict(self) -> Dict[str, Any]:
        """Return the certificate as a JSON-ready dictionary.

        The scope is a decoded object (or ``None``). The dictionary also carries
        the method, the conditions behind the bound, and the family warning.
        """
        return {
            "method": _METHOD,
            "n": self.n,
            "n_violations": self.n_harmed,
            "n_unknown": self.n_unknown,
            "estimate": self.estimate,
            "upper": self.upper,
            "alpha": self.alpha,
            "confidence": self.level,
            "limit": self.limit,
            "status": self.status,
            "scope": None if self.scope is None else json.loads(self.scope),
            "conditions": CONDITIONS,
            "family_warning": FAMILY_WARNING,
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Return :meth:`to_dict` as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "Certificate":
        """Rebuild a certificate from :meth:`to_json` output.

        The rebuilt certificate recomputes ``estimate`` and ``upper`` from the
        counts and ``alpha``. The stored values must agree with the recomputed
        ones within 1e-12 and 1e-9, ``confidence`` must equal ``1 - alpha``
        within 1e-12, and ``status`` must match the status of the rebuilt
        certificate. A mismatch raises ``ValueError``. The JSON ``scope`` must
        be an object or ``null``; any other type raises ``TypeError``.
        ``from_json(c.to_json()) == c`` holds when the same SciPy computes
        both.
        """
        try:
            data = json.loads(text, parse_constant=_reject_constant)
        except RecursionError:
            raise ValueError("certificate JSON is nested too deeply") from None
        if not isinstance(data, dict):
            raise ValueError("certificate JSON must be an object")
        required = ("n", "n_violations", "n_unknown", "alpha", "estimate", "upper")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"certificate JSON is missing fields: {', '.join(missing)}")

        n = _check_count(data["n"], "n")
        n_harmed = _check_count(data["n_violations"], "n_violations")
        n_unknown = _check_count(data["n_unknown"], "n_unknown")
        alpha = check_open_unit(data["alpha"], "alpha")
        estimate = _json_real(data["estimate"], "estimate")
        upper = _json_real(data["upper"], "upper")
        if n == 0:
            raise ValueError("n must be at least 1")
        if n_harmed > n:
            raise ValueError(f"n_violations must not exceed n, got {n_harmed} > {n}")
        if n_unknown > n_harmed:
            raise ValueError(f"n_unknown must not exceed n_violations, got {n_unknown} > {n_harmed}")
        expected_estimate = n_harmed / n
        if not abs(estimate - expected_estimate) <= 1e-12:
            raise ValueError(f"estimate {estimate!r} does not equal n_violations / n = {expected_estimate!r}")
        expected = clopper_pearson_upper(n_harmed, n, alpha)
        if not abs(upper - expected) <= 1e-9:
            raise ValueError(f"upper {upper!r} does not match the exact bound {expected!r} at alpha={alpha}")
        if "confidence" in data:
            confidence = _json_real(data["confidence"], "confidence")
            if not abs(confidence - (1.0 - alpha)) <= 1e-12:
                raise ValueError(f"confidence {confidence!r} does not equal 1 - alpha = {1.0 - alpha!r}")
        scope = data.get("scope")
        if scope is not None and not isinstance(scope, dict):
            raise TypeError(f"scope must be a JSON object or null, got {type(scope).__name__}")

        cert = cls(
            n=n,
            n_harmed=n_harmed,
            n_unknown=n_unknown,
            alpha=alpha,
            estimate=expected_estimate,
            upper=expected,
            limit=data.get("limit"),
            scope=scope,
        )
        if "status" in data and data["status"] != cert.status:
            raise ValueError(f"status {data['status']!r} does not match the rebuilt status {cert.status!r}")
        return cert

    def to_markdown(self) -> str:
        """Render the certificate as Markdown from its public fields."""
        percent = _percent(self.confidence)
        upper = _bound_text(self.upper, self.limit)
        if self.limit is None:
            status = "no limit set"
            verdict = f"The upper bound is {upper} at {percent} one-sided confidence."
        elif self.upper <= self.limit:
            status = "passed"
            verdict = (
                f"The upper bound {upper} is at most the limit {self.limit!r} "
                f"at {percent} one-sided confidence."
            )
        else:
            status = "not certified"
            verdict = (
                f"The upper bound {upper} exceeds the limit {self.limit!r} "
                f"at {percent} one-sided confidence."
            )
        parts = [
            f"**Status:** {status}",
            verdict,
            "\n".join(
                [
                    f"- Episodes: {self.n}",
                    f"- Violations counted: {self.n_violations} (estimate {self.estimate:.4g})",
                    f"- Unknown outcomes: {self.n_unknown}",
                    f"- Method: {_METHOD}",
                ]
                + ([f"- Scope: {_code_span(self.scope)}"] if self.scope is not None else [])
            ),
            CONDITIONS,
            FAMILY_WARNING,
        ]
        if self.n_unknown > 0:
            parts.append(
                f"Episodes with no known outcome: {self.n_unknown} of {self.n}, each counted as a violation."
            )
        return "\n\n".join(parts) + "\n"


def certify(
    n_harmed: int,
    n: int,
    *,
    n_unknown: int = 0,
    alpha: Optional[float] = None,
    confidence: Optional[float] = None,
    limit: Optional[float] = None,
    scope: Optional[Any] = None,
) -> Certificate:
    """Certify the violation rate from counts.

    Parameters
    ----------
    n_harmed:
        Episodes known to be violations.
    n:
        Total number of episodes.
    n_unknown:
        Episodes with unknown outcome; they are counted as violations.
    alpha:
        One-sided error level (default 0.05, a 95% bound).
    confidence:
        One-sided confidence level in ``(0, 1)``, equal to ``1 - alpha``. Pass
        ``alpha`` or ``confidence``, not both; ``confidence=0.95`` and
        ``alpha=0.05`` give equal certificates.
    limit:
        Risk limit in ``(0, 1)``. When set, the certificate passes if the upper
        bound is at most ``limit``. Choose it before seeing the episodes.
    scope:
        Non-empty mapping with string keys and JSON values that names the
        configuration the certificate covers (for example model, tools,
        permissions, prompt hash, task set, gate threshold, and violation
        rule).

    Notes
    -----
    The bound is one-sided. A two-sided routine such as statsmodels
    ``proportion_confint(..., method="beta")`` at ``alpha=0.05`` returns the
    one-sided 97.5% bound; the one-sided 95% bound needs ``alpha=0.10`` there.
    """
    n_harmed = _check_count(n_harmed, "n_harmed")
    n_unknown = _check_count(n_unknown, "n_unknown")
    n = _check_count(n, "n")
    alpha = _resolve_alpha(alpha, confidence)
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
        alpha=alpha,
        estimate=k / n,
        upper=clopper_pearson_upper(k, n, alpha),
        limit=limit,
        scope=scope,
    )


def _outcome_counts(outcomes: object) -> Tuple[int, int, int]:
    """Return ``(n, n_known_violations, n_missing)`` for a label sequence."""
    if isinstance(outcomes, (str, bytes, bytearray, Mapping)):
        raise TypeError(
            f"outcomes must be a list, tuple, or NumPy array of labels, got {type(outcomes).__name__}"
        )
    if isinstance(outcomes, np.ndarray):
        if outcomes.ndim != 1:
            raise ValueError(f"outcomes must be one-dimensional, got an array with ndim={outcomes.ndim}")
        items = outcomes.tolist()
    elif isinstance(outcomes, (list, tuple)):
        items = list(outcomes)
    else:
        raise TypeError(
            f"outcomes must be a list, tuple, or NumPy array of labels, got {type(outcomes).__name__}; "
            "convert a pandas Series with series.to_numpy(dtype=float, na_value=np.nan) "
            "and any other sequence with list()"
        )
    if not items:
        raise ValueError("outcomes is empty: at least one episode is required")

    n_violations = 0
    n_missing = 0
    for i, value in enumerate(items):
        if value is None:
            n_missing += 1
        elif isinstance(value, (bool, np.bool_)):
            n_violations += bool(value)
        elif isinstance(value, numbers.Real):
            if value != value:
                n_missing += 1
            elif value == 0:
                pass
            elif value == 1:
                n_violations += 1
            else:
                raise ValueError(f"outcomes[{i}] must be 0 or 1, got {value!r}")
        else:
            raise TypeError(
                f"outcomes[{i}] must be a bool, 0, 1, 0.0, 1.0, None, or NaN, got {value!r}"
            )
    return len(items), n_violations, n_missing


def certify_violations(
    outcomes: Sequence[Any],
    *,
    limit: float,
    confidence: float = 0.95,
    scope: Optional[Any] = None,
) -> Certificate:
    """Certify a violation rate from per-episode 0/1 outcomes.

    Parameters
    ----------
    outcomes:
        Non-empty one-dimensional list, tuple, or NumPy array with one entry
        per episode: ``True`` or 1 for a violation, ``False`` or 0 for none,
        and ``None`` or NaN for a missing label. Labels are checked by value,
        so ``np.array([0, 1])`` and ``np.array([False, True])`` are accepted.
        Strings, bytes, mappings, nested input, and other entries raise. Convert
        a pandas Series with ``series.to_numpy(dtype=float, na_value=np.nan)``,
        which also maps ``pd.NA`` (a missing label) to NaN, and any other
        sequence with ``list()``.
    limit:
        Risk limit in ``(0, 1)``, chosen before the episodes were seen.
    confidence:
        One-sided confidence level in ``(0, 1)`` (default 0.95).
    scope:
        Non-empty mapping naming the configuration the certificate covers.

    Returns
    -------
    Certificate
        A missing label counts as a violation and appears in ``n_unknown``. The
        worst-case imputation keeps the bound valid whatever caused the gap.

    Notes
    -----
    A passed certificate states that the violation rate is at most ``limit`` at
    ``confidence``. It rests on three conditions: independent episodes, a
    sample size fixed in advance with every drawn episode counted, and a
    deployment, violation rule, limit, and confidence chosen before the
    episodes were seen. A failed certificate does not show that the rate
    exceeds ``limit``.
    """
    limit = check_open_unit(limit, "limit")
    confidence = check_open_unit(confidence, "confidence")
    n, n_violations, n_missing = _outcome_counts(outcomes)
    return certify(
        n_violations, n, n_unknown=n_missing, confidence=confidence, limit=limit, scope=scope
    )


def sample_size(limit: float, *, confidence: float = 0.95, max_violations: int = 0) -> int:
    """Smallest number of episodes at which ``max_violations`` still pass.

    Finds the smallest ``n`` with
    ``clopper_pearson_upper(max_violations, n, confidence=confidence) <= limit``.

    Parameters
    ----------
    limit:
        Risk limit in ``(0, 1)``.
    confidence:
        One-sided confidence level in ``(0, 1)`` (default 0.95).
    max_violations:
        Number of violations the certificate must tolerate (default 0).

    Returns
    -------
    int
        For example, 299 at ``limit=0.01`` and ``confidence=0.95`` with no
        violations, and 473 with one.

    Raises
    ------
    ValueError
        If ``limit``, ``confidence``, or ``max_violations`` is out of range, or
        if more than 10**12 episodes would be needed.

    Notes
    -----
    The result assumes that the observed number of violations equals
    ``max_violations``. It is not a power calculation: when the true rate is
    below ``limit`` but not zero, more violations can occur and the certificate
    can fail at this ``n``. For example, at ``limit=0.01`` and
    ``confidence=0.95``, 299 episodes pass only when no violation occurs, which
    happens with probability about 0.55 when the true rate is 0.002.
    """
    limit = check_open_unit(limit, "limit")
    confidence = check_open_unit(confidence, "confidence")
    k = _check_count(max_violations, "max_violations")
    alpha = _resolve_alpha(None, confidence)

    def passes(n: int) -> bool:
        return clopper_pearson_upper(k, n, alpha) <= limit

    # The bound at fixed k decreases in n, and n = k gives 1. Double until the
    # bound passes, then bisect between the last failing and the first passing n.
    low = k
    high = k + 1
    while not passes(high):
        if high >= _MAX_SAMPLE_SIZE:
            raise ValueError(
                f"limit {limit} needs more than {_MAX_SAMPLE_SIZE} episodes at confidence {confidence} "
                f"with {k} violations"
            )
        low = high
        high = min(2 * high, _MAX_SAMPLE_SIZE)
    while high - low > 1:
        mid = (low + high) // 2
        if passes(mid):
            high = mid
        else:
            low = mid
    return high


def certify_replay(
    replay: StopReplay,
    alpha: Optional[float] = None,
    *,
    confidence: Optional[float] = None,
    limit: Optional[float] = None,
    scope: Optional[Any] = None,
) -> Certificate:
    """Certify the replayed violation rate of a stopping gate.

    Counts the episodes whose replayed harm is ``True`` or unknown, whatever
    ``unknown`` policy the replay used. The certificate covers the stated
    valuation (no harm after a stop); a declared post-stop harm probability
    ``p`` is not part of it.

    Parameters
    ----------
    replay:
        Result of :func:`agent_risk.replay_stop` over the certification log.
    alpha, confidence:
        One-sided error level or confidence level; pass at most one (default
        ``alpha=0.05``).
    limit:
        Risk limit in ``(0, 1)``. A replay with ``valuation.post_stop_harm > 0``
        raises when a limit is set, because the binary count omits the
        declared post-stop harm.
    scope:
        Non-empty mapping naming the configuration the certificate covers,
        including the gate threshold.

    Raises
    ------
    ValueError
        If ``limit`` is set and ``replay.valuation.post_stop_harm > 0``. Replay
        with ``post_stop_harm = 0``, or certify the worst-case count of
        episodes whose replayed harm is ``True`` or unknown plus the stopped
        episodes whose replayed harm is ``False``, as
        ``certify(k, n, limit=...)``.
    """
    if not isinstance(replay, StopReplay):
        raise TypeError(f"replay must be a StopReplay, got {type(replay).__name__}")
    if limit is not None:
        limit = check_open_unit(limit, "limit")
        p = replay.valuation.post_stop_harm
        if p > 0:
            n_stop_clean = sum(e.branch == "stop" and e.harm is False for e in replay.episodes)
            n_bad = replay.n_harm_known + replay.n_unknown
            k = n_bad + n_stop_clean
            raise ValueError(
                f"a limit cannot be certified from a replay with post_stop_harm={p}: the binary count "
                "omits the declared post-stop harm. Either replay with post_stop_harm=0, or certify "
                f"the worst-case count k={k} (replayed harm True or unknown: {n_bad}, plus stopped "
                f"episodes whose replayed harm is False: {n_stop_clean}) as "
                f"certify({k}, {replay.n}, limit=...), which covers every post-stop harm in (0, 1]."
            )
    return certify(
        replay.n_harm_known,
        replay.n,
        n_unknown=replay.n_unknown,
        alpha=alpha,
        confidence=confidence,
        limit=limit,
        scope=scope,
    )


def certify_retry(
    bounds: RetryBounds,
    *,
    limit: float,
    confidence: float = 0.95,
    scope: Optional[Any] = None,
) -> Certificate:
    """Certify the upper count of a retry gate's violation rate.

    The certificate is a conservative bound: a continuing episode with no harm
    before its first block continues off the log and counts as a violation. The
    known violations are ``n_harm_agree + n_harm_before_block`` and
    ``n_unknown`` is ``n_continue - n_harm_before_block``, so ``n_harmed``
    equals ``n_harm_agree + n_continue``. Unknown harm is already rejected by
    :func:`agent_risk.retry_harm_bounds`.

    Parameters
    ----------
    bounds:
        Result of :func:`agent_risk.retry_harm_bounds` over the certification
        log.
    limit:
        Risk limit in ``(0, 1)``, chosen before the log was seen.
    confidence:
        One-sided confidence level in ``(0, 1)`` (default 0.95).
    scope:
        Non-empty mapping naming the configuration the certificate covers,
        including the gate threshold.
    """
    if not isinstance(bounds, RetryBounds):
        raise TypeError(f"bounds must be a RetryBounds, got {type(bounds).__name__}")
    limit = check_open_unit(limit, "limit")
    confidence = check_open_unit(confidence, "confidence")
    known = bounds.n_harm_agree + bounds.n_harm_before_block
    n_unknown = bounds.n_continue - bounds.n_harm_before_block
    return certify(
        known, bounds.n, n_unknown=n_unknown, confidence=confidence, limit=limit, scope=scope
    )


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
    ``False``, and counts it as harmed otherwise. The returned certificates
    carry no limit.
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
