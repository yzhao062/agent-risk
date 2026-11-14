"""Tests for statistical certificates: boundaries, exact coverage, and the public API."""

import ast
import dataclasses
import importlib
import inspect
import json
import math

import numpy as np
import pytest
from scipy.stats import binom

import agent_risk
from agent_risk import (
    Certificate,
    Episode,
    StopValuation,
    certify,
    certify_replay,
    certify_retry,
    certify_violations,
    clopper_pearson_upper,
    compare_logging_schemes,
    replay_stop,
    retry_harm_bounds,
    sample_size,
)
from agent_risk.certify import CONDITIONS, FAMILY_WARNING

from toydata import TOY_THETA

LIMIT = 0.01
CONFIDENCE = 0.95


def _passes(k, n, limit, confidence):
    return clopper_pearson_upper(k, n, confidence=confidence) <= limit


def _max_passing_k(n, limit, confidence):
    """Largest k whose bound is at most ``limit`` (-1 if none); the bound increases in k."""
    k = -1
    while k + 1 <= n and _passes(k + 1, n, limit, confidence):
        k += 1
    return k


def _clean_episodes(n):
    return [Episode([0.1, 0.2], [False, False], [1, 1], failed=False) for _ in range(n)]


def _tampered(cert, **changes):
    data = json.loads(cert.to_json())
    for key, value in changes.items():
        if value is _DROP:
            data.pop(key)
        else:
            data[key] = value
    return json.dumps(data)


_DROP = object()

SCOPE = {"model": "m-1", "tools": ["search", "shell"], "gate": {"theta": 0.94, "cap": 2}}


# ---------------------------------------------------------------------------
# Boundaries and the sample-size function
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_violations, n, expected",
    [(0, 298, "not_certified"), (0, 299, "passed"), (1, 472, "not_certified"), (1, 473, "passed")],
)
def test_boundaries_at_one_percent_and_95(n_violations, n, expected):
    c = certify(n_violations, n, confidence=CONFIDENCE, limit=LIMIT)
    assert c.status == expected
    assert c.passed is (expected == "passed")


@pytest.mark.parametrize(
    "n_violations, n, expected",
    [(0, 298, "not_certified"), (0, 299, "passed"), (1, 472, "not_certified"), (1, 473, "passed")],
)
def test_boundaries_from_outcome_labels(n_violations, n, expected):
    outcomes = [True] * n_violations + [False] * (n - n_violations)
    c = certify_violations(outcomes, limit=LIMIT, confidence=CONFIDENCE)
    assert (c.n, c.n_harmed, c.n_unknown) == (n, n_violations, 0)
    assert c.status == expected


@pytest.mark.parametrize("n, expected", [(298, "not_certified"), (299, "passed")])
def test_boundaries_from_a_replayed_log(n, expected):
    replay = replay_stop(_clean_episodes(n), 0.5)
    assert certify_replay(replay, limit=LIMIT, confidence=CONFIDENCE).status == expected


@pytest.mark.parametrize("max_violations, expected", [(0, 299), (1, 473), (2, 628)])
def test_sample_size_reference_values(max_violations, expected):
    n = sample_size(LIMIT, confidence=CONFIDENCE, max_violations=max_violations)
    assert isinstance(n, int)
    assert n == expected


@pytest.mark.parametrize("max_violations", [0, 1, 2, 5])
@pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("limit", [0.001, 0.01, 0.05, 0.2, 0.9])
def test_sample_size_is_the_smallest_passing_n(limit, confidence, max_violations):
    n = sample_size(limit, confidence=confidence, max_violations=max_violations)
    assert n > max_violations
    assert _passes(max_violations, n, limit, confidence)
    if n - 1 >= 1:
        assert not _passes(max_violations, n - 1, limit, confidence)
    else:
        assert n - 1 == max_violations == 0


def test_sample_size_one_episode_can_suffice():
    # A single clean episode gives the bound 1 - alpha = 0.8 at confidence 0.8.
    assert sample_size(0.9, confidence=0.8) == 1


def test_sample_size_is_monotone_in_its_inputs():
    base = sample_size(0.02, confidence=0.9, max_violations=1)
    assert sample_size(0.01, confidence=0.9, max_violations=1) > base
    assert sample_size(0.02, confidence=0.99, max_violations=1) > base
    assert sample_size(0.02, confidence=0.9, max_violations=2) > base


def test_sample_size_rejects_an_unreachable_limit():
    with pytest.raises(ValueError, match="episodes"):
        sample_size(1e-15)


@pytest.mark.parametrize(
    "args, kwargs, exc",
    [
        ((0.0,), {}, ValueError),
        ((1.0,), {}, ValueError),
        ((None,), {}, TypeError),
        (("0.01",), {}, TypeError),
        ((True,), {}, TypeError),
        ((0.01,), {"confidence": 1.0}, ValueError),
        ((0.01,), {"confidence": 0.0}, ValueError),
        ((0.01,), {"confidence": None}, TypeError),
        ((0.01,), {"max_violations": -1}, ValueError),
        ((0.01,), {"max_violations": 1.0}, TypeError),
        ((0.01,), {"max_violations": True}, TypeError),
        ((0.01, 0.95), {}, TypeError),  # confidence is keyword-only
    ],
)
def test_sample_size_validation(args, kwargs, exc):
    with pytest.raises(exc):
        sample_size(*args, **kwargs)


# ---------------------------------------------------------------------------
# Closed form and the all-violations edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 5, 59, 299, 1000])
@pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99, 0.999])
def test_zero_violation_bound_matches_closed_form(n, confidence):
    expected = 1 - (1 - confidence) ** (1 / n)
    assert abs(clopper_pearson_upper(0, n, confidence=confidence) - expected) < 1e-12
    assert abs(certify(0, n, confidence=confidence).upper - expected) < 1e-12


@pytest.mark.parametrize("n", [1, 3, 50, 299])
def test_every_episode_a_violation_gives_the_trivial_bound(n):
    c = certify(n, n, confidence=CONFIDENCE, limit=0.99)
    assert c.upper == 1.0
    assert c.status == "not_certified" and c.passed is False
    assert certify_violations([1] * n, limit=0.99).status == "not_certified"


def test_all_unknown_labels_give_the_trivial_bound():
    c = certify_violations([None, float("nan"), None], limit=0.5)
    assert (c.n, c.n_harmed, c.n_unknown) == (3, 3, 3)
    assert c.upper == 1.0 and c.status == "not_certified"


# ---------------------------------------------------------------------------
# Exact binomial coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("limit", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("n", [10, 50, 299, 1000])
def test_pass_probability_is_at_most_alpha_at_and_above_the_limit(n, limit, confidence):
    k_star = _max_passing_k(n, limit, confidence)
    for p in (limit, 1.5 * limit, 2 * limit, 3 * limit):
        pass_probability = binom.cdf(k_star, n, p) if k_star >= 0 else 0.0
        assert pass_probability <= (1 - confidence) + 1e-12, (n, limit, confidence, p)


def test_exact_coverage_is_nearly_tight_at_the_limit():
    # k_star = 0 at n = 299: the pass probability at p = limit is 0.99^299, just under alpha.
    assert _max_passing_k(299, LIMIT, CONFIDENCE) == 0
    at_limit = binom.cdf(0, 299, LIMIT)
    assert 0.049 < at_limit <= 0.05 + 1e-12


def test_pass_probability_is_high_when_the_rate_is_well_below_the_limit():
    # A procedure that never passes, or ignores the confidence, fails this check.
    k_star = _max_passing_k(299, LIMIT, CONFIDENCE)
    assert binom.cdf(k_star, 299, 0.001) == pytest.approx(0.74, abs=0.01)


def test_pass_probability_follows_the_confidence():
    # At n = 299 a 99% certificate cannot pass at limit 0.01, a 90% one passes more often than a 95% one.
    assert _max_passing_k(299, LIMIT, 0.99) == -1
    assert _max_passing_k(299, LIMIT, 0.90) >= _max_passing_k(299, LIMIT, 0.95)
    assert certify(0, 299, confidence=0.99, limit=LIMIT).status == "not_certified"
    assert certify(0, 299, confidence=0.90, limit=LIMIT).status == "passed"


# ---------------------------------------------------------------------------
# certify_violations: labels
# ---------------------------------------------------------------------------


def test_missing_labels_count_as_violations_and_as_unknown():
    c = certify_violations([0, None, 0], limit=0.5)
    assert (c.n, c.n_harmed, c.n_unknown) == (3, 1, 1)
    c = certify_violations(np.array([0.0, 1.0, np.nan]), limit=0.5)
    assert (c.n, c.n_harmed, c.n_unknown) == (3, 2, 1)


@pytest.mark.parametrize(
    "outcomes, n_harmed, n_unknown",
    [
        (np.array([0, 1]), 1, 0),
        (np.array([False, True]), 1, 0),
        (np.array([0, 1], dtype=np.int8), 1, 0),
        (np.array([1.0, 0.0, 1.0]), 2, 0),
        ((0, 1, 1), 2, 0),
        ([True, False, True, True], 3, 0),
        ([1, 0.0, True, False], 2, 0),
        ([np.int64(0), np.int64(1), np.float32(1.0), np.bool_(True), np.bool_(False)], 3, 0),
        ([np.float64(np.nan), np.float32(np.nan), None, 0], 3, 3),
    ],
)
def test_labels_are_checked_by_value(outcomes, n_harmed, n_unknown):
    c = certify_violations(outcomes, limit=0.5)
    assert c.n == len(outcomes)
    assert (c.n_harmed, c.n_unknown) == (n_harmed, n_unknown)


def test_outcomes_match_the_count_interface():
    c = certify_violations([1, 0, None, 0, 0, 0, 0, 0, 0, 0], limit=0.5, confidence=0.9, scope=SCOPE)
    assert c == certify(1, 10, n_unknown=1, confidence=0.9, limit=0.5, scope=SCOPE)


_CONVERSION_HINT = r"series\.to_numpy\(dtype=float, na_value=np\.nan\).*list\(\)"


class _SeriesLike:
    """Stand-in for a pandas Series: iterable, with a to_numpy method."""

    def __iter__(self):
        return iter([0, 1, 0])

    def __len__(self):
        return 3

    def to_numpy(self):
        return np.array([0, 1, 0])


@pytest.mark.parametrize(
    "outcomes, exc",
    [
        ("0101", TypeError),
        (b"\x00\x01", TypeError),
        (bytearray(b"\x00\x01"), TypeError),
        ({"a": 0, "b": 1}, TypeError),
        ({0, 1}, TypeError),
        (None, TypeError),
        (0, TypeError),
        (iter([0, 1]), TypeError),
        (range(3), TypeError),
        ([[0, 1], [1, 0]], TypeError),
        (np.array([[0, 1], [1, 0]]), ValueError),
        (np.array(1), ValueError),
        ([], ValueError),
        ((), ValueError),
        (np.array([]), ValueError),
        ([2], ValueError),
        ([0, 1, 2], ValueError),
        ([0.5], ValueError),
        ([-1], ValueError),
        ([float("inf")], ValueError),
        (np.array([0, 2]), ValueError),
        (["a"], TypeError),
        (["0", "1"], TypeError),
        (np.array(["a", "b"]), TypeError),
        ([[1]], TypeError),
        ([{"x": 1}], TypeError),
        ([1 + 0j], TypeError),
    ],
)
def test_invalid_outcomes_raise_the_documented_error(outcomes, exc):
    with pytest.raises(exc):
        certify_violations(outcomes, limit=0.5)


def test_series_like_input_is_rejected_with_a_conversion_hint():
    with pytest.raises(TypeError, match=_CONVERSION_HINT):
        certify_violations(_SeriesLike(), limit=0.5)
    assert certify_violations(_SeriesLike().to_numpy(), limit=0.5).n_harmed == 1


def test_pandas_series_needs_conversion():
    pd = pytest.importorskip("pandas")
    series = pd.Series([0, 1, 0, None])
    with pytest.raises(TypeError, match=_CONVERSION_HINT):
        certify_violations(series, limit=0.5)
    c = certify_violations(series.to_numpy(dtype=float), limit=0.5)
    assert (c.n, c.n_harmed, c.n_unknown) == (4, 2, 1)
    assert certify_violations(list(series), limit=0.5) == c


def test_nullable_boolean_series_converts_with_na_value():
    pd = pytest.importorskip("pandas")
    series = pd.Series([False, True, None], dtype="boolean")
    c = certify_violations(series.to_numpy(dtype=float, na_value=np.nan), limit=0.5)
    assert (c.n, c.n_harmed, c.n_unknown) == (3, 2, 1)


def test_invalid_label_error_names_the_index():
    with pytest.raises(ValueError, match=r"outcomes\[2\]"):
        certify_violations([0, 1, 2], limit=0.5)
    with pytest.raises(TypeError, match=r"outcomes\[1\]"):
        certify_violations([0, "x"], limit=0.5)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"limit": None}, TypeError),
        ({"limit": 0.0}, ValueError),
        ({"limit": 1.0}, ValueError),
        ({"limit": 1.5}, ValueError),
        ({"limit": "0.01"}, TypeError),
        ({"limit": True}, TypeError),
        ({"limit": 0.1, "confidence": None}, TypeError),
        ({"limit": 0.1, "confidence": 1.0}, ValueError),
        ({"limit": 0.1, "confidence": 0.0}, ValueError),
        ({"limit": 0.1, "confidence": "0.95"}, TypeError),
    ],
)
def test_certify_violations_validates_limit_and_confidence(kwargs, exc):
    with pytest.raises(exc):
        certify_violations([0, 0, 1], **kwargs)


def test_certify_violations_requires_a_limit():
    with pytest.raises(TypeError):
        certify_violations([0, 0, 1])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        certify_violations([0, 0, 1], 0.5)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# alpha and confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k, n", [(0, 10), (3, 100), (1, 3), (7, 7)])
def test_confidence_and_alpha_give_equal_certificates(k, n):
    assert certify(k, n, confidence=0.95) == certify(k, n, alpha=0.05)
    assert certify(k, n, confidence=0.9, limit=0.5) == certify(k, n, alpha=0.1, limit=0.5)
    assert clopper_pearson_upper(k, n, confidence=0.95) == clopper_pearson_upper(k, n, alpha=0.05)


@pytest.mark.parametrize(
    "confidence, alpha", [(0.8, 0.2), (0.9, 0.1), (0.95, 0.05), (0.99, 0.01), (0.999, 0.001), (0.75, 0.25)]
)
def test_confidence_converts_to_the_decimal_alpha(confidence, alpha):
    c = certify(2, 20, confidence=confidence)
    assert c.alpha == alpha
    assert c.confidence == pytest.approx(confidence, abs=1e-15)


def test_the_four_alpha_and_confidence_combinations():
    assert certify(2, 50).alpha == 0.05  # neither: the default
    assert certify(2, 50, alpha=0.1).alpha == 0.1  # alpha only
    assert certify(2, 50, confidence=0.9).alpha == 0.1  # confidence only
    with pytest.raises(TypeError, match="not both"):  # both
        certify(2, 50, alpha=0.05, confidence=0.95)
    with pytest.raises(TypeError, match="not both"):
        clopper_pearson_upper(2, 50, alpha=0.05, confidence=0.95)
    with pytest.raises(TypeError, match="not both"):
        clopper_pearson_upper(2, 50, 0.05, confidence=0.95)
    with pytest.raises(TypeError, match="not both"):
        certify_replay(replay_stop(_clean_episodes(3), 0.5), alpha=0.05, confidence=0.95)


def test_default_alpha_is_five_percent_everywhere(toy_episodes):
    assert clopper_pearson_upper(1, 3) == clopper_pearson_upper(1, 3, alpha=0.05)
    replay = replay_stop(toy_episodes, TOY_THETA)
    assert certify_replay(replay).alpha == 0.05
    assert certify_replay(replay) == certify_replay(replay, confidence=0.95)


def test_old_positional_forms_still_work(toy_episodes):
    assert clopper_pearson_upper(1, 3, 0.05) == clopper_pearson_upper(1, 3, alpha=0.05)
    assert clopper_pearson_upper(1, 3, 0.01) == clopper_pearson_upper(1, 3, confidence=0.99)
    replay = replay_stop(toy_episodes, TOY_THETA)
    assert certify_replay(replay, 0.05) == certify_replay(replay, alpha=0.05)
    assert certify_replay(replay, 0.01).alpha == 0.01


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        ({"alpha": 0.0}, ValueError),
        ({"alpha": 1.0}, ValueError),
        ({"alpha": 1.5}, ValueError),
        ({"alpha": True}, TypeError),
        ({"alpha": "0.05"}, TypeError),
        ({"confidence": 0.0}, ValueError),
        ({"confidence": 1.0}, ValueError),
        ({"confidence": -0.5}, ValueError),
        ({"confidence": True}, TypeError),
        ({"confidence": "0.95"}, TypeError),
        ({"confidence": 1e-17}, ValueError),  # 1 - confidence rounds to 1
    ],
)
def test_alpha_and_confidence_validation(kwargs, exc):
    with pytest.raises(exc):
        certify(1, 10, **kwargs)
    with pytest.raises(exc):
        clopper_pearson_upper(1, 10, **kwargs)


def test_count_validation():
    with pytest.raises(ValueError, match="exceed"):
        certify(3, 5, n_unknown=3)
    with pytest.raises(ValueError, match="at least 1"):
        certify(0, 0, confidence=0.9)
    with pytest.raises(TypeError):
        certify(1.0, 5)
    with pytest.raises(ValueError):
        certify(-1, 5)


def test_compare_logging_schemes_keeps_its_behavior(toy_episodes):
    cmp = compare_logging_schemes(toy_episodes, TOY_THETA)
    for cert in (cmp.per_step, cmp.episode_flag):
        assert cert.limit is None and cert.scope is None
        assert cert.status is None and cert.passed is None
        assert cert.alpha == 0.05
    assert cmp.per_step == certify(1, 3)
    assert cmp.episode_flag == certify(2, 3)
    assert compare_logging_schemes(toy_episodes, TOY_THETA, alpha=0.1).per_step == certify(1, 3, alpha=0.1)
    with pytest.raises(TypeError):
        compare_logging_schemes(toy_episodes, TOY_THETA, confidence=0.95)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# certify_replay
# ---------------------------------------------------------------------------


def test_certify_replay_with_a_limit_equals_certify_on_the_counts(toy_episodes):
    replay = replay_stop(toy_episodes, TOY_THETA)
    for limit in (0.5, 0.99):
        c = certify_replay(replay, limit=limit, scope=SCOPE)
        expected = certify(replay.n_harm_known, replay.n, n_unknown=replay.n_unknown, limit=limit, scope=SCOPE)
        assert c == expected
    assert certify_replay(replay, limit=0.5).status == "not_certified"
    assert certify_replay(replay, limit=0.99).status == "passed"


def test_certify_replay_with_unknown_harm_equals_certify_on_the_counts():
    eps = [
        Episode([0.1, 0.2], [None, False], [1, 1], failed=False),
        Episode([0.1, 0.9], [True, False], [1, 1], failed=False),
        Episode([0.1, 0.2], [False, False], [1, 1], failed=False),
    ]
    for policy in ("harm", "no_harm"):
        replay = replay_stop(eps, 0.5, unknown=policy)
        c = certify_replay(replay, confidence=0.9, limit=0.8)
        assert (c.n_harmed, c.n_unknown) == (2, 1)
        assert c == certify(replay.n_harm_known, replay.n, n_unknown=replay.n_unknown, confidence=0.9, limit=0.8)


def test_certify_replay_without_a_limit_ignores_post_stop_harm(toy_episodes):
    stated = certify_replay(replay_stop(toy_episodes, TOY_THETA))
    for p in (0.2, 1.0):
        replay = replay_stop(toy_episodes, TOY_THETA, valuation=StopValuation(post_stop_harm=p))
        c = certify_replay(replay)
        assert c == stated
        assert c.limit is None and c.status is None


@pytest.mark.parametrize("p", [1e-9, 0.2, 1.0])
def test_certify_replay_with_a_limit_rejects_post_stop_harm(toy_episodes, p):
    replay = replay_stop(toy_episodes, TOY_THETA, valuation=StopValuation(post_stop_harm=p))
    with pytest.raises(ValueError) as info:
        certify_replay(replay, limit=0.5)
    message = str(info.value)
    # Substitute 1: replay with no post-stop harm.
    assert "post_stop_harm=0" in message
    # Substitute 2: the worst-case count, here 1 harmed plus 1 clean stopped episode, out of 3.
    assert "k=2" in message and "certify(2, 3, limit=...)" in message


def test_post_stop_harm_message_counts_unknown_and_clean_stops():
    eps = [
        Episode([0.99, 0.1], [None, None], [1, 1], failed=False),  # stops at step 0, clean before the stop
        Episode([0.1, 0.2], [None, False], [1, 1], failed=False),  # agrees, harm unknown
        Episode([0.1, 0.9], [True, False], [1, 1], failed=False),  # stops with harm before the stop
        Episode([0.1, 0.2], [False, False], [1, 1], failed=False),  # agrees, no harm
    ]
    replay = replay_stop(eps, 0.5, valuation=StopValuation(post_stop_harm=0.3))
    n_bad = sum(e.harm is True or e.harm is None for e in replay.episodes)
    n_clean_stops = sum(e.branch == "stop" and e.harm is False for e in replay.episodes)
    k = n_bad + n_clean_stops
    assert (n_bad, n_clean_stops, k) == (2, 1, 3)
    with pytest.raises(ValueError) as info:
        certify_replay(replay, limit=0.5)
    assert f"k={k}" in str(info.value)
    assert f"certify({k}, 4, limit=...)" in str(info.value)
    # The named substitute is a valid call and gives a certificate on the worst-case count.
    c = certify(k, replay.n, limit=0.99)
    assert c.n_harmed == k


@pytest.mark.parametrize("unknown", ["harm", "no_harm"])
def test_post_stop_harm_message_counts_each_episode_once_under_either_policy(unknown):
    eps = [
        Episode([0.1, 0.99], [None, False], [1, 1], failed=False),  # stops at step 1, harm before it unknown
        Episode([0.99, 0.1], [False, True], [1, 1], failed=False),  # stops at step 0, clean before the stop
        Episode([0.1, 0.2], [None, False], [1, 1], failed=False),  # agrees, harm unknown
        Episode([0.1, 0.2], [True, False], [1, 1], failed=False),  # agrees, harmed
        Episode([0.1, 0.2], [False, False], [1, 1], failed=False),  # agrees, no harm
    ]
    replay = replay_stop(eps, 0.5, unknown=unknown, valuation=StopValuation(post_stop_harm=0.3))
    # Episodes 0 to 3 are harmed, unknown, or stopped clean. Under "no_harm" the replay's own
    # n_stop_clean also counts episode 0, so adding it would count that episode twice.
    with pytest.raises(ValueError) as info:
        certify_replay(replay, limit=0.5)
    assert "k=4" in str(info.value) and "certify(4, 5, limit=...)" in str(info.value)


def test_certify_violations_default_confidence_is_95_percent():
    assert certify_violations([False] * 299, limit=LIMIT).status == "passed"
    assert certify_violations([False] * 298, limit=LIMIT).status == "not_certified"
    assert certify_violations([False] * 299, limit=LIMIT).alpha == 0.05


def test_deeply_nested_json_raises_value_error():
    with pytest.raises(ValueError, match="nested too deeply"):
        certify(0, 10, scope="[" * 100000)
    with pytest.raises(ValueError, match="nested too deeply"):
        Certificate.from_json("[" * 100000)


def test_numpy_scalar_in_scope_error_names_numpy():
    with pytest.raises(TypeError, match=r"numpy\.") as info:
        certify(0, 10, scope={"a": np.bool_(True)})
    assert ".item()" in str(info.value)


def test_certify_replay_limit_and_replay_validation(toy_episodes):
    replay = replay_stop(toy_episodes, TOY_THETA)
    with pytest.raises(TypeError):
        certify_replay(toy_episodes, limit=0.5)  # type: ignore[arg-type]
    for limit, exc in [(0.0, ValueError), (1.0, ValueError), ("0.5", TypeError), (True, TypeError)]:
        with pytest.raises(exc):
            certify_replay(replay, limit=limit)


# ---------------------------------------------------------------------------
# certify_retry
# ---------------------------------------------------------------------------


def test_certify_retry_certifies_the_upper_count_and_reports_imputed_episodes(toy_episodes):
    bounds = retry_harm_bounds(toy_episodes, TOY_THETA, cap=2)
    assert (bounds.n, bounds.n_continue, bounds.n_harm_agree, bounds.n_harm_before_block) == (3, 2, 0, 1)
    c = certify_retry(bounds, limit=0.99, scope=SCOPE)
    assert c.n == 3
    assert c.n_harmed == bounds.n_harm_agree + bounds.n_continue == 2
    assert c.n_unknown == bounds.n_continue - bounds.n_harm_before_block == 1
    assert c == certify(1, 3, n_unknown=1, confidence=0.95, limit=0.99, scope=SCOPE)
    # The bound equals the exact bound on the upper count.
    assert c.upper == certify(round(bounds.upper * bounds.n), bounds.n).upper
    assert c.status == "passed"


def test_certify_retry_with_every_episode_continuing_is_vacuous(toy_episodes):
    bounds = retry_harm_bounds(toy_episodes, 0.05)
    c = certify_retry(bounds, limit=0.99)
    assert (c.n, c.n_harmed, c.n_unknown) == (3, 3, 3)
    assert c.upper == 1.0 and c.status == "not_certified"


def test_certify_retry_without_continuing_episodes_equals_certify(toy_episodes):
    bounds = retry_harm_bounds(toy_episodes, 0.99)
    assert bounds.n_continue == 0
    c = certify_retry(bounds, limit=0.99, confidence=0.9)
    assert c == certify(2, 3, confidence=0.9, limit=0.99)
    assert c.n_unknown == 0


@pytest.mark.parametrize("n, expected", [(298, "not_certified"), (299, "passed")])
def test_certify_retry_boundary(n, expected):
    bounds = retry_harm_bounds(_clean_episodes(n), 0.5, cap=1)
    assert certify_retry(bounds, limit=LIMIT).status == expected


def test_certify_retry_validation(toy_episodes):
    bounds = retry_harm_bounds(toy_episodes, TOY_THETA)
    with pytest.raises(TypeError):
        certify_retry(toy_episodes, limit=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        certify_retry(replay_stop(toy_episodes, TOY_THETA), limit=0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        certify_retry(bounds)  # type: ignore[call-arg]
    for kwargs, exc in [
        ({"limit": None}, TypeError),
        ({"limit": 0.0}, ValueError),
        ({"limit": 0.5, "confidence": 1.0}, ValueError),
        ({"limit": 0.5, "confidence": None}, TypeError),
    ]:
        with pytest.raises(exc):
            certify_retry(bounds, **kwargs)


def test_retry_module_does_not_import_certify():
    source = inspect.getsource(importlib.import_module("agent_risk.retry"))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert "certify" not in (node.module or "")
            assert all(alias.name != "certify" for alias in node.names)
        elif isinstance(node, ast.Import):
            assert all("certify" not in alias.name for alias in node.names)


# ---------------------------------------------------------------------------
# Certificate fields
# ---------------------------------------------------------------------------


def test_status_and_passed_are_none_without_a_limit():
    c = certify(0, 100)
    assert c.limit is None
    assert c.status is None and c.passed is None
    assert c.to_dict()["status"] is None and c.to_dict()["limit"] is None


def test_status_with_a_limit():
    good = certify(0, 299, confidence=CONFIDENCE, limit=LIMIT)
    bad = certify(0, 298, confidence=CONFIDENCE, limit=LIMIT)
    assert (good.passed, good.status) == (True, "passed")
    assert (bad.passed, bad.status) == (False, "not_certified")
    assert good.limit == LIMIT


def test_a_bound_equal_to_the_limit_passes():
    upper = clopper_pearson_upper(2, 40)
    at_limit = certify(2, 40, limit=upper)
    assert (at_limit.passed, at_limit.status) == (True, "passed")
    below = certify(2, 40, limit=math.nextafter(upper, 0.0))
    assert (below.passed, below.status) == (False, "not_certified")


def test_aliases():
    c = certify(2, 50, n_unknown=1, confidence=0.9)
    assert c.n_violations == c.n_harmed == 3
    assert c.confidence == c.level == pytest.approx(0.9)
    assert c.level == 1.0 - c.alpha
    for name in ("n_violations", "confidence", "level", "passed", "status"):
        with pytest.raises(AttributeError):
            setattr(c, name, 1)


def test_certificate_is_frozen():
    c = certify(1, 10, limit=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.limit = 0.9  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.scope = None  # type: ignore[misc]


def test_existing_fields_keep_their_order():
    names = [f.name for f in dataclasses.fields(Certificate)]
    assert names == ["n", "n_harmed", "n_unknown", "alpha", "estimate", "upper", "limit", "scope"]


@pytest.mark.parametrize(
    "limit, exc", [(0.0, ValueError), (1.0, ValueError), (-0.1, ValueError), ("0.1", TypeError), (True, TypeError)]
)
def test_limit_validation(limit, exc):
    with pytest.raises(exc):
        certify(1, 10, limit=limit)
    with pytest.raises(exc):
        Certificate(10, 1, 0, 0.05, 0.1, 0.3, limit=limit)


def test_limit_is_stored_as_a_float():
    assert type(certify(1, 10, limit=np.float32(0.5)).limit) is float
    assert type(Certificate(10, 1, 0, 0.05, 0.1, 0.3, limit=np.float64(0.5)).limit) is float


def test_to_dict_keys_and_values():
    c = certify(1, 50, n_unknown=1, confidence=0.9, limit=0.2, scope=SCOPE)
    d = c.to_dict()
    assert list(d) == [
        "method",
        "n",
        "n_violations",
        "n_unknown",
        "estimate",
        "upper",
        "alpha",
        "confidence",
        "limit",
        "status",
        "scope",
        "conditions",
        "family_warning",
    ]
    assert d["method"] == "one-sided exact Clopper-Pearson"
    assert (d["n"], d["n_violations"], d["n_unknown"]) == (50, 2, 1)
    assert d["estimate"] == 2 / 50 and d["upper"] == c.upper
    assert d["alpha"] == 0.1 and d["confidence"] == pytest.approx(0.9)
    assert d["limit"] == 0.2 and d["status"] == c.status
    assert d["scope"] == SCOPE
    assert d["conditions"] == CONDITIONS and d["family_warning"] == FAMILY_WARNING
    assert certify(1, 50).to_dict()["scope"] is None


def test_fixed_sentences():
    assert CONDITIONS == (
        "The bound assumes that the episodes are independent draws from the task distribution, "
        "that their number was fixed before any outcome was seen and every drawn episode is counted, "
        "and that the deployment, the violation rule, the limit, and the confidence were chosen "
        "before the certification episodes were seen."
    )
    assert FAMILY_WARNING == (
        "Separate certificates each control their own error, but they do not control the chance "
        "that at least one incorrect certificate passes."
    )


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_reordered_scope_keys_match_and_compare_equal():
    a = certify(0, 299, limit=LIMIT, scope={"model": "m", "gate": {"theta": 0.9, "cap": 2}, "tools": ["a", "b"]})
    b = certify(0, 299, limit=LIMIT, scope={"tools": ["a", "b"], "gate": {"cap": 2, "theta": 0.9}, "model": "m"})
    assert a.scope == b.scope == '{"gate":{"cap":2,"theta":0.9},"model":"m","tools":["a","b"]}'
    assert a == b and hash(a) == hash(b)
    assert a.matches_scope(b) and b.matches_scope(a)
    assert a.matches_scope({"model": "m", "tools": ["a", "b"], "gate": {"theta": 0.9, "cap": 2}})


def test_unequal_scope_values_do_not_match():
    a = certify(0, 299, scope={"model": "m", "theta": 0.9})
    assert not a.matches_scope({"model": "m", "theta": 0.95})
    assert not a.matches_scope({"model": "m"})
    assert not a.matches_scope({"model": "m", "theta": 0.9, "extra": 1})
    other = certify(0, 299, scope={"model": "m2", "theta": 0.9})
    assert not a.matches_scope(other) and a != other
    # List order is part of the value.
    assert not certify(0, 9, scope={"t": ["a", "b"]}).matches_scope({"t": ["b", "a"]})


def test_a_missing_scope_matches_nothing():
    plain = certify(0, 299)
    scoped = certify(0, 299, scope={"model": "m"})
    assert plain.scope is None
    assert not plain.matches_scope(plain)
    assert not plain.matches_scope(scoped)
    assert not scoped.matches_scope(plain)
    assert not plain.matches_scope({"model": "m"})
    assert scoped.matches_scope(scoped)


@pytest.mark.parametrize("other", [None, "model", '{"model":"m"}', 5, [("model", "m")], object()])
def test_matches_scope_rejects_other_types(other):
    with pytest.raises(TypeError):
        certify(0, 10, scope={"model": "m"}).matches_scope(other)


def test_matches_scope_normalizes_a_mapping_argument():
    c = certify(0, 10, scope={"model": "m"})
    with pytest.raises(ValueError):
        c.matches_scope({})
    with pytest.raises(ValueError):
        c.matches_scope({"x": float("nan")})
    with pytest.raises(TypeError):
        c.matches_scope({1: "m"})


@pytest.mark.parametrize("scope", [{}, "", "{}", "[]", '"text"', "not json", '{"a": NaN}', '{"a": Infinity}'])
def test_empty_or_malformed_scope_raises_value_error(scope):
    with pytest.raises(ValueError):
        certify(0, 10, scope=scope)
    with pytest.raises(ValueError):
        Certificate(10, 0, 0, 0.05, 0.0, 0.2, scope=scope)


def test_cyclic_scope_raises_value_error():
    cycle = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError):
        certify(0, 10, scope=cycle)


@pytest.mark.parametrize(
    "scope",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": float("-inf")},
        {"x": [1, float("nan")]},
        {"x": {"y": float("inf")}},
    ],
)
def test_non_finite_scope_values_raise_value_error(scope):
    with pytest.raises(ValueError):
        certify(0, 10, scope=scope)


@pytest.mark.parametrize(
    "scope",
    [
        {1: "a"},
        {None: "a"},
        {"a": {1: 2}},
        {"a": [{2: 3}]},
        {"a": {1, 2}},
        {"a": frozenset([1])},
        {"a": b"bytes"},
        {"a": object()},
        {"a": complex(1, 1)},
        {"a": np.float64(1.0)},
        {"a": np.int64(1)},
        {"a": np.bool_(True)},
        {"a": np.str_("s")},
        {"a": np.array([1, 2])},
        {"a": [np.float32(0.5)]},
        [("a", 1)],
        5,
        1.5,
        True,
        b'{"a":1}',
    ],
)
def test_unsupported_scope_values_raise_type_error(scope):
    with pytest.raises(TypeError):
        certify(0, 10, scope=scope)


def test_json_value_types_are_accepted_in_a_scope():
    scope = {
        "none": None,
        "flag": True,
        "count": 3,
        "rate": 0.5,
        "name": "café",
        "list": [1, "a", None],
        "tuple": (1, 2),
        "nested": {"b": [{"c": False}]},
    }
    c = certify(0, 10, scope=scope)
    expected = {**scope, "tuple": [1, 2]}
    assert json.loads(c.scope) == expected
    assert c.to_dict()["scope"] == expected
    assert "café" in c.scope  # stored unescaped
    assert certify(0, 10, scope=c.scope) == c  # the stored string is itself a valid scope


def test_scope_may_be_given_as_a_json_object_string():
    c = certify(0, 10, scope='{"b": 1, "a": {"y": 2, "x": 1}}')
    assert c.scope == '{"a":{"x":1,"y":2},"b":1}'
    assert c == certify(0, 10, scope={"a": {"x": 1, "y": 2}, "b": 1})


def test_certificates_are_hashable():
    a = certify(0, 299, limit=LIMIT, scope={"x": 1, "y": 2})
    b = certify(0, 299, limit=LIMIT, scope={"y": 2, "x": 1})
    c = certify(0, 299, limit=LIMIT)
    assert len({a, b}) == 1
    assert len({a, b, c}) == 2
    assert {a: "first"}[b] == "first"


def test_replace_keeps_and_normalizes_the_scope():
    c = certify(0, 299, limit=LIMIT, scope={"b": 1, "a": 2})
    kept = dataclasses.replace(c, limit=0.5)
    assert kept.scope == c.scope and kept.limit == 0.5
    assert dataclasses.replace(c, scope=None).scope is None
    assert dataclasses.replace(c, scope={"z": [1]}).scope == '{"z":[1]}'
    with pytest.raises(ValueError):
        dataclasses.replace(c, limit=1.5)
    with pytest.raises(ValueError):
        dataclasses.replace(c, scope={})


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cert",
    [
        certify(0, 299),
        certify(3, 120, alpha=0.1),
        certify(0, 299, confidence=0.95, limit=0.01),
        certify(0, 298, confidence=0.95, limit=0.01),
        certify(7, 7, confidence=0.99, limit=0.5),
        certify(2, 80, n_unknown=3, confidence=0.9, limit=0.2, scope=SCOPE),
        certify(0, 10, n_unknown=0, scope={"only": "scope"}),
        certify(1, 1000, confidence=0.999, limit=0.05, scope={"unicode": "café", "n": [1, 2.5, None, True]}),
    ],
)
def test_json_round_trip(cert):
    text = cert.to_json()
    rebuilt = Certificate.from_json(text)
    assert rebuilt == cert
    assert rebuilt.to_json() == text
    assert Certificate.from_json(cert.to_json(indent=2)) == cert
    assert json.loads(text) == cert.to_dict()


def test_to_json_indent_and_strictness():
    c = certify(1, 10)
    assert "\n" not in c.to_json()
    assert "\n" in c.to_json(indent=2)
    assert "NaN" not in c.to_json() and "Infinity" not in c.to_json()
    with pytest.raises(TypeError):
        c.to_json(2)  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [
        {"upper": 0.5},
        {"upper": 1.0},
        {"estimate": 0.123},
        {"n_violations": 3},
        {"n": 300},
        {"alpha": 0.1},
        {"confidence": 0.9},
        {"status": "not_certified"},
        {"status": "passed", "limit": 0.0001},
        {"limit": 0.0001},
        {"status": None},
    ],
)
def test_tampered_certificate_is_rejected(changes):
    original = certify(0, 299, confidence=0.95, limit=0.01, scope=SCOPE)
    with pytest.raises(ValueError):
        Certificate.from_json(_tampered(original, **changes))


def test_tampered_status_on_a_limitless_certificate_is_rejected():
    original = certify(0, 299)
    assert Certificate.from_json(_tampered(original, status=None)) == original
    for status in ("passed", "not_certified"):
        with pytest.raises(ValueError, match="status"):
            Certificate.from_json(_tampered(original, status=status))


def test_tampered_status_of_a_failed_certificate_is_rejected():
    original = certify(0, 298, confidence=0.95, limit=0.01)
    assert json.loads(original.to_json())["status"] == "not_certified"
    with pytest.raises(ValueError, match="status"):
        Certificate.from_json(_tampered(original, status="passed"))


def test_tampered_upper_names_the_field():
    with pytest.raises(ValueError, match="upper"):
        Certificate.from_json(_tampered(certify(1, 50), upper=0.001))


def test_tampered_counts_are_rejected():
    original = certify(2, 50, n_unknown=1)
    with pytest.raises(ValueError):
        Certificate.from_json(_tampered(original, n_violations=51, estimate=51 / 50))
    with pytest.raises(ValueError):
        Certificate.from_json(_tampered(original, n_unknown=5))
    with pytest.raises(ValueError):
        Certificate.from_json(_tampered(certify(0, 1), n=0))


@pytest.mark.parametrize("field", ["n", "n_violations", "n_unknown", "alpha", "estimate", "upper"])
def test_missing_core_field_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        Certificate.from_json(_tampered(certify(1, 50), **{field: _DROP}))


@pytest.mark.parametrize(
    "text", ["", "not json", "[]", "1", "null", '{"n": NaN}', '{"upper": Infinity}', '{"upper": -Infinity}']
)
def test_malformed_certificate_json_is_rejected(text):
    with pytest.raises(ValueError):
        Certificate.from_json(text)


def test_non_finite_constant_in_otherwise_valid_json_is_rejected():
    text = certify(1, 50).to_json().replace('"upper":', '"upper": NaN, "x":', 1)
    with pytest.raises(ValueError):
        Certificate.from_json(text)


def test_wrong_json_types_are_rejected():
    original = certify(1, 50)
    for changes in ({"n": "50"}, {"n": 50.0}, {"n_violations": True}, {"upper": "x"}, {"alpha": "0.05"}):
        with pytest.raises(TypeError):
            Certificate.from_json(_tampered(original, **changes))


def test_scope_in_json_is_validated():
    original = certify(0, 10, scope={"a": 1})
    with pytest.raises(ValueError):
        Certificate.from_json(_tampered(original, scope={}))
    with pytest.raises(TypeError):
        Certificate.from_json(_tampered(original, scope=[1]))


def test_string_scope_in_json_is_rejected():
    original = certify(0, 10, scope={"a": 1})
    with pytest.raises(TypeError, match="str"):
        Certificate.from_json(_tampered(original, scope='{"a": 1}'))


@pytest.mark.parametrize("direction", [0.0, 1.0])
def test_stored_estimate_one_ulp_away_is_accepted(direction):
    original = certify(2, 40, confidence=0.9, limit=0.2, scope=SCOPE)
    nudged = math.nextafter(original.estimate, direction)
    assert nudged != original.estimate
    rebuilt = Certificate.from_json(_tampered(original, estimate=nudged))
    assert rebuilt == original
    assert rebuilt.estimate == 2 / 40


def test_stored_upper_cannot_flip_the_status():
    k, n = 2, 40
    expected_upper = clopper_pearson_upper(k, n, confidence=CONFIDENCE)
    limit = expected_upper - 5e-10
    original = certify(k, n, limit=limit)
    assert original.status == "not_certified"
    assert original.upper == expected_upper

    with pytest.raises(ValueError, match="status"):
        Certificate.from_json(_tampered(original, upper=limit, status="passed"))

    rebuilt = Certificate.from_json(_tampered(original, upper=limit))
    assert rebuilt.upper == expected_upper
    assert rebuilt.status == "not_certified"
    assert rebuilt == original


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_markdown_for_a_passed_certificate():
    md = certify(0, 299, confidence=CONFIDENCE, limit=LIMIT, scope={"model": "m"}).to_markdown()
    assert "passed" in md
    assert "95%" in md
    assert "0.01" in md
    assert "one-sided exact Clopper-Pearson" in md
    assert '{"model":"m"}' in md
    assert "299" in md
    assert CONDITIONS in md and FAMILY_WARNING in md
    assert "no known outcome" not in md


def test_markdown_for_a_failed_certificate():
    md = certify(0, 298, confidence=CONFIDENCE, limit=LIMIT).to_markdown()
    assert "not certified" in md
    assert "exceeds" in md
    assert CONDITIONS in md and FAMILY_WARNING in md
    assert "Scope" not in md


def test_markdown_without_a_limit():
    md = certify(2, 50, confidence=0.9).to_markdown()
    assert "no limit set" in md
    assert "90%" in md
    assert CONDITIONS in md and FAMILY_WARNING in md


def test_markdown_mentions_unknown_outcomes_only_when_there_are_some():
    md = certify(2, 50, n_unknown=1).to_markdown()
    assert "Episodes with no known outcome: 1 of 50, each counted as a violation." in md
    c = certify_violations([0, None, None, 0, 1], limit=0.9)
    assert "Episodes with no known outcome: 2 of 5, each counted as a violation." in c.to_markdown()
    assert "no known outcome" not in certify(3, 50).to_markdown()


@pytest.mark.parametrize(
    "n, sentence",
    [
        (298, "The upper bound 0.010002 exceeds the limit 0.01 "),
        (299, "The upper bound 0.009969 is at most the limit 0.01 "),
    ],
)
def test_markdown_prints_a_bound_that_orders_like_the_status(n, sentence):
    # At n = 298 the bound is 0.0100024; four digits would print it as equal to the limit.
    assert sentence in certify(0, n, confidence=CONFIDENCE, limit=LIMIT).to_markdown()


def test_markdown_confidence_never_rounds_up_to_100_percent():
    md = certify(0, 10**8, confidence=0.9999999, limit=0.5).to_markdown()
    assert "99.99999%" in md and "100%" not in md


def test_markdown_scope_survives_backticks():
    md = certify(0, 10, scope={"p": "a`b"}).to_markdown()
    assert '- Scope: ``{"p":"a`b"}``' in md


def test_markdown_is_rebuilt_from_public_fields():
    c = certify(2, 80, n_unknown=3, confidence=0.9, limit=0.2, scope=SCOPE)
    assert Certificate.from_json(c.to_json()).to_markdown() == c.to_markdown()
    assert c.to_markdown().endswith("\n")


# ---------------------------------------------------------------------------
# Exports and statsmodels
# ---------------------------------------------------------------------------


def test_new_names_are_exported():
    certify_module = importlib.import_module("agent_risk.certify")
    for name in ("certify_violations", "certify_retry", "sample_size"):
        assert name in certify_module.__all__
        assert name in agent_risk.__all__
        assert callable(getattr(agent_risk, name))
    assert certify_module.certify_retry is agent_risk.certify_retry
    names = [n for n in agent_risk.__all__ if n != "__version__"]
    lower = [n for n in names if n[0].islower()]
    assert lower == sorted(lower)
    assert agent_risk.Certificate is Certificate


@pytest.mark.parametrize("confidence", [0.8, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("k, n", [(0, 10), (0, 299), (1, 473), (2, 10), (5, 40), (39, 40), (100, 1000)])
def test_matches_one_sided_statsmodels(k, n, confidence):
    sm = pytest.importorskip("statsmodels.stats.proportion")
    # statsmodels' interval is two-sided: its upper end at alpha is the one-sided bound at 1 - alpha / 2.
    _, upper = sm.proportion_confint(k, n, alpha=2 * (1 - confidence), method="beta")
    assert clopper_pearson_upper(k, n, confidence=confidence) == pytest.approx(upper, abs=1e-12)
    assert certify(k, n, confidence=confidence).upper == pytest.approx(upper, abs=1e-12)


def test_two_sided_alpha_is_not_the_one_sided_alpha():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    _, upper_two_sided_05 = sm.proportion_confint(3, 100, alpha=0.05, method="beta")
    assert clopper_pearson_upper(3, 100, confidence=0.975) == pytest.approx(upper_two_sided_05, abs=1e-12)
    assert clopper_pearson_upper(3, 100, confidence=0.95) < upper_two_sided_05 - 1e-4
