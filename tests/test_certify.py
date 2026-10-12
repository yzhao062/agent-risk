import math

import pytest
from scipy.stats import binom

from agent_risk import (
    Episode,
    certify,
    certify_replay,
    clopper_pearson_upper,
    compare_logging_schemes,
    replay_stop,
)

from toydata import TOY_THETA


@pytest.mark.parametrize("n", [1, 5, 59, 300])
@pytest.mark.parametrize("alpha", [0.05, 0.01])
def test_zero_events_matches_closed_form(n, alpha):
    # With k = 0 the exact bound solves (1 - U)^n = alpha.
    assert clopper_pearson_upper(0, n, alpha) == pytest.approx(1 - alpha ** (1 / n), rel=1e-10)


def test_rule_of_three_scale():
    # 0 of 59 harmed gives a 95% upper bound just below 5%.
    assert clopper_pearson_upper(0, 59) < 0.05 < clopper_pearson_upper(0, 58)


@pytest.mark.parametrize("k, n", [(1, 3), (4, 20), (17, 100), (99, 100)])
def test_bound_solves_the_binomial_tail(k, n):
    u = clopper_pearson_upper(k, n, 0.05)
    assert binom.cdf(k, n, u) == pytest.approx(0.05, abs=1e-9)
    assert k / n < u < 1


def test_all_harmed_gives_one():
    assert clopper_pearson_upper(7, 7) == 1.0


def test_bound_is_monotone():
    values = [clopper_pearson_upper(k, 20) for k in range(21)]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert clopper_pearson_upper(3, 20, 0.01) > clopper_pearson_upper(3, 20, 0.05)


def test_matches_statsmodels_if_available():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for k, n in [(0, 10), (2, 10), (5, 40), (39, 40)]:
        # A one-sided level 1 - alpha bound is the upper end of a two-sided 1 - 2 alpha interval.
        _, upper = sm.proportion_confint(k, n, alpha=0.10, method="beta")
        assert clopper_pearson_upper(k, n, 0.05) == pytest.approx(upper, abs=1e-12)


@pytest.mark.parametrize(
    "args, exc",
    [
        ((4, 3), ValueError),
        ((0, 0), ValueError),
        ((-1, 3), ValueError),
        ((1.0, 3), TypeError),
        ((True, 3), TypeError),
        ((1, 3, 0.0), ValueError),
        ((1, 3, 1.0), ValueError),
        ((1, 3, "0.05"), TypeError),
    ],
)
def test_clopper_pearson_validation(args, exc):
    with pytest.raises(exc):
        clopper_pearson_upper(*args)


def test_certify_counts_unknown_as_harmed():
    c = certify(2, 50, n_unknown=3)
    assert c.n_harmed == 5 and c.n_unknown == 3 and c.n == 50
    assert c.estimate == pytest.approx(0.1)
    assert c.upper == pytest.approx(clopper_pearson_upper(5, 50))
    assert c.level == pytest.approx(0.95)
    with pytest.raises(ValueError, match="exceed"):
        certify(48, 50, n_unknown=3)
    with pytest.raises(ValueError, match="at least 1"):
        certify(0, 0)


def test_certify_replay_toy(toy_episodes):
    c = certify_replay(replay_stop(toy_episodes, TOY_THETA))
    assert (c.n_harmed, c.n) == (1, 3)
    assert c.upper == pytest.approx(clopper_pearson_upper(1, 3))
    with pytest.raises(TypeError):
        certify_replay(toy_episodes)  # type: ignore[arg-type]


def test_certify_replay_counts_unknown_whatever_the_policy():
    eps = [
        Episode([0.1, 0.2], [None, False], [1, 1], failed=False),
        Episode([0.1, 0.2], [False, False], [1, 1], failed=False),
    ]
    for policy in ("harm", "no_harm"):
        c = certify_replay(replay_stop(eps, 0.5, unknown=policy))
        assert (c.n_harmed, c.n_unknown) == (1, 1)


def test_logging_schemes_toy(toy_episodes):
    cmp = compare_logging_schemes(toy_episodes, TOY_THETA)
    # Per-step log: only episode 0 is harmed before its stop.
    assert cmp.per_step.n_harmed == 1
    # Flag-only log: episode 1 is flagged and its harm cannot be placed after the stop.
    assert cmp.episode_flag.n_harmed == 2
    assert cmp.n_timing_unknown == 2
    assert cmp.n_extra == 1
    assert cmp.episode_flag.upper > cmp.per_step.upper


def test_logging_schemes_agree_without_stops(toy_episodes):
    cmp = compare_logging_schemes(toy_episodes, 0.99)
    assert cmp.per_step == cmp.episode_flag
    assert cmp.n_timing_unknown == 0 and cmp.n_extra == 0


def test_logging_schemes_all_unknown():
    eps = [
        Episode([0.99, 0.1], [None, None], [1, 1], failed=False),
        Episode([0.1, 0.1], [None, None], [1, 1], failed=False),
    ]
    cmp = compare_logging_schemes(eps, 0.5)
    # The per-step log clears the episode stopped at step 0; the flag-only log cannot.
    assert (cmp.per_step.n_harmed, cmp.per_step.n_unknown) == (1, 1)
    assert (cmp.episode_flag.n_harmed, cmp.episode_flag.n_unknown) == (2, 2)
    assert cmp.episode_flag.upper == 1.0
    assert cmp.n_timing_unknown == 1


def test_logging_schemes_flag_false_clears_a_stop():
    eps = [Episode([0.99, 0.1], [False, False], [1, 1], failed=False)]
    cmp = compare_logging_schemes(eps, 0.5)
    assert cmp.per_step.n_harmed == cmp.episode_flag.n_harmed == 0
    assert cmp.n_timing_unknown == 0


def test_logging_schemes_validation(toy_episodes):
    with pytest.raises(ValueError, match="empty"):
        compare_logging_schemes([], 0.5)
    with pytest.raises(ValueError, match="alpha"):
        compare_logging_schemes(toy_episodes, 0.5, alpha=1.5)


def test_certificate_upper_is_finite(toy_episodes):
    c = certify_replay(replay_stop(toy_episodes, 0.05))
    assert c.n_harmed == 0
    assert math.isfinite(c.upper) and 0 < c.upper < 1
