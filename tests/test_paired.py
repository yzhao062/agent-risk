import math

import pytest
from scipy.stats import norm

from agent_risk import PairedInterval, is_equivalent, newcombe_paired_interval, wilson_interval

# Newcombe (1998), Statistics in Medicine 17:2635-2650, Table III: 95% intervals
# for the cell counts e, f, g, h (both, first only, second only, neither) by
# method 8 (score intervals, phi as estimated) and method 10 (score intervals,
# continuity-corrected phi). Table II of the same article covers methods 1-7.
# Published values are rounded to four decimals.
NEWCOMBE_TABLE_III = [
    # e, f, g, h, method 8, method 10
    (36, 12, 2, 0, (0.0569, 0.3404), (0.0569, 0.3404)),
    (20, 12, 2, 16, (0.0618, 0.3242), (0.0562, 0.3292)),
    (18, 12, 2, 18, (0.0618, 0.3239), (0.0562, 0.3290)),
    (36, 14, 0, 0, (0.1528, 0.4167), (0.1528, 0.4167)),
    (35, 14, 0, 1, (0.1573, 0.4149), (0.1461, 0.4175)),
    (18, 14, 0, 18, (0.1504, 0.3910), (0.1441, 0.3963)),
    (2, 97, 1, 0, (0.8721, 0.9854), (0.8721, 0.9854)),
    (1, 97, 1, 1, (0.8737, 0.9850), (0.8736, 0.9850)),
    (0, 29, 1, 0, (0.6666, 0.9882), (0.6666, 0.9882)),
    (2, 98, 0, 0, (0.9178, 0.9945), (0.9178, 0.9945)),
    (1, 98, 0, 1, (0.9174, 0.9916), (0.9171, 0.9916)),
    (0, 30, 0, 0, (0.8395, 1.0), (0.8395, 1.0)),
    (54, 0, 0, 0, (-0.0664, 0.0664), (-0.0664, 0.0664)),
    (53, 0, 0, 1, (-0.0640, 0.0640), (-0.0729, 0.0729)),
    (30, 0, 0, 24, (-0.0074, 0.0074), (-0.0358, 0.0358)),
    (29, 0, 0, 25, (-0.0049, 0.0049), (-0.0354, 0.0354)),
    (28, 0, 0, 26, (-0.0025, 0.0025), (-0.0352, 0.0352)),
    (27, 0, 0, 27, (0.0, 0.0), (-0.0351, 0.0351)),
]

# Half a unit in the fourth decimal, plus floating-point slack.
ROUNDING = 5e-5 + 1e-12


@pytest.mark.parametrize("e, f, g, h, m8, m10", NEWCOMBE_TABLE_III)
def test_method_8_matches_published_table(e, f, g, h, m8, m10):
    ci = newcombe_paired_interval(e, f, g, h, phi_correction=False)
    assert ci.method == "newcombe8"
    assert ci.lower == pytest.approx(m8[0], abs=ROUNDING)
    assert ci.upper == pytest.approx(m8[1], abs=ROUNDING)


@pytest.mark.parametrize("e, f, g, h, m8, m10", NEWCOMBE_TABLE_III)
def test_method_10_matches_published_table(e, f, g, h, m8, m10):
    ci = newcombe_paired_interval(e, f, g, h)
    assert ci.method == "newcombe10"
    assert ci.estimate == pytest.approx((f - g) / (e + f + g + h))
    if (e, f, g, h) == (1, 97, 1, 1):
        # Here e*h < f*g, so the correction does not apply and method 10 equals
        # method 8 by definition; the table prints 0.8737 for method 8 and
        # 0.8736 for method 10. The computed limit is 0.873672.
        assert ci.interval == newcombe_paired_interval(e, f, g, h, phi_correction=False).interval
        assert ci.lower == pytest.approx(m10[0], abs=1e-4)
        assert ci.lower == pytest.approx(m8[0], abs=ROUNDING)
    else:
        assert ci.lower == pytest.approx(m10[0], abs=ROUNDING)
    assert ci.upper == pytest.approx(m10[1], abs=ROUNDING)


def test_phi_estimates():
    # Zero marginal total: phi is set to 0.
    assert newcombe_paired_interval(36, 14, 0, 0).phi == 0.0
    # e*h > f*g: method 10 subtracts n/2 from the numerator.
    assert newcombe_paired_interval(53, 0, 0, 1, phi_correction=False).phi == pytest.approx(1.0)
    assert newcombe_paired_interval(53, 0, 0, 1).phi == pytest.approx(26 / 53)
    # e*h < f*g: no correction.
    a = newcombe_paired_interval(36, 12, 2, 0)
    assert a.phi == pytest.approx(-24 / math.sqrt(48 * 2 * 38 * 12))


def test_swapping_outcomes_negates_the_interval():
    a = newcombe_paired_interval(20, 12, 2, 16)
    b = newcombe_paired_interval(20, 2, 12, 16)
    assert b.lower == pytest.approx(-a.upper)
    assert b.upper == pytest.approx(-a.lower)


def test_level_controls_width():
    wide = newcombe_paired_interval(20, 12, 2, 16, level=0.99)
    mid = newcombe_paired_interval(20, 12, 2, 16, level=0.95)
    narrow = newcombe_paired_interval(20, 12, 2, 16, level=0.90)
    assert wide.lower < mid.lower < narrow.lower < narrow.estimate
    assert narrow.estimate < narrow.upper < mid.upper < wide.upper


def test_interval_stays_in_range():
    for counts in [(0, 5, 0, 0), (0, 0, 5, 0), (0, 1, 0, 0), (1, 0, 0, 0), (0, 3, 2, 0)]:
        ci = newcombe_paired_interval(*counts)
        assert -1.0 <= ci.lower <= ci.estimate <= ci.upper <= 1.0


@pytest.mark.parametrize(
    "args, kwargs, exc",
    [
        ((0, 0, 0, 0), {}, ValueError),
        ((-1, 2, 3, 4), {}, ValueError),
        ((1.5, 2, 3, 4), {}, TypeError),
        ((1, 2, 3, 4), {"level": 1.0}, ValueError),
        ((1, 2, 3, 4), {"level": 0}, ValueError),
        ((1, 2, 3, 4), {"phi_correction": 1}, TypeError),
    ],
)
def test_newcombe_validation(args, kwargs, exc):
    with pytest.raises(exc):
        newcombe_paired_interval(*args, **kwargs)


def test_wilson_closed_forms():
    z = norm.ppf(0.975)
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0 and hi == pytest.approx(z**2 / (30 + z**2))
    lo, hi = wilson_interval(30, 30)
    assert hi == 1.0 and lo == pytest.approx(30 / (30 + z**2))
    with pytest.raises(ValueError):
        wilson_interval(4, 3)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_wilson_matches_statsmodels_if_available():
    sm = pytest.importorskip("statsmodels.stats.proportion")
    for x, n in [(0, 10), (3, 10), (48, 50), (17, 101)]:
        for level in (0.9, 0.95):
            expected = sm.proportion_confint(x, n, alpha=1 - level, method="wilson")
            got = wilson_interval(x, n, level)
            assert got == pytest.approx(expected, abs=1e-12)


def test_is_equivalent():
    ci = newcombe_paired_interval(40, 1, 2, 57, level=0.90)
    assert isinstance(ci, PairedInterval)
    assert is_equivalent(ci, 0.10) is True
    assert is_equivalent(ci, 0.01) is False
    assert is_equivalent((-0.02, 0.03), 0.05) is True
    # The margin is an open bound.
    assert is_equivalent((-0.05, 0.03), 0.05) is False
    assert is_equivalent((0.06, 0.08), 0.05) is False


@pytest.mark.parametrize(
    "interval, margin, exc",
    [
        ((0.1, -0.1), 0.2, ValueError),
        ((-0.1, 0.1), 0.0, ValueError),
        ((-0.1, 0.1), -0.2, ValueError),
        ((-0.1,), 0.2, TypeError),
        (0.1, 0.2, TypeError),
        ((float("nan"), 0.1), 0.2, TypeError),
        ((-0.1, 0.1), "0.2", TypeError),
    ],
)
def test_is_equivalent_validation(interval, margin, exc):
    with pytest.raises(exc):
        is_equivalent(interval, margin)
