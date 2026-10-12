import pytest

from agent_risk import Episode, replay_stop, retry_harm_bounds

from toydata import TOY_THETA


def test_toy_retry_bounds(toy_episodes):
    b = retry_harm_bounds(toy_episodes, TOY_THETA, cap=2)
    assert b.branches == ("continue", "continue", "agree")
    assert b.first_block == (2, 2, None)
    assert b.n == 3 and b.n_agree == 1 and b.n_continue == 2
    # Episode 0 has harm before the first block, so it is harmed at both ends.
    assert b.n_harm_before_block == 1 and b.n_harm_agree == 0
    assert b.lower == pytest.approx(1 / 3)
    assert b.upper == pytest.approx(2 / 3)
    assert b.interval == (b.lower, b.upper)
    assert b.width == pytest.approx(1 / 3)


def test_interval_does_not_depend_on_cap(toy_episodes):
    one = retry_harm_bounds(toy_episodes, TOY_THETA, cap=1)
    many = retry_harm_bounds(toy_episodes, TOY_THETA, cap=50)
    assert one.interval == many.interval
    assert many.cap == 50


def test_lower_end_matches_stopping_replay(toy_episodes):
    # With binary harm, harm before the first exceedance is the same quantity in both replays.
    for theta in (0.05, 0.15, 0.25, TOY_THETA, 0.99):
        assert retry_harm_bounds(toy_episodes, theta).lower == pytest.approx(
            replay_stop(toy_episodes, theta).p_harm_stated
        )


def test_theta_above_every_score_is_point_identified(toy_episodes):
    b = retry_harm_bounds(toy_episodes, 0.99)
    assert b.n_continue == 0
    assert b.lower == b.upper == pytest.approx(2 / 3)


def test_theta_below_every_score_is_vacuous_above(toy_episodes):
    b = retry_harm_bounds(toy_episodes, 0.05)
    assert b.first_block == (0, 0, 0)
    assert b.lower == 0.0 and b.upper == 1.0


def test_unknown_harm_is_rejected_where_read():
    with pytest.raises(ValueError, match=r"binary harm.*episodes\[0\]\.harm\[0\]"):
        retry_harm_bounds([Episode([0.1, 0.2], [None, False], [1, 1], failed=False)], 0.5)
    with pytest.raises(ValueError, match="binary harm"):
        retry_harm_bounds([Episode([0.1, 0.9], [None, False], [1, 1], failed=False)], 0.5)
    # Entries at or after the first block are never read.
    b = retry_harm_bounds([Episode([0.1, 0.9, 0.1], [False, None, None], [1, 1, 1], failed=False)], 0.5)
    assert b.interval == (0.0, 1.0)


def test_all_unknown_is_rejected():
    eps = [Episode([0.1, 0.2], [None, None], [1, 1], failed=False)]
    with pytest.raises(ValueError, match="binary harm"):
        retry_harm_bounds(eps, 0.5)


@pytest.mark.parametrize("cap, exc", [(0, ValueError), (-1, ValueError), (1.0, TypeError), (True, TypeError)])
def test_cap_validation(toy_episodes, cap, exc):
    with pytest.raises(exc, match="cap"):
        retry_harm_bounds(toy_episodes, TOY_THETA, cap=cap)


def test_empty_input():
    with pytest.raises(ValueError, match="empty"):
        retry_harm_bounds([], 0.5)
