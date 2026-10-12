import pytest

from agent_risk import Episode, StopValuation, replay_stop

from toydata import TOY_COST, TOY_SCORES, TOY_THETA


def test_toy_stops_at_step_2_and_keeps_only_prior_harm(toy_episodes):
    r = replay_stop(toy_episodes, TOY_THETA)
    assert r.branches == ("stop", "stop", "agree")
    assert r.stop_steps == (2, 2, None)
    # Harm at step 1 is before the stop and counts.
    assert r.episodes[0].harm is True
    # Harm at step 3 is after the stop and is removed.
    assert r.episodes[1].harm is False
    assert r.episodes[2].harm is False


def test_toy_aggregates(toy_episodes):
    r = replay_stop(toy_episodes, TOY_THETA)
    assert r.n == 3 and r.n_stop == 2 and r.n_agree == 1
    assert r.agreement_mass == pytest.approx(1 / 3)
    assert r.stop_mass == pytest.approx(2 / 3)
    assert r.n_harm == 1 and r.n_harm_known == 1 and r.n_unknown == 0
    assert r.p_harm_stated == pytest.approx(1 / 3)
    assert r.p_harm == pytest.approx(1 / 3)
    # Stopped episodes fail under the default valuation; the agreeing one keeps failed=False.
    assert [e.failed for e in r.episodes] == [True, True, False]
    assert r.p_failure == pytest.approx(2 / 3)
    # Cost through the stop: 10 + 20 + 30 = 60 for each stopped episode; 30 for the agreeing one.
    assert [e.cost for e in r.episodes] == [60.0, 60.0, 30.0]
    assert r.mean_cost == pytest.approx(50.0)
    assert r.n_stop_clean == 1
    assert r.stop_clean_share == pytest.approx(1 / 3)


def test_cost_at_stop_rules(toy_episodes):
    before = replay_stop(toy_episodes, TOY_THETA, cost_at_stop="before_stop")
    assert [e.cost for e in before.episodes] == [30.0, 30.0, 30.0]
    custom = replay_stop(toy_episodes, TOY_THETA, cost_at_stop=lambda ep, sigma: 7.0 * (sigma + 1))
    assert [e.cost for e in custom.episodes] == [21.0, 21.0, 30.0]
    with pytest.raises(ValueError, match="cost_at_stop"):
        replay_stop(toy_episodes, TOY_THETA, cost_at_stop="all")
    with pytest.raises(ValueError, match="non-negative"):
        replay_stop(toy_episodes, TOY_THETA, cost_at_stop=lambda ep, sigma: -1.0)
    with pytest.raises(TypeError, match="real number"):
        replay_stop(toy_episodes, TOY_THETA, cost_at_stop=lambda ep, sigma: "1")


def test_declared_valuation(toy_episodes):
    r = replay_stop(toy_episodes, TOY_THETA, valuation=StopValuation(failed=False, post_stop_harm=0.25))
    assert r.p_failure == 0.0
    assert r.p_harm_stated == pytest.approx(1 / 3)
    assert r.p_harm == pytest.approx(1 / 3 + 0.25 * (1 / 3))
    full = replay_stop(toy_episodes, TOY_THETA, valuation=StopValuation(post_stop_harm=1.0))
    assert full.p_harm == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "kwargs, exc",
    [
        (dict(failed=1), TypeError),
        (dict(post_stop_harm=1.5), ValueError),
        (dict(post_stop_harm=-0.1), ValueError),
        (dict(post_stop_harm="0.1"), TypeError),
    ],
)
def test_stop_valuation_validation(kwargs, exc):
    with pytest.raises(exc):
        StopValuation(**kwargs)


def test_unknown_harm_policies():
    eps = [
        Episode([0.1, 0.99], [None, False], [1, 1], failed=False),  # unknown before the stop
        Episode([0.1, 0.99], [False, None], [1, 1], failed=False),  # unknown at the stop: removed
        Episode([0.1, 0.2], [False, None], [1, 1], failed=False),  # agrees with unknown harm
        Episode([0.1, 0.2], [True, None], [1, 1], failed=False),  # agrees, known harm
    ]
    r = replay_stop(eps, 0.5)
    assert [e.harm for e in r.episodes] == [None, False, None, True]
    assert r.n_unknown == 2 and r.n_harm_known == 1
    assert r.n_harm == 3 and r.p_harm == pytest.approx(3 / 4)
    assert r.n_stop_clean == 1
    lenient = replay_stop(eps, 0.5, unknown="no_harm")
    assert lenient.n_harm == 1 and lenient.p_harm == pytest.approx(1 / 4)
    assert lenient.n_stop_clean == 2
    with pytest.raises(ValueError, match="unknown"):
        replay_stop(eps, 0.5, unknown="skip")


def test_all_unknown():
    eps = [
        Episode([0.1, 0.2], [None, None], [1, 1], failed=False),
        Episode([0.99, 0.2], [None, None], [1, 1], failed=False),
    ]
    r = replay_stop(eps, 0.5)
    # The stop at step 0 resolves the second episode: nothing executed, no harm.
    assert [e.harm for e in r.episodes] == [None, False]
    assert r.n_unknown == 1
    assert r.p_harm == pytest.approx(0.5)
    assert replay_stop(eps, 0.5, unknown="no_harm").p_harm == 0.0


def test_theta_above_every_score_reproduces_the_log(toy_episodes):
    r = replay_stop(toy_episodes, 0.99)
    assert r.branches == ("agree",) * 3
    assert r.stop_mass == 0.0 and r.agreement_mass == 1.0
    assert [e.harm for e in r.episodes] == [True, True, False]
    assert r.p_harm == pytest.approx(2 / 3)
    assert r.p_failure == 0.0
    assert r.mean_cost == pytest.approx((100 + 100 + 30) / 3)


def test_theta_below_every_score_stops_everything_at_step_0(toy_episodes):
    r = replay_stop(toy_episodes, 0.05)
    assert r.branches == ("stop",) * 3
    assert r.stop_steps == (0, 0, 0)
    assert r.p_harm == 0.0
    assert r.p_failure == 1.0
    assert [e.cost for e in r.episodes] == [10.0, 10.0, 10.0]
    assert r.stop_clean_share == 1.0


def test_score_equal_to_theta_does_not_stop():
    e = Episode([0.5, 0.5], [False, True], [1, 1], failed=False)
    r = replay_stop([e], 0.5)
    assert r.branches == ("agree",)
    assert r.p_harm == 1.0


def test_zero_step_episode_agrees():
    r = replay_stop([Episode([], [], [], failed=True)], 0.5)
    assert r.branches == ("agree",)
    assert r.p_harm == 0.0 and r.p_failure == 1.0 and r.mean_cost == 0.0


def test_input_validation(toy_episodes):
    with pytest.raises(ValueError, match="empty"):
        replay_stop([], TOY_THETA)
    with pytest.raises(TypeError, match="single Episode"):
        replay_stop(toy_episodes[0], TOY_THETA)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"episodes\[1\]"):
        replay_stop([toy_episodes[0], {"scores": TOY_SCORES}], TOY_THETA)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="NaN"):
        replay_stop(toy_episodes, float("nan"))
    with pytest.raises(TypeError, match="theta"):
        replay_stop(toy_episodes, True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="valuation"):
        replay_stop(toy_episodes, TOY_THETA, valuation=0.0)  # type: ignore[arg-type]


def test_replay_does_not_mutate_inputs(toy_episodes):
    before = list(toy_episodes)
    replay_stop(toy_episodes, TOY_THETA)
    assert toy_episodes == before
    assert toy_episodes[0].cost == tuple(TOY_COST)
