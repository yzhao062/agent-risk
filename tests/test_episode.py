import dataclasses
import json

import numpy as np
import pytest

from agent_risk import Episode, any_harm, load_episodes, load_jsonl


def test_episode_coerces_inputs():
    e = Episode(np.array([0.1, 0.9]), [0, 1], (1, 2), failed=1)
    assert e.scores == (0.1, 0.9)
    assert e.harm == (False, True)
    assert e.cost == (1.0, 2.0)
    assert e.failed is True
    assert e.n_steps == 2
    assert e.total_cost == 3.0


def test_episode_accepts_numpy_bools_and_unknown():
    e = Episode([0.5, 0.5, 0.5], [np.bool_(True), None, np.bool_(False)], [0, 0, 0], np.bool_(False))
    assert e.harm == (True, None, False)
    assert e.failed is False


def test_episode_zero_steps():
    e = Episode([], [], [], failed=False)
    assert e.n_steps == 0
    assert e.harm_flag is False
    assert e.first_exceedance(0.5) is None
    assert e.total_cost == 0.0


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        (dict(scores=[0.1], harm=[False, True], cost=[1], failed=False), ValueError, "same length"),
        (dict(scores=[1.2], harm=[False], cost=[1], failed=False), ValueError, r"\[0, 1\]"),
        (dict(scores=[-0.1], harm=[False], cost=[1], failed=False), ValueError, r"\[0, 1\]"),
        (dict(scores=[float("nan")], harm=[False], cost=[1], failed=False), ValueError, "finite"),
        (dict(scores=[0.1], harm=[False], cost=[-1], failed=False), ValueError, "non-negative"),
        (dict(scores=[0.1], harm=[False], cost=[float("inf")], failed=False), ValueError, "finite"),
        (dict(scores=[0.1], harm=[2], cost=[1], failed=False), ValueError, "harm"),
        (dict(scores=[0.1], harm=["yes"], cost=[1], failed=False), ValueError, "harm"),
        (dict(scores=[0.1], harm=[False], cost=[1], failed="no"), ValueError, "failed"),
        (dict(scores=[0.1], harm=[False], cost=[1], failed=None), ValueError, "failed"),
        (dict(scores="0.1", harm=[False], cost=[1], failed=False), TypeError, "scores"),
        (dict(scores=[True], harm=[False], cost=[1], failed=False), TypeError, "scores"),
        (dict(scores=[0.1], harm=None, cost=[1], failed=False), TypeError, "harm"),
    ],
)
def test_episode_validation(kwargs, exc, match):
    with pytest.raises(exc, match=match):
        Episode(**kwargs)


def test_episode_is_frozen():
    e = Episode([0.1], [False], [1], failed=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.failed = True  # type: ignore[misc]


def test_any_harm_three_valued():
    assert any_harm([]) is False
    assert any_harm([False, False]) is False
    assert any_harm([False, None]) is None
    assert any_harm([None, True]) is True
    assert any_harm([None, None]) is None


def test_harm_before_and_flag():
    e = Episode([0.1, 0.2, 0.3], [None, True, False], [1, 1, 1], failed=False)
    assert e.harm_before(0) is False
    assert e.harm_before(1) is None
    assert e.harm_before(2) is True
    assert e.harm_before(3) is True
    assert e.harm_flag is True
    with pytest.raises(ValueError):
        e.harm_before(4)
    with pytest.raises(TypeError):
        e.harm_before(1.0)  # type: ignore[arg-type]


def test_first_exceedance_is_strict():
    e = Episode([0.1, 0.94, 0.95], [False] * 3, [1] * 3, failed=False)
    assert e.first_exceedance(0.94) == 2
    assert e.first_exceedance(0.95) is None
    assert e.first_exceedance(-1.0) == 0
    with pytest.raises(ValueError, match="NaN"):
        e.first_exceedance(float("nan"))
    with pytest.raises(TypeError):
        e.first_exceedance("0.5")  # type: ignore[arg-type]


def test_load_episodes_columnar_and_steps():
    records = [
        {"scores": [0.1, 0.97], "harm": [False, None], "cost": [5, 6], "failed": True},
        {
            "steps": [
                {"score": 0.2, "harm": 0, "cost": 1},
                {"score": 0.3, "harm": 1, "cost": 2},
            ],
            "failed": False,
        },
    ]
    eps = load_episodes(records)
    assert eps[0] == Episode([0.1, 0.97], [False, None], [5, 6], True)
    assert eps[1] == Episode([0.2, 0.3], [False, True], [1, 2], False)


@pytest.mark.parametrize(
    "record, match",
    [
        ({"scores": [0.1], "harm": [False], "cost": [1]}, "failed"),
        ({"scores": [0.1], "harm": [False], "failed": False}, "cost"),
        ({"steps": [{"score": 0.1, "harm": False}], "failed": False}, "cost"),
        ({"steps": [], "scores": [], "failed": False}, "either"),
        ({"scores": [0.1], "harm": [False], "cost": [1, 2], "failed": False}, "same length"),
    ],
)
def test_load_episodes_errors_name_the_record(record, match):
    with pytest.raises(ValueError, match=r"record 1: .*" + match):
        load_episodes([{"scores": [], "harm": [], "cost": [], "failed": False}, record])


def test_load_episodes_rejects_single_mapping():
    with pytest.raises(TypeError):
        load_episodes({"scores": [], "harm": [], "cost": [], "failed": False})  # type: ignore[arg-type]


def test_load_jsonl_from_path_and_lines(tmp_path):
    lines = [
        json.dumps({"scores": [0.1], "harm": [None], "cost": [3], "failed": False}),
        "",
        json.dumps({"scores": [0.99], "harm": [True], "cost": [4], "failed": True}),
    ]
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from_path = load_jsonl(path)
    from_str_path = load_jsonl(str(path))
    from_lines = load_jsonl(lines)
    assert from_path == from_str_path == from_lines
    assert len(from_path) == 2
    assert from_path[0].harm == (None,)


def test_load_jsonl_reports_line_numbers():
    with pytest.raises(ValueError, match="line 2: invalid JSON"):
        load_jsonl(['{"scores": [], "harm": [], "cost": [], "failed": false}', "{not json"])
    with pytest.raises(ValueError, match="line 1: .*failed"):
        load_jsonl(['{"scores": [], "harm": [], "cost": []}'])
