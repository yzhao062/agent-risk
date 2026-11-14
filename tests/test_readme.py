"""Run every Python example in README.md and check the printed output."""

import contextlib
import io
import re
from pathlib import Path

import pytest

from agent_risk import (
    certify_replay,
    certify_violations,
    clopper_pearson_upper,
    replay_stop,
    sample_size,
)

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_text():
    if not README.exists():
        pytest.skip("README.md is not available next to the tests")
    return README.read_text(encoding="utf-8")


def _python_blocks():
    return re.findall(r"```python\n(.*?)```", _readme_text(), flags=re.DOTALL)


def _block_with(marker):
    blocks = [b for b in _python_blocks() if marker in b]
    assert len(blocks) == 1, f"expected one README block containing {marker!r}"
    return blocks[0]


def _exec(block, name):
    """Run a block; return its printed lines and its namespace."""
    out = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(out):
        exec(compile(block, name, "exec"), namespace)
    return out.getvalue().splitlines(), namespace


def _run(block, name):
    return _exec(block, name)[0]


def test_readme_has_examples():
    assert len(_python_blocks()) >= 3


def test_workflow_example_output_matches_comments():
    block = _block_with("certify_violations")
    printed, ns = _exec(block, "README workflow example")
    assert printed == ["59", "not_certified 0.142", "passed 0.045"]
    for line in printed:
        assert line in block  # the inline comment states the printed value
    assert len(block.strip().splitlines()) <= 20
    # Facts that only the comments and the prose after the block state, built from computed values.
    today = ns["today"]
    replay = replay_stop(ns["log"], theta=0.9)
    assert (today.n_harmed, today.n, replay.n_stop) == (20, 200, 16)
    assert f"{today.n_harmed} of {today.n} violate" in block
    assert f"stops the {replay.n_stop} caught ({replay.p_failure:.0%})" in block
    assert f"this gate stops {replay.p_failure:.0%} of the episodes" in _readme_text()
    assert f"{len(ns['log'])} episodes" in _readme_text()
    # The prose after the block: a gate that stops every episode would pass.
    stop_all = certify_replay(replay_stop(ns["log"], theta=0.0), limit=0.05)
    assert stop_all.status == "passed"


def test_replay_report_example_output_matches_comments():
    block = _block_with("replay.branches")
    printed = _run(block, "README replay report example")
    for line in printed:
        assert line in block  # the inline comment states the printed value
    assert printed == [
        "('stop', 'stop', 'agree') (2, 2, None)",
        "0.333",
        "0.667",
        "50.0",
        "0.667 0.333",
        "1 3 0.865",
        "[0.333, 0.667]",
    ]
    assert len(block.strip().splitlines()) <= 22


def test_motivating_numbers_match_the_library():
    # The opening: 0 violations in 50 episodes cannot support a 5% limit at 95% confidence,
    # 59 clean episodes can, a 1% limit needs 299, and a wrong pass has probability at most 5%.
    clean, limit, strict, confidence = 50, 0.05, 0.01, 0.95
    upper = clopper_pearson_upper(0, clean, confidence=confidence)
    assert round(upper, 3) == 0.058 and upper > limit
    enough = sample_size(limit, confidence=confidence)
    assert enough == 59 and clopper_pearson_upper(0, enough, confidence=confidence) <= limit
    assert sample_size(strict, confidence=confidence) == 299
    alpha = certify_violations([False] * enough, limit=limit, confidence=confidence).alpha
    text = _readme_text()
    for phrase in (
        f"0 violations in {clean} episodes",
        f"At {confidence:.0%} confidence, {clean} clean independent episodes",
        f"below about {upper:.0%}",
        f"cannot support a {limit:.0%} limit",
        f"{enough} clean episodes can",
        f"a {strict:.0%} limit needs {sample_size(strict, confidence=confidence)}",
        f"a {confidence:.0%} certificate passes at most {alpha:.0%} of the time",
    ):
        assert phrase in text, phrase


def test_examples_run():
    for block in _python_blocks():
        lines = _run(block, "README example")
        if "is_equivalent" in block:
            assert lines == ["-0.01 -0.043 0.023", "True"]
            for line in lines:
                assert line in block  # the inline comment states the printed value
