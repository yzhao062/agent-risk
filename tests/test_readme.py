"""Run every Python example in README.md and check the quickstart output."""

import contextlib
import io
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def _python_blocks():
    if not README.exists():
        pytest.skip("README.md is not available next to the tests")
    text = README.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_readme_has_examples():
    assert len(_python_blocks()) >= 3


def test_quickstart_output_matches_comments():
    quickstart = _python_blocks()[0]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(quickstart, "README quickstart", "exec"), {})
    lines = out.getvalue().splitlines()
    assert lines == [
        "('stop', 'stop', 'agree') (2, 2, None)",
        "0.333",
        "0.667",
        "50.0",
        "0.667 0.333",
        "1 3 0.865",
        "[0.333, 0.667]",
    ]
    assert len(quickstart.strip().splitlines()) <= 22


def test_other_examples_run():
    for block in _python_blocks()[1:]:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(block, "README example", "exec"), {})
        if "is_equivalent" in block:
            assert out.getvalue().splitlines() == ["-0.01 -0.043 0.023", "True"]
