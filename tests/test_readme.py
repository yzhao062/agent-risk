"""Run every Python example in README.md and check that it prints what its comments state."""

import contextlib
import io
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def _readme_text():
    if not README.exists():
        pytest.skip("README.md is not available next to the tests")
    return README.read_text(encoding="utf-8")


def _python_blocks():
    return re.findall(r"```python\n(.*?)```", _readme_text(), flags=re.DOTALL)


def _run(block):
    out = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(out):
        exec(compile(block, "README example", "exec"), namespace)
    return out.getvalue().splitlines(), namespace


def test_examples_print_what_their_comments_state():
    blocks = _python_blocks()
    assert blocks
    for block in blocks:
        printed, _ = _run(block)
        prints = [line for line in block.splitlines() if line.lstrip().startswith("print(")]
        assert len(printed) == len(prints), block
        for line, value in zip(prints, printed):
            comment = line.split("#", 1)[1].strip()
            assert comment == value or (comment.startswith(value) and comment[len(value)] in ": "), (line, value)


def test_quoted_certificate_is_the_real_output():
    block = next(b for b in _python_blocks() if "to_markdown()" in b)
    _, namespace = _run(block)
    after = _readme_text().split(block, 1)[1]
    quote = re.search(r"(?:^>.*\n)+", after, flags=re.MULTILINE).group(0)
    shown = "\n".join(re.sub(r"^> ?", "", line) for line in quote.splitlines())
    assert shown.strip() == namespace["report"].strip()
