"""Offline tests for agent.executor's code extraction.

The hardened extractor must (a) return exactly what the frozen baseline returned for a normally
formatted reply and (b) additionally recover the script from the malformed/odd shapes that used to
fall back to the raw reply and fail with a SyntaxError on line 1.
"""

from __future__ import annotations

import ast
import contextlib
import random
import re
import signal
import time

import pytest

from agent.executor import extract_code, parse_reply

# The frozen Development baseline's extractor, verbatim, as the reference for "unchanged behaviour".
_BASELINE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _baseline_extract(reply: str) -> str:
    match = _BASELINE_FENCE_RE.search(reply)
    return match.group(1) if match else reply


CODE = "import json\nvalue = 1 + 1\nprint(json.dumps({'value': value}))\n"

# Replies the baseline already handled: the new extractor must return the identical string.
_WELL_FORMED = {
    "clean block": f"```python\n{CODE}```\n",
    "clean block, no trailing newline": f"```python\n{CODE}```",
    "prose before and after": f"Here is the solution:\n```python\n{CODE}```\nHope this helps.",
    "unlabelled fence": f"```\n{CODE}```\n",
    "CRLF line endings": f"```python\r\n{CODE.replace(chr(10), chr(13) + chr(10))}```\r\n",
    "trailing spaces after the tag": f"```python  \n{CODE}```\n",
    "closing fence glued to the last line": "```python\nimport json\nprint(1)```",
    "no fence at all (raw code)": CODE,
    "empty reply": "",
}


@pytest.mark.parametrize("name", sorted(_WELL_FORMED))
def test_well_formed_replies_extract_exactly_as_the_baseline_did(name):
    reply = _WELL_FORMED[name]
    assert extract_code(reply) == _baseline_extract(reply)


def test_leading_blank_lines_after_the_fence_do_not_change_the_program():
    reply = f"```python\n\n\n{CODE}```\n"
    assert ast.dump(ast.parse(extract_code(reply))) == ast.dump(ast.parse(_baseline_extract(reply)))


@pytest.mark.parametrize("tag", ["Python", "PYTHON", "py", "python3", "Python3", "py3", "ipython"])
def test_language_tag_variants_are_recognised(tag):
    """These made the baseline return the whole reply (fence lines included): a guaranteed
    SyntaxError on line 1 that burnt the one repair."""
    reply = f"```{tag}\n{CODE}```\n"
    assert extract_code(reply) == CODE
    assert _baseline_extract(reply) != CODE  # the baseline really did fail on these


def test_a_reply_cut_off_inside_the_block_yields_the_code_so_far_and_says_it_is_incomplete():
    reply = "```python\nimport json\ndef f(x):\n    return x +"
    parsed = parse_reply(reply)
    assert parsed.code == "import json\ndef f(x):\n    return x +"
    assert parsed.complete is False and parsed.fenced is True


def test_a_closed_block_is_complete():
    parsed = parse_reply(f"```python\n{CODE}```\n")
    assert parsed.complete is True and parsed.fenced is True and parsed.n_blocks == 1


def test_a_reply_with_no_fence_is_returned_whole_and_not_marked_fenced():
    parsed = parse_reply(CODE)
    assert parsed.code == CODE and parsed.fenced is False and parsed.complete is True


def test_a_following_shell_block_is_ignored():
    reply = f"```python\n{CODE}```\nRun it with:\n```bash\npython solution.py\n```\n"
    assert extract_code(reply) == CODE


def test_json_only_replies_fall_back_to_the_raw_reply_like_the_baseline():
    reply = '```json\n{"a": 1}\n```\n'
    assert extract_code(reply) == reply


def test_the_longest_python_block_wins_over_a_sketch_or_a_usage_snippet():
    sketch_first = f"Plan:\n```python\n# pseudo\n```\nFull code:\n```python\n{CODE}```\n"
    usage_last = f"```python\n{CODE}```\nExample call:\n```python\nmain()\n```\n"
    assert extract_code(sketch_first) == CODE
    assert extract_code(usage_last) == CODE


def test_an_equal_length_tie_goes_to_the_later_block():
    reply = "```python\nx = 1\n```\n```python\ny = 2\n```\n"
    assert extract_code(reply) == "y = 2\n"


def test_reasoning_text_is_dropped_before_choosing_a_block():
    """A draft fence inside the reasoning must never be taken as the answer."""
    long_draft = "```python\n" + "draft = 0\n" * 50 + "```\n"
    reply = f"<think>Let me try:\n{long_draft}Wrong, rewrite.</think>\n```python\n{CODE}```\n"
    assert extract_code(reply) == CODE


def test_an_unopened_think_close_tag_drops_everything_before_it():
    """With the chat template that opens the tag in the prompt, the reply holds only `</think>`."""
    long_draft = "```python\n" + "draft = 0\n" * 50 + "```\n"
    reply = f"Reasoning...\n{long_draft}</think>\n```python\n{CODE}```\n"
    assert extract_code(reply) == CODE


def test_reasoning_that_never_finished_yields_no_code():
    reply = "<think>Thinking about the Black-Scholes formula\n```python\nd1 = ..."
    assert extract_code(reply) == ""


def test_a_longer_outer_fence_can_contain_a_shorter_inner_one():
    reply = "````python\ntext = '''\n```\nnot a closer\n```\n'''\nprint(text)\n````\n"
    code = extract_code(reply)
    assert ast.parse(code) is not None
    assert "not a closer" in code and code.rstrip().endswith("print(text)")


def test_tilde_fences_work():
    assert extract_code(f"~~~python\n{CODE}~~~\n") == CODE


def test_a_fence_opened_at_the_end_of_a_prose_line_is_found():
    assert extract_code(f"Here is the code: ```python\n{CODE}```\n") == CODE


def test_extraction_never_alters_the_code_it_returns():
    """The extractor only chooses a block; it must not strip, re-indent or repair the code."""
    code = "def f():\n    return 1\n\n\nif __name__ == '__main__':\n    f()\n"
    assert extract_code(f"```python\n{code}```\n") == code


# ---------------------------------------------------------------------------------------------
# Regression: fence detection must be linear-time
#
# The first version used unanchored regex `.search` calls for "a fence at the end of a line". On a long
# run of backticks (a repetition loop at the 4,000-token cap) that is quadratic: 16,000 characters took
# ~6 s, 32,000 ~24 s and 64,000 ~97 s per parse, against ~0.001 s for the baseline's regex. Every input
# below is 32,000 characters; the fixed scan handles each in milliseconds, the old one took 10-25 s.
# ---------------------------------------------------------------------------------------------

_TICKS = "`" * 3
_N = 32_000
_TIME_LIMIT_SEC = 2.0  # ~200x the measured time of the fixed code; the quadratic version needs >10 s here

_PATHOLOGICAL = {
    "one line of backticks, then junk": "`" * _N + "z!",
    "one line of tildes, then junk": "~" * _N + "z!",
    "open fence, then a long backtick line with junk": f"{_TICKS}python\nx = 1\n" + "`" * _N + "z!\n",
    "open fence, then a long tilde line with junk": f"{_TICKS}python\nx = 1\n" + "~" * _N + "z!\n",
    "open fence, then a long backtick line that is a valid closer": f"{_TICKS}python\nx = 1\n" + "`" * _N + "\n",
    "long backtick run followed by a long word": "`" * (_N // 2) + "a" * (_N // 2),
    "long backtick run followed by a long tag and junk": "`" * (_N // 2) + "a" * (_N // 2) + "!",
    "long run of backticks then long whitespace then junk": "`" * (_N // 2) + " " * (_N // 2) + "z",
    "word characters then one backtick": "a" * _N + "`",
    "alternating backticks and tildes": "`~" * (_N // 2),
    "many short fence-like runs on one line": (_TICKS + " a ") * (_N // 7),
    "many one-fence lines": (_TICKS + "\n") * (_N // 4),
    "many fence-open lines": (_TICKS + "python\n") * (_N // 10),
    "very long single line of code, no fence": "x = 1; " * (_N * 5),
    "long open-fence body of backtick-ended lines": f"{_TICKS}python\n" + ("print('x')`\n" * (_N // 12)),
}


@contextlib.contextmanager
def _deadline(seconds: float):
    """Fail (rather than hang) if the block is still running after `seconds`. SIGALRM interrupts even a
    runaway regex, so a regression fails in seconds instead of stalling the suite for minutes. Where
    SIGALRM is unavailable the elapsed-time assertions in the tests still apply."""
    if not hasattr(signal, "setitimer"):
        yield
        return

    def _expired(signum, frame):
        raise AssertionError(f"still running after {seconds}s: fence detection is no longer linear-time")

    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.parametrize("name", sorted(_PATHOLOGICAL))
def test_pathological_fence_like_replies_are_parsed_in_linear_time(name):
    reply = _PATHOLOGICAL[name]
    started = time.perf_counter()
    with _deadline(_TIME_LIMIT_SEC + 1):
        parsed = parse_reply(reply)
    elapsed = time.perf_counter() - started
    assert isinstance(parsed.code, str) and isinstance(parsed.complete, bool)
    assert elapsed < _TIME_LIMIT_SEC, f"{name}: {elapsed:.1f}s"


def test_time_does_not_grow_quadratically_with_the_length_of_a_backtick_run():
    """Quadrupling a pathological input must not make it 16x slower (it takes ~0.5 s -> ~8 s if it does)."""

    def seconds_for(n: int) -> float:
        started = time.perf_counter()
        with _deadline(10):
            parse_reply("`" * n + "z!")
        return time.perf_counter() - started

    small, large = seconds_for(10_000), seconds_for(40_000)
    assert large < max(0.5, small * 8)  # linear: ~4x (both are milliseconds); quadratic: ~16x and tens of seconds


# The regexes the linear scan replaced, kept here verbatim as the reference for "unchanged behaviour".
_OLD_PROSE_FENCE_RE = re.compile(r"(`{3,}|~{3,})([\w+.#-]*)[ \t]*$")
_OLD_CODE_FENCE_RE = re.compile(r"(`{3,}|~{3,})[ \t]*$")


def _random_fence_like_line(rng: random.Random) -> str:
    atoms = ["`", "``", _TICKS, "````", "~", "~~~", "~~~~", "python", "py", "Python", "x", "3", "_", "+", ".", "#", "-", " ", "\t", "é", "字", "!", "(", ")", "="]
    return "".join(rng.choice(atoms) for _ in range(rng.randint(0, 9)))


def test_the_linear_scan_finds_exactly_what_the_regexes_it_replaced_found():
    from agent.executor import _trailing_fence

    rng = random.Random(20261012)
    checked_hits = 0
    for _ in range(40_000):
        line = _random_fence_like_line(rng)

        old = _OLD_PROSE_FENCE_RE.search(line)
        new = _trailing_fence(line, with_tag=True)
        assert (None if old is None else (old.group(1), old.group(2), old.start(1))) == new, repr(line)

        old = _OLD_CODE_FENCE_RE.search(line)
        new = _trailing_fence(line, with_tag=False)
        assert (None if old is None else (old.group(1), "", old.start(1))) == new, repr(line)
        checked_hits += old is not None
    assert checked_hits > 1_000  # the generator really exercises the matching branch, not just the misses


def test_extraction_results_are_unchanged_by_the_performance_fix_on_odd_but_short_replies():
    """The same parse_reply contract on the odd shapes the earlier tests cover, spelled out once more."""
    assert extract_code(f"Here is the code: {_TICKS}python\nx = 1\n{_TICKS}\n") == "x = 1\n"
    assert extract_code(f"{_TICKS}python\nprint(1){_TICKS}") == "print(1)"
    assert extract_code(f"{_TICKS}python\nx = 1\n{_TICKS} \ntrailing") == "x = 1\n"
    unclosed = parse_reply(f"{_TICKS}Python\nx = 1\n")
    assert (unclosed.code, unclosed.complete, unclosed.fenced) == ("x = 1\n", False, True)
