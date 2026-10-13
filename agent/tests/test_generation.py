"""Offline tests for agent.generation: bounded continuation of replies the House route cut off.

A scripted fake stands in for ModelClient (no network, no openai package). `ModelReply` is the real
class, so `finish_reason` is carried exactly as the real client carries it.
"""

from __future__ import annotations

import ast

import pytest

from agent import generation
from agent.executor import extract_code
from agent.generation import MAX_CONTINUATIONS, generate_script
from agent.model_client import ModelReply

SCRIPT_LINES = [
    "import json",
    "import math",
    "",
    "def compute(x):",
    "    # a long enough line to make a repeated seam meaningful",
    "    return math.sqrt(x) * 2.0",
    "",
    "values = [compute(v) for v in range(1, 6)]",
    "result = {'total': sum(values), 'n': len(values)}",
    "print(json.dumps(result))",
]
SCRIPT = "\n".join(SCRIPT_LINES) + "\n"
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def _reply(text: str, reason: str | None = "stop") -> ModelReply:
    return ModelReply(text, finish_reason=reason)


def _fenced(code: str) -> str:
    return f"```python\n{code}```\n"


def _cut(lines: int, partial: str = "") -> ModelReply:
    """The first `lines` script lines, then half a line, cut by the length limit (fence left open)."""
    return _reply("```python\n" + "\n".join(SCRIPT_LINES[:lines]) + "\n" + partial, "length")


class _Client:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def generate_code(self, messages):
        self.calls.append([dict(m) for m in messages])
        step = self._replies.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def test_a_complete_reply_is_used_as_is_with_one_request():
    client = _Client([_reply(_fenced(SCRIPT))])
    result = generate_script(client, MESSAGES)
    assert result.code == SCRIPT == extract_code(_fenced(SCRIPT))
    assert len(client.calls) == 1
    assert result.continuations == 0 and not result.still_truncated and result.notes == []
    assert result.trace_fields() == {"finish_reason": "stop"}


def test_a_reply_without_metadata_is_never_continued_even_if_it_looks_cut_off():
    """Fakes and older clients return plain str: no finish_reason, so truncation is not confirmed."""
    client = _Client(["```python\nimport json\nx = (1 +"])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 1 and result.continuations == 0


@pytest.mark.parametrize("reason", ["stop", None, "tool_calls", "content_filter"])
def test_an_open_fence_without_a_length_finish_is_not_continued(reason):
    client = _Client([_reply("```python\nimport json\nprint(1)\n", reason)])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 1
    assert result.code == "import json\nprint(1)\n" and not result.still_truncated


def test_a_length_finish_with_a_closed_block_is_not_continued():
    """The limit hit trailing prose after the code was complete: nothing to continue."""
    client = _Client([_reply(_fenced(SCRIPT) + "Explanation: this script", "length")])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 1 and result.code == SCRIPT


def test_a_truncated_reply_is_continued_and_the_pieces_are_joined():
    client = _Client([_cut(7, partial="values = [compute(v) fo"), _reply(_fenced("\n".join(SCRIPT_LINES[7:]) + "\n"))])
    result = generate_script(client, MESSAGES)

    assert result.code == SCRIPT
    ast.parse(result.code)
    assert len(client.calls) == 2 and result.continuations == 1
    assert result.still_truncated is False and result.finish_reason == "stop"

    sent = client.calls[1]
    assert sent[:2] == MESSAGES  # the original conversation, untouched
    assert sent[2]["role"] == "assistant" and sent[3]["role"] == "user"
    assert sent[2]["content"].endswith("\n")
    assert "values = [compute(v) fo" not in sent[2]["content"]  # the cut-off line is not sent back
    assert "cut off by the output-length limit" in sent[3]["content"]
    assert MESSAGES == [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def test_lines_repeated_at_the_seam_are_dropped():
    repeated = "\n".join(SCRIPT_LINES[4:]) + "\n"  # starts 3 lines before the cut
    client = _Client([_cut(7), _reply(_fenced(repeated))])
    assert generate_script(client, MESSAGES).code == SCRIPT


def test_a_continuation_that_restarts_the_script_replaces_it_instead_of_appending():
    client = _Client([_cut(6), _reply(_fenced(SCRIPT))])
    result = generate_script(client, MESSAGES)
    assert result.code == SCRIPT
    ast.parse(result.code)


def test_two_continuations_assemble_three_pieces():
    client = _Client(
        [
            _cut(4),
            _reply("```python\n" + "\n".join(SCRIPT_LINES[4:8]) + "\nresult = {'tot", "length"),
            _reply(_fenced("\n".join(SCRIPT_LINES[8:]) + "\n")),
        ]
    )
    result = generate_script(client, MESSAGES)
    assert result.code == SCRIPT and result.continuations == 2 and len(client.calls) == 3
    # The third request carries the whole conversation so far, in order.
    assert [m["role"] for m in client.calls[2]] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_continuations_are_capped_and_the_script_is_reported_as_still_cut_off():
    always_cut = [_cut(3 + i) for i in range(10)]
    client = _Client(always_cut)
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 1 + MAX_CONTINUATIONS
    assert result.still_truncated is True and result.continuations == MAX_CONTINUATIONS
    assert any("cap" in n for n in result.notes)


def test_a_failed_continuation_keeps_the_script_so_far_and_does_not_raise():
    client = _Client([_cut(7, partial="values = [comp"), ConnectionError("endpoint down")])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 2  # exactly one attempt: no retry loop of our own
    assert result.still_truncated is True and result.continuations == 0
    assert any("ConnectionError" in n for n in result.notes)
    assert result.code.startswith("import json")


def test_the_first_requests_failure_still_propagates():
    client = _Client([ConnectionError("endpoint down")])
    with pytest.raises(ConnectionError):
        generate_script(client, MESSAGES)


def test_no_continuation_is_started_without_enough_unit_budget():
    client = _Client([_cut(7), _reply(_fenced("unused"))])
    clock = iter([0.0, 0.0])
    result = generate_script(client, MESSAGES, deadline=100.0, now=lambda: next(clock))
    assert len(client.calls) == 1
    assert result.still_truncated is True and any("budget" in n for n in result.notes)
    # unchanged baseline behaviour for an unrecoverable truncation: the reply as received
    assert result.code == extract_code(_cut(7))


def test_a_continuation_is_allowed_when_the_budget_is_ample():
    client = _Client([_cut(7), _reply(_fenced("\n".join(SCRIPT_LINES[7:]) + "\n"))])
    result = generate_script(client, MESSAGES, deadline=10_000.0, now=lambda: 0.0)
    assert result.code == SCRIPT


def test_raw_code_without_a_fence_that_hits_the_limit_is_continued():
    head = "import json\nimport math\n\ndef compute(x):\n    return math.sqrt(x)\n\nvalues = ["
    client = _Client([_reply(head, "length"), _reply(_fenced("values = [compute(v) for v in range(5)]\nprint(values)\n"))])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 2
    assert result.code.endswith("print(values)\n") and "def compute" in result.code


def test_a_cut_off_reply_that_is_not_code_is_not_continued():
    client = _Client([_reply("Sure! First I will explain the approach in detail and then", "length")])
    generate_script(client, MESSAGES)
    assert len(client.calls) == 1


def test_unfinished_reasoning_has_nothing_to_continue():
    client = _Client([_reply("<think>The Black-Scholes formula needs d1 and", "length")])
    result = generate_script(client, MESSAGES)
    assert len(client.calls) == 1 and result.code == ""


def test_a_short_repeated_line_at_the_seam_is_kept_because_it_may_be_real_code():
    accumulated = "def f():\n    x = 1\n    return x\n"
    piece = "    return x\n\ny = f()\n"
    assert generation._splice(accumulated, piece).count("return x") == 2


def test_a_long_repeated_line_at_the_seam_is_dropped():
    accumulated = "def f():\n    value = compute_something_long(1, 2, 3)\n"
    piece = "    value = compute_something_long(1, 2, 3)\n    return value\n"
    assert generation._splice(accumulated, piece).count("compute_something_long") == 1


def test_completion_tokens_are_summed_across_the_replies_for_the_trace():
    first = ModelReply("```python\n" + "\n".join(SCRIPT_LINES[:7]) + "\nvalues = [comp", "length", 4000)
    second = ModelReply("```python\n" + "\n".join(SCRIPT_LINES[7:]) + "\n```\n", "stop", 123)
    result = generate_script(_Client([first, second]), MESSAGES)
    assert result.completion_tokens == 4123
    assert result.trace_fields()["completion_tokens"] == 4123 and result.trace_fields()["continuations"] == 1


def test_a_degenerate_cut_off_reply_does_not_stall_the_pipeline():
    """A repetition loop at the output cap (here a 32,000-character backtick line) is parsed several times
    per generation (initial reply, continuation check, fold). With a quadratic fence scan that cost ~40 s
    for one generation; it must stay far below any realistic unit budget."""
    import signal
    import time

    degenerate = ModelReply("```python\nx = 1\n" + "`" * 32_000 + "z", "length", 4000)
    client = _Client([degenerate] * 6)

    def _expired(signum, frame):
        raise AssertionError("generation still running after 10s: fence detection is no longer linear-time")

    started = time.perf_counter()
    previous = signal.signal(signal.SIGALRM, _expired) if hasattr(signal, "setitimer") else None
    if previous is not None:
        signal.setitimer(signal.ITIMER_REAL, 10)
    try:
        result = generate_script(client, MESSAGES)
    finally:
        if previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    elapsed = time.perf_counter() - started
    assert len(client.calls) == 1 + MAX_CONTINUATIONS and result.still_truncated is True
    assert elapsed < 5.0, f"{elapsed:.1f}s"
