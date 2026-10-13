"""Generate one script from the House model, finishing it when -- and only when -- it was cut off.

The House route caps every reply at 4,000 output tokens. A script longer than that arrives truncated
(`finish_reason == "length"`) and cannot run. A complete reply is used exactly as before. A reply the
route itself reports as cut off by the length limit, with its code block still open, is continued:

  * at most `MAX_CONTINUATIONS` extra requests per generation (so at most 3 requests for a generation
    and 6 for a whole unit, against the 25 allowed; `ModelClient` enforces the 25),
  * only while the unit's absolute deadline leaves at least `_MIN_CONTINUATION_WINDOW_SEC`,
  * each one a normal chat turn: the model's own text (minus a final line that was cut mid-way) as the
    assistant message, then a request to carry on from the next line.

The pieces are joined by `_splice`, which drops lines the model repeated at the seam and recognises a
continuation that restarted the script from the top. Nothing here judges the code: the assembled script
is run and self-checked like any other, and one that is still unusable (a continuation failed, or the
cap was reached) goes to the existing one-shot repair exactly as a truncated reply always did.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.executor import parse_reply

MAX_CONTINUATIONS = 2
_MIN_CONTINUATION_WINDOW_SEC = 300.0

_CONTINUE_PROMPT = (
    "Your previous reply was cut off by the output-length limit. Continue the Python script from exactly "
    "where it stopped: output ONLY the remaining code, beginning with the very next line (a last line that "
    "was cut mid-way has been removed from what you already wrote, so rewrite that line first). Do not "
    "repeat earlier code, do not restart the script and do not add explanations. Put the continuation in a "
    "single ```python fenced block. The final script is your earlier code followed by this continuation."
)

_OVERLAP_MAX_LINES = 12
_OVERLAP_MIN_CHARS = 12  # a repeated seam shorter than this (`return x`, `)`) may be legitimate code
_RESTART_LINES = 3


@dataclass
class GeneratedScript:
    code: str
    finish_reason: str | None = None  # of the last reply received
    completion_tokens: int | None = None  # summed over the replies, when the route reported them
    continuations: int = 0
    still_truncated: bool = False  # the script is known to end mid-way (cap reached / continuation failed)
    notes: list[str] = field(default_factory=list)

    def trace_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"finish_reason": self.finish_reason}
        if self.completion_tokens is not None:
            fields["completion_tokens"] = self.completion_tokens
        if self.continuations or self.still_truncated or self.notes:
            fields.update(continuations=self.continuations, still_truncated=self.still_truncated, notes=self.notes)
        return fields


def _cut_at_last_line(text: str) -> str:
    """Drop a final line that has no newline: it may have been cut mid-token."""
    end = text.rfind("\n")
    return text if end == -1 else text[: end + 1]


def _significant(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.strip()]


def _splice(accumulated: str, piece: str) -> str:
    """`accumulated` followed by `piece`, without the lines the model repeated at the seam; or `piece`
    alone when it restarted the script from the top instead of continuing. Neither text is otherwise
    altered (not even blank lines at the seam: one may sit inside a multi-line string)."""
    if not accumulated.strip():
        return piece
    acc_lines = accumulated.splitlines()
    piece_lines = piece.splitlines()
    kept = piece.splitlines(keepends=True)

    head_acc, head_piece = _significant(acc_lines)[:_RESTART_LINES], _significant(piece_lines)[:_RESTART_LINES]
    if len(head_acc) == _RESTART_LINES and head_acc == head_piece:
        return piece  # a full rewrite: the new text replaces the old rather than being appended to it

    for k in range(min(_OVERLAP_MAX_LINES, len(acc_lines), len(piece_lines)), 0, -1):
        tail = [line.rstrip() for line in acc_lines[-k:]]
        head = [line.rstrip() for line in piece_lines[:k]]
        if tail == head and sum(len(line.strip()) for line in tail) >= _OVERLAP_MIN_CHARS:
            kept = kept[k:]
            break
    base = accumulated if accumulated.endswith("\n") else accumulated + "\n"
    return base + "".join(kept)


_LOOKS_LIKE_PYTHON_RE = re.compile(r"(?m)^(?:import |from [\w.]+ import |def |class |#!)")


def _needs_continuation(reply: Any, text: str) -> bool:
    """Confirmed truncation: the route says the reply hit the length limit AND the code is not closed
    (an open fence, or raw code with no fence at all)."""
    if getattr(reply, "finish_reason", None) != "length" or not text.strip():
        return False
    parsed = parse_reply(text)
    if parsed.fenced:
        return not parsed.complete
    return bool(_LOOKS_LIKE_PYTHON_RE.search(parsed.code))


def _fold(accumulated: str | None, text: str, truncated: bool) -> str:
    """Add one reply to the script so far. A reply that is itself cut off contributes only its complete
    lines (the same text that is sent back to the model if another continuation follows)."""
    piece = parse_reply(_cut_at_last_line(text) if truncated else text).code
    return piece if accumulated is None else _splice(accumulated, piece)


def generate_script(
    client: Any,
    messages: list[dict[str, str]],
    *,
    deadline: float | None = None,
    now: Callable[[], float] | None = None,
) -> GeneratedScript:
    """One generation. The first request's failure propagates (the caller decides what that means);
    a failed continuation is recorded and the script so far is returned."""
    reply = client.generate_code(messages)
    text = str(reply)
    result = GeneratedScript(
        code=parse_reply(text).code,
        finish_reason=getattr(reply, "finish_reason", None),
        completion_tokens=getattr(reply, "completion_tokens", None),
    )

    truncated = _needs_continuation(reply, text)
    if not truncated:
        return result  # the normal case: exactly the pre-continuation behaviour

    unmodified_first_reply = result.code
    accumulated = _fold(None, text, True)
    conversation = list(messages)
    while truncated:
        if result.continuations >= MAX_CONTINUATIONS:
            result.notes.append("continuation cap reached; the script is still cut off")
        elif deadline is not None and now is not None and deadline - now() < _MIN_CONTINUATION_WINDOW_SEC:
            result.notes.append("continuation skipped: too little unit budget left")
        else:
            conversation = conversation + [
                {"role": "assistant", "content": _cut_at_last_line(text)},
                {"role": "user", "content": _CONTINUE_PROMPT},
            ]
            try:
                reply = client.generate_code(conversation)
            except Exception as exc:  # noqa: BLE001 - keep the script so far; the repair path takes over
                result.notes.append(f"continuation request failed: {type(exc).__name__}: {exc}")
            else:
                result.continuations += 1
                result.finish_reason = getattr(reply, "finish_reason", None)
                used = getattr(reply, "completion_tokens", None)
                if used is not None:
                    result.completion_tokens = (result.completion_tokens or 0) + used
                text = str(reply)
                truncated = _needs_continuation(reply, text)
                accumulated = _fold(accumulated, text, truncated)
                result.code = accumulated
                continue
        # a guard stopped the loop or the continuation request failed
        result.still_truncated = True
        result.code = accumulated if result.continuations else unmodified_first_reply
        return result
    return result
