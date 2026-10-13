"""Extracts the model's code and runs it, capturing exit status / stdout / stderr.

No repair here: a failing script's output is captured and logged by the caller, not acted on.
The generated script is written to a scratch location outside out_dir, so it can never itself be
mistaken for a deliverable.

Extraction is deliberately forgiving about HOW a reply is wrapped but never about WHAT it contains:
it recovers the code from replies with prose around the fence, a differently-cased or abbreviated
language tag, several fenced blocks, reasoning text, or a fence that was never closed. It does not
repair code: whatever it returns is exactly the characters the model wrote inside the block.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_TIMEOUT_SEC = 1800.0

# A fence on its own line (up to 3 spaces of indent). Anchored and linear.
_FENCE_LINE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*([\w+.#-]*)[ \t]*$")


def _trailing_fence(line: str, with_tag: bool) -> tuple[str, str, int] | None:
    """(fence, tag, start) when `line` ENDS with a run of 3+ backticks/tildes, optionally followed by a
    language tag ("Here is the code: ```python"; or, with with_tag=False, "print(1)```"). A single
    right-to-left scan: linear in the line length, unlike an unanchored regex search, which is quadratic
    on a long run of backticks."""
    end = len(line.rstrip(" \t"))
    i = end
    if with_tag:
        while i > 0 and (line[i - 1].isalnum() or line[i - 1] in "_+.#-"):
            i -= 1
    if i == 0 or line[i - 1] not in "`~":
        return None
    mark = line[i - 1]
    j = i
    while j > 0 and line[j - 1] == mark:
        j -= 1
    if i - j < 3:
        return None
    return line[j:i], line[i:end], j


_CLOSE_TAG = "</think>"
_OPEN_TAG = "<think>"


@dataclass
class ExecutionResult:
    exit_code: int | None  # None if the process timed out
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class ParsedReply:
    code: str
    #: False when the chosen block was opened but never closed (a reply cut off mid-code, or a model that
    #: forgot the closing fence). True for a closed block and for a reply with no fence at all.
    complete: bool
    #: True when `code` came out of a fenced block; False when it is the (reasoning-stripped) raw reply.
    fenced: bool
    n_blocks: int


def _strip_reasoning(text: str) -> str:
    """Drop reasoning text: everything up to the last closing think tag, and from an unclosed opening
    tag onward (reasoning that was cut off before it produced an answer). Replies from a model with
    thinking disabled contain neither tag and pass through unchanged."""
    cut = text.lower().rfind(_CLOSE_TAG)
    if cut != -1:
        text = text[cut + len(_CLOSE_TAG) :]
    opened = text.lower().find(_OPEN_TAG)
    if opened != -1:
        text = text[:opened]
    return text


def _is_python_tag(tag: str) -> bool:
    tag = tag.lower()
    return tag == "" or tag.startswith("py") or tag == "ipython"


def _blocks(text: str) -> list[tuple[str, str, bool]]:
    """Every fenced block as (language tag, code, closed)."""
    blocks: list[tuple[str, str, bool]] = []
    fence: str | None = None
    tag = ""
    body: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if fence is None:
            opener = _FENCE_LINE_RE.match(bare)
            if opener:
                fence, tag, body = opener.group(1), opener.group(2), []
            else:
                prose = _trailing_fence(bare, with_tag=True)
                if prose:
                    fence, tag, body = prose[0], prose[1], []
            continue
        closer = _FENCE_LINE_RE.match(bare)
        if closer and not closer.group(2) and closer.group(1)[0] == fence[0] and len(closer.group(1)) >= len(fence):
            blocks.append((tag, "".join(body), True))
            fence = None
            continue
        trailing = _trailing_fence(bare, with_tag=False)
        if trailing and trailing[0][0] == fence[0] and len(trailing[0]) >= len(fence):
            body.append(bare[: trailing[2]])
            blocks.append((tag, "".join(body), True))
            fence = None
            continue
        body.append(line)
    if fence is not None:
        blocks.append((tag, "".join(body), False))
    return blocks


def parse_reply(model_reply: str) -> ParsedReply:
    """Pick the script out of a model reply.

    Reasoning text is removed first. Among the fenced blocks whose language is Python (or
    unlabelled), the one with the most code wins -- a one-line usage example or a pseudo-code sketch
    never beats the real script -- and a later block wins a tie. Blocks in other languages (bash,
    json, ...) are ignored. A reply with no usable fence is returned whole, exactly as before.
    """
    text = _strip_reasoning(model_reply)
    blocks = _blocks(text)
    candidates = [(i, b) for i, b in enumerate(blocks) if _is_python_tag(b[0])]
    if not candidates:
        return ParsedReply(code=text, complete=True, fenced=False, n_blocks=len(blocks))
    _, (_, code, closed) = max(candidates, key=lambda c: (len(re.sub(r"\s+", "", c[1][1])), c[0]))
    return ParsedReply(code=code, complete=closed, fenced=True, n_blocks=len(blocks))


def extract_code(model_reply: str) -> str:
    """The script in a model reply (see parse_reply); the raw reply when it holds no fenced block."""
    return parse_reply(model_reply).code


def run_generated_code(
    code: str, out_dir: str | Path, timeout_sec: float = _DEFAULT_TIMEOUT_SEC
) -> ExecutionResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="t1-agent-scratch-") as scratch:
        script_path = Path(scratch) / "generated_solution.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                exit_code=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )

        return ExecutionResult(exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
