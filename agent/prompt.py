"""Builds the messages sent to the House model.

This is where the generic output-path rule lives: the model is told the literal, absolute
out_dir this process was actually given, and instructed to substitute it for whatever output
root instruction.md names (/app/output, /output, or anything else) -- keeping only the
filename/relative path that follows. The physical output root always comes from out_dir; the
task instruction only ever determines the relative filename and required content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.input_resolver import ResolvedInput

if TYPE_CHECKING:
    from agent.executor import ExecutionResult
    from agent.self_check import SelfCheckResult

_SYSTEM_PROMPT = """\
You are a quantitative-finance coding agent. You will be given one task's full instructions, a \
factual summary of its input files, and the exact input and output paths available in the \
current runtime. Write a single, self-contained Python 3 script that solves the task.

Rules:
- Read input data only from the exact absolute input paths given to you below. Do not guess or \
invent other paths.
- The task's own instructions may name an output directory such as /app/output or /output. \
Ignore that literal root. Instead, write every deliverable under the exact absolute output \
directory given to you below, using only the filename (or relative path) that the instructions \
specify after their stated output root. For example, if the instructions say to write to \
/app/output/results.json and the given output directory is /tmp/xyz, write to \
/tmp/xyz/results.json.
- Produce exactly the deliverable files, columns/keys, and formats the instructions require -- \
no more, no fewer.
- The script must run standalone with `python script.py` and must not require any interactive \
input, network access, or command-line arguments.
- Output only the Python code, in a single ```python fenced code block, with no other prose.
"""


_REPAIR_SYSTEM_PROMPT = (
    _SYSTEM_PROMPT
    + """
You are now fixing your own previous attempt at this same task. You will also be given your \
previous code and a concrete description of what was wrong with it -- an execution failure, or \
a structural self-check finding. Return one complete, corrected, self-contained Python script \
-- not a diff or a patch -- that fixes the described problem while still meeting every rule \
above.
"""
)


def _format_inputs_block(resolved_inputs: list[ResolvedInput], summaries: list[dict[str, Any]]) -> str:
    input_lines = []
    for resolved, summary in zip(resolved_inputs, summaries):
        input_lines.append(
            f"- absolute path: {resolved.resolved_path}\n"
            f"  (manifest-relative name: {resolved.manifest_path})\n"
            f"  factual summary: {json.dumps(summary, default=str)}"
        )
    return "\n".join(input_lines) if input_lines else "(this task declares no input files)"


def build_messages(
    instruction_text: str,
    resolved_inputs: list[ResolvedInput],
    summaries: list[dict[str, Any]],
    out_dir: str | Path,
) -> list[dict[str, str]]:
    out_dir = str(Path(out_dir))
    inputs_block = _format_inputs_block(resolved_inputs, summaries)

    user_prompt = f"""\
## Task instructions

{instruction_text}

## Runtime input files (use these exact absolute paths)

{inputs_block}

## Runtime output directory (write every deliverable under this exact path)

{out_dir}
"""

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_self_check_failures(self_check_result: "SelfCheckResult") -> str:
    failures = [o for o in self_check_result.outcomes if o.status == "fail"]
    if not failures:
        return "(no specific self-check failures were recorded)"
    return "\n".join(f"- {o.name}: {o.detail}" for o in failures)


def build_repair_messages(
    instruction_text: str,
    resolved_inputs: list[ResolvedInput],
    summaries: list[dict[str, Any]],
    out_dir: str | Path,
    previous_code: str,
    execution_result: "ExecutionResult",
    self_check_result: "SelfCheckResult",
) -> list[dict[str, str]]:
    out_dir = str(Path(out_dir))
    inputs_block = _format_inputs_block(resolved_inputs, summaries)

    execution_summary = (
        f"exit_code={execution_result.exit_code}, timed_out={execution_result.timed_out}\n"
        f"stderr (truncated): {execution_result.stderr[-2000:]}"
    )
    self_check_summary = _format_self_check_failures(self_check_result)

    user_prompt = f"""\
## Task instructions

{instruction_text}

## Runtime input files (use these exact absolute paths)

{inputs_block}

## Runtime output directory (write every deliverable under this exact path)

{out_dir}

## Your previous code

```python
{previous_code}
```

## What went wrong

Execution result:
{execution_summary}

Self-check findings:
{self_check_summary}

Return one complete, corrected Python script that fixes these problems.
"""

    return [
        {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
