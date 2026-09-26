"""Extracts the model's code and runs it, capturing exit status / stdout / stderr.

No repair here: a failing script's output is captured and logged by the caller, not acted on.
The generated script is written to a scratch location outside out_dir, so it can never itself be
mistaken for a deliverable.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DEFAULT_TIMEOUT_SEC = 1800.0


@dataclass
class ExecutionResult:
    exit_code: int | None  # None if the process timed out
    stdout: str
    stderr: str
    timed_out: bool = False


def extract_code(model_reply: str) -> str:
    """Return the contents of the first ```python fenced block, or the raw reply if there is none."""
    match = _CODE_FENCE_RE.search(model_reply)
    return match.group(1) if match else model_reply


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
