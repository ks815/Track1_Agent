"""Researcher-side invocation of the official local checker.

Runs a unit's own checks/test.sh inside finance-bench-sandbox:latest, exactly as README step 6
documents (the same flow this session already ran manually against the exemplar). This module is
imported only by this eval harness, never by anything under agent/.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

_IMAGE = "finance-bench-sandbox:latest"
_TIMEOUT_SEC = 300.0


@dataclass
class CheckerResult:
    ran: bool
    reward: float | None
    detail: str
    checks_passed: int | None = None
    checks_failed: int | None = None
    checks_skipped: int | None = None


def run_official_checker(unit_dir: str | Path, out_dir: str | Path) -> CheckerResult:
    unit_dir = Path(unit_dir).resolve()
    out_dir = Path(out_dir).resolve()

    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--network=none",
            "-e", "OUTPUT_DIR=/app/output",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{unit_dir}:/input:ro",
            "-v", f"{out_dir}:/app/output",
            "-v", f"{out_dir}:/output",
            _IMAGE, "bash", "/input/checks/test.sh",
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SEC,
    )

    reward_path = out_dir / "reward.json"
    if not reward_path.is_file():
        return CheckerResult(
            ran=False, reward=None,
            detail=f"no reward.json produced; docker exit {proc.returncode}; "
                    f"stderr: {proc.stderr[-1000:]}",
        )

    payload = json.loads(reward_path.read_text())
    report_path = out_dir / "pytest_report.json"
    passed = failed = skipped = None
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        summary = report.get("summary", {})
        passed = summary.get("passed")
        failed = summary.get("failed")
        skipped = summary.get("skipped")

    return CheckerResult(
        ran=True,
        reward=payload.get("reward"),
        detail=payload.get("details", ""),
        checks_passed=passed,
        checks_failed=failed,
        checks_skipped=skipped,
    )
