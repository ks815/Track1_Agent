"""Deterministic input resolution from manifest.json.

manifest.json is the primary and only source of truth for which files are inputs: it lists each
file's bundle-relative path and its role. A path stated in instruction.md prose (which, per the
verified Track 1 contract, sometimes names a location -- e.g. /app/data/... -- that only exists
under the organizer's own checker presentation machinery, never in this container) is never
trusted here. This module resolves every manifest-declared input against the real task_dir the
harness handed us, and verifies each one actually exists before the rest of the pipeline runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResolvedInput:
    manifest_path: str  # bundle-relative path, exactly as manifest.json states it
    resolved_path: Path  # task_dir / manifest_path, verified to exist


def resolve_inputs(task_dir: str | Path, manifest: dict[str, Any]) -> list[ResolvedInput]:
    """Resolve every manifest entry with role == "input" to a real, existing path.

    Raises FileNotFoundError if a manifest-declared input file is absent -- Phase 1 has no
    repair path for a broken/incomplete task bundle, so this fails loudly rather than silently
    solving against fewer inputs than the task declares.
    """
    task_dir = Path(task_dir)
    files = manifest.get("files", [])
    resolved: list[ResolvedInput] = []

    for entry in files:
        if entry.get("role") != "input":
            continue  # explicitly excludes reference/other roles -- never solver material

        manifest_path = entry.get("path")
        if not manifest_path:
            raise ValueError(f"manifest.json input entry has no 'path': {entry!r}")

        candidate = (task_dir / manifest_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"manifest.json declares input {manifest_path!r} but no file exists at "
                f"{candidate} under task_dir {task_dir}"
            )

        resolved.append(ResolvedInput(manifest_path=manifest_path, resolved_path=candidate))

    return resolved
