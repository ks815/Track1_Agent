"""Generic task reader: instruction.md + card.toml + manifest.json only.

Deliberately never reads anything under checks/ or checks/reference_data/ -- those are the
grader's own material and, on the hidden eval split, are not even mounted into this container.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # Python < 3.11 (the competition sandbox is 3.13; this is for local dev only)
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class TaskBundle:
    task_dir: Path
    instruction_text: str
    card: dict[str, Any]
    manifest: dict[str, Any]


def read_task(task_dir: str | Path) -> TaskBundle:
    """Read the three solver-visible task files. Raises if any is missing or unparsable."""
    task_dir = Path(task_dir)

    instruction_path = task_dir / "instruction.md"
    card_path = task_dir / "card.toml"
    manifest_path = task_dir / "manifest.json"

    if not instruction_path.is_file():
        raise FileNotFoundError(f"instruction.md not found under task_dir: {instruction_path}")
    if not card_path.is_file():
        raise FileNotFoundError(f"card.toml not found under task_dir: {card_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found under task_dir: {manifest_path}")

    instruction_text = instruction_path.read_text(encoding="utf-8")
    card = tomllib.loads(card_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return TaskBundle(
        task_dir=task_dir,
        instruction_text=instruction_text,
        card=card,
        manifest=manifest,
    )
