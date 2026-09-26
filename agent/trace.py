"""A structured trace of one solve() run's pipeline stages.

Logged to stderr for local debugging -- never written into out_dir, which must contain only
genuine deliverables.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

_TRUNCATE_CHARS = 300


@dataclass
class Trace:
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, **fields: Any) -> None:
        self.stages.append({"stage": stage, **fields})

    def log_summary(self, file: TextIO = sys.stderr) -> None:
        print("[agent] trace:", file=file)
        for entry in self.stages:
            stage = entry["stage"]
            rest = {k: v for k, v in entry.items() if k != "stage"}
            bits = []
            for key, value in rest.items():
                text = str(value)
                if len(text) > _TRUNCATE_CHARS:
                    text = text[:_TRUNCATE_CHARS] + "...(truncated)"
                bits.append(f"{key}={text}")
            print(f"  - {stage}: {', '.join(bits)}", file=file)
