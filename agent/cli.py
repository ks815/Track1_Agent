"""CLI entry point implementing the verified Track 1 interface: solve --task-dir <path> --out <path>."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from agent.solve import solve


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Track 1 Phase 1 agent")
    parser.add_argument("verb", choices=["solve"])
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    solve(args.task_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
