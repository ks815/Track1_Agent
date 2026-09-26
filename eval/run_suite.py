"""Drives the Phase 3 coverage suite (see suite.py) through harness.run_unit for all 6 units,
and writes the raw results to eval/results/.

Usage (from the repo root, with PYTHONPATH set to the repo root):
    python -m eval.run_suite
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from eval.harness import run_unit
from eval.suite import SUITE

_RESULTS_DIR = Path("eval/results")


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="t1-eval-"))

    all_extractions = []
    all_records = []
    all_diagnostics = {}

    try:
        for entry in SUITE:
            print(f"=== {entry.unit_id} ===", file=sys.stderr)
            extraction, records, diagnostics = run_unit(entry.unit_id, work_root)
            all_extractions.append(dataclasses.asdict(extraction))
            all_records.extend(dataclasses.asdict(r) for r in records)
            all_diagnostics[entry.unit_id] = diagnostics
            for r in records:
                print(
                    f"  [{r.condition}] exec={r.final_exec} reward={r.checker_reward} "
                    f"repair={r.repair_triggered} calls={r.model_calls} category={r.failure_category}",
                    file=sys.stderr,
                )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Mechanism-validation results only (stubbed generation). Not a pass@1 measurement.",
        "suite": [dataclasses.asdict(e) for e in SUITE],
        "contract_extraction": all_extractions,
        "records": all_records,
        "diagnostics": all_diagnostics,
    }

    out_path = _RESULTS_DIR / f"phase3_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
