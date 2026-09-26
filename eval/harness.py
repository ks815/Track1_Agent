"""Runs conditions A/B/C for one unit using the real agent.* modules.

Only the model call is stood in for (see stub_generation.py); everything else -- task reading,
input resolution, summarization, execution, self-check, the repair-trigger logic -- is the real,
unmodified code from agent/. Nothing here is imported by anything under agent/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.executor import ExecutionResult, run_generated_code
from agent.input_resolver import resolve_inputs
from agent.self_check import (
    SelfCheckResult,
    _extract_json_example_keys,
    _extract_output_filenames,
    _extract_schema_rows,
    self_check,
)
from agent.summarizer import summarize
from agent.task_reader import read_task

from eval.checker import CheckerResult, run_official_checker
from eval.stub_generation import _OUT_DIR_TOKEN, build_placeholder_script

# The placeholder scripts are trivial (a handful of file writes); a short timeout is a harness
# convenience, not a claim about real generated-code runtime under the unit's actual
# [agent].timeout_sec.
_MECHANISM_TIMEOUT_SEC = 60.0


@dataclass
class ContractExtraction:
    unit_id: str
    candidate_filenames: list[str]
    schema_keys: list[str]
    json_example_keys: list[str]
    n_resolved_inputs: int


@dataclass
class RunRecord:
    unit_id: str
    condition: str
    output_types: str
    initial_exec: str
    self_check_findings: str
    repair_triggered: bool
    final_exec: str
    checker_reward: float | None
    checker_ran: bool
    failure_category: str
    model_calls: int
    notes: str = ""


def _snapshot(out_dir: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(out_dir.iterdir()) if p.is_file()}


def _exec_label(result: ExecutionResult) -> str:
    if result.timed_out:
        return "timeout"
    if result.exit_code != 0:
        return f"failure(exit={result.exit_code})"
    return "success"


def _findings_summary(check: SelfCheckResult) -> str:
    return "; ".join(f"{o.name}={o.status}" for o in check.outcomes)


def _failure_category(
    exec_result: ExecutionResult, check: SelfCheckResult, checker: CheckerResult
) -> str:
    if exec_result.timed_out:
        return "timeout"
    if exec_result.exit_code != 0:
        return "execution_error"
    if not checker.ran:
        return "checker_infra_error"
    if checker.reward == 1.0:
        return "checker_pass"
    fail_names = [o.name for o in check.outcomes if o.status == "fail"]
    if any(n == "output_files_exist" for n in fail_names):
        return "missing_deliverable"
    if any(n.startswith("readable:") for n in fail_names):
        return "unreadable_output"
    if any(n.startswith("required_keys:") for n in fail_names):
        return "schema_mismatch"
    if any(n.startswith("sign_range:") for n in fail_names):
        return "sign_range_violation"
    if any(n.startswith("no_nan_inf_empty:") for n in fail_names):
        return "nan_inf_empty"
    if "row_count_parity" in fail_names:
        return "row_count_mismatch"
    return "checker_wrong_numeric"  # self-check saw nothing wrong but the real checker still failed


def run_unit(unit_id: str, work_root: Path) -> tuple[ContractExtraction, list[RunRecord], dict[str, Any]]:
    unit_dir = Path("units") / unit_id
    bundle = read_task(unit_dir)
    resolved = resolve_inputs(bundle.task_dir, bundle.manifest)
    summaries = [summarize(r) for r in resolved]

    candidate_filenames = _extract_output_filenames(bundle.instruction_text)
    schema_rows = _extract_schema_rows(bundle.instruction_text)
    json_keys = _extract_json_example_keys(bundle.instruction_text)
    extraction = ContractExtraction(
        unit_id=unit_id,
        candidate_filenames=sorted(candidate_filenames),
        schema_keys=sorted(key for key, _ in schema_rows),
        json_example_keys=sorted(json_keys),
        n_resolved_inputs=len(resolved),
    )
    output_types = sorted({f.rsplit(".", 1)[-1] for f in candidate_filenames}) or ["unknown"]

    naive_template = build_placeholder_script(candidate_filenames, schema_rows, json_keys, style="naive")
    repaired_template = build_placeholder_script(candidate_filenames, schema_rows, json_keys, style="repaired")

    records: list[RunRecord] = []
    diagnostics: dict[str, Any] = {}

    # --- Condition A: initial generation only. No self-check call is part of A's own pipeline;
    # self_check is invoked here ONLY to compute a comparable failure_category label for the
    # report table, never to alter out_A or to gate anything.
    out_A = work_root / f"{unit_id}__A"
    out_A.mkdir(parents=True)
    code_A = naive_template.replace(_OUT_DIR_TOKEN, str(out_A))
    exec_A = run_generated_code(code_A, out_A, timeout_sec=_MECHANISM_TIMEOUT_SEC)
    # Snapshot the deliverable BEFORE the official checker runs -- the checker writes
    # reward.json/pytest_report.json into this same directory, which must never be counted as
    # part of "the generated deliverable" when comparing against condition B below.
    snapshot_A_deliverables = _snapshot(out_A)
    diagnostic_check_A = self_check(bundle, resolved, summaries, out_A, exec_A)
    checker_A = run_official_checker(unit_dir, out_A)
    records.append(
        RunRecord(
            unit_id=unit_id,
            condition="A",
            output_types=",".join(output_types),
            initial_exec=_exec_label(exec_A),
            self_check_findings="(not run as part of condition A's own pipeline)",
            repair_triggered=False,
            final_exec=_exec_label(exec_A),
            checker_reward=checker_A.reward,
            checker_ran=checker_A.ran,
            failure_category=_failure_category(exec_A, diagnostic_check_A, checker_A),
            model_calls=1,
        )
    )

    # --- Condition B: self-check only (diagnostic) -- must not alter the deliverable.
    out_B = work_root / f"{unit_id}__B"
    out_B.mkdir(parents=True)
    code_B = naive_template.replace(_OUT_DIR_TOKEN, str(out_B))
    exec_B = run_generated_code(code_B, out_B, timeout_sec=_MECHANISM_TIMEOUT_SEC)
    snapshot_before_check = _snapshot(out_B)
    check_B = self_check(bundle, resolved, summaries, out_B, exec_B)
    snapshot_after_check = _snapshot(out_B)
    diagnostics["B_self_check_side_effect_free"] = snapshot_before_check == snapshot_after_check
    diagnostics["A_B_byte_identical"] = snapshot_A_deliverables == snapshot_after_check
    checker_B = run_official_checker(unit_dir, out_B)
    records.append(
        RunRecord(
            unit_id=unit_id,
            condition="B",
            output_types=",".join(output_types),
            initial_exec=_exec_label(exec_B),
            self_check_findings=_findings_summary(check_B),
            repair_triggered=False,
            final_exec=_exec_label(exec_B),
            checker_reward=checker_B.reward,
            checker_ran=checker_B.ran,
            failure_category=_failure_category(exec_B, check_B, checker_B),
            model_calls=1,
            notes=(
                f"side_effect_free={diagnostics['B_self_check_side_effect_free']}, "
                f"byte_identical_to_A={diagnostics['A_B_byte_identical']}"
            ),
        )
    )

    # --- Condition C: self-check + at most one repair.
    out_C = work_root / f"{unit_id}__C"
    out_C.mkdir(parents=True)
    code_C1 = naive_template.replace(_OUT_DIR_TOKEN, str(out_C))
    exec_C1 = run_generated_code(code_C1, out_C, timeout_sec=_MECHANISM_TIMEOUT_SEC)
    check_C1 = self_check(bundle, resolved, summaries, out_C, exec_C1)

    model_calls = 1
    repair_triggered = False
    final_exec_result, final_check_result = exec_C1, check_C1
    if not check_C1.passed:
        repair_triggered = True
        code_C2 = repaired_template.replace(_OUT_DIR_TOKEN, str(out_C))
        exec_C2 = run_generated_code(code_C2, out_C, timeout_sec=_MECHANISM_TIMEOUT_SEC)
        check_C2 = self_check(bundle, resolved, summaries, out_C, exec_C2)
        model_calls = 2
        final_exec_result, final_check_result = exec_C2, check_C2

    checker_C = run_official_checker(unit_dir, out_C)
    diagnostics["C_final_out_dir_listing"] = sorted(p.name for p in out_C.iterdir())
    records.append(
        RunRecord(
            unit_id=unit_id,
            condition="C",
            output_types=",".join(output_types),
            initial_exec=_exec_label(exec_C1),
            self_check_findings=_findings_summary(final_check_result),
            repair_triggered=repair_triggered,
            final_exec=_exec_label(final_exec_result),
            checker_reward=checker_C.reward,
            checker_ran=checker_C.ran,
            failure_category=_failure_category(final_exec_result, final_check_result, checker_C),
            model_calls=model_calls,
        )
    )

    return extraction, records, diagnostics
