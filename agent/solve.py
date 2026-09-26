"""Orchestrates the pipeline: read task -> resolve inputs -> summarize -> prompt ->
generate -> execute -> self-check -> (at most one repair) -> execute -> self-check -> trace.

task_dir and out_dir are opaque parameters handed in by the CLI -- never hardcoded to /input or
/app/output or /output.
"""

from __future__ import annotations

from pathlib import Path

from agent.executor import extract_code, run_generated_code
from agent.input_resolver import resolve_inputs
from agent.model_client import ModelClient
from agent.prompt import build_messages, build_repair_messages
from agent.self_check import self_check
from agent.summarizer import summarize
from agent.task_reader import read_task
from agent.trace import Trace

_DEFAULT_TIMEOUT_SEC = 1800.0
_LOG_PREVIEW_CHARS = 4000


def solve(task_dir: str | Path, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = Trace()

    # A broken/unreadable task bundle is a real infra problem -- nothing meaningful can be
    # attempted without it, so this is allowed to raise and crash the process.
    bundle = read_task(task_dir)
    resolved_inputs = resolve_inputs(bundle.task_dir, bundle.manifest)
    summaries = [summarize(r) for r in resolved_inputs]
    timeout_sec = bundle.card.get("agent", {}).get("timeout_sec", _DEFAULT_TIMEOUT_SEC)

    client = ModelClient()

    messages = build_messages(bundle.instruction_text, resolved_inputs, summaries, out_dir)
    code = extract_code(client.generate_code(messages))
    trace.record("initial_generation", code=code)

    result = run_generated_code(code, out_dir, timeout_sec=timeout_sec)
    trace.record("execution", exit_code=result.exit_code, timed_out=result.timed_out, stderr=result.stderr)

    check = self_check(bundle, resolved_inputs, summaries, out_dir, result)
    trace.record("self_check", passed=check.passed, outcomes=[o.__dict__ for o in check.outcomes])

    if not check.passed:
        repair_messages = build_repair_messages(
            bundle.instruction_text,
            resolved_inputs,
            summaries,
            out_dir,
            previous_code=code,
            execution_result=result,
            self_check_result=check,
        )
        code = extract_code(client.generate_code(repair_messages))
        trace.record("repair_generation", code=code)

        result = run_generated_code(code, out_dir, timeout_sec=timeout_sec)
        check = self_check(bundle, resolved_inputs, summaries, out_dir, result)

    # "final_execution"/"final_self_check" are recorded unconditionally -- when no repair ran,
    # this repeats the same result as "execution"/"self_check" above, which is intentional: the
    # trace always has all six named stages in order, whether or not a repair happened.
    trace.record("final_execution", exit_code=result.exit_code, timed_out=result.timed_out, stderr=result.stderr)
    trace.record("final_self_check", passed=check.passed, outcomes=[o.__dict__ for o in check.outcomes])

    # A failing/erroring generated program is not this wrapper's failure: the checker's own gates
    # will correctly score a missing/wrong deliverable. Log for local debugging and exit 0
    # regardless.
    trace.log_summary()
