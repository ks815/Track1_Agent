"""Offline unit tests for agent.self_check.

Pure Python + pandas/pyarrow (both already required by the base image, and present on any host
that can run agent.summarizer) -- no Docker, no model, no network. Synthetic instruction texts
and synthetic out_dir contents only; never reads any real unit's checks/.
"""

from __future__ import annotations

from pathlib import Path

from agent.executor import ExecutionResult
from agent.self_check import self_check
from agent.task_reader import TaskBundle

_EXEMPLAR_LIKE_INSTRUCTION = """\
## Inputs

| Column | Type | Unit / Convention |
|---|---|---|
| option_id | str | Unique row identifier |
| S | float64 | Spot price |
| K | float64 | Strike price |

## Deliverables

### `/output/results.parquet`

| Column | Type | Unit / Convention |
|---|---|---|
| option_id | str | Matches input option_id |
| price | float64 | USD, >= 0 |
| delta | float64 | Dimensionless; in (0, 1) for calls |
"""


def _bundle(instruction_text: str) -> TaskBundle:
    return TaskBundle(task_dir=Path("/nonexistent"), instruction_text=instruction_text, card={}, manifest={})


def _ok_result() -> ExecutionResult:
    return ExecutionResult(exit_code=0, stdout="", stderr="")


def test_input_schema_table_never_leaks_into_expected_output_keys(tmp_path):
    """The Inputs table's S/K columns must never become "required output keys" -- only rows
    inside a heading that itself names output/deliverables count."""
    (tmp_path / "results.parquet").write_bytes(b"")  # existence only; parse will fail, that's fine here
    from agent.self_check import _extract_schema_rows

    rows = _extract_schema_rows(_EXEMPLAR_LIKE_INSTRUCTION)
    keys = {k for k, _ in rows}
    assert "S" not in keys and "K" not in keys
    assert {"option_id", "price", "delta"} <= keys


def test_passing_output_produces_no_failures(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        {
            "option_id": ["opt-1", "opt-2"],
            "price": [1.5, 2.5],
            "delta": [0.4, 0.6],
        }
    )
    df.to_parquet(tmp_path / "results.parquet", index=False)

    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, _ok_result())
    failures = [o for o in result.outcomes if o.status == "fail"]
    assert result.passed, failures
    names = {o.name for o in result.outcomes}
    assert "output_files_exist" in names
    assert any(n.startswith("sign_range:") for n in names)


def test_negative_price_violates_stated_nonnegativity(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"option_id": ["opt-1"], "price": [-3.0], "delta": [0.4]})
    df.to_parquet(tmp_path / "results.parquet", index=False)

    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, _ok_result())
    assert not result.passed
    sign_failures = [o for o in result.outcomes if o.name.startswith("sign_range:") and o.status == "fail"]
    assert sign_failures, result.outcomes


def test_delta_outside_open_interval_is_flagged(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"option_id": ["opt-1"], "price": [1.0], "delta": [1.4]})
    df.to_parquet(tmp_path / "results.parquet", index=False)

    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, _ok_result())
    assert not result.passed


def test_missing_required_file_fails(tmp_path):
    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, _ok_result())
    assert not result.passed
    outcome = next(o for o in result.outcomes if o.name == "output_files_exist")
    assert outcome.status == "fail"


def test_execution_failure_short_circuits_before_any_output_inspection(tmp_path):
    (tmp_path / "results.parquet").write_bytes(b"garbage")
    bad_result = ExecutionResult(exit_code=1, stdout="", stderr="Traceback: boom")
    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, bad_result)
    assert not result.passed
    assert result.outcomes == [result.outcomes[0]]  # exactly one outcome: "execution"
    assert result.outcomes[0].name == "execution"


def test_timeout_short_circuits(tmp_path):
    timed_out_result = ExecutionResult(exit_code=None, stdout="", stderr="", timed_out=True)
    result = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], [], tmp_path, timed_out_result)
    assert not result.passed
    assert result.outcomes[0].detail == "generated code timed out"


def test_instruction_with_no_derivable_schema_is_not_applicable_not_fail(tmp_path):
    vague = "Do something useful with the data and write your answer somewhere sensible."
    (tmp_path / "anything.txt").write_text("some output")

    result = self_check(_bundle(vague), [], [], tmp_path, _ok_result())
    files_outcome = next(o for o in result.outcomes if o.name == "output_files_exist")
    assert files_outcome.status == "not_applicable"
    keys_outcomes = [o for o in result.outcomes if o.name.startswith("required_keys:")]
    assert all(o.status == "not_applicable" for o in keys_outcomes)


def test_empty_out_dir_with_no_derivable_schema_still_fails(tmp_path):
    vague = "Do something useful with the data."
    result = self_check(_bundle(vague), [], [], tmp_path, _ok_result())
    outcome = next(o for o in result.outcomes if o.name == "output_files_exist")
    assert outcome.status == "fail"


def test_json_fenced_example_keys_are_derived_and_checked():
    instruction = """\
## Output

Save to `/app/output/results.json`:

```json
{
  "AAPL": {
    "total_return": 0.1,
    "sharpe_ratio": 0.5
  }
}
```
"""
    from agent.self_check import _extract_json_example_keys

    keys = _extract_json_example_keys(instruction)
    assert {"AAPL", "total_return", "sharpe_ratio"} <= keys


def test_nested_json_output_with_all_keys_passes(tmp_path):
    import json

    instruction = """\
## Output

Save to `/app/output/results.json`:

```json
{
  "AAPL": {
    "total_return": 0.1,
    "sharpe_ratio": 0.5
  }
}
```
"""
    (tmp_path / "results.json").write_text(
        json.dumps({"AAPL": {"total_return": 0.2, "sharpe_ratio": 0.7}})
    )
    result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]


def test_missing_json_key_fails():
    import json
    import tempfile

    instruction = """\
## Output

Save to `/app/output/results.json`:

```json
{
  "AAPL": {
    "total_return": 0.1,
    "sharpe_ratio": 0.5
  }
}
```
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "results.json").write_text(json.dumps({"AAPL": {"total_return": 0.2}}))
        result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
        assert not result.passed
        outcome = next(o for o in result.outcomes if o.name.startswith("required_keys:"))
        assert outcome.status == "fail"
        assert "sharpe_ratio" in outcome.detail


def test_nan_in_json_output_fails():
    instruction = "## Output\n\nSave to `/app/output/results.json`."
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "results.json").write_text(json.dumps({"value": float("nan")}))
        # json.dumps of NaN produces the literal `NaN`, which json.loads (used by self_check)
        # accepts back (Python's json module is permissive here, matching common agent output).
        result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
        nan_outcomes = [o for o in result.outcomes if o.name.startswith("no_nan_inf_empty:")]
        assert nan_outcomes and nan_outcomes[0].status == "fail"


def test_row_count_parity_fires_only_when_instruction_states_it(tmp_path):
    import pandas as pd

    instruction_with_parity = _EXEMPLAR_LIKE_INSTRUCTION + "\nThe output has one row per input option.\n"
    df = pd.DataFrame({"option_id": ["a", "b"], "price": [1.0, 2.0], "delta": [0.4, 0.6]})
    df.to_parquet(tmp_path / "results.parquet", index=False)

    summaries_matching = [{"n_rows": 2}]
    result = self_check(_bundle(instruction_with_parity), [], summaries_matching, tmp_path, _ok_result())
    row_outcome = next(o for o in result.outcomes if o.name == "row_count_parity")
    assert row_outcome.status == "pass"

    summaries_mismatched = [{"n_rows": 5}]
    result2 = self_check(_bundle(instruction_with_parity), [], summaries_mismatched, tmp_path, _ok_result())
    row_outcome2 = next(o for o in result2.outcomes if o.name == "row_count_parity")
    assert row_outcome2.status == "fail"
    assert not result2.passed

    result3 = self_check(_bundle(_EXEMPLAR_LIKE_INSTRUCTION), [], summaries_mismatched, tmp_path, _ok_result())
    row_outcome3 = next(o for o in result3.outcomes if o.name == "row_count_parity")
    assert row_outcome3.status == "not_applicable"
