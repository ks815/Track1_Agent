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
    """An UNCONDITIONAL stated interval is a hard constraint on every row. (The conditional form,
    "in (0, 1) for calls", is deliberately not enforced -- see the conditional-convention tests.)"""
    import pandas as pd

    unconditional = _EXEMPLAR_LIKE_INSTRUCTION.replace("in (0, 1) for calls", "in (0, 1)")
    assert unconditional != _EXEMPLAR_LIKE_INSTRUCTION
    df = pd.DataFrame({"option_id": ["opt-1"], "price": [1.0], "delta": [1.4]})
    df.to_parquet(tmp_path / "results.parquet", index=False)

    result = self_check(_bundle(unconditional), [], [], tmp_path, _ok_result())
    assert not result.passed
    assert any(o.name.startswith("sign_range:") and o.status == "fail" for o in result.outcomes)


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


# A multi-output task whose files have different schemas, written in the three attribution styles
# seen in real instructions: filename in the heading (`### File 2: ...`), filename in a "Save to"
# line under a filename-less heading, and a table under a filename heading. The HTML deliverable
# has no schema at all.
_HETEROGENEOUS_INSTRUCTION = """\
## Output

### File 1: `/app/output/calibration.json`

```json
{
    "kappa": <float>,
    "theta": <float>
}
```

### File 2: `/app/output/bond_prices.csv`

| Column | Type | Unit / Convention |
|---|---|---|
| tau | float64 | Years, > 0 |
| zcb_price | float64 | Price, in (0, 1) |

### Summary JSON

Save to `/app/output/summary.json` with this structure:

```json
{
    "mean_yield_error": <float>,
    "yield_curve_upward_sloping": <bool>
}
```

### Chart

Save an interactive chart to `/app/output/curve.html`.
"""


def _write_correct_heterogeneous_outputs(out_dir: Path) -> None:
    import json

    import pandas as pd

    (out_dir / "calibration.json").write_text(json.dumps({"kappa": 0.5, "theta": 0.04}))
    pd.DataFrame({"tau": [0.25, 1.0], "zcb_price": [0.99, 0.95]}).to_csv(
        out_dir / "bond_prices.csv", index=False
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"mean_yield_error": 0.001, "yield_curve_upward_sloping": True})
    )
    (out_dir / "curve.html").write_text("<html><body>chart</body></html>")


def test_multi_output_task_checks_each_file_only_against_its_own_keys(tmp_path):
    """Regression: keys/columns from the whole task contract used to be required of EVERY output
    file, so a fully correct set of heterogeneous files failed self-check (and burned the one
    repair call on false findings)."""
    _write_correct_heterogeneous_outputs(tmp_path)

    result = self_check(_bundle(_HETEROGENEOUS_INSTRUCTION), [], [], tmp_path, _ok_result())

    failures = [o for o in result.outcomes if o.status == "fail"]
    assert result.passed, failures
    by_name = {o.name: o for o in result.outcomes}
    assert by_name["required_keys:calibration.json"].status == "pass"
    assert by_name["required_keys:bond_prices.csv"].status == "pass"
    assert by_name["required_keys:summary.json"].status == "pass"
    # The HTML deliverable has nothing attributed to it: not applicable, never a false fail.
    assert by_name["required_keys:curve.html"].status == "not_applicable"


def test_multi_output_task_still_flags_a_file_missing_its_own_key(tmp_path):
    import json

    _write_correct_heterogeneous_outputs(tmp_path)
    (tmp_path / "summary.json").write_text(json.dumps({"mean_yield_error": 0.001}))

    result = self_check(_bundle(_HETEROGENEOUS_INSTRUCTION), [], [], tmp_path, _ok_result())

    assert not result.passed
    failures = {o.name: o for o in result.outcomes if o.status == "fail"}
    assert set(failures) == {"required_keys:summary.json"}
    detail = failures["required_keys:summary.json"].detail
    assert "yield_curve_upward_sloping" in detail
    # Another file's keys must not be reported as missing from this one.
    assert "kappa" not in detail and "tau" not in detail


def test_multi_output_task_applies_sign_range_conventions_per_file(tmp_path):
    """A column's stated convention applies to the file whose table states it, not to a
    same-named column in a different file."""
    import pandas as pd

    instruction = """\
## Deliverables

### `/output/a.csv`

| Column | Type | Unit / Convention |
|---|---|---|
| value | float64 | USD, >= 0 |

### `/output/b.csv`

| Column | Type | Unit / Convention |
|---|---|---|
| value | float64 | Signed P&L, may be negative |
"""
    pd.DataFrame({"value": [1.0, 2.0]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"value": [-5.0, 3.0]}).to_csv(tmp_path / "b.csv", index=False)

    result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]

    pd.DataFrame({"value": [-1.0, 2.0]}).to_csv(tmp_path / "a.csv", index=False)
    result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
    failures = {o.name for o in result.outcomes if o.status == "fail"}
    assert failures == {"sign_range:a.csv"}


def test_unattributable_keys_apply_only_when_there_is_a_single_candidate_output(tmp_path):
    import json

    # The fence precedes the only filename mention, so it cannot be attributed by position --
    # but with exactly one candidate deliverable, the keys can only describe that file.
    single = """\
## Output

Example of the result:

```json
{"alpha": 1.0, "beta": 2.0}
```

Write it to `/app/output/results.json`.
"""
    (tmp_path / "results.json").write_text(json.dumps({"alpha": 1.0}))
    result = self_check(_bundle(single), [], [], tmp_path, _ok_result())
    assert not result.passed
    assert "beta" in next(o for o in result.outcomes if o.name == "required_keys:results.json").detail

    # With two candidate deliverables the same unattributable fence is dropped, not guessed at.
    multi = single.replace(
        "Write it to `/app/output/results.json`.",
        "Write it to `/app/output/results.json`, plus a log at `/app/output/log.txt`.",
    )
    (tmp_path / "log.txt").write_text("done")
    result = self_check(_bundle(multi), [], [], tmp_path, _ok_result())
    key_outcomes = [o for o in result.outcomes if o.name.startswith("required_keys:")]
    assert key_outcomes and all(o.status == "not_applicable" for o in key_outcomes)
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]


# The deliverable schema of the Black-Scholes exemplar, with the convention cells verbatim --
# including the conditional delta interval written with a Unicode minus sign.
_BLACK_SCHOLES_INSTRUCTION = """\
## Deliverables

### `/output/results.parquet`

A Parquet file with one row per input option, preserving `option_id` order.

| Column | Type | Unit / Convention |
|---|---|---|
| `option_id` | `str` | Matches input `option_id` |
| `price` | `float64` | USD, ≥ 0 |
| `delta` | `float64` | Dimensionless; ∈ (0, 1) for calls, ∈ (−1, 0) for puts |
| `gamma` | `float64` | Per USD; > 0 for both calls and puts |
| `vega` | `float64` | USD per unit of σ; > 0 for both calls and puts |
| `theta` | `float64` | USD per calendar day; typically < 0 for long positions |
"""


def _black_scholes_frame():
    """A CORRECT closed-form Black-Scholes result for a mixed call/put book that includes very
    deep out-of-the-money options, where gamma/vega are strictly positive but astronomically
    small (down to ~1e-200) and put deltas are negative."""
    import math

    import pandas as pd

    def cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def pdf(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    rows = []
    spot, rate, sigma = 100.0, 0.05, 0.2
    for strike in (30.0, 60.0, 80.0, 100.0, 120.0, 200.0, 300.0):
        for maturity in (14 / 365, 0.5, 2.0):
            for kind in ("call", "put"):
                sqrt_t = math.sqrt(maturity)
                d1 = (math.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt_t)
                d2 = d1 - sigma * sqrt_t
                disc = strike * math.exp(-rate * maturity)
                theta_common = -spot * pdf(d1) * sigma / (2.0 * sqrt_t)
                if kind == "call":
                    price = spot * cdf(d1) - disc * cdf(d2)
                    delta = cdf(d1)
                    theta = (theta_common - rate * disc * cdf(d2)) / 365.0
                else:
                    price = disc * cdf(-d2) - spot * cdf(-d1)
                    delta = cdf(d1) - 1.0
                    theta = (theta_common + rate * disc * cdf(-d2)) / 365.0
                rows.append(
                    {
                        "option_id": f"{kind}-{strike:g}-{maturity:.3f}",
                        "price": max(price, 0.0),
                        "delta": delta,
                        "gamma": pdf(d1) / (spot * sigma * sqrt_t),
                        "vega": spot * pdf(d1) * sqrt_t,
                        "theta": theta,
                    }
                )
    return pd.DataFrame(rows)


def _failed_names(result):
    return [o.name for o in result.outcomes if o.status == "fail"]


def test_correct_black_scholes_book_passes_self_check(tmp_path):
    """Regression: a correct mixed call/put Black-Scholes output (negative put deltas, tiny but
    strictly positive gamma/vega on deep-OTM options) used to fail self-check on three counts --
    delta's first interval was applied to puts, and a fixed 1e-6 tolerance treated positive values
    below it as "not > 0" -- sending a correct solution through a needless repair."""
    df = _black_scholes_frame()
    assert (df["delta"] < 0).any() and (df["delta"] > 0).any()  # genuinely mixed
    assert 0.0 < df["gamma"].min() < 1e-6 and 0.0 < df["vega"].min() < 1e-6  # genuinely tiny
    df.to_parquet(tmp_path / "results.parquet", index=False)

    result = self_check(
        _bundle(_BLACK_SCHOLES_INSTRUCTION), [], [{"n_rows": len(df)}], tmp_path, _ok_result()
    )

    assert result.passed, [o for o in result.outcomes if o.status == "fail"]
    by_name = {o.name: o for o in result.outcomes}
    assert by_name["sign_range:results.parquet"].status == "pass"
    assert by_name["row_count_parity"].status == "pass"


def test_real_exemplar_instruction_accepts_a_correct_black_scholes_book(tmp_path):
    """The same regression against the exemplar's actual instruction.md (read-only; it is the
    solver-visible task text). Skipped where the units are not present, e.g. inside the image."""
    import pytest

    instruction_path = (
        Path(__file__).resolve().parents[2] / "units" / "t1-EXAMPLE-bs-greeks-pde" / "instruction.md"
    )
    if not instruction_path.is_file():
        pytest.skip("units/ not present in this environment")

    df = _black_scholes_frame()
    df.to_parquet(tmp_path / "results.parquet", index=False)
    result = self_check(
        _bundle(instruction_path.read_text(encoding="utf-8")),
        [],
        [{"n_rows": len(df)}],
        tmp_path,
        _ok_result(),
    )
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]


def test_conditional_or_multi_interval_convention_is_not_enforced_on_every_row(tmp_path):
    import pandas as pd

    def check(convention: str, delta_values: list[float]):
        instruction = (
            "## Deliverables\n\n### `/output/results.csv`\n\n"
            "| Column | Type | Unit / Convention |\n|---|---|---|\n"
            f"| `delta` | `float64` | {convention} |\n"
        )
        pd.DataFrame({"delta": delta_values}).to_csv(tmp_path / "results.csv", index=False)
        return self_check(_bundle(instruction), [], [], tmp_path, _ok_result())

    mixed_book = [0.4, -0.4, 0.9, -0.1]
    # Two intervals (one written with a Unicode minus): neither can be applied to every row.
    assert check("in (0, 1) for calls, in (-1, 0) for puts", mixed_book).passed
    assert check("∈ (0, 1) for calls, ∈ (−1, 0) for puts", mixed_book).passed
    # One interval, but scoped to a subset of rows by a qualifier.
    assert check("in (0, 1) for calls", mixed_book).passed
    assert check("in (0, 1) when the position is long", mixed_book).passed
    # The same interval with no qualifier is a hard constraint on every row.
    unconditional = check("in (0, 1)", mixed_book)
    assert not unconditional.passed
    assert "sign_range:results.csv" in _failed_names(unconditional)
    assert check("in (0, 1)", [0.4, 0.9]).passed


def test_sign_conventions_conditional_on_a_subset_are_not_enforced(tmp_path):
    import pandas as pd

    instruction = (
        "## Deliverables\n\n### `/output/results.csv`\n\n"
        "| Column | Type | Unit / Convention |\n|---|---|---|\n"
        "| `payoff` | `float64` | >= 0 for long positions |\n"
    )
    pd.DataFrame({"payoff": [1.0, -2.0]}).to_csv(tmp_path / "results.csv", index=False)
    assert self_check(_bundle(instruction), [], [], tmp_path, _ok_result()).passed


def test_strict_positivity_ignores_tiny_positives_but_keeps_hard_violations(tmp_path):
    import pandas as pd

    instruction = (
        "## Deliverables\n\n### `/output/results.csv`\n\n"
        "| Column | Type | Unit / Convention |\n|---|---|---|\n"
        "| `gamma` | `float64` | Per USD; > 0 for both calls and puts |\n"
    )

    def check(values: list[float]):
        pd.DataFrame({"gamma": values}).to_csv(tmp_path / "results.csv", index=False)
        return self_check(_bundle(instruction), [], [], tmp_path, _ok_result())

    # Strictly positive, however small, satisfies "> 0" -- including values that underflow to
    # exactly 0.0 or carry float noise just below it, as long as the column is genuinely positive.
    assert check([0.02, 3.6e-205, 5.5e-192, 1e-9]).passed
    assert check([0.02, 0.0, 0.01]).passed  # underflow on one deep-OTM row
    assert check([0.02, -1e-12, 0.01]).passed  # float noise, far inside the tolerance
    # Hard violations still fail: clearly negative values, and a column with nothing positive.
    assert not check([0.02, -0.5, 0.01]).passed
    assert not check([0.0, 0.0, 0.0]).passed
    assert not check([-1e-3]).passed
    assert not check([0.0]).passed


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


# --- Legitimate NaN -------------------------------------------------------------------------
# The generic NaN check rejects any NaN. Some instructions prescribe NaN for a named column
# ("NaN if not available", "set `param2` to NaN"), and then every CORRECT output contains it.
# NaN is tolerated only where the instruction says so, for that column of that file.

_BULLET_FORM_INSTRUCTION = """\
## Output

### File 1: `/app/output/bond_prices.csv`

Columns: `tau, zcb_price, market_yield, yield_diff`

- tau: time to maturity in years
- zcb_price: zero-coupon bond price P(tau)
- market_yield: observed market yield from DGS columns, NaN if not available
- yield_diff: model_yield - market_yield (NaN if market_yield is NaN)

### File 2: `/app/output/forward_rates.csv`

Columns: `tau1, tau2, forward_rate`
"""


def _bond_frame(**overrides):
    import pandas as pd

    nan = float("nan")
    data = {
        "tau": [0.25, 1.0, 2.0],
        "zcb_price": [0.99, 0.96, 0.92],
        "market_yield": [nan, 0.037, 0.038],
        "yield_diff": [nan, 0.003, 0.002],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _write_forward_rates(out_dir):
    import pandas as pd

    pd.DataFrame({"tau1": [0.25], "tau2": [1.0], "forward_rate": [0.04]}).to_csv(
        out_dir / "forward_rates.csv", index=False
    )


def _nan_outcome(result, filename):
    return next(o for o in result.outcomes if o.name == f"no_nan_inf_empty:{filename}")


def test_nan_explicitly_allowed_for_a_column_passes(tmp_path):
    _bond_frame().to_csv(tmp_path / "bond_prices.csv", index=False)
    _write_forward_rates(tmp_path)

    result = self_check(_bundle(_BULLET_FORM_INSTRUCTION), [], [], tmp_path, _ok_result())

    assert _nan_outcome(result, "bond_prices.csv").status == "pass"
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]


def test_unexpected_nan_in_another_column_still_fails(tmp_path):
    nan = float("nan")
    _bond_frame(zcb_price=[0.99, nan, 0.92]).to_csv(tmp_path / "bond_prices.csv", index=False)
    _write_forward_rates(tmp_path)

    result = self_check(_bundle(_BULLET_FORM_INSTRUCTION), [], [], tmp_path, _ok_result())

    outcome = _nan_outcome(result, "bond_prices.csv")
    assert outcome.status == "fail" and not result.passed
    assert "zcb_price" in outcome.detail
    # The permitted columns are not what is reported.
    assert "market_yield" not in outcome.detail and "yield_diff" not in outcome.detail


def test_nan_in_a_file_with_no_permission_still_fails(tmp_path):
    import pandas as pd

    _bond_frame().to_csv(tmp_path / "bond_prices.csv", index=False)
    pd.DataFrame({"tau1": [0.25], "tau2": [1.0], "forward_rate": [float("nan")]}).to_csv(
        tmp_path / "forward_rates.csv", index=False
    )

    result = self_check(_bundle(_BULLET_FORM_INSTRUCTION), [], [], tmp_path, _ok_result())

    assert _nan_outcome(result, "bond_prices.csv").status == "pass"
    assert _nan_outcome(result, "forward_rates.csv").status == "fail"


def test_permission_is_per_file_not_task_wide(tmp_path):
    """Permission stated for market_yield in file 1 must not excuse a same-named column elsewhere."""
    import pandas as pd

    instruction = _BULLET_FORM_INSTRUCTION.replace(
        "Columns: `tau1, tau2, forward_rate`", "Columns: `tau1, market_yield`"
    )
    _bond_frame().to_csv(tmp_path / "bond_prices.csv", index=False)
    pd.DataFrame({"tau1": [0.25], "market_yield": [float("nan")]}).to_csv(
        tmp_path / "forward_rates.csv", index=False
    )

    result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())

    assert _nan_outcome(result, "bond_prices.csv").status == "pass"
    assert _nan_outcome(result, "forward_rates.csv").status == "fail"


def test_nan_permission_stated_as_an_inline_sentence_and_in_a_table_row(tmp_path):
    import pandas as pd

    nan = float("nan")
    sentence = """\
## Deliverables

- **`fits.csv`**: Columns: `copula`, `param1`, `param2`, `loglik`. For copulas with one parameter, set `param2` to NaN.
"""
    pd.DataFrame({"copula": ["a", "b"], "param1": [0.5, 0.6], "param2": [nan, 6.0], "loglik": [1.0, 2.0]}).to_csv(
        tmp_path / "fits.csv", index=False
    )
    assert self_check(_bundle(sentence), [], [], tmp_path, _ok_result()).passed

    pd.DataFrame({"copula": ["a", "b"], "param1": [nan, 0.6], "param2": [nan, 6.0], "loglik": [1.0, 2.0]}).to_csv(
        tmp_path / "fits.csv", index=False
    )
    unexpected = self_check(_bundle(sentence), [], [], tmp_path, _ok_result())
    assert not unexpected.passed and "param1" in _nan_outcome(unexpected, "fits.csv").detail

    table = """\
## Deliverables

### `/output/ic.csv`

| Column | Type | Unit / Convention |
|---|---|---|
| `date` | `str` | ISO date |
| `ic` | `float64` | NaN when the correlation is undefined |
"""
    pd.DataFrame({"date": ["d1", "d2"], "ic": [nan, 0.1]}).to_csv(tmp_path / "ic.csv", index=False)
    assert self_check(_bundle(table), [], [], tmp_path, _ok_result()).passed


def test_nan_in_json_output_is_allowed_only_for_the_named_key(tmp_path):
    import json

    instruction = """\
## Output

Save to `/app/output/results.json`:

- ratio: float, or NaN if the denominator is zero
- total: float
"""
    (tmp_path / "results.json").write_text(json.dumps({"ratio": float("nan"), "total": 1.0}))
    assert self_check(_bundle(instruction), [], [], tmp_path, _ok_result()).passed

    (tmp_path / "results.json").write_text(json.dumps({"ratio": 0.5, "total": float("nan")}))
    result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
    assert not result.passed and "total" in _nan_outcome(result, "results.json").detail


def test_text_that_does_not_grant_nan_permission_does_not_weaken_the_check(tmp_path):
    """Forbidding text, prose that names no column, and statements in INPUT sections are not
    permission for NaN in an output column."""
    import pandas as pd

    nan = float("nan")
    cases = {
        "forbidden": "- score: finite float (not NaN)\n- other: float",
        "must_use_null": "- score: MUST use JSON null for unavailable values (not 0, not NaN, not omitted)\n- other: float",
        "no_column_named": "Non-applicable methods should return NaN.\n- score: float\n- other: float",
        "no_nan_allowed": "- score: float; no NaN allowed\n- other: float",
        "when_input_is_nan": "- score: float; if the input is NaN, drop the row\n- other: float",
    }
    pd.DataFrame({"score": [nan, 1.0], "other": [1.0, 2.0]}).to_csv(tmp_path / "out.csv", index=False)
    for name, bullets in cases.items():
        instruction = f"## Output\n\nSave to `/app/output/out.csv`:\n\n{bullets}\n"
        result = self_check(_bundle(instruction), [], [], tmp_path, _ok_result())
        assert _nan_outcome(result, "out.csv").status == "fail", name

    in_input_section = (
        "## Inputs\n\n- score: float, NaN if missing\n\n"
        "## Output\n\nSave to `/app/output/out.csv`:\n\n- score: float\n- other: float\n"
    )
    result = self_check(_bundle(in_input_section), [], [], tmp_path, _ok_result())
    assert _nan_outcome(result, "out.csv").status == "fail"


def test_inf_is_never_excused_by_a_nan_permission(tmp_path):
    _bond_frame(market_yield=[float("inf"), 0.037, 0.038]).to_csv(tmp_path / "bond_prices.csv", index=False)
    _write_forward_rates(tmp_path)

    result = self_check(_bundle(_BULLET_FORM_INSTRUCTION), [], [], tmp_path, _ok_result())

    outcome = _nan_outcome(result, "bond_prices.csv")
    assert outcome.status == "fail" and "market_yield" in outcome.detail


def test_real_units_that_prescribe_nan_accept_a_correct_output(tmp_path):
    """The two real units whose correct outputs necessarily contain NaN, using their actual
    instruction.md (read-only). Skipped where the units are not present, e.g. inside the image."""
    import json

    import pandas as pd
    import pytest

    units = Path(__file__).resolve().parents[2] / "units"
    if not (units / "t1-cir-bond-pricing" / "instruction.md").is_file():
        pytest.skip("units/ not present in this environment")
    nan = float("nan")

    # t1-cir-bond-pricing: market_yield / yield_diff are NaN for maturities with no DGS column.
    cir_dir = tmp_path / "cir"
    cir_dir.mkdir()
    taus = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    pd.DataFrame(
        {
            "tau": taus,
            "zcb_price": [0.99] * 8,
            "zero_rate": [0.01] * 8,
            "model_yield": [0.04] * 8,
            "market_yield": [nan, nan, 0.037, 0.038, nan, 0.040, nan, 0.043],
            "yield_diff": [nan, nan, 0.003, 0.002, nan, 0.000, nan, -0.003],
        }
    ).to_csv(cir_dir / "bond_prices.csv", index=False)
    pd.DataFrame({"tau1": taus[:-1], "tau2": taus[1:], "forward_rate": [0.04] * 7}).to_csv(
        cir_dir / "forward_rates.csv", index=False
    )
    calibration_keys = "n_obs kappa theta sigma log_likelihood feller_condition_satisfied feller_ratio r_current r_mean r_std r_min r_max"
    summary_keys = "kappa theta sigma r_current feller_satisfied mean_yield_error max_yield_error yield_curve_upward_sloping"
    (cir_dir / "calibration.json").write_text(json.dumps({k: 1.0 for k in calibration_keys.split()}))
    (cir_dir / "summary.json").write_text(json.dumps({k: 1.0 for k in summary_keys.split()}))
    cir_text = (units / "t1-cir-bond-pricing" / "instruction.md").read_text(encoding="utf-8")
    result = self_check(_bundle(cir_text), [], [], cir_dir, _ok_result())
    assert result.passed, [o for o in result.outcomes if o.status == "fail"]

    # ...while an unexpected NaN in a column the instruction does not excuse is still caught.
    bad = pd.read_csv(cir_dir / "bond_prices.csv")
    bad.loc[2, "zcb_price"] = nan
    bad.to_csv(cir_dir / "bond_prices.csv", index=False)
    result = self_check(_bundle(cir_text), [], [], cir_dir, _ok_result())
    assert not result.passed and "zcb_price" in _nan_outcome(result, "bond_prices.csv").detail

    # t1-copula-equity-fitting: param2 is NaN for one-parameter copulas.
    copula_dir = tmp_path / "copula"
    copula_dir.mkdir()
    rows = [
        {"stock1": f"S{i}", "stock2": f"S{j}", "copula": cop, "param1": 0.5,
         "param2": 6.0 if cop == "student_t" else nan, "loglik": 10.0, "aic": -18.0, "bic": -15.0}
        for i in range(5) for j in range(i + 1, 5) for cop in ("gaussian", "student_t", "clayton", "gumbel")
    ]
    pd.DataFrame(rows).to_csv(copula_dir / "copula_fits.csv", index=False)
    copula_text = (units / "t1-copula-equity-fitting" / "instruction.md").read_text(encoding="utf-8")
    from agent.self_check import _extract_output_filenames

    for name in _extract_output_filenames(copula_text) - {"copula_fits.csv"}:
        (copula_dir / name).write_text(json.dumps({"x": 1.0}) if name.endswith(".json") else "x\n1\n")
    result = self_check(_bundle(copula_text), [], [], copula_dir, _ok_result())
    assert _nan_outcome(result, "copula_fits.csv").status == "pass"

    rows[0]["loglik"] = nan
    pd.DataFrame(rows).to_csv(copula_dir / "copula_fits.csv", index=False)
    result = self_check(_bundle(copula_text), [], [], copula_dir, _ok_result())
    assert _nan_outcome(result, "copula_fits.csv").status == "fail"
    assert "loglik" in _nan_outcome(result, "copula_fits.csv").detail


def test_a_hyphenated_word_is_not_mistaken_for_a_bullet_about_a_column():
    """"- ATM-only methods should have NaN ..." is prose about methods, not a bullet whose leading
    identifier ("ATM") is an output column; it must grant nothing."""
    from agent.self_check import _extract_nan_allowances

    instruction = (
        "## Output\n\nSave to `/app/output/approx.csv`:\n\n"
        "- ATM-only methods should have NaN for rows where moneyness is not 1.00\n"
        "- sigma: float, NaN if the method is not applicable\n"
    )
    per_file, unattributed = _extract_nan_allowances(instruction, {"approx.csv"})
    assert per_file == {"approx.csv": {"sigma"}} and unattributed == set()
