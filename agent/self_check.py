"""Generic structural self-check.

Only ever looks at what the solver itself already has: this task's own instruction.md, the
resolved input files, and whatever the generated code actually wrote under out_dir. Never reads
checks/, checks/reference_data/, or any frozen TaskContract JSON.

Every instruction-text heuristic below is fixed and task-agnostic: the same code runs for every
task, and none of it branches on a task id or hardcodes a task's own column/key names. When a
heuristic can't derive anything concrete from a given task's text, the corresponding check
reports "not_applicable" rather than a false "fail".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.executor import ExecutionResult
from agent.input_resolver import ResolvedInput
from agent.task_reader import TaskBundle

# Written by the sealed checker (checks/test.sh), never by the solver -- a fixed,
# competition-wide convention (see qfbench2_track_coding.scoring's own g1_schema gate), not a
# per-task fact. Must never be treated as a required solver deliverable.
_NEVER_SOLVER_OUTPUTS = {"reward.json", "pytest_report.json"}

_NUMERIC_EPSILON = 1e-6


@dataclass
class CheckOutcome:
    name: str
    status: str  # "pass" | "fail" | "not_applicable"
    detail: str


@dataclass
class SelfCheckResult:
    passed: bool
    outcomes: list[CheckOutcome]


# ---------------------------------------------------------------------------
# Instruction-text parsing (fixed, generic heuristics -- see module docstring)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.*)$")
_PATH_TOKEN_RE = re.compile(r"[\w./\-]+\.(?:json|csv|parquet|html|txt|md)", re.IGNORECASE)
_OUTPUT_ROOT_PATH_RE = re.compile(
    r"/(?:app/)?output/[\w./\-]+\.(?:json|csv|parquet|html|txt|md)", re.IGNORECASE
)
_TABLE_ROW_RE = re.compile(r"(?m)^\|(.+)\|[ \t]*$")
_TABLE_SEPARATOR_RE = re.compile(r"^[\s|:-]+$")
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ROW_CORRESPONDENCE_RE = re.compile(
    r"one row (?:per|for each)|same number of rows|preserving[^.\n]{0,30}order", re.IGNORECASE
)
_RANGE_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_GE_ZERO_RE = re.compile(r">=\s*0|≥\s*0|non-negative", re.IGNORECASE)
_GT_ZERO_RE = re.compile(r"(?<!>)>\s*0\b|(?<!non-)\bpositive\b", re.IGNORECASE)


def _section_spans(text: str) -> list[tuple[int, str, str]]:
    """Every heading's (level, heading_text, subtree_text) -- subtree runs until the next
    heading at the same level or shallower, so a qualifying heading's nested sub-headings and
    their bodies are included verbatim."""
    matches = list(_HEADING_RE.finditer(text))
    spans = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2)
        start = match.end()
        end = len(text)
        for later in matches[i + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        spans.append((level, heading_text, text[start:end]))
    return spans


def _output_sections(instruction_text: str) -> list[str]:
    """Text of every heading (any level) whose own title names output/deliverables and does not
    name input -- e.g. "## Deliverables", "## Output Files", "### Output: `results.json`". A
    nested sub-heading is captured through its qualifying ancestor's subtree text; an
    independently-qualifying nested heading (e.g. "### Output: `x.json`" under a neutral parent)
    is also captured on its own, correctly bounded to just its own subtree."""
    sections = []
    for _level, heading_text, body in _section_spans(instruction_text):
        lower = heading_text.lower()
        if ("output" in lower or "deliverable" in lower) and "input" not in lower:
            sections.append(heading_text + "\n" + body)
    return sections


def _extract_output_filenames(instruction_text: str) -> set[str]:
    filenames: set[str] = set()

    # Unambiguous anywhere in the document: a path literally rooted at /output/ or /app/output/.
    for match in _OUTPUT_ROOT_PATH_RE.finditer(instruction_text):
        filenames.add(Path(match.group(0)).name)

    # Section-aware: any path-like token inside a heading that itself names output/deliverables,
    # even when the path itself has no "output" substring (e.g. "## Output: `/app/curve.json`").
    for section in _output_sections(instruction_text):
        for match in _PATH_TOKEN_RE.finditer(section):
            filenames.add(Path(match.group(0)).name)

    return filenames - _NEVER_SOLVER_OUTPUTS


def _table_data_rows(text: str) -> list[list[str]]:
    """Cell lists for every DATA row of every markdown table in `text` -- skips each table's own
    header row and its `|---|---|` separator row, using the separator as the marker (a table
    with no separator is skipped entirely: not a table this parser can trust)."""
    lines = text.splitlines()
    data_rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]) and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(
            _TABLE_ROW_RE.match(lines[i + 1]).group(1) if _TABLE_ROW_RE.match(lines[i + 1]) else ""
        ):
            j = i + 2
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                cells = [c.strip() for c in _TABLE_ROW_RE.match(lines[j]).group(1).split("|")]
                data_rows.append(cells)
                j += 1
            i = j
        else:
            i += 1
    return data_rows


def _extract_schema_rows(instruction_text: str) -> list[tuple[str, str]]:
    """(key, convention_text) pairs from markdown tables inside output-labeled sections only --
    never from an input schema table, so e.g. an options.parquet input table's S/K/T/r/sigma
    columns never leak in as "expected output keys". Deduplicated by key (first occurrence wins)
    since a nested heading (e.g. "### `/output/x.json`" under "## Deliverables") can
    independently qualify as its own output section, overlapping its parent's."""
    seen: dict[str, str] = {}
    for section in _output_sections(instruction_text):
        for cells in _table_data_rows(section):
            if len(cells) < 2:
                continue
            key = cells[0].strip("` ")
            if not key or not re.match(r"^[A-Za-z_][\w.]*$", key):
                continue  # not a plausible identifier
            seen.setdefault(key, cells[-1])
    return list(seen.items())


def _extract_json_example_keys(instruction_text: str) -> set[str]:
    keys: set[str] = set()
    for section in _output_sections(instruction_text):
        for match in _JSON_FENCE_RE.finditer(section):
            try:
                example = json.loads(_strip_json_placeholders(match.group(1)))
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(example, dict):
                keys.update(example.keys())
                for value in example.values():
                    if isinstance(value, dict):
                        keys.update(value.keys())
    return keys


def _strip_json_placeholders(text: str) -> str:
    """instruction.md JSON examples use non-JSON placeholders like <float> or 0.0XXXXXX; replace
    common placeholder shapes with a valid JSON literal so the example still parses."""
    text = re.sub(r"<[^>]*>", "0", text)
    text = re.sub(r"\bNaN\b", "0", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # trailing commas before a closing brace/bracket
    text = re.sub(r"\.\.\.", "", text)
    return text


# ---------------------------------------------------------------------------
# Output inspection (fully generic -- never reads instruction_text)
# ---------------------------------------------------------------------------


def _try_parse(path: Path) -> tuple[Any, str | None]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            import pandas as pd

            return pd.read_csv(path), None
        if suffix == ".parquet":
            import pandas as pd

            return pd.read_parquet(path), None
        if suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8")), None
        return path.read_text(encoding="utf-8", errors="replace"), None
    except Exception as exc:  # noqa: BLE001 - any parse failure is a legitimate check result
        return None, f"{path.name} could not be parsed as {suffix or 'text'}: {exc}"


def _find_nan_inf(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, float):
        if math.isnan(value):
            hits.append(f"{path} is NaN")
        elif math.isinf(value):
            hits.append(f"{path} is Inf")
    elif isinstance(value, dict):
        for key, sub in value.items():
            hits.extend(_find_nan_inf(sub, f"{path}.{key}"))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            hits.extend(_find_nan_inf(sub, f"{path}[{i}]"))
    return hits


def _check_nan_inf_empty(value: Any) -> str | None:
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            if len(value) == 0:
                return "output table has zero rows"
            numeric = value.select_dtypes(include="number")
            nan_cols = [c for c in numeric.columns if numeric[c].isna().any()]
            inf_cols = [c for c in numeric.columns if not pd.Series.isna(numeric[c]).all() and (numeric[c].abs() == float("inf")).any()]
            if nan_cols or inf_cols:
                return f"NaN in columns {nan_cols}; Inf in columns {inf_cols}"
            return None
    except ImportError:
        pass

    if isinstance(value, dict):
        if not value:
            return "output object is empty"
        hits = _find_nan_inf(value)
        return "; ".join(hits) if hits else None
    if isinstance(value, list):
        if not value:
            return "output list is empty"
        hits = _find_nan_inf(value)
        return "; ".join(hits) if hits else None
    return None  # plain text or unrecognized shape: nothing generic to check


def _actual_keys(value: Any) -> set[str]:
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return set(value.columns)
    except ImportError:
        pass
    if isinstance(value, dict):
        keys = set(value.keys())
        for sub in value.values():
            if isinstance(sub, dict):
                keys.update(sub.keys())
        return keys
    return set()


def _check_keys_present(value: Any, expected_keys: set[str]) -> tuple[set[str], set[str]]:
    actual = _actual_keys(value)
    return expected_keys - actual, expected_keys & actual


def _lookup_values(value: Any, key: str) -> list[Any]:
    """Numeric value(s) for `key`: a CSV column's values, a top-level scalar, values one level
    into a nested dict, or a scalar found inside a list of records."""
    found: list[Any] = []
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            if key in value.columns:
                found.extend(value[key].tolist())
            return [v for v in found if isinstance(v, (int, float))]
    except ImportError:
        pass

    if isinstance(value, dict):
        if key in value and isinstance(value[key], (int, float)):
            found.append(value[key])
        for sub in value.values():
            if isinstance(sub, dict) and key in sub and isinstance(sub[key], (int, float)):
                found.append(sub[key])
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and key in item and isinstance(item[key], (int, float)):
                found.append(item[key])
    return found


def _check_sign_range_constraints(value: Any, schema_rows: list[tuple[str, str]]) -> list[str]:
    violations: list[str] = []
    for key, convention_text in schema_rows:
        numbers = _lookup_values(value, key)
        if not numbers:
            continue

        range_match = _RANGE_RE.search(convention_text)
        if range_match:
            lo, hi = float(range_match.group(1)), float(range_match.group(2))
            bad = [n for n in numbers if not (lo - _NUMERIC_EPSILON <= n <= hi + _NUMERIC_EPSILON)]
            if bad:
                violations.append(f"{key} stated as in ({lo}, {hi}) but found {bad[:5]}")
            continue

        if _GE_ZERO_RE.search(convention_text):
            bad = [n for n in numbers if n < -_NUMERIC_EPSILON]
            if bad:
                violations.append(f"{key} stated as >= 0 but found {bad[:5]}")
            continue

        if _GT_ZERO_RE.search(convention_text):
            bad = [n for n in numbers if n <= _NUMERIC_EPSILON]
            if bad:
                violations.append(f"{key} stated as > 0 but found {bad[:5]}")
            continue

    return violations


def _check_row_count_parity(
    instruction_text: str, summaries: list[dict[str, Any]], parsed_outputs: dict[str, Any]
) -> CheckOutcome:
    if not _ROW_CORRESPONDENCE_RE.search(instruction_text):
        return CheckOutcome(
            "row_count_parity", "not_applicable", "instruction states no row-correspondence property"
        )

    tabular_input_counts = [s["n_rows"] for s in summaries if "n_rows" in s]
    if len(tabular_input_counts) != 1:
        return CheckOutcome(
            "row_count_parity", "not_applicable",
            f"expected exactly one tabular input to compare against, found {len(tabular_input_counts)}",
        )

    try:
        import pandas as pd

        tabular_outputs = [v for v in parsed_outputs.values() if isinstance(v, pd.DataFrame)]
    except ImportError:
        tabular_outputs = []

    if len(tabular_outputs) != 1:
        return CheckOutcome(
            "row_count_parity", "not_applicable",
            f"expected exactly one tabular output to compare against, found {len(tabular_outputs)}",
        )

    input_rows = tabular_input_counts[0]
    output_rows = len(tabular_outputs[0])
    if input_rows != output_rows:
        return CheckOutcome(
            "row_count_parity", "fail",
            f"instruction implies row correspondence with the input; input has {input_rows} rows, "
            f"output has {output_rows}",
        )
    return CheckOutcome("row_count_parity", "pass", f"input and output both have {input_rows} rows")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def self_check(
    bundle: TaskBundle,
    resolved_inputs: list[ResolvedInput],
    summaries: list[dict[str, Any]],
    out_dir: str | Path,
    execution_result: ExecutionResult,
) -> SelfCheckResult:
    del resolved_inputs  # kept in the signature for symmetry/future use; row-parity uses summaries
    out_dir = Path(out_dir)
    outcomes: list[CheckOutcome] = []

    if execution_result.timed_out:
        outcomes.append(CheckOutcome("execution", "fail", "generated code timed out"))
        return SelfCheckResult(False, outcomes)
    if execution_result.exit_code != 0:
        outcomes.append(
            CheckOutcome(
                "execution",
                "fail",
                f"generated code exited with code {execution_result.exit_code}; "
                f"stderr: {execution_result.stderr[-2000:]}",
            )
        )
        return SelfCheckResult(False, outcomes)
    outcomes.append(CheckOutcome("execution", "pass", "generated code exited 0"))

    existing_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    candidate_filenames = _extract_output_filenames(bundle.instruction_text)

    if candidate_filenames:
        missing = sorted(candidate_filenames - set(existing_files))
        if missing:
            outcomes.append(
                CheckOutcome(
                    "output_files_exist",
                    "fail",
                    f"instruction names {sorted(candidate_filenames)}; missing: {missing}; "
                    f"out_dir contains: {existing_files}",
                )
            )
        else:
            outcomes.append(
                CheckOutcome(
                    "output_files_exist",
                    "pass",
                    f"all instruction-named files present: {sorted(candidate_filenames)}",
                )
            )
    elif not existing_files:
        outcomes.append(
            CheckOutcome("output_files_exist", "fail", "out_dir is empty; no deliverable was written")
        )
    else:
        outcomes.append(
            CheckOutcome(
                "output_files_exist",
                "not_applicable",
                f"could not derive expected filenames from instruction.md; out_dir contains: {existing_files}",
            )
        )

    schema_rows = _extract_schema_rows(bundle.instruction_text)
    json_keys = _extract_json_example_keys(bundle.instruction_text)
    expected_keys = json_keys | {key for key, _ in schema_rows}

    files_to_inspect = [f for f in existing_files if not candidate_filenames or f in candidate_filenames]
    if not files_to_inspect:
        files_to_inspect = existing_files

    parsed_outputs: dict[str, Any] = {}
    for filename in files_to_inspect:
        value, error = _try_parse(out_dir / filename)
        if error:
            outcomes.append(CheckOutcome(f"readable:{filename}", "fail", error))
            continue
        outcomes.append(CheckOutcome(f"readable:{filename}", "pass", f"parsed as {type(value).__name__}"))
        parsed_outputs[filename] = value

        nan_issue = _check_nan_inf_empty(value)
        if nan_issue is None:
            outcomes.append(CheckOutcome(f"no_nan_inf_empty:{filename}", "pass", "no NaN/Inf/empty detected"))
        else:
            outcomes.append(CheckOutcome(f"no_nan_inf_empty:{filename}", "fail", nan_issue))

        if expected_keys:
            missing_keys, present_keys = _check_keys_present(value, expected_keys)
            if missing_keys:
                outcomes.append(
                    CheckOutcome(
                        f"required_keys:{filename}",
                        "fail",
                        f"instruction-derived keys/columns missing: {sorted(missing_keys)}; "
                        f"present: {sorted(present_keys)}",
                    )
                )
            else:
                outcomes.append(
                    CheckOutcome(
                        f"required_keys:{filename}",
                        "pass",
                        f"all instruction-derived keys present: {sorted(present_keys)}",
                    )
                )
        else:
            outcomes.append(
                CheckOutcome(
                    f"required_keys:{filename}", "not_applicable", "no keys/columns derivable from instruction.md"
                )
            )

        if schema_rows:
            violations = _check_sign_range_constraints(value, schema_rows)
            if violations:
                outcomes.append(CheckOutcome(f"sign_range:{filename}", "fail", "; ".join(violations)))
            else:
                outcomes.append(
                    CheckOutcome(f"sign_range:{filename}", "pass", "no stated sign/range constraints violated")
                )
        else:
            outcomes.append(
                CheckOutcome(
                    f"sign_range:{filename}", "not_applicable", "no sign/range convention text found in instruction.md"
                )
            )

    outcomes.append(_check_row_count_parity(bundle.instruction_text, summaries, parsed_outputs))

    passed = not any(o.status == "fail" for o in outcomes)
    return SelfCheckResult(passed, outcomes)
