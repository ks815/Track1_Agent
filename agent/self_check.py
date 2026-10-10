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
from dataclasses import dataclass, field
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
# Any "(a, b)" interval, also with a Unicode minus/en dash as the sign: _RANGE_RE above only
# understands ASCII "-", so it silently misses "(−1, 0)". Used only to COUNT intervals in a cell.
_ANY_INTERVAL_RE = re.compile(r"\(\s*[-−–]?\d+(?:\.\d+)?\s*,\s*[-−–]?\d+(?:\.\d+)?\s*\)")
# A qualifier that scopes a convention to a subset of rows ("for calls", "if long", "when ...").
# "for all/every/each/both/any" is NOT a qualifier -- it states the rule holds everywhere.
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|when|whenever|where|unless|except|otherwise|depending|only)\b"
    r"|\bfor\s+(?!(?:all|every|each|both|any)\b)",
    re.IGNORECASE,
)


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


def _tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """(text preceding the table, DATA-row cell lists) for every markdown table in `text` --
    skips each table's own header row and its `|---|---|` separator row, using the separator as
    the marker (a table with no separator is skipped entirely: not a table this parser can
    trust). The preceding text lets a caller work out which file a table describes."""
    lines = text.splitlines()
    tables: list[tuple[str, list[list[str]]]] = []
    i = 0
    while i < len(lines):
        if _TABLE_ROW_RE.match(lines[i]) and i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(
            _TABLE_ROW_RE.match(lines[i + 1]).group(1) if _TABLE_ROW_RE.match(lines[i + 1]) else ""
        ):
            data_rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                cells = [c.strip() for c in _TABLE_ROW_RE.match(lines[j]).group(1).split("|")]
                data_rows.append(cells)
                j += 1
            tables.append(("\n".join(lines[:i]), data_rows))
            i = j
        else:
            i += 1
    return tables


def _schema_pairs(data_rows: list[list[str]]) -> list[tuple[str, str]]:
    """(key, convention_text) for each table data row whose first cell is a plausible identifier."""
    pairs = []
    for cells in data_rows:
        if len(cells) < 2:
            continue
        key = cells[0].strip("` ")
        if not key or not re.match(r"^[A-Za-z_][\w.]*$", key):
            continue  # not a plausible identifier
        pairs.append((key, cells[-1]))
    return pairs


def _json_fence_keys(fence_body: str) -> set[str]:
    try:
        example = json.loads(_strip_json_placeholders(fence_body))
    except (json.JSONDecodeError, ValueError):
        return set()
    keys: set[str] = set()
    if isinstance(example, dict):
        keys.update(example.keys())
        for value in example.values():
            if isinstance(value, dict):
                keys.update(value.keys())
    return keys


def _extract_schema_rows(instruction_text: str) -> list[tuple[str, str]]:
    """(key, convention_text) pairs from markdown tables inside output-labeled sections only --
    never from an input schema table, so e.g. an options.parquet input table's S/K/T/r/sigma
    columns never leak in as "expected output keys". Deduplicated by key (first occurrence wins)
    since a nested heading (e.g. "### `/output/x.json`" under "## Deliverables") can
    independently qualify as its own output section, overlapping its parent's.

    Task-wide view: it does not say which file a key belongs to. self_check itself uses
    _extract_expectations_by_file for that."""
    seen: dict[str, str] = {}
    for section in _output_sections(instruction_text):
        for _preceding, data_rows in _tables(section):
            for key, convention in _schema_pairs(data_rows):
                seen.setdefault(key, convention)
    return list(seen.items())


def _extract_json_example_keys(instruction_text: str) -> set[str]:
    """Task-wide view (see _extract_schema_rows): every fenced-JSON example key, any file."""
    keys: set[str] = set()
    for section in _output_sections(instruction_text):
        for match in _JSON_FENCE_RE.finditer(section):
            keys.update(_json_fence_keys(match.group(1)))
    return keys


@dataclass
class _FileExpectations:
    """What the instruction states about ONE output file's keys/columns and value conventions."""

    json_keys: set[str] = field(default_factory=set)
    schema_rows: dict[str, str] = field(default_factory=dict)  # key -> convention text

    @property
    def keys(self) -> set[str]:
        return self.json_keys | set(self.schema_rows)

    def merge(self, other: "_FileExpectations") -> None:
        self.json_keys |= other.json_keys
        for key, convention in other.schema_rows.items():
            self.schema_rows.setdefault(key, convention)


# A phrase that PERMITS a NaN value, each with the NaN token as group 1: "NaN if not available",
# "may be NaN", "set to NaN" (markdown emphasis/backticks are stripped before matching).
_NAN_PERMIT_PATTERNS = [
    re.compile(
        r"\b(NaN)\b\s*\)?\s*(?:if|when|where|whenever|for|otherwise|unless|allowed|permitted|acceptable)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:may|can|could|might|should|must|will|shall)\s+(?:be|contain|have|return)\s+(?:an?\s+)?(NaN)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:set|returned|assigned|filled|left|reported)\s+(?:\w+\s+){0,2}?(?:to|as|with)\s+(NaN)\b",
        re.IGNORECASE,
    ),
]
# Any of these shortly BEFORE a NaN means the sentence forbids it ("no NaN", "not NaN").
_NAN_NEGATOR_RE = re.compile(
    r"\b(?:no|not|never|nor|neither|without|forbid\w*|disallow\w*|avoid|don't|cannot|can't)\b", re.IGNORECASE
)
_NAN_NEGATOR_WINDOW = 40
# "- market_yield: ...", "- **`ic`** — ..." -- a bullet whose leading identifier names a column/key.
# The separator is a colon or a dash set off by whitespace, so "- ATM-only methods ..." is prose,
# not a bullet about a column called "ATM".
_BULLET_LEAD_RE = re.compile(
    r"^\s*(?:[-*+]|\d+[.)])\s+(?:\*\*)?`?([A-Za-z_][\w.]*)`?(?:\*\*)?\s*(?::|\s[—–-])(?:\s+|$)(.*)$"
)
# "set `param2` to NaN" -- the column is named right in the instruction sentence.
_SET_COLUMN_TO_NAN_RE = re.compile(
    r"\b(?:set|assign|fill|leave)\s+`?([A-Za-z_][\w.]*)`?\s+(?:to|as|with)\s+`?NaN\b", re.IGNORECASE
)
_FILENAME_SUFFIXES = {".json", ".csv", ".parquet", ".html", ".txt", ".md"}


def _states_nan_is_allowed(text: str) -> bool:
    """True if `text` (one bullet/row/sentence) says NaN is an acceptable value, and does not
    forbid it. Markdown emphasis and backticks are ignored."""
    plain = text.replace("`", "").replace("*", "")
    for pattern in _NAN_PERMIT_PATTERNS:
        for match in pattern.finditer(plain):
            before = plain[max(0, match.start(1) - _NAN_NEGATOR_WINDOW) : match.start(1)]
            if not _NAN_NEGATOR_RE.search(before):
                return True
    return False


def _extract_nan_allowances(
    instruction_text: str, candidate_filenames: set[str]
) -> tuple[dict[str, set[str]], set[str]]:
    """Columns/keys the instruction explicitly says may be NaN, per output file. Returns
    (allowed names keyed by filename, allowed names not attributable to any file).

    Deliberately narrow, so the NaN check is not weakened: only text inside an OUTPUT section
    counts, and it must name the column -- as a bullet's leading identifier ("- market_yield: ...,
    NaN if not available"), as a table row's first cell, or as "set `param2` to NaN". Prose that
    names no column ("non-applicable methods return NaN"), text that forbids NaN ("not NaN"), and
    anything in an input section grants nothing."""
    per_file: dict[str, set[str]] = {}
    unattributed: set[str] = set()

    def grant(column: str, preceding_text: str) -> None:
        owner = _last_named_output_file(preceding_text, candidate_filenames)
        (per_file.setdefault(owner, set()) if owner else unattributed).add(column)

    for section in _output_sections(instruction_text):
        offset = 0
        for line in section.splitlines(keepends=True):
            lead = _BULLET_LEAD_RE.match(line.rstrip("\r\n"))
            if lead and Path(lead.group(1)).suffix.lower() not in _FILENAME_SUFFIXES:
                if _states_nan_is_allowed(lead.group(2)):
                    grant(lead.group(1), section[:offset])
            offset += len(line)

        for match in _SET_COLUMN_TO_NAN_RE.finditer(section):
            before = section[max(0, match.start() - _NAN_NEGATOR_WINDOW) : match.start()]
            if not _NAN_NEGATOR_RE.search(before):
                grant(match.group(1), section[: match.start()])

        for preceding, data_rows in _tables(section):
            for cells in data_rows:
                key = cells[0].strip("` ") if cells else ""
                if re.match(r"^[A-Za-z_][\w.]*$", key) and _states_nan_is_allowed(" ".join(cells[1:])):
                    grant(key, preceding)

    return per_file, unattributed


def _last_named_output_file(preceding_text: str, candidate_filenames: set[str]) -> str | None:
    """The deliverable filename mentioned closest before a fence/table, or None. A fence or table
    is taken to describe the file the text most recently named -- in the heading ("### File 2:
    `/app/output/x.csv`", "### Output: `x.csv`") or in a "Save to `.../x.csv`" line."""
    owner = None
    for match in _PATH_TOKEN_RE.finditer(preceding_text):
        name = Path(match.group(0)).name
        if name in candidate_filenames:
            owner = name
    return owner


def _extract_expectations_by_file(
    instruction_text: str, candidate_filenames: set[str]
) -> tuple[dict[str, _FileExpectations], _FileExpectations]:
    """Key/column/convention expectations, attributed per output file by position in the
    instruction text. Returns (expectations keyed by filename, expectations that could not be
    attributed to any file).

    Without this, the keys of every file in a multi-output task get required of every file --
    e.g. a calibration.json's parameter names demanded as columns of a CSV, or of an HTML
    deliverable. A fenced JSON example can only describe a .json file; a fence whose nearest
    preceding filename is not a .json file is treated as unattributed."""
    per_file: dict[str, _FileExpectations] = {}
    unattributed = _FileExpectations()

    def expectations_for(owner: str | None) -> _FileExpectations:
        return per_file.setdefault(owner, _FileExpectations()) if owner else unattributed

    for section in _output_sections(instruction_text):
        for match in _JSON_FENCE_RE.finditer(section):
            keys = _json_fence_keys(match.group(1))
            if not keys:
                continue
            owner = _last_named_output_file(section[: match.start()], candidate_filenames)
            if owner is not None and not owner.lower().endswith(".json"):
                owner = None
            expectations_for(owner).json_keys |= keys
        for preceding, data_rows in _tables(section):
            owner = _last_named_output_file(preceding, candidate_filenames)
            for key, convention in _schema_pairs(data_rows):
                expectations_for(owner).schema_rows.setdefault(key, convention)

    return per_file, unattributed


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


def _find_nan_inf(
    value: Any, path: str = "$", allowed_nan: frozenset[str] = frozenset(), key: str | None = None
) -> list[str]:
    """`key` is the nearest enclosing object key; a NaN under a key the instruction explicitly
    allows to be NaN is not reported. Inf is always reported."""
    hits: list[str] = []
    if isinstance(value, float):
        if math.isnan(value):
            if key not in allowed_nan:
                hits.append(f"{path} is NaN")
        elif math.isinf(value):
            hits.append(f"{path} is Inf")
    elif isinstance(value, dict):
        for sub_key, sub in value.items():
            hits.extend(_find_nan_inf(sub, f"{path}.{sub_key}", allowed_nan, sub_key))
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            hits.extend(_find_nan_inf(sub, f"{path}[{i}]", allowed_nan, key))
    return hits


def _check_nan_inf_empty(value: Any, allowed_nan: set[str] | frozenset[str] = frozenset()) -> str | None:
    """`allowed_nan`: columns/keys the instruction explicitly says may be NaN (see
    _extract_nan_allowances). Everything else being NaN, and any Inf, is still reported."""
    allowed = frozenset(allowed_nan)
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            if len(value) == 0:
                return "output table has zero rows"
            numeric = value.select_dtypes(include="number")
            nan_cols = [c for c in numeric.columns if numeric[c].isna().any() and c not in allowed]
            inf_cols = [c for c in numeric.columns if not pd.Series.isna(numeric[c]).all() and (numeric[c].abs() == float("inf")).any()]
            if nan_cols or inf_cols:
                return f"NaN in columns {nan_cols}; Inf in columns {inf_cols}"
            return None
    except ImportError:
        pass

    if isinstance(value, dict):
        if not value:
            return "output object is empty"
        hits = _find_nan_inf(value, allowed_nan=allowed)
        return "; ".join(hits) if hits else None
    if isinstance(value, list):
        if not value:
            return "output list is empty"
        hits = _find_nan_inf(value, allowed_nan=allowed)
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


def _is_conditional_convention(convention_text: str) -> bool:
    """True when a convention cell cannot be read as one rule for every value in the column: it
    states several intervals ("(0, 1) for calls, (-1, 0) for puts") or scopes a rule to a subset
    ("for calls", "if long"). Checking such a cell against the whole column would flag correct
    output, so the check is skipped -- fail safe, like every other heuristic in this module."""
    return len(_ANY_INTERVAL_RE.findall(convention_text)) > 1 or bool(
        _CONDITIONAL_RE.search(convention_text)
    )


def _check_sign_range_constraints(value: Any, schema_rows: list[tuple[str, str]]) -> list[str]:
    violations: list[str] = []
    for key, convention_text in schema_rows:
        numbers = _lookup_values(value, key)
        if not numbers:
            continue
        if _is_conditional_convention(convention_text):
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
            # A strictly positive value satisfies "> 0" however small it is (1e-205 is a normal
            # deep-out-of-the-money gamma), so the tolerance must not turn tiny positives into
            # violations. What is flagged: values clearly below zero, and a column in which
            # nothing is positive at all. A stray exact 0.0 among positive values is underflow,
            # indistinguishable from a bug at this level, so it is not flagged.
            bad = [n for n in numbers if n < -_NUMERIC_EPSILON]
            if bad:
                violations.append(f"{key} stated as > 0 but found {bad[:5]}")
            elif not any(n > 0 for n in numbers):
                violations.append(f"{key} stated as > 0 but no value is positive: {numbers[:5]}")
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

    expectations_by_file, unattributed = _extract_expectations_by_file(
        bundle.instruction_text, candidate_filenames
    )
    # Keys/conventions the text states but does not tie to a file can only be applied safely when
    # there is exactly one possible deliverable; with several, guessing would reintroduce
    # cross-file false failures, so they are dropped (the affected checks report not_applicable).
    possible_deliverables = candidate_filenames or set(existing_files)
    if len(possible_deliverables) == 1 and unattributed.keys:
        (only_file,) = possible_deliverables
        expectations_by_file.setdefault(only_file, _FileExpectations()).merge(unattributed)

    nan_allowed_by_file, nan_unattributed = _extract_nan_allowances(
        bundle.instruction_text, candidate_filenames
    )
    if len(possible_deliverables) == 1 and nan_unattributed:
        (only_file,) = possible_deliverables
        nan_allowed_by_file.setdefault(only_file, set()).update(nan_unattributed)

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

        nan_issue = _check_nan_inf_empty(value, nan_allowed_by_file.get(filename, set()))
        if nan_issue is None:
            outcomes.append(CheckOutcome(f"no_nan_inf_empty:{filename}", "pass", "no NaN/Inf/empty detected"))
        else:
            outcomes.append(CheckOutcome(f"no_nan_inf_empty:{filename}", "fail", nan_issue))

        file_expectations = expectations_by_file.get(filename, _FileExpectations())
        expected_keys = file_expectations.keys
        schema_rows = list(file_expectations.schema_rows.items())

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
                    f"required_keys:{filename}",
                    "not_applicable",
                    "no keys/columns attributable to this file in instruction.md",
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
                    f"sign_range:{filename}",
                    "not_applicable",
                    "no sign/range convention text attributable to this file in instruction.md",
                )
            )

    outcomes.append(_check_row_count_parity(bundle.instruction_text, summaries, parsed_outputs))

    passed = not any(o.status == "fail" for o in outcomes)
    return SelfCheckResult(passed, outcomes)
