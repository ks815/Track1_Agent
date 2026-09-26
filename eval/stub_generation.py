"""A generic, non-per-task stand-in for the House model, used only by this evaluation harness.

Not a solver: it does not read or reason about a task's actual math. It only proves the harness's
plumbing by producing *some* deterministic, structurally plausible output for whatever filenames
and keys `agent.self_check`'s own contract-extraction found in a given unit's instruction.md --
the same extraction the real agent uses. The exact same code path runs for every unit; nothing
here branches on a task id.

Two styles:
  "naive"    -- placeholder values that ignore any stated sign/range convention (deliberately
               likely to trip agent.self_check's sign_range check on units that have one).
  "repaired" -- placeholder values chosen to satisfy the same convention-text patterns
               agent.self_check._check_sign_range_constraints already parses (reused, not
               reimplemented), standing in for a model repair attempt informed by that feedback.

Every generated script contains the literal token __OUT_DIR__, substituted by the caller with the
real absolute out_dir for a given run -- mirroring how a real model's response would hardcode
whatever out_dir string appeared in its own prompt.
"""

from __future__ import annotations

from agent.self_check import _GE_ZERO_RE, _GT_ZERO_RE, _RANGE_RE

_OUT_DIR_TOKEN = "__OUT_DIR__"


def _placeholder_value(convention_text: str, style: str) -> float:
    if style == "naive":
        return 0.0
    range_match = _RANGE_RE.search(convention_text)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return (lo + hi) / 2.0
    if _GE_ZERO_RE.search(convention_text):
        return 0.0
    if _GT_ZERO_RE.search(convention_text):
        return 1.0
    return 0.0


def build_placeholder_script(
    candidate_filenames: set[str],
    schema_rows: list[tuple[str, str]],
    json_keys: set[str],
    style: str,
) -> str:
    """A Python script (as text) that writes a placeholder for every candidate output filename.

    JSON/CSV files get one flat record built from the union of schema-table keys and
    fenced-JSON-example keys (the same `expected_keys` set agent.self_check itself checks
    against); HTML/other files get a minimal, clearly-a-placeholder body. If no filenames were
    derivable at all, writes one generic fallback file instead.
    """
    values: dict[str, float] = {key: _placeholder_value(text, style) for key, text in schema_rows}
    for key in json_keys:
        values.setdefault(key, 0.0 if style == "naive" else 1.0)

    lines = [
        "import json",
        "import pathlib",
        f"out_dir = pathlib.Path({_OUT_DIR_TOKEN!r})",
        "out_dir.mkdir(parents=True, exist_ok=True)",
        f"record = {values!r}",
    ]

    targets = sorted(candidate_filenames) or ["placeholder_output.json"]
    for filename in targets:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix == "json":
            lines.append(f"(out_dir / {filename!r}).write_text(json.dumps(record))")
        elif suffix == "csv":
            lines.append(
                f"(out_dir / {filename!r}).write_text("
                "','.join(record.keys()) + '\\n' + ','.join(str(v) for v in record.values()))"
                if values
                else f"(out_dir / {filename!r}).write_text('value\\n0')"
            )
        elif suffix == "parquet":
            lines.append("import pandas as _pd")
            lines.append(
                f"_pd.DataFrame([record] if record else [{{'value': 0}}]).to_parquet("
                f"out_dir / {filename!r}, index=False)"
            )
        else:
            lines.append(
                f"(out_dir / {filename!r}).write_text('<html><body>placeholder</body></html>')"
            )

    lines.append("print('placeholder script finished')")
    return "\n".join(lines)
