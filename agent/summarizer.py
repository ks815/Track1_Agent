"""Lightweight, format-generic factual summaries of resolved input files.

Purely structural/descriptive: filename, format, columns/keys, row count, dtypes, missing-value
counts. Never computes or exposes anything that could function as a reference answer, and never
opens anything outside the resolved input path it is given.
"""

from __future__ import annotations

import json
from typing import Any

from agent.input_resolver import ResolvedInput

_PREVIEW_BYTES = 500

# .pqt is a Parquet alias seen in this corpus (e.g. t1-double-sort, t1-residual-momentum,
# t1-stable-residual -- 3 units, actually more common here than the ".parquet" spelling itself,
# which appears in only 1). Same file format, same structured summary; dispatch on either
# spelling to the same parquet reader.
_PARQUET_EXTENSIONS = {".parquet", ".pqt"}


def summarize(resolved_input: ResolvedInput) -> dict[str, Any]:
    """Dispatch on file extension. Falls back to a generic byte/text preview for unknown types."""
    path = resolved_input.resolved_path
    suffix = path.suffix.lower()

    base: dict[str, Any] = {
        "manifest_path": resolved_input.manifest_path,
        "filename": path.name,
        "bytes": path.stat().st_size,
    }

    try:
        if suffix == ".csv":
            base.update(_summarize_csv(path))
        elif suffix in _PARQUET_EXTENSIONS:
            base.update(_summarize_parquet(path))
        elif suffix == ".json":
            base.update(_summarize_json(path))
        else:
            base.update(_summarize_generic(path))
    except Exception as exc:  # noqa: BLE001 - a summary failure must not crash the pipeline
        base["summary_error"] = f"{type(exc).__name__}: {exc}"

    return base


def _summarize_csv(path) -> dict[str, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    return {
        "format": "csv",
        "n_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
    }


def _summarize_parquet(path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "format": "parquet",
        "n_rows": pf.metadata.num_rows,
        "columns": schema.names,
        "dtypes": {name: str(schema.field(name).type) for name in schema.names},
    }


def _summarize_json(path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        first_keys = list(data[0].keys()) if data and isinstance(data[0], dict) else None
        return {"format": "json", "shape": "list", "n_items": len(data), "item_keys": first_keys}
    if isinstance(data, dict):
        return {"format": "json", "shape": "object", "top_level_keys": list(data.keys())}
    return {"format": "json", "shape": type(data).__name__}


def _summarize_generic(path) -> dict[str, Any]:
    with path.open("rb") as f:
        raw = f.read(_PREVIEW_BYTES)
    preview = raw.decode("utf-8", errors="replace")
    return {"format": path.suffix.lstrip(".") or "unknown", "text_preview": preview}
