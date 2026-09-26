"""Regression test: summarizer.py must treat .pqt as a Parquet alias.

Before this fix, a .pqt file fell through to the generic byte-preview branch and produced raw
binary garbage instead of a structured summary (see t1-double-sort, t1-residual-momentum,
t1-stable-residual -- 3 real units in this corpus use the .pqt spelling).
"""

from __future__ import annotations

from agent.input_resolver import ResolvedInput
from agent.summarizer import summarize


def test_pqt_extension_gets_the_same_structured_summary_as_parquet(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"stock_id": ["a", "b", "c"], "beta": [0.1, 0.2, 0.3]})

    parquet_path = tmp_path / "data.parquet"
    pqt_path = tmp_path / "data.pqt"
    df.to_parquet(parquet_path, index=False)
    df.to_parquet(pqt_path, index=False)

    parquet_summary = summarize(ResolvedInput(manifest_path="data.parquet", resolved_path=parquet_path))
    pqt_summary = summarize(ResolvedInput(manifest_path="data.pqt", resolved_path=pqt_path))

    # Same structured content (format/rows/columns/dtypes), modulo the filename fields.
    for key in ("format", "n_rows", "columns", "dtypes"):
        assert pqt_summary[key] == parquet_summary[key], key

    assert pqt_summary["format"] == "parquet"
    assert pqt_summary["n_rows"] == 3
    assert set(pqt_summary["columns"]) == {"stock_id", "beta"}
    assert "text_preview" not in pqt_summary
    assert "summary_error" not in pqt_summary


def test_pqt_extension_is_case_insensitive(tmp_path):
    import pandas as pd

    path = tmp_path / "data.PQT"
    pd.DataFrame({"x": [1, 2]}).to_parquet(path, index=False)

    result = summarize(ResolvedInput(manifest_path="data.PQT", resolved_path=path))
    assert result["format"] == "parquet"
    assert result["n_rows"] == 2
