"""The Phase 3 coverage-oriented evaluation suite.

Six units, picked for structural coverage (category weight, output shape, checker mechanism,
contract-wording pattern) -- not a statistically representative sample of the 86-unit corpus, and
not picked for any expectation of how easy they are to solve.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteEntry:
    unit_id: str
    category: str
    why: str


SUITE: list[SuiteEntry] = [
    SuiteEntry(
        "t1-EXAMPLE-bs-greeks-pde",
        "derivatives-pricing",
        "Already studied. Single parquet deliverable; plain pytest invariant checker. Baseline reference point.",
    ),
    SuiteEntry(
        "t1-bollinger-backtest-aapl",
        "backtesting",
        "Already studied. Five deliverables across four formats (json/csv/csv/html/json); plain pytest checker.",
    ),
    SuiteEntry(
        "t1-alpha-hedge-strategy",
        "cross-domain",
        "Already studied. Nested JSON + a separate 'intermediates' file; the one checker "
        "*architecture* genuinely different from plain pytest (generic multi-phase verifier, "
        "tolerance-based, ships checks/reference_data/) -- 11/86 units use this pattern.",
    ),
    SuiteEntry(
        "t1-double-sort",
        "factor-research",
        "New category. Single flat CSV, nothing else -- the simplest, most different output shape "
        "in the suite. Its one input file is stock_chars.pqt (non-'.parquet' extension), "
        "surfacing a real gap in agent/summarizer.py's and agent/self_check.py's extension dispatch.",
    ),
    SuiteEntry(
        "t1-cir-bond-pricing",
        "fixed-income",
        "New category. Three separate deliverables (json+csv+csv) under explicit 'File 1/File "
        "2/File 3' headings -- a materially different multi-deliverable contract style from "
        "t1-bollinger-backtest-aapl's single 'Step 4' section.",
    ),
    SuiteEntry(
        "t1-var-es-estimation",
        "risk-management",
        "New category. manifest.json declares zero input files -- the task must synthesize its "
        "own data. Three small deliverables scattered across three independent 'Output:' "
        "sub-headings under different Steps, rather than one consolidated section.",
    ),
]
