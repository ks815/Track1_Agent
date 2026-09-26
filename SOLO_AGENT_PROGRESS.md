# Track 1 Solo Agent — Project Log

This document summarizes the engineering work done in this workspace on top of the official
Agenthon 2026 Track 1 public starter kit: an audit of the real competition contract, a working
solo-capable agent (`agent/`), and an evaluation harness for it (`eval/`). It's written to be
readable by someone picking this up cold — a teammate, a future version of yourself, or a
reviewer.

**Everything here is original work added on top of the organizer-provided repository.** The
`units/`, `docs/`, `templates/`, `qfbench2_track_coding/`, `examples/`, `baselines/`, `.github/`,
and root-level `README.md`/`AGENTS.md`/etc. are the official Track 1 starter kit and were not
modified.

---

## 1. Where this started: verifying the real contract

Before writing any code, the actual scored submission contract was verified line-by-line against
source (not assumed from the docs' prose, since several of them turned out to disagree with each
other — see below). Key facts established, each with a file/line citation recorded in this
session's transcript:

- A submission is a Docker image implementing exactly `solve --task-dir <path> --out <path>`.
- At scoring time, `--task-dir` points at **the raw unit directory itself** (`instruction.md`,
  `card.toml`, `environment/`, `checks/`, `manifest.json`) mounted read-only — not a rebuilt copy.
- `--out` is bound to **both** `/app/output` and `/output` on the same host directory — either
  path is safe to write to.
- The only reachable model is the organizer-hosted **House model**, via
  `$MODEL_ENDPOINT/v1/chat/completions` with a `$MODEL_TOKEN` bearer and `$MODEL_NAME` — capped
  at **25 admitted requests per unit**, 4,000 output tokens per call. No vendor API is reachable.
- **A genuinely important, non-obvious finding**: many units' `instruction.md` state an input
  path like `/app/data/returns.csv`, but that path is a leftover from this repo's Harbor/QFBench
  heritage — it does **not** exist in the real submission container. The only reliable way to
  find a task's real input files at solve time is `manifest.json`'s bundle-relative paths,
  resolved against whatever `task_dir` the harness actually passes. This one finding drove a core
  design rule for the agent (§2).
- 86 public units ship in this repo (87 were migrated from QFBench v1; one was later withdrawn).

## 2. Phase 1 — the competition floor

**Goal:** the smallest agent that legitimately implements the real contract, with no shortcuts —
generic across every unit, not tuned to the one worked example the starter kit ships.

**Design rule:** `task_dir` and `out_dir` are opaque parameters from the CLI. Nothing in the agent
hardcodes `/input`, `/app/output`, or `/output`, even though those are what the harness happens to
pass today.

| Module | Job |
|---|---|
| `agent/task_reader.py` | Reads `instruction.md`, `card.toml`, `manifest.json`. Nothing else — never `checks/`. |
| `agent/input_resolver.py` | Resolves every `manifest.json` entry with `role == "input"` against the real `task_dir`, verifies each file exists. Never trusts a literal path from `instruction.md` prose. |
| `agent/summarizer.py` | Opens each resolved input and produces a factual summary — row count, columns, dtypes, null counts — dispatched by file format. |
| `agent/prompt.py` | Builds the message sent to the model: the task instructions, the input summaries, and the *real* output directory — with an explicit rule telling the model to substitute the real `out_dir` for whatever output root the instructions happen to name. |
| `agent/model_client.py` | A thin client for the House model endpoint. One function, one call. |
| `agent/executor.py` | Extracts code from the model's reply and runs it as a subprocess, capturing exit code / stdout / stderr. |
| `agent/solve.py` | Wires the above together end to end. |
| `agent/cli.py` | The `solve --task-dir --out` entry point. |
| `agent/Dockerfile` | `FROM finance-bench-sandbox:latest` (the shared organizer base image) + the one missing dependency (`openai`). |

**Verified**, not just written: built both Docker images locally, confirmed `solve --help` and the
real entry point work inside the container, confirmed task reading and input resolution succeed
against real units with the real (unstubbed) image, and confirmed output is written *only* to
whatever `out_dir` is passed — including a test where `/app/output` and `/output` were both bound
to empty decoy directories and only the actual (arbitrary, non-standard) `--out` path received
anything.

Also ran the organizer's own local checker flow (`checks/test.sh` inside
`finance-bench-sandbox:latest`) against the repo's built-in deterministic example solver, as a
sanity check on the harness/mount conventions — reward 1.0, matching the documented expectation.

## 3. Phase 2 — self-check and one repair

**Goal:** after the model generates code and it runs, check the result *generically* — using only
information the solver itself has (the task's own `instruction.md`, the resolved inputs, and
whatever got written to `out_dir`) — and if something looks wrong, give the model exactly one
more try with a concrete description of the problem.

`agent/self_check.py` runs a mix of two kinds of checks:

- **Fully generic** (same for every task): did the code crash or time out? Is whatever got
  written actually readable in its format? Does it contain NaN/Inf/empty values?
- **Instruction-derived** (parsed from *that task's own* `instruction.md` at runtime, through
  fixed heuristics that are identical for every task — never a per-task code path): what
  filenames does the instruction actually name as deliverables? What columns/keys does its own
  schema table or JSON example show? Does it say the output should have one row per input row?
  Does a column's own stated convention (`>= 0`, `> 0`, an interval like `(0, 1)`) hold?

Every instruction-derived check is designed to fail *safe*: if the heuristic can't find anything
concrete in a given task's text, it reports "not applicable," never a false failure.

If self-check fails (or the code crashed), `agent/solve.py` builds one repair prompt — the
original instructions, the previous code, and the concrete failure — asks the model once more,
runs the result, and checks it again. No further repair loop.

**Verified**: an offline test suite (`agent/tests/`, 21 tests) covers the House-model client
(mocked HTTP, confirming the request URL/auth/model name are built correctly and that failures
raise clearly) and the self-check heuristics (synthetic instructions + synthetic outputs). The
extraction heuristics were also run against all 86 real units' `instruction.md` files: every
single one yielded at least one candidate output filename, and for the three tasks studied in
depth, the derived filenames and keys matched hand-built reference extractions almost exactly,
with no leakage from input-schema tables into output-key checks.

## 4. Phase 3 — evaluation harness

**Goal:** validate the whole pipeline against a representative slice of real units — without
access to the real House model, which only exists inside the organizer's network.

Six units were chosen for structural coverage (not because they're easy, and not a claim of
statistical representativeness):

| Unit | Category | Why it was picked |
|---|---|---|
| `t1-EXAMPLE-bs-greeks-pde` | derivatives-pricing | The baseline reference case. |
| `t1-bollinger-backtest-aapl` | backtesting | Five deliverables across four file formats. |
| `t1-alpha-hedge-strategy` | cross-domain | The one checker *architecture* genuinely different from plain pytest checks (tolerance-based). |
| `t1-double-sort` | factor-research | Simplest possible output (one CSV); its input file uses the `.pqt` extension (see §5). |
| `t1-cir-bond-pricing` | fixed-income | Four deliverables under explicit "File 1/2/3/4" headings. |
| `t1-var-es-estimation` | risk-management | Zero declared input files — the task has to generate its own data. |

A harness under `eval/` (separate from `agent/` — nothing in `agent/` imports from `eval/` or
reads `checks/`) drives each unit through three conditions using the real `agent.*` modules, with
a generic, deterministic placeholder generator standing in for the model:

- **A** — generate once, execute, check with the official local checker.
- **B** — same, plus self-check runs but doesn't act on anything (a diagnostic-only condition).
- **C** — self-check plus, if it fails, one repair round.

**Result: the mechanism works.** Across all 6 units: input resolution and output-contract
extraction were correct (including two cases where the extractor found a genuine deliverable a
manual read had missed); A and B produced byte-identical output every time; self-check never had
a side effect; repair triggered exactly when self-check failed and never more than once.

**What this did *not* measure, and cannot measure locally**: whether self-check and repair
actually improve real accuracy. The placeholder generator isn't a model — every "reward" number
from these local runs reflects its designed lack of real computation, not agent quality. That
question stays open until an organizer development run with real House-model credentials.

## 5. A concrete bug found and fixed: `.pqt` files

The evaluation surfaced a real, non-hypothetical gap: `agent/summarizer.py` only recognized the
`.parquet` extension. `t1-double-sort`'s input file is `stock_chars.pqt` — and a corpus check
showed `.pqt` is actually *more common* in this repo than `.parquet` (3 units vs. 1). Files with
that extension fell through to a generic byte-preview fallback, handing the model ~500 bytes of
raw binary garbage instead of column names and row counts.

Fixed: `.pqt` is now treated as a Parquet alias, dispatched to the same structured reader.
Confirmed directly — `t1-double-sort`'s input now summarizes as 104,250 rows across 38 named,
typed columns instead of unreadable bytes. A regression test (`agent/tests/test_summarizer.py`)
locks this in. Input resolution, self-check, and the rest of the architecture were untouched.

A read-only follow-up check found two more extensions with the *same* failure mode —
`.xlsx` (1 unit) and `.zip` (1 unit), both binary formats — flagged but **not yet fixed**, pending
a decision on priority. Three other unhandled extensions (`.jsonl`, `.tsv`, `.xml`, one unit
each) are plain text, so the existing fallback degrades gracefully there and isn't a comparable
problem.

## 6. Current state, plainly

**What's built and verified:** a complete, generic Phase 1+2 agent (read task → resolve inputs →
summarize → prompt → generate → execute → self-check → at most one repair → final check), an
offline test suite, two working Docker images, and an evaluation harness proving the whole
mechanism runs correctly against 6 structurally diverse real units.

**What's explicitly not yet known:** real pass@1 with the actual House model. Nothing in this
project has run against real model credentials yet — that can only happen inside the organizer's
Development network.

**Known open items, not yet acted on:**
- `.xlsx`/`.zip` inputs get a garbled preview, same as the `.pqt` bug did (fixed for `.pqt` only).
- `self_check.py` checks every output file in a multi-file unit against the *same* combined set of
  expected keys, rather than mapping specific keys to specific files — likely to cause
  false-positive "missing key" failures (and therefore an unnecessary repair call) on multi-file
  units like `t1-cir-bond-pricing`. Not fixed; noted as a Phase 4 candidate.

## Repository map

```
agent/            the submission agent (new)
├── *.py            task reading, input resolution, summarization, prompting,
│                   model client, execution, self-check, repair orchestration
├── tests/          offline unit tests (21, no network/model needed)
├── Dockerfile      builds the submission image
└── README.md       build/run instructions

eval/             the Phase 3 evaluation harness (new, separate from agent/)
├── suite.py         the 6-unit coverage suite
├── stub_generation.py, checker.py, harness.py, run_suite.py
├── results/         raw JSON output from evaluation runs
└── README.md

SOLO_AGENT_PROGRESS.md   this file

units/, docs/, templates/, qfbench2_track_coding/, examples/, baselines/, .github/,
README.md, AGENTS.md, SUBMISSION_CLI.md, ...   the official Track 1 starter kit (unmodified)
```

## Building and running

See `agent/README.md` for building the two Docker images and running the agent locally (the
House model itself can only be reached from inside the organizer's network), and `eval/README.md`
for re-running the evaluation suite.
