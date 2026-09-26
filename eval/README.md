# Phase 3 evaluation harness

**This directory is not part of the submission.** It is not copied into `agent/Dockerfile`'s
image and nothing under `agent/` imports anything from here — the dependency runs one way only:
this harness imports and drives the real `agent.*` modules as a library.

It exists to answer two different questions, kept strictly separate:

1. **Mechanical validation** (what this harness can actually answer today, offline): does the
   real pipeline — task reading, manifest-based input resolution, prompt/contract extraction,
   execution, self-check, the at-most-one-repair loop, and the official local checker — work
   correctly against real unit directories? This needs no House-model access; the "model" step is
   played by a deterministic, generic stand-in (`stub_generation.py`) so results are reproducible.
2. **Competition effectiveness**: does self-check + repair actually improve real House-model
   pass@1? **This harness cannot answer that question and does not attempt to.** The stand-in
   generator is not a model — it is a fixed, generic placeholder-writer, the same code shape for
   every unit, used only to exercise the harness deterministically. Any number this harness
   produces is a mechanism-validation result, never a pass@1 estimate.

## What this harness may touch that `agent/` may not

`eval/checker.py` invokes each unit's own `checks/test.sh` inside `finance-bench-sandbox:latest`,
researcher-side, exactly as README step 6 documents — the same way earlier verification in this
session ran the exemplar's checker manually. This is the *only* place `checks/` is read anywhere
in this project. `agent/self_check.py` and the rest of the shipped runtime never see it.

## Layout

- `suite.py` — the 6 selected units and why each was picked.
- `stub_generation.py` — a generic (non-per-task) placeholder code generator standing in for the
  House model in local runs.
- `checker.py` — researcher-side invocation of the official local checker via Docker.
- `harness.py` — runs conditions A/B/C for one unit using the real `agent.*` modules.
- `run_suite.py` — drives all 6 units, writes `results/phase3_<timestamp>.json`.
