# Track 1 agent

Reads a task generically, resolves its real inputs from `manifest.json`, summarizes them
factually, asks the House model for a Python solution, runs it, and generically self-checks the
result (`agent/self_check.py`): required-output-file presence, format readability, NaN/Inf/empty
output, instruction-derived required keys/columns, row-count parity, and instruction-stated
sign/range constraints — all derived from *this task's own* `instruction.md` at runtime, never
from `checks/`, reference data, or any precomputed per-task material. If the self-check fails
(or the generated code errored), the model gets **at most one** repair call with the previous
code and the concrete failure, then a final execution and self-check. Every stage — initial
generation, execution, self-check, optional repair, final execution, final self-check — is
logged to stderr as a trace (`agent/trace.py`). No further repair loop, no per-task logic
anywhere in this package.

Offline, network-free unit tests live in `agent/tests/` (`test_model_client.py` mocks the House
HTTP call; `test_self_check.py` exercises the self-check heuristics against synthetic
instructions) — run with `python -m pytest agent/tests/` wherever `openai`+`pytest` are
installed (baked into this image; not on a bare host).

## Build

```bash
# from the repo root, once:
docker build -t finance-bench-sandbox:latest -f docker/sandbox.Dockerfile .

# this agent image:
docker build -f agent/Dockerfile -t my-agent:latest agent/
```

## Run locally against a practice unit

This needs `MODEL_ENDPOINT` / `MODEL_TOKEN` / `MODEL_NAME`, which only exist inside the
organizer's restricted network — there is no local stand-in for the House model. Without them,
`agent/model_client.py` will raise a `KeyError` at the model-call step. Everything before that
step (`task_reader`, `input_resolver`, `summarizer`, `prompt`) is pure and network-free, and can
be exercised locally without Docker:

```bash
python -c "
from agent.task_reader import read_task
from agent.input_resolver import resolve_inputs
from agent.summarizer import summarize
from agent.prompt import build_messages

bundle = read_task('units/t1-EXAMPLE-bs-greeks-pde')
resolved = resolve_inputs(bundle.task_dir, bundle.manifest)
summaries = [summarize(r) for r in resolved]
print(build_messages(bundle.instruction_text, resolved, summaries, '/tmp/out')[1]['content'])
"
```

Full end-to-end run (once you have House credentials, e.g. inside the real competition network):

```bash
docker run --rm --network qfb2-eval \
  -e HTTP_PROXY -e HTTPS_PROXY -e MODEL_ENDPOINT -e MODEL_TOKEN -e MODEL_NAME \
  -v "$(pwd)/units/t1-EXAMPLE-bs-greeks-pde:/input:ro" \
  -v /tmp/out:/app/output -v /tmp/out:/output \
  my-agent:latest solve --task-dir /input --out /app/output
```

Then check the output with the unit's own checker per the main repo README, step 6.
