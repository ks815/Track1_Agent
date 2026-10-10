"""Offline tests for agent.solve.solve: the repair step must never make things worse.

A scripted fake replaces the House-model client (no network, no openai package needed); the real
task reader, executor and self-check run against a small synthetic task in a temp dir. Never reads
any real unit's checks/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import solve as solve_module

_INSTRUCTION = """\
## Output

Save to `/app/output/result.json`:

```json
{
  "value": <float>
}
```
"""

# Runs fine (exit 0) and writes a readable file -- but without the required "value" key, so
# self-check fails and a repair is attempted.
_FIRST_ATTEMPT = 'import json; open({out!r} + "/result.json", "w").write(json.dumps({{"wrong": 1.0}}))'
_GOOD = 'import json; open({out!r} + "/result.json", "w").write(json.dumps({{"value": 2.0}}))'
_STILL_WRONG = 'import json; open({out!r} + "/result.json", "w").write(json.dumps({{"other": 3.0}}))'
# Corrupts the first-attempt file, leaves a stray file behind, then crashes.
_CORRUPT_THEN_CRASH = (
    'open({out!r} + "/result.json", "wb").write(b"garbage")\n'
    'open({out!r} + "/stray.tmp", "w").write("partial")\n'
    'raise RuntimeError("repair blew up")'
)
_RAISE = object()  # sentinel: this model call raises


class _ScriptedClient:
    """Stands in for ModelClient: each generate_code call plays back the next scripted reply."""

    def __init__(self, out_dir: Path, script: list):
        self._out = str(out_dir)
        self._script = list(script)
        self.calls = 0

    def generate_code(self, messages):
        self.calls += 1
        step = self._script.pop(0)
        if step is _RAISE:
            raise ConnectionError("House endpoint unreachable")
        return f"```python\n{step.format(out=self._out)}\n```"


@pytest.fixture(autouse=True)
def _tiny_repair_window(monkeypatch):
    """Production never starts a repair with < 120 s of unit budget left. The older tests here use a
    60 s synthetic ceiling, so lower the floor for them; the time-budget tests set it back explicitly."""
    monkeypatch.setattr(solve_module, "_MIN_REPAIR_WINDOW_SEC", 1.0, raising=False)


@pytest.fixture
def task_dir(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text(_INSTRUCTION)
    (task / "card.toml").write_text("[agent]\ntimeout_sec = 60\n")
    (task / "manifest.json").write_text(json.dumps({"files": []}))
    return task


@pytest.fixture
def traces(monkeypatch):
    """Every Trace solve() creates, so tests can assert on the recorded pipeline stages."""
    created: list = []

    class _SpyTrace(solve_module.Trace):
        def __init__(self):
            super().__init__()
            created.append(self)

    monkeypatch.setattr(solve_module, "Trace", _SpyTrace)
    return created


def _stages(traces) -> list[str]:
    (trace,) = traces
    return [entry["stage"] for entry in trace.stages]


def _stage(traces, name: str) -> dict:
    (trace,) = traces
    (entry,) = [e for e in trace.stages if e["stage"] == name]
    return entry


def _run_solve(monkeypatch, task_dir, tmp_path, script):
    out_dir = tmp_path / "out"
    client = _ScriptedClient(out_dir, script)
    monkeypatch.setattr(solve_module, "ModelClient", lambda: client)
    solve_module.solve(task_dir, out_dir)
    return out_dir, client


def _listing(out_dir: Path) -> list[str]:
    return sorted(p.name for p in out_dir.iterdir())


def test_passing_first_attempt_makes_no_repair_call(monkeypatch, task_dir, tmp_path):
    out_dir, client = _run_solve(monkeypatch, task_dir, tmp_path, [_GOOD])
    assert client.calls == 1
    assert json.loads((out_dir / "result.json").read_text()) == {"value": 2.0}


def test_successful_repair_replaces_the_artifact(monkeypatch, task_dir, tmp_path):
    out_dir, client = _run_solve(monkeypatch, task_dir, tmp_path, [_FIRST_ATTEMPT, _GOOD])
    assert client.calls == 2
    assert json.loads((out_dir / "result.json").read_text()) == {"value": 2.0}


def test_at_most_one_repair_even_when_the_repair_is_still_wrong(monkeypatch, task_dir, tmp_path):
    out_dir, client = _run_solve(
        monkeypatch, task_dir, tmp_path, [_FIRST_ATTEMPT, _STILL_WRONG, _GOOD]
    )
    assert client.calls == 2  # initial + exactly one repair; the third scripted reply is never used
    assert json.loads((out_dir / "result.json").read_text()) == {"other": 3.0}


def test_repair_model_call_failure_preserves_the_first_attempt_artifact(
    monkeypatch, task_dir, tmp_path, traces
):
    """If the repair request raises (endpoint down, rate limit, malformed reply), the first
    attempt's output stays exactly as written and solve() returns normally -- so the CLI exits 0
    and the checker scores what is on disk, as it does for any other downstream failure."""
    out_dir, client = _run_solve(monkeypatch, task_dir, tmp_path, [_FIRST_ATTEMPT, _RAISE])

    assert client.calls == 2  # initial + the one failed repair attempt; no retry loop of our own
    assert json.loads((out_dir / "result.json").read_text()) == {"wrong": 1.0}
    assert _listing(out_dir) == ["result.json"]
    assert "ConnectionError" in _stage(traces, "repair_generation_failed")["error"]
    assert _stages(traces)[-2:] == ["final_execution", "final_self_check"]  # trace still completed


def test_repair_run_that_crashes_restores_the_first_attempt_artifact(
    monkeypatch, task_dir, tmp_path, traces
):
    """A repair script can fail AFTER overwriting the first attempt's files. When the first
    attempt itself ran to completion, that artifact is put back byte-for-byte."""
    out_dir, client = _run_solve(
        monkeypatch, task_dir, tmp_path, [_FIRST_ATTEMPT, _CORRUPT_THEN_CRASH]
    )

    assert client.calls == 2
    assert json.loads((out_dir / "result.json").read_text()) == {"wrong": 1.0}
    assert _listing(out_dir) == ["result.json"]  # the repair's stray file is gone
    discarded = _stage(traces, "repair_discarded")
    assert discarded["exit_code"] == 1
    # The reported final state is the restored first attempt's, not the discarded repair's.
    assert _stage(traces, "final_execution")["exit_code"] == 0


def test_crashing_repair_is_kept_when_the_first_attempt_never_ran_to_completion(
    monkeypatch, task_dir, tmp_path
):
    """No valid first-attempt artifact to protect: behaviour is unchanged (nothing is restored)."""
    out_dir, client = _run_solve(
        monkeypatch, task_dir, tmp_path, ["raise RuntimeError('first attempt crashed')", _CORRUPT_THEN_CRASH]
    )
    assert client.calls == 2
    assert (out_dir / "result.json").read_bytes() == b"garbage"



def test_one_failed_repair_request_costs_exactly_one_request_end_to_end(monkeypatch, task_dir, tmp_path):
    """Through the REAL ModelClient (HTTP stubbed at the transport): the initial request succeeds,
    the repair request gets HTTP 500. That is exactly two requests on the wire -- the SDK does not
    resend the failed repair -- and the first attempt's artifact survives."""
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai is only installed inside the agent Docker image")

    import httpx
    import openai

    out_dir = tmp_path / "out"
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if len(sent) == 1:
            code = _FIRST_ATTEMPT.format(out=str(out_dir))
            return httpx.Response(
                200,
                json={
                    "id": "c1", "object": "chat.completion", "created": 0, "model": "m",
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": f"```python\n{code}\n```"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        return httpx.Response(500, json={"error": {"message": "upstream failure"}})

    real_openai_cls = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_openai_cls(**{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(handler))}),
    )
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model.internal:8443")
    monkeypatch.setenv("MODEL_TOKEN", "test-bearer-token")
    monkeypatch.setenv("MODEL_NAME", "test-model-v1")

    solve_module.solve(task_dir, out_dir)

    assert len(sent) == 2  # 1 initial + 1 repair; no hidden SDK retries
    assert json.loads((out_dir / "result.json").read_text()) == {"wrong": 1.0}


# --- Per-unit time budget --------------------------------------------------------------------
# The card's [agent].timeout_sec is the platform's ceiling for the WHOLE container (pull, model calls and
# every run). One absolute deadline = 90% of it; the first run may use at most half of that; a repair
# needs >= 120 s left both before and after its model call and then gets all that remains.

from agent.executor import ExecutionResult  # noqa: E402


class _FakeClock:
    def __init__(self):
        self.t = 5000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class _TimedClient(_ScriptedClient):
    """A scripted model client whose calls take (fake) time."""

    def __init__(self, out_dir, script, clock, latencies):
        super().__init__(out_dir, script)
        self._clock = clock
        self._latencies = list(latencies)

    def generate_code(self, messages):
        self._clock.advance(self._latencies.pop(0) if self._latencies else 0.0)
        return super().generate_code(messages)


def _run_timed(monkeypatch, task_dir, tmp_path, *, ceiling, script, latencies=(), runs=()):
    """Solve with a fake clock. `runs`: one entry per generated-code run -- {"hang": True} (runs to its
    timeout and is killed) or {"elapsed": seconds}. Returns (out_dir, client, timeouts given to runs)."""
    (task_dir / "card.toml").write_text(f"[agent]\ntimeout_sec = {ceiling}\n")
    monkeypatch.setattr(solve_module, "_MIN_REPAIR_WINDOW_SEC", 120.0)
    clock = _FakeClock()
    monkeypatch.setattr(solve_module, "_now", clock)

    real_run = solve_module.run_generated_code
    plan, timeouts = list(runs), []

    def fake_run(code, out_dir, timeout_sec):
        timeouts.append(timeout_sec)
        step = plan.pop(0) if plan else {}
        if step.get("hang"):
            clock.advance(timeout_sec)
            return ExecutionResult(exit_code=None, stdout="", stderr="", timed_out=True)
        result = real_run(code, out_dir, timeout_sec=timeout_sec)
        clock.advance(step.get("elapsed", 1.0))
        return result

    monkeypatch.setattr(solve_module, "run_generated_code", fake_run)
    out_dir = tmp_path / "out"
    client = _TimedClient(out_dir, script, clock, latencies)
    monkeypatch.setattr(solve_module, "ModelClient", lambda: client)
    solve_module.solve(task_dir, out_dir)
    return out_dir, client, timeouts


def test_first_run_is_capped_at_half_of_the_usable_unit_budget(monkeypatch, task_dir, tmp_path):
    """Ceiling 1800 s -> usable 90% = 1620 s -> the first run may use at most 810 s, not the full 1800 s."""
    _out, _client, timeouts = _run_timed(monkeypatch, task_dir, tmp_path, ceiling=1800, script=[_GOOD])
    assert timeouts == [pytest.approx(810.0)]


@pytest.mark.parametrize("ceiling, expected_cap", [(1200, 540.0), (2400, 1080.0), (3600, 1620.0), (5400, 2430.0)])
def test_first_run_cap_scales_with_the_cards_ceiling(monkeypatch, task_dir, tmp_path, ceiling, expected_cap):
    _out, _client, timeouts = _run_timed(monkeypatch, task_dir, tmp_path, ceiling=ceiling, script=[_GOOD])
    assert timeouts == [pytest.approx(expected_cap)]


def test_a_slow_initial_model_call_shrinks_the_first_run_window(monkeypatch, task_dir, tmp_path):
    """The deadline is absolute, so time already spent on the model call is not given to the script twice."""
    _out, _client, timeouts = _run_timed(
        monkeypatch, task_dir, tmp_path, ceiling=1800, script=[_GOOD], latencies=[1000.0]
    )
    assert timeouts == [pytest.approx(1620.0 - 1000.0)]  # less than the 810 s cap


def test_a_hung_first_script_leaves_the_repair_the_rest_of_the_budget(monkeypatch, task_dir, tmp_path, traces):
    out_dir, client, timeouts = _run_timed(
        monkeypatch, task_dir, tmp_path, ceiling=1800, script=[_FIRST_ATTEMPT, _GOOD],
        latencies=[0.0, 60.0], runs=[{"hang": True}, {"elapsed": 5.0}],
    )
    assert client.calls == 2
    # first run killed at its 810 s cap; repair model call 60 s; the repair run gets everything left of 1620 s.
    assert timeouts == [pytest.approx(810.0), pytest.approx(1620.0 - 810.0 - 60.0)]
    assert json.loads((out_dir / "result.json").read_text()) == {"value": 2.0}
    assert "repair_skipped" not in _stages(traces)


def test_repair_is_not_attempted_when_too_little_budget_remains(monkeypatch, task_dir, tmp_path, traces):
    """After a 780 s model call and an 810 s first run only 30 s of the 1620 s budget is left: no repair
    request is spent, and the first attempt's artifact stays."""
    out_dir, client, timeouts = _run_timed(
        monkeypatch, task_dir, tmp_path, ceiling=1800, script=[_FIRST_ATTEMPT, _GOOD],
        latencies=[780.0], runs=[{"elapsed": 810.0}],
    )
    assert client.calls == 1 and len(timeouts) == 1
    assert json.loads((out_dir / "result.json").read_text()) == {"wrong": 1.0}
    skipped = _stage(traces, "repair_skipped")
    assert skipped["remaining_sec"] == pytest.approx(30.0)
    assert _stages(traces)[-2:] == ["final_execution", "final_self_check"]


def test_repair_run_is_not_started_if_its_model_call_used_up_the_window(monkeypatch, task_dir, tmp_path, traces):
    """The repair request itself may be slow. If that leaves < 120 s, the new script is discarded rather
    than started, so a run cannot be cut off by the platform mid-write; the first artifact stays."""
    out_dir, client, timeouts = _run_timed(
        monkeypatch, task_dir, tmp_path, ceiling=1800, script=[_FIRST_ATTEMPT, _GOOD],
        latencies=[0.0, 1550.0], runs=[{"elapsed": 5.0}],
    )
    assert client.calls == 2 and len(timeouts) == 1  # the repair request happened; its run did not
    assert json.loads((out_dir / "result.json").read_text()) == {"wrong": 1.0}
    assert _stage(traces, "repair_skipped")["remaining_sec"] == pytest.approx(1620.0 - 5.0 - 1550.0)


def test_a_really_hung_first_script_still_leaves_time_for_a_repair_inside_the_ceiling(
    monkeypatch, task_dir, tmp_path
):
    """Real clocks, a real hanging script, a 10 s ceiling. Before the policy the first run was given the
    whole 10 s, so unit time was already gone when the repair began; now the first run is killed at 45% of
    the ceiling, the repair succeeds, and the whole solve finishes before the platform's ceiling."""
    import time

    (task_dir / "card.toml").write_text("[agent]\ntimeout_sec = 10\n")
    out_dir = tmp_path / "out"
    client = _ScriptedClient(out_dir, ["import time; time.sleep(60)", _GOOD])
    monkeypatch.setattr(solve_module, "ModelClient", lambda: client)

    started = time.monotonic()
    solve_module.solve(task_dir, out_dir)
    elapsed = time.monotonic() - started

    assert client.calls == 2
    assert json.loads((out_dir / "result.json").read_text()) == {"value": 2.0}
    assert elapsed < 10.0, f"solve took {elapsed:.1f}s, past the 10 s ceiling"


# --- "Never crash": an initial model failure ------------------------------------------------------
# The Track 1 pack: an uncaught exception forfeits the unit; exit 0. We cannot produce a first attempt
# without the model, and we never fabricate a placeholder deliverable -- so the right behaviour is to
# record the failure, write nothing, and exit 0 (the checker then scores the missing output as zero,
# exactly as it would score a crash).


def test_initial_model_failure_writes_nothing_and_does_not_raise(monkeypatch, task_dir, tmp_path, traces):
    out_dir = tmp_path / "out"
    client = _ScriptedClient(out_dir, [_RAISE])
    monkeypatch.setattr(solve_module, "ModelClient", lambda: client)

    solve_module.solve(task_dir, out_dir)  # must not raise

    assert client.calls == 1  # no hidden retry, and no "repair" used as a second try
    assert _listing(out_dir) == []  # nothing fabricated
    assert "ConnectionError" in _stage(traces, "initial_generation_failed")["error"]
    stages = _stages(traces)
    assert "execution" not in stages and "repair_generation" not in stages


def test_failure_to_construct_the_model_client_is_handled_the_same_way(monkeypatch, task_dir, tmp_path, traces):
    def no_endpoint():
        raise KeyError("MODEL_ENDPOINT")

    monkeypatch.setattr(solve_module, "ModelClient", no_endpoint)
    out_dir = tmp_path / "out"

    solve_module.solve(task_dir, out_dir)

    assert _listing(out_dir) == []
    assert "KeyError" in _stage(traces, "initial_generation_failed")["error"]


def test_the_cli_exits_zero_when_the_initial_model_call_fails(monkeypatch, task_dir, tmp_path):
    from agent.cli import main

    out_dir = tmp_path / "out"
    monkeypatch.setattr(solve_module, "ModelClient", lambda: _ScriptedClient(out_dir, [_RAISE]))

    assert main(["solve", "--task-dir", str(task_dir), "--out", str(out_dir)]) == 0
    assert _listing(out_dir) == []


def test_an_initial_http_500_costs_exactly_one_request_and_exits_zero(monkeypatch, task_dir, tmp_path):
    """Through the REAL ModelClient with the transport stubbed: the House route answers 500 to the first
    request. One request on the wire (no SDK retry), no output, and the CLI path returns 0."""
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai is only installed inside the agent Docker image")

    import httpx
    import openai

    from agent.cli import main

    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500, json={"error": {"message": "upstream failure"}})

    real_openai_cls = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_openai_cls(**{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(handler))}),
    )
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model.internal:8443")
    monkeypatch.setenv("MODEL_TOKEN", "test-bearer-token")
    monkeypatch.setenv("MODEL_NAME", "test-model-v1")
    out_dir = tmp_path / "out"

    assert main(["solve", "--task-dir", str(task_dir), "--out", str(out_dir)]) == 0
    assert len(sent) == 1
    assert _listing(out_dir) == []


def test_every_request_disables_thinking_including_the_repair(monkeypatch, task_dir, tmp_path):
    """End to end through the real ModelClient: the initial AND the repair request both carry the
    documented `chat_template_kwargs.enable_thinking = false`."""
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai is only installed inside the agent Docker image")

    import httpx
    import openai

    out_dir = tmp_path / "out"
    bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        code = (_FIRST_ATTEMPT if len(bodies) == 1 else _GOOD).format(out=str(out_dir))
        return httpx.Response(
            200,
            json={
                "id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": f"```python\n{code}\n```"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    real_openai_cls = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_openai_cls(**{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(handler))}),
    )
    monkeypatch.setenv("MODEL_ENDPOINT", "http://model.internal:8443")
    monkeypatch.setenv("MODEL_TOKEN", "test-bearer-token")
    monkeypatch.setenv("MODEL_NAME", "test-model-v1")

    solve_module.solve(task_dir, out_dir)

    assert len(bodies) == 2  # initial + repair
    for body in bodies:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
