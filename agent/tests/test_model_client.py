"""Offline unit tests for agent.model_client.ModelClient.

No real network call is ever made: openai's own HTTP transport is swapped for
httpx.MockTransport, so the request that would otherwise go out over the audited proxy is
intercepted and answered in-process. This only verifies ModelClient's existing behavior -- it
does not add a second model abstraction or change how ModelClient itself is built.

Requires the `openai` package (only installed inside the agent Docker image, per
docker/requirements-sandbox.txt + agent/Dockerfile) -- skipped entirely if it is absent, e.g. on
a bare host during Phase 1/2 development.
"""

from __future__ import annotations

import importlib.util
import json

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openai") is None,
    reason="openai is only installed inside the agent Docker image",
)


def _install_mock_transport(monkeypatch, handler):
    """Make every OpenAI(...) call ModelClient makes route through `handler` instead of the
    real network, while still passing through and letting us inspect base_url/api_key."""
    import openai

    real_openai_cls = openai.OpenAI
    captured_kwargs: dict = {}

    def factory(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handler))
        return real_openai_cls(**kwargs)

    monkeypatch.setattr(openai, "OpenAI", factory)
    return captured_kwargs


def _set_env(monkeypatch, *, endpoint="http://model.internal:8443", token="test-bearer-token", name="test-model-v1"):
    monkeypatch.setenv("MODEL_ENDPOINT", endpoint)
    monkeypatch.setenv("MODEL_TOKEN", token)
    monkeypatch.setenv("MODEL_NAME", name)


def _chat_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model-v1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def test_request_url_bearer_and_model_are_constructed_correctly(monkeypatch):
    from agent.model_client import ModelClient

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return _chat_completion_response("```python\nprint('hi')\n```")

    captured_kwargs = _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch, endpoint="http://model.internal:8443", token="test-bearer-token", name="test-model-v1")

    messages = [{"role": "user", "content": "solve the task"}]
    ModelClient().generate_code(messages)

    # The actual HTTP request ModelClient's call produced.
    assert seen["url"] == "http://model.internal:8443/v1/chat/completions"
    assert seen["authorization"] == "Bearer test-bearer-token"
    assert seen["body"]["model"] == "test-model-v1"
    assert seen["body"]["messages"] == messages

    # And, redundantly, the SDK constructor arguments ModelClient itself chose.
    assert captured_kwargs["base_url"] == "http://model.internal:8443/v1"
    assert captured_kwargs["api_key"] == "test-bearer-token"


def test_no_v1_suffix_is_never_sent_even_with_a_trailing_slash_endpoint(monkeypatch):
    """MODEL_ENDPOINT is an origin with no path; a trailing slash must not produce //v1."""
    from agent.model_client import ModelClient

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return _chat_completion_response("code")

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch, endpoint="http://model.internal:8443/")

    ModelClient().generate_code([{"role": "user", "content": "x"}])
    assert seen["url"] == "http://model.internal:8443/v1/chat/completions"


def test_generated_code_text_is_extracted_from_the_response(monkeypatch):
    from agent.model_client import ModelClient

    expected_content = "```python\nimport pandas as pd\nprint('generated')\n```"

    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_completion_response(expected_content)

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)

    result = ModelClient().generate_code([{"role": "user", "content": "x"}])
    assert result == expected_content


def test_http_error_response_fails_clearly(monkeypatch):
    """A non-2xx response must raise, never be swallowed into an empty/garbage string."""
    from agent.model_client import ModelClient
    import openai

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "upstream failure"}})

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)

    with pytest.raises(openai.APIStatusError):
        ModelClient().generate_code([{"role": "user", "content": "x"}])


def test_missing_env_vars_fail_clearly(monkeypatch):
    """No MODEL_ENDPOINT/MODEL_TOKEN/MODEL_NAME (e.g. running outside the restricted network)
    must raise immediately, not silently proceed."""
    from agent.model_client import ModelClient

    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("MODEL_TOKEN", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    with pytest.raises(KeyError):
        ModelClient()


def test_default_client_constructs_with_the_installed_httpx(monkeypatch):
    """Regression: openai 1.40.0 passes `proxies=` to httpx.Client, an argument httpx 0.28 removed,
    so a bare ModelClient() raised TypeError inside the agent image and `solve` never reached the
    model. Every other test here injects its own http_client and so bypasses openai's default
    client construction -- this one deliberately does not. Constructing a client makes no
    network call."""
    _set_env(monkeypatch)
    from agent.model_client import ModelClient

    ModelClient()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(lambda request: httpx.Response(500, json={"error": {"message": "boom"}}), id="http-500"),
        pytest.param(lambda request: httpx.Response(503, json={"error": {"message": "busy"}}), id="http-503"),
        pytest.param(lambda request: httpx.Response(429, json={"error": {"message": "slow"}}), id="http-429"),
        pytest.param(lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")), id="connection-error"),
    ],
)
def test_a_failed_request_is_sent_exactly_once_not_retried_by_the_sdk(monkeypatch, failure):
    """The House allowance is 25 ADMITTED requests per unit, and an admitted request is charged
    even when it fails. openai's default max_retries=2 would silently resend a 5xx/429/connection
    failure up to twice, so one logical call could cost three requests. ModelClient makes exactly
    one request per call and surfaces the failure to the caller (solve() decides what to do)."""
    import openai

    from agent.model_client import ModelClient

    _set_env(monkeypatch)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return failure(request)

    _install_mock_transport(monkeypatch, handler)

    with pytest.raises(openai.OpenAIError):
        ModelClient().generate_code([{"role": "user", "content": "hello"}])

    assert len(sent) == 1


def test_client_is_constructed_with_sdk_retries_disabled(monkeypatch):
    import openai  # noqa: F401  (ensures the skip-guard module is importable)

    _set_env(monkeypatch)
    captured = _install_mock_transport(monkeypatch, lambda request: _chat_completion_response("x"))
    from agent.model_client import ModelClient

    ModelClient()
    assert captured["max_retries"] == 0


def _capture_request_body(monkeypatch) -> dict:
    """Run one ModelClient call against a stubbed transport; return the raw JSON body it sent."""
    from agent.model_client import ModelClient

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["raw"] = request.content
        seen["body"] = json.loads(request.content)
        return _chat_completion_response("```python\nprint('hi')\n```")

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)
    ModelClient().generate_code([{"role": "user", "content": "solve the task"}])
    return seen


def test_request_body_disables_thinking_exactly_as_the_house_guide_documents(monkeypatch):
    """HOUSE-MODEL.md ("Development thinking controls"): the House model thinks by default; to turn it
    off for a request, the JSON body carries `chat_template_kwargs` at the TOP LEVEL with
    `enable_thinking: false`. The OpenAI client sends that via extra_body (a nested "extra_body" key
    would not do it, and omitting it keeps thinking on, which can spend the 4,000-token output cap
    before any code is emitted)."""
    seen = _capture_request_body(monkeypatch)
    body = seen["body"]

    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["chat_template_kwargs"]["enable_thinking"] is False  # a real JSON false, not "false"/0
    assert "extra_body" not in body  # merged at the top level, not nested
    assert b'"enable_thinking": false' in seen["raw"] or b'"enable_thinking":false' in seen["raw"]


def test_disabling_thinking_changes_nothing_else_in_the_request(monkeypatch):
    """Only the documented field is added: same model, messages and 4,000-token cap as before, no
    tools (vendor-side tools must be off), no `n` (multiple alternatives are refused), and none of the
    untested House pass-through options (`low_effort`, `reasoning_budget`)."""
    body = _capture_request_body(monkeypatch)["body"]

    assert set(body) == {"model", "max_tokens", "messages", "chat_template_kwargs"}
    assert body["model"] == "test-model-v1"
    assert body["max_tokens"] == 4000
    assert body["messages"] == [{"role": "user", "content": "solve the task"}]
    assert set(body["chat_template_kwargs"]) == {"enable_thinking"}


# ---------------------------------------------------------------------------------------------
# Response metadata (finish_reason) and the per-unit request backstop
# ---------------------------------------------------------------------------------------------


def _response(content, finish_reason="stop", completion_tokens=7) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model-v1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 1, "completion_tokens": completion_tokens, "total_tokens": 1 + completion_tokens},
        },
    )


def test_the_reply_carries_the_routes_finish_reason_and_token_count_and_is_still_a_plain_string(monkeypatch):
    from agent.model_client import ModelClient, ModelReply

    _install_mock_transport(monkeypatch, lambda request: _response("```python\nx = (1 +", "length", 4000))
    _set_env(monkeypatch)
    reply = ModelClient().generate_code([{"role": "user", "content": "x"}])

    assert isinstance(reply, ModelReply) and isinstance(reply, str)
    assert reply == "```python\nx = (1 +" and reply.startswith("```python")
    assert reply.finish_reason == "length" and reply.completion_tokens == 4000


def test_a_missing_finish_reason_or_usage_is_reported_as_none_not_guessed(monkeypatch):
    from agent.model_client import ModelClient

    def handler(request):
        return httpx.Response(
            200,
            json={"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "code"}, "finish_reason": None}]},
        )

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)
    reply = ModelClient().generate_code([{"role": "user", "content": "x"}])
    assert reply == "code" and reply.finish_reason is None and reply.completion_tokens is None


def test_empty_content_still_comes_back_as_an_empty_string(monkeypatch):
    from agent.model_client import ModelClient

    _install_mock_transport(monkeypatch, lambda request: _response(None, "stop"))
    _set_env(monkeypatch)
    assert ModelClient().generate_code([{"role": "user", "content": "x"}]) == ""


def test_requests_are_counted_including_failed_ones_and_the_house_allowance_is_a_hard_stop(monkeypatch):
    import openai

    from agent import model_client
    from agent.model_client import ModelClient

    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(500, json={"error": {"message": "boom"}}) if len(sent) == 1 else _response("ok")

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)
    client = ModelClient()
    with pytest.raises(openai.OpenAIError):
        client.generate_code([{"role": "user", "content": "x"}])
    assert client.requests_made == 1  # an admitted request is charged even when it fails

    monkeypatch.setattr(model_client, "MAX_REQUESTS_PER_UNIT", 3)
    client.generate_code([{"role": "user", "content": "x"}])
    client.generate_code([{"role": "user", "content": "x"}])
    assert client.requests_made == 3
    with pytest.raises(RuntimeError, match="allowance"):
        client.generate_code([{"role": "user", "content": "x"}])
    assert len(sent) == 3  # the refused call never reached the wire


def test_the_allowance_constant_matches_the_documented_house_limit():
    from agent.model_client import MAX_REQUESTS_PER_UNIT

    assert MAX_REQUESTS_PER_UNIT == 25


def test_every_request_of_a_continued_generation_disables_thinking_and_keeps_the_cap(monkeypatch):
    """Two requests through the real client: the continuation turn carries the model's own partial reply
    as the assistant message, and is otherwise an ordinary request (thinking off, 4,000 cap, no tools)."""
    from agent.generation import generate_script
    from agent.model_client import ModelClient

    bodies: list[dict] = []
    replies = [
        _response("```python\nimport json\nx = 1\ny = (2 +", "length", 4000),
        _response("```python\ny = 2\nprint(json.dumps([x, y]))\n```", "stop", 20),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return replies[len(bodies) - 1]

    _install_mock_transport(monkeypatch, handler)
    _set_env(monkeypatch)
    client = ModelClient()
    result = generate_script(client, [{"role": "user", "content": "solve"}])

    assert len(bodies) == 2 and client.requests_made == 2
    for body in bodies:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] == 4000
        assert set(body) == {"model", "max_tokens", "messages", "chat_template_kwargs"}
    assert [m["role"] for m in bodies[1]["messages"]] == ["user", "assistant", "user"]
    assert bodies[1]["messages"][1]["content"] == "```python\nimport json\nx = 1\n"  # cut-off line not sent back
    assert result.code == "import json\nx = 1\ny = 2\nprint(json.dumps([x, y]))\n"
