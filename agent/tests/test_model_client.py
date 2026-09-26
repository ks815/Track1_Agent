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
