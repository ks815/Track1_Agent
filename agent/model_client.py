"""Thin client for the organizer-hosted House model.

Reaches the model the only way the restricted-network contract allows: $MODEL_ENDPOINT (an
origin, with the OpenAI-compatible API served under /v1) using $MODEL_TOKEN as the bearer, and
$MODEL_NAME as the pinned model id. No vendor-side tools are enabled. One call, one request: SDK
retries are disabled (see _MAX_RETRIES).
"""

from __future__ import annotations

import os

_DEFAULT_MAX_TOKENS = 4000
# The House allowance is 25 ADMITTED requests per unit and an admitted request is charged even if
# it fails, so openai's default max_retries=2 would silently turn one failed call into three
# charged requests. Exactly one request per call; the caller (solve) decides what a failure means.
_MAX_RETRIES = 0
# The House model THINKS BY DEFAULT (docs/HOUSE-MODEL.md, "Development thinking controls"), and
# reasoning is generated output, so it competes for the 4,000-token output cap with the code we
# actually need. The documented off switch is a top-level `chat_template_kwargs.enable_thinking =
# false` in the JSON body, which the OpenAI client sends via extra_body. The other documented
# pass-through options (`low_effort`, `reasoning_budget`) are untested by the operator and not used.
_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


class ModelClient:
    def __init__(self) -> None:
        # Imported here, not at module load time: `openai` is only installed inside the agent
        # Docker image (see Dockerfile). A lazy import lets `agent.cli`/`agent.solve` be imported
        # and the CLI's argparse layer run (e.g. `solve --help`) on a host that lacks the
        # package, and it only fails once something actually tries to talk to the model.
        from openai import OpenAI

        endpoint = os.environ["MODEL_ENDPOINT"].rstrip("/")
        self._client = OpenAI(
            base_url=f"{endpoint}/v1", api_key=os.environ["MODEL_TOKEN"], max_retries=_MAX_RETRIES
        )
        self._model = os.environ["MODEL_NAME"]

    def generate_code(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            messages=messages,
            extra_body=_EXTRA_BODY,
        )
        return response.choices[0].message.content or ""
