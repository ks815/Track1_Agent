"""Thin client for the organizer-hosted House model.

Reaches the model the only way the restricted-network contract allows: $MODEL_ENDPOINT (an
origin, with the OpenAI-compatible API served under /v1) using $MODEL_TOKEN as the bearer, and
$MODEL_NAME as the pinned model id. No vendor-side tools are enabled. One call, no retry.
"""

from __future__ import annotations

import os

_DEFAULT_MAX_TOKENS = 4000


class ModelClient:
    def __init__(self) -> None:
        # Imported here, not at module load time: `openai` is only installed inside the agent
        # Docker image (see Dockerfile). A lazy import lets `agent.cli`/`agent.solve` be imported
        # and the CLI's argparse layer run (e.g. `solve --help`) on a host that lacks the
        # package, and it only fails once something actually tries to talk to the model.
        from openai import OpenAI

        endpoint = os.environ["MODEL_ENDPOINT"].rstrip("/")
        self._client = OpenAI(base_url=f"{endpoint}/v1", api_key=os.environ["MODEL_TOKEN"])
        self._model = os.environ["MODEL_NAME"]

    def generate_code(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=_DEFAULT_MAX_TOKENS,
            messages=messages,
        )
        return response.choices[0].message.content or ""
