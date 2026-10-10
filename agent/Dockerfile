# Phase 1 competition-floor agent image.
# Build the shared base once from the repo root first:
#   docker build -t finance-bench-sandbox:latest -f docker/sandbox.Dockerfile .
# Then build this image with this directory as build context:
#   docker build -f agent/Dockerfile -t my-agent:latest agent/
FROM finance-bench-sandbox:latest

# The base already has numpy/pandas/scipy/pyarrow/pytest (docker/requirements-sandbox.txt).
# The only thing it lacks for an api-category agent is an OpenAI-compatible client for the
# House endpoint ($MODEL_ENDPOINT via the audited proxy). Vendor SDKs are of no use here.
# httpx is pinned below 0.28 on purpose: openai 1.40.0 passes `proxies=` to httpx.Client, which
# httpx 0.28 removed, so with an unpinned (latest) httpx every `ModelClient()` raised TypeError and
# the agent exited before making a single model call. httpx is used by nothing else in this image.
RUN pip install --no-cache-dir openai==1.40.0 httpx==0.27.2

WORKDIR /agent
COPY . /agent/agent/

LABEL qfbench2.interface_version="2.0"

ENTRYPOINT ["python", "-m", "agent.cli"]
CMD ["solve", "--task-dir", "/input", "--out", "/app/output"]
