# gliner-haystack

Serve GLiNER2.5 with vLLM Factory using Docker Compose. GPU and CPU images are available:

| Image | Example Compose file | Requirements |
| --- | --- | --- |
| `ghcr.io/maxdswain/vllm-gliner:latest` (GPU) | [compose.yaml](compose.yaml) | NVIDIA GPU and NVIDIA Container Toolkit |
| `ghcr.io/maxdswain/vllm-gliner:latest-cpu` (x86-64 CPU) | [compose.cpu.yaml](compose.cpu.yaml) | x86-64 CPU |

## Haystack components

Install with `uv sync --group dev --group test`. The text component returns labeled entity spans;
the document component stores them in copied documents' `meta["entities"]`:

```python
from haystack import Document
from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor, GLiNERTextExtractor

labels = ["person", "organization", "location"]
text_extractor = GLiNERTextExtractor(api_base_url="http://localhost:8000", http_client_kwargs={"timeout": 30.0})
document_extractor = GLiNERDocumentExtractor(max_concurrency=5)
try:
    entities = text_extractor.run("Alice works at Acme in Paris.", labels)["entities"]
    documents = document_extractor.run([Document(content="Alice works at Acme in Paris.")], labels)["documents"]
finally:
    text_extractor.close()
    document_extractor.close()
```

Both components support `run_async` via Haystack 3's `Pipeline.run_async`. Document `run_async` sends requests
concurrently, with a semaphore enforcing `max_concurrency`; synchronous `run` processes documents sequentially.
Call `warm_up()` / `warm_up_async()` to prepare reusable HTTP clients if desired (runs also warm up lazily).
Call `close()` / `close_async()` to release them when done. HTTP client options such as `timeout` can be passed
via `http_client_kwargs` in either constructor.

Offline tests, lint, formatting and type checks: `uv run --group dev --group test poe check`.
To run tests against a live server:
`GLINER_INTEGRATION_URL=http://localhost:8000 uv run --group dev --group test poe test-integration`.
Integration tests skip when the variable is unset.

## Python package releases

`.github/workflows/publish-package.yml` runs offline checks on pull requests and pushes to `main`
(Python 3.11 and 3.13). After successful checks on `main`, git-cliff finds releasable
Conventional Commits (`feat`, `fix`, `perf`, or breaking changes). If there are any, the
workflow updates `pyproject.toml` and `uv.lock`, builds distributions, commits the version,
creates a `vX.Y.Z` tag and GitHub Release with generated notes, then publishes those
artifacts to PyPI using the `pypi` GitHub environment and PyPI Trusted Publishing. The
initial release is `v0.1.0`; Docker-scoped commits do not release the Python package.
The release job needs permission to push its version commit and tag to `main` (check branch
protection rules). GitHub's `GITHUB_TOKEN` does not trigger another workflow when it creates
a release, so the PyPI publish job runs in the same workflow after the release job succeeds.

## Quick start

Install Docker with Compose, then:

```sh
git clone https://github.com/maxdswain/gliner-haystack.git
cd gliner-haystack

# GPU (default)
docker compose up

# Or CPU
docker compose -f compose.cpu.yaml up
```

The first start downloads the model. Once the server is ready, try:

```sh
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true}}'
```
