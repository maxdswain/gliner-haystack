# gliner-haystack

[Haystack](https://haystack.deepset.ai/) components for extracting named entities with
[GLiNER2 and GLiNER2.5](https://github.com/fastino-ai/GLiNER2). Extract entities from text
or enrich Haystack `Document` metadata, with synchronous and asynchronous pipeline support.

The Python package is a **client** for a compatible GLiNER server exposing vLLM Factory's
`POST /pooling` endpoint; it does not download or run a model in the Python process.
You can connect to an existing server or use the [Docker images](#serving-gliner-with-docker)
provided by this repository for a production-ready, higher-throughput inference backend.

## Install

```sh
pip install gliner-haystack
# Or: uv add gliner-haystack
```

Requires Python 3.11+ and Haystack 3.x. Start a compatible GLiNER2 or GLiNER2.5 server before
running the examples below. The default endpoint is `http://localhost:8000` and the
model name used in requests is `/models/current` (matching the supplied Compose files).

## Import

```python
from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor, GLiNERTextExtractor
```

## Usage

### Extract entities from text

```python
from haystack_integrations.components.extractors.gliner import GLiNERTextExtractor

extractor = GLiNERTextExtractor(api_base_url="http://localhost:8000")
result = extractor.run(
    text="Alice works at Acme in Paris.",
    labels=["person", "organization", "location"],
)
print(result["entities"])
extractor.close()
```

`entities` is a mapping from label to a list of records. For example (confidence values
are illustrative):

```python
{
    "person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.98}],
    "organization": [{"text": "Acme", "start": 15, "end": 19, "confidence": 0.94}],
    "location": [{"text": "Paris", "start": 23, "end": 28, "confidence": 0.96}],
}
```

Labels are **run inputs**, so you can reuse the same component with different labels.
The extractor requests character spans and confidence from the server; returned records
reflect the server's actual detections.

### Enrich Haystack documents

```python
from haystack import Document
from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor

extractor = GLiNERDocumentExtractor(api_base_url="http://localhost:8000")
result = extractor.run(
    documents=[Document(content="Alice works at Acme in Paris.", meta={"source": "example"})],
    labels=["person", "organization", "location"],
)
enriched = result["documents"][0]
print(enriched.meta["entities"])
print(enriched.meta["source"])  # "example"
extractor.close()
```

The component returns new documents, preserves their existing metadata, and adds
`meta["entities"]`; it does not change the input documents or their text.

### Use in a Haystack pipeline

```python
from haystack import Pipeline
from haystack_integrations.components.extractors.gliner import GLiNERTextExtractor

extractor = GLiNERTextExtractor()
pipeline = Pipeline()
pipeline.add_component("entities", extractor)
result = pipeline.run({
    "entities": {
        "text": "Alice works at Acme in Paris.",
        "labels": ["person", "organization", "location"],
    }
})
print(result["entities"]["entities"])
extractor.close()
```

In Haystack 3, `Pipeline.run_async(...)` calls the components' `run_async(...)` methods.
To process multiple documents concurrently, use `GLiNERDocumentExtractor.run_async`:

```python
import asyncio
from haystack import Document
from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor

async def main():
    extractor = GLiNERDocumentExtractor(max_concurrency=5)
    result = await extractor.run_async(
        documents=[Document(content="Alice works at Acme."), Document(content="Bob lives in Paris.")],
        labels=["person", "organization", "location"],
    )
    await extractor.close_async()
    return result["documents"]

documents = asyncio.run(main())
```

The async document extractor uses a shared HTTP client and a semaphore to limit
in-flight requests to `max_concurrency` (default: 5), preserving document order.
Synchronous `run` sends requests sequentially. Clients are initialized lazily; call
`warm_up()` or `warm_up_async()` to prepare them ahead of time, and `close()` or
`close_async()` when finished.

### Configuration

Both components accept `api_base_url` (default `http://localhost:8000`), `model`
(default `/models/current`), `threshold` (default `0.5`), and `http_client_kwargs`
for [httpx2](https://github.com/pydantic/httpx2) client options. For example:

```python
extractor = GLiNERTextExtractor(
    api_base_url="https://inference.example.com",
    model="/models/current",
    threshold=0.3,
    http_client_kwargs={"timeout": 120.0},
)
```

Without `http_client_kwargs`, the client timeout is 30 seconds. The document component
also accepts `max_concurrency` for async extraction. Both components require nonempty
`labels` on every `run` / `run_async` call.

### Compatible model families

The components use vLLM Factory's **schema-based entity extraction** contract. They
work with GLiNER2.5 boundary models (for example, `fastino/gliner2.5-multi-v1`
with `deberta_gliner25_io`) and GLiNER2 schema models (for example,
`fastino/gliner2-large-v1` with `deberta_gliner2_io`). The `model` constructor
argument must match the model name exposed by your server; the supplied Compose files
serve either model as `/models/current`. Both plugins accept the components' entity
schema, threshold, and span/confidence options. The GLiNER2 plugin ignores the
GLiNER2.5-specific `runtime="classic"` request field.

vLLM Factory also serves older span-based GLiNER models, but those plugins take a
**different** `labels` request and return flat entity lists; these two components do
not support that response format. They are entity extractors, not wrappers for other
GLiNER2 tasks such as relation extraction or classification.

## Serving GLiNER with Docker

For production inference, this repository publishes vLLM Factory-based images
configured for GLiNER2.5 by default, for NVIDIA GPUs and x86-64 CPUs. Both serve `POST /pooling` on port 8000.
The GPU image is intended for higher-throughput deployments; the CPU image also
allows development and integration testing on machines without a supported GPU.

| Image | Compose file | Requirements |
| --- | --- | --- |
| `ghcr.io/maxdswain/vllm-gliner:latest` | `compose.yaml` | Supported NVIDIA GPU and NVIDIA Container Toolkit |
| `ghcr.io/maxdswain/vllm-gliner:latest-cpu` | `compose.cpu.yaml` | x86-64 CPU |

```sh
# From a clone of this repository, choose one:
docker compose up -d                      # GPU
# docker compose -f compose.cpu.yaml up -d  # CPU instead
```

The first startup downloads and prepares `fastino/gliner2.5-multi-v1` in a persistent
Hugging Face cache; wait until the server is ready before sending requests. With either
Compose file, the Python examples above work without changes. To serve GLiNER2 instead,
change the Compose file's environment entries to `MODEL_ID: fastino/gliner2-large-v1`,
`MODEL_PLUGIN: deberta_gliner2`, and `IO_PLUGIN: deberta_gliner2_io` (keep the
`/models/current` path). To build an image yourself,
initialize the submodule and build with `docker build -t vllm-gliner .`; the Dockerfile
supports a `VLLM_BASE_IMAGE` build argument for selecting the CPU base image.

You can also call the server directly:

```sh
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/current","data":{"text":"Alice works at Acme in Paris.","schema":{"entities":{"person":"","organization":"","location":""}},"runtime":"classic","include_spans":true,"include_confidence":true}}'
```

## Development

```sh
uv sync --group dev --group test
uv run --group dev --group test poe check
```

`poe check` runs offline pytest tests, Ruff lint/format checks (120-character lines),
and Pyrefly type checks. For integration tests against a running GLiNER server:

```sh
GLINER_INTEGRATION_URL=http://localhost:8000 uv run --group dev --group test poe test-integration
```

Without `GLINER_INTEGRATION_URL`, integration tests skip. CPU-only serving works for
these tests; a GPU is not required.

Python package checks and publishing are defined in `.github/workflows/publish-package.yml`.
After checks pass on `main`, releasable Conventional Commits drive git-cliff release
notes, a version bump, a GitHub Release, and PyPI Trusted Publishing via the `pypi`
GitHub environment. Docker image publishing is a separate workflow that rebuilds only
when image inputs change.
