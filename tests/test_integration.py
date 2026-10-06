"""Live GLiNER2.5 pooling checks; enable with GLINER_INTEGRATION_URL."""

import asyncio
import os

import pytest
from haystack import Document

from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor, GLiNERTextExtractor

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def server_url() -> str:
    url = os.environ.get("GLINER_INTEGRATION_URL")
    if not url:
        pytest.skip("Set GLINER_INTEGRATION_URL to run against a live vLLM GLiNER2.5 server")
    return url


def test_live_text_extraction(server_url: str) -> None:
    extractor = GLiNERTextExtractor(api_base_url=server_url, http_client_kwargs={"timeout": 180})
    try:
        entities = extractor.run("Alice works at Acme in Paris.", ["person", "organization", "location"])["entities"]
    finally:
        extractor.close()
    assert isinstance(entities, dict)
    assert "person" in entities
    assert any(entity["text"] == "Alice" for entity in entities["person"])
    for records in entities.values():
        for entity in records:
            assert "start" in entity and "end" in entity and "confidence" in entity


def test_live_async_text_extraction(server_url: str) -> None:
    extractor = GLiNERTextExtractor(api_base_url=server_url, http_client_kwargs={"timeout": 180})

    async def extract() -> dict:
        try:
            return (await extractor.run_async("Alice works at Acme in Paris.", ["person", "organization", "location"]))[
                "entities"
            ]
        finally:
            await extractor.close_async()

    entities = asyncio.run(extract())
    assert any(entity["text"] == "Alice" for entity in entities["person"])


def test_live_async_document_extraction(server_url: str) -> None:
    documents = [
        Document(content="Alice works at Acme in Paris.", meta={"source": "integration"}),
        Document(content="Bob works at Acme in London."),
    ]
    extractor = GLiNERDocumentExtractor(api_base_url=server_url, http_client_kwargs={"timeout": 180}, max_concurrency=2)

    async def extract() -> list[Document]:
        try:
            return (await extractor.run_async(documents, ["person", "organization", "location"]))["documents"]
        finally:
            await extractor.close_async()

    results = asyncio.run(extract())
    assert len(results) == 2
    assert results[0].meta["source"] == "integration"
    assert any(entity["text"] == "Alice" for entity in results[0].meta["entities"]["person"])
    assert results[1].meta["entities"]["person"]
    assert all("entities" not in doc.meta for doc in documents)
