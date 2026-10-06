"""Haystack entity extractors backed by vLLM Factory's GLiNER2.5 pooling API."""

import asyncio
from dataclasses import replace
from typing import Any

import httpx2
from haystack import Document, component


def _validate_labels(labels: list[str]) -> None:
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be a non-empty list of non-empty strings")


@component
class GLiNERTextExtractor:
    """Extract labeled entities from text using the vLLM GLiNER2.5 server.

    The ``entities`` mapping has one list per label. Each record includes
    ``text``, ``start``, ``end``, and ``confidence`` when provided by the server.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model: str = "/models/current",
        threshold: float = 0.5,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.http_client_kwargs = dict(http_client_kwargs) if http_client_kwargs is not None else {"timeout": 30.0}
        self.client: httpx2.Client | None = None
        self.async_client: httpx2.AsyncClient | None = None

    def warm_up(self) -> None:
        """Create and retain the synchronous HTTP client."""
        if self.client is None:
            self.client = httpx2.Client(**self.http_client_kwargs)

    async def warm_up_async(self) -> None:
        """Create and retain the asynchronous HTTP client."""
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(**self.http_client_kwargs)

    def close(self) -> None:
        """Release the synchronous client's connections."""
        if self.client is not None:
            self.client.close()
            self.client = None

    async def close_async(self) -> None:
        """Release the asynchronous client's connections."""
        if self.async_client is not None:
            await self.async_client.aclose()
            self.async_client = None

    def _payload(self, text: str, labels: list[str]) -> dict[str, Any]:
        _validate_labels(labels)
        return {
            "model": self.model,
            "data": {
                "text": text,
                "schema": {"entities": {label: "" for label in labels}},
                "runtime": "classic",
                "threshold": self.threshold,
                "include_spans": True,
                "include_confidence": True,
            },
        }

    @staticmethod
    def _entities(response: httpx2.Response) -> dict[str, list[dict[str, Any]]]:
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ValueError("Invalid vLLM GLiNER pooling response: expected a data object")  # noqa: TRY004
        entities = payload["data"].get("entities")
        if not isinstance(entities, dict) or any(
            not isinstance(label, str)
            or not isinstance(records, list)
            or any(not isinstance(record, dict) for record in records)
            for label, records in entities.items()
        ):
            raise ValueError("Invalid vLLM GLiNER pooling response: expected labeled entity lists")
        return entities

    @component.output_types(entities=dict[str, list[dict[str, Any]]])
    def run(self, text: str, labels: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Call ``POST /pooling`` synchronously."""
        payload = self._payload(text, labels)
        self.warm_up()
        assert self.client is not None
        response = self.client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"entities": self._entities(response)}

    @component.output_types(entities=dict[str, list[dict[str, Any]]])
    async def run_async(self, text: str, labels: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Call ``POST /pooling`` without blocking the event loop."""
        payload = self._payload(text, labels)
        await self.warm_up_async()
        assert self.async_client is not None
        response = await self.async_client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"entities": self._entities(response)}


@component
class GLiNERDocumentExtractor:
    """Return copies of documents with labeled entities in ``meta['entities']``."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model: str = "/models/current",
        threshold: float = 0.5,
        http_client_kwargs: dict[str, Any] | None = None,
        max_concurrency: int = 5,
    ) -> None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.text_extractor = GLiNERTextExtractor(api_base_url, model, threshold, http_client_kwargs)
        self.max_concurrency = max_concurrency

    def warm_up(self) -> None:
        """Prepare the synchronous HTTP client."""
        self.text_extractor.warm_up()

    async def warm_up_async(self) -> None:
        """Prepare the asynchronous HTTP client."""
        await self.text_extractor.warm_up_async()

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        self.text_extractor.close()

    async def close_async(self) -> None:
        """Close the asynchronous HTTP client."""
        await self.text_extractor.close_async()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], labels: list[str]) -> dict[str, list[Document]]:
        """Extract entities synchronously per document, without mutating inputs."""
        _validate_labels(labels)
        result = []
        for document in documents:
            if document.content is None:
                raise ValueError("GLiNERDocumentExtractor requires text document content")
            entities = self.text_extractor.run(document.content, labels)["entities"]
            result.append(replace(document, meta={**document.meta, "entities": entities}))
        return {"documents": result}

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document], labels: list[str]) -> dict[str, list[Document]]:
        """Extract concurrently, bounding in-flight requests with a semaphore."""
        _validate_labels(labels)
        for document in documents:
            if document.content is None:
                raise ValueError("GLiNERDocumentExtractor requires text document content")

        semaphore = asyncio.Semaphore(self.max_concurrency)
        extractor = self.text_extractor
        if not documents:
            return {"documents": []}
        await self.warm_up_async()
        assert extractor.async_client is not None
        client = extractor.async_client

        async def extract(document: Document) -> Document:
            async with semaphore:
                response = await client.post(
                    f"{extractor.api_base_url}/pooling", json=extractor._payload(document.content or "", labels)
                )
                entities = extractor._entities(response)
            return replace(document, meta={**document.meta, "entities": entities})

        return {"documents": list(await asyncio.gather(*(extract(document) for document in documents)))}
