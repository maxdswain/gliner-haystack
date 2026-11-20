"""Provide Haystack entity extractors for vLLM Factory's GLiNER2 and GLiNER2.5 plugins."""

import asyncio
from dataclasses import replace
from typing import Any

import httpx2
from haystack import Document, component


def _validate_labels(labels: list[str]) -> None:
    """Reject empty or invalid entity labels before sending a request.

    Parameters
    ----------
    labels : list[str]
        Entity types to extract from the input text.

    Raises
    ------
    ValueError
        If no nonempty string labels are supplied.
    """
    if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must be a non-empty list of non-empty strings")


class _PoolingClient:
    """Shared transport for schema-based GLiNER components."""

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model: str = "/models/current",
        threshold: float = 0.5,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Configure the remote GLiNER text extractor.

        Parameters
        ----------
        api_base_url : str
            Base URL of the vLLM Factory server, without the ``/pooling`` path.
            Defaults to ``http://localhost:8000``.
        model : str
            Model identifier advertised by the server. The supplied Docker
            image uses ``/models/current``.
        threshold : float
            Minimum entity confidence, between 0 and 1, passed to the server.
        http_client_kwargs : dict[str, Any] | None
            Options passed to both ``httpx2.Client`` and ``httpx2.AsyncClient``.
            Defaults to a 30-second timeout; specify, for example,
            ``{"timeout": 120.0}`` to override it.

        Raises
        ------
        ValueError
            If ``threshold`` is outside the interval from 0 to 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.threshold = threshold
        self.http_client_kwargs = dict(http_client_kwargs) if http_client_kwargs is not None else {"timeout": 30.0}
        self.client: httpx2.Client | None = None
        self.async_client: httpx2.AsyncClient | None = None

    def warm_up(self) -> None:
        """Initialize a reusable synchronous HTTP client, if needed.

        The first ``run`` call also warms up the component automatically.
        """
        if self.client is None:
            self.client = httpx2.Client(**self.http_client_kwargs)

    async def warm_up_async(self) -> None:
        """Initialize a reusable asynchronous HTTP client, if needed.

        The first ``run_async`` call also warms up the component automatically.
        """
        if self.async_client is None:
            self.async_client = httpx2.AsyncClient(**self.http_client_kwargs)

    def close(self) -> None:
        """Close the synchronous HTTP client and release its connections.

        Subsequent ``run`` calls create a new client as needed.
        """
        if self.client is not None:
            self.client.close()
            self.client = None

    async def close_async(self) -> None:
        """Close the asynchronous HTTP client and release its connections.

        Subsequent ``run_async`` calls create a new client as needed.
        """
        if self.async_client is not None:
            await self.async_client.aclose()
            self.async_client = None

    def _schema_payload(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": self.model,
            "data": {
                "text": text,
                "schema": schema,
                "runtime": "classic",
                "threshold": self.threshold,
                "include_spans": True,
                "include_confidence": True,
            },
        }


@component
class GLiNERTextExtractor(_PoolingClient):
    """Extract labeled entities from text using a remote GLiNER2 or GLiNER2.5 server.

    Pass entity labels to each call. The ``entities`` output maps each label to
    records containing text, character offsets, and confidence.

    Examples
    --------
    With a pooling server running on localhost:8000, extract mentions and use
    their offsets to recover the exact substring in the source text::

        from haystack_integrations.components.extractors.gliner import GLiNERTextExtractor

        extractor = GLiNERTextExtractor()
        text = "Alice works for Acme."
        result = extractor.run(text=text, labels=["person", "organization"])
        for label, mentions in result["entities"].items():
            for mention in mentions:
                span = text[mention["start"] : mention["end"]]
                print(label, span)  # e.g. person Alice
    """

    def _payload(self, text: str, labels: list[str]) -> dict[str, Any]:
        """Build a schema-based pooling request for text and entity labels.

        Parameters
        ----------
        text : str
            Source text to analyze.
        labels : list[str]
            Entity types to extract.

        Returns
        -------
        dict[str, Any]
            JSON request body with the model, schema, and extraction options.
        """
        _validate_labels(labels)
        return self._schema_payload(text, {"entities": {label: "" for label in labels}})

    @staticmethod
    def _entities(response: httpx2.Response) -> dict[str, list[dict[str, Any]]]:
        """Validate a pooling response and return its labeled entity records.

        Parameters
        ----------
        response : httpx2.Response
            HTTP response from a compatible GLiNER ``/pooling`` endpoint.

        Returns
        -------
        dict[str, list[dict[str, Any]]]
            Entity records grouped by label.

        Raises
        ------
        httpx2.HTTPStatusError
            If the server returns an error status.
        ValueError
            If the response does not contain a labeled entity mapping.
        """
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
        """Extract entities from text synchronously.

        Parameters
        ----------
        text : str
            Text to analyze.
        labels : list[str]
            Nonempty names of entity types to find, such as ``["person"]``.

        Returns
        -------
        dict[str, dict[str, list[dict[str, Any]]]]
            ``{"entities": {label: [records]}}`` with the server's spans
            and confidence values for each detected entity.

        Raises
        ------
        ValueError
            If the labels are invalid or the server's response is malformed.
        httpx2.HTTPStatusError
            If the server returns an error status.
        """
        payload = self._payload(text, labels)
        self.warm_up()
        assert self.client is not None
        response = self.client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"entities": self._entities(response)}

    @component.output_types(entities=dict[str, list[dict[str, Any]]])
    async def run_async(self, text: str, labels: list[str]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Extract entities from text without blocking the event loop.

        This is the asynchronous counterpart of ``run`` and returns the same
        ``entities`` mapping. Use ``await`` to retrieve the result.

        Parameters
        ----------
        text : str
            Text to analyze.
        labels : list[str]
            Nonempty names of entity types to find, such as ``["person"]``.

        Returns
        -------
        dict[str, dict[str, list[dict[str, Any]]]]
            ``{"entities": {label: [records]}}`` with the server's spans
            and confidence values for each detected entity.

        Raises
        ------
        ValueError
            If the labels are invalid or the server's response is malformed.
        httpx2.HTTPStatusError
            If the server returns an error status.
        """
        payload = self._payload(text, labels)
        await self.warm_up_async()
        assert self.async_client is not None
        response = await self.async_client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"entities": self._entities(response)}


@component
class GLiNERRelationExtractor(_PoolingClient):
    """Extract directed relations from text using a relation-capable GLiNER pooling server.

    Each output record contains ``relation_type``, ``head`` and ``tail``. Endpoints
    retain the server's text, confidence and offsets when available. These are
    mentions in the input text, not resolved knowledge graph identities.

    Examples
    --------
    With a relation-capable pooling server running on localhost:8000, request
    a predicate and read the detected mentions in subject-to-object order::

        from haystack_integrations.components.extractors.gliner import GLiNERRelationExtractor

        extractor = GLiNERRelationExtractor()
        result = extractor.run(text="Alice works for Acme.", relation_types=["works_for"])
        for relation in result["relations"]:
            subject = relation["head"]["text"]
            predicate = relation["relation_type"]
            obj = relation["tail"]["text"]
            print(subject, predicate, obj)  # e.g. Alice works_for Acme

    If no relation is detected, ``result["relations"]`` is an empty list.
    """

    def _payload(self, text: str, relation_types: list[str]) -> dict[str, Any]:
        if (
            not isinstance(relation_types, list)
            or not relation_types
            or any(not isinstance(name, str) or not name.strip() for name in relation_types)
        ):
            raise ValueError("relation_types must be a non-empty list of non-empty strings")
        return self._schema_payload(text, {"relations": relation_types})

    @staticmethod
    def _relations(response: httpx2.Response) -> list[dict[str, Any]]:
        response.raise_for_status()
        groups = response.json()["data"]["relation_extraction"]
        return [
            {"relation_type": name, "head": record["head"], "tail": record["tail"]}
            for name, records in groups.items()
            for record in records
        ]

    @component.output_types(relations=list[dict[str, Any]])
    def run(self, text: str, relation_types: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Extract directed head/tail mentions for the supplied relation types."""
        payload = self._payload(text, relation_types)
        self.warm_up()
        assert self.client is not None
        response = self.client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"relations": self._relations(response)}

    @component.output_types(relations=list[dict[str, Any]])
    async def run_async(self, text: str, relation_types: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Extract directed head/tail mentions without blocking the event loop."""
        payload = self._payload(text, relation_types)
        await self.warm_up_async()
        assert self.async_client is not None
        response = await self.async_client.post(f"{self.api_base_url}/pooling", json=payload)
        return {"relations": self._relations(response)}


@component
class GLiNERDocumentExtractor:
    """Enrich copies of Haystack documents with GLiNER2 or GLiNER2.5 entities.

    The component sends each document's text to a vLLM Factory ``/pooling``
    server and adds the labeled entity mapping to ``meta["entities"]`` of
    each returned document. Existing metadata and document content are
    preserved, and input documents are not modified. Use ``run_async`` to
    process a batch concurrently; ``max_concurrency`` limits simultaneous
    HTTP requests.

    Examples
    --------
    Add person and organization entities to document metadata::

        from haystack import Document
        from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor

        extractor = GLiNERDocumentExtractor(max_concurrency=5)
        result = extractor.run(
            documents=[Document(content="Alice works at Acme.")],
            labels=["person", "organization"],
        )
        documents = result["documents"]
        entities = documents[0].meta["entities"]
        extractor.close()

    Use ``await extractor.run_async(documents=..., labels=...)`` inside an
    async application or a Haystack 3 ``Pipeline.run_async`` call.
    """

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model: str = "/models/current",
        threshold: float = 0.5,
        http_client_kwargs: dict[str, Any] | None = None,
        max_concurrency: int = 5,
    ) -> None:
        """Configure the remote GLiNER document extractor.

        Parameters
        ----------
        api_base_url : str
            Base URL of the vLLM Factory server, without the ``/pooling`` path.
            Defaults to ``http://localhost:8000``.
        model : str
            Model identifier advertised by the server. The supplied Docker
            image uses ``/models/current``.
        threshold : float
            Minimum entity confidence, between 0 and 1, passed to the server.
        http_client_kwargs : dict[str, Any] | None
            Options passed to both ``httpx2.Client`` and ``httpx2.AsyncClient``.
            Defaults to a 30-second timeout.
        max_concurrency : int
            Maximum number of concurrent HTTP requests in ``run_async``;
            ``run`` always processes documents sequentially. Defaults to 5.

        Raises
        ------
        ValueError
            If ``max_concurrency`` is not a positive integer or ``threshold``
            is outside the interval from 0 to 1.
        """
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        self.text_extractor = GLiNERTextExtractor(api_base_url, model, threshold, http_client_kwargs)
        self.max_concurrency = max_concurrency

    def warm_up(self) -> None:
        """Initialize a reusable synchronous HTTP client for document requests.

        The first ``run`` call also warms up the component automatically.
        """
        self.text_extractor.warm_up()

    async def warm_up_async(self) -> None:
        """Initialize a reusable asynchronous HTTP client for document requests.

        The first nonempty ``run_async`` call also warms up the component.
        """
        await self.text_extractor.warm_up_async()

    def close(self) -> None:
        """Close the synchronous HTTP client after processing documents."""
        self.text_extractor.close()

    async def close_async(self) -> None:
        """Close the asynchronous HTTP client after processing documents."""
        await self.text_extractor.close_async()

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], labels: list[str]) -> dict[str, list[Document]]:
        """Add entities to copies of documents, processing one at a time.

        Parameters
        ----------
        documents : list[Document]
            Text documents to analyze in input order.
        labels : list[str]
            Nonempty names of entity types to extract from every document.

        Returns
        -------
        dict[str, list[Document]]
            ``{"documents": enriched_documents}`` with entities stored in
            each document's ``meta["entities"]``.

        Raises
        ------
        ValueError
            If labels are invalid, a document has no text content, or a
            response is malformed.
        httpx2.HTTPStatusError
            If the server returns an error status.
        """
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
        """Add entities to document copies with bounded concurrent requests.

        The shared async HTTP client sends at most ``max_concurrency`` requests
        simultaneously. Returned documents retain the input order.

        Parameters
        ----------
        documents : list[Document]
            Text documents to analyze in input order.
        labels : list[str]
            Nonempty names of entity types to extract from every document.

        Returns
        -------
        dict[str, list[Document]]
            ``{"documents": enriched_documents}`` with entities stored in
            each document's ``meta["entities"]``.

        Raises
        ------
        ValueError
            If labels are invalid, a document has no text content, or a
            response is malformed.
        httpx2.HTTPStatusError
            If the server returns an error status.
        """
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
