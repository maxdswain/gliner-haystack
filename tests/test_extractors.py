"""Offline component tests: no model download or GPU required."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx2
import pytest
from haystack import Document, Pipeline

from haystack_integrations.components.extractors.gliner import GLiNERDocumentExtractor, GLiNERTextExtractor

ENTITIES = {"person": [{"text": "Alice", "start": 0, "end": 5, "confidence": 0.98}]}


def response(payload: dict, status: int = 200) -> httpx2.Response:
    return httpx2.Response(status, json=payload, request=httpx2.Request("POST", "http://localhost:8000/pooling"))


def test_text_extractor_sends_gliner25_pooling_request() -> None:
    extractor = GLiNERTextExtractor(
        api_base_url="http://localhost:8000/", threshold=0.3, http_client_kwargs={"timeout": 4}
    )
    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        client_cls.return_value.post.return_value = response({"data": {"entities": ENTITIES}})
        result = extractor.run("Alice works at Acme", ["person", "organization"])
        extractor.close()

    assert result == {"entities": ENTITIES}
    client_cls.assert_called_once_with(timeout=4)
    url, kwargs = client_cls.return_value.post.call_args
    assert url == ("http://localhost:8000/pooling",)
    assert kwargs["json"] == {
        "model": "/models/current",
        "data": {
            "text": "Alice works at Acme",
            "schema": {"entities": {"person": "", "organization": ""}},
            "runtime": "classic",
            "threshold": 0.3,
            "include_spans": True,
            "include_confidence": True,
        },
    }


def test_document_extractor_preserves_inputs_and_metadata() -> None:
    original = Document(content="Alice", meta={"source": "test"})
    extractor = GLiNERDocumentExtractor()
    with patch.object(extractor.text_extractor, "run", return_value={"entities": ENTITIES}) as run:
        documents = extractor.run([original, Document(content="Bob")], ["person"])["documents"]

    assert run.call_count == 2
    run.assert_any_call("Alice", ["person"])
    assert documents[0].id == original.id
    assert documents[0].meta == {"source": "test", "entities": ENTITIES}
    assert original.meta == {"source": "test"}
    assert documents[0] is not original
    assert documents[1].meta == {"entities": ENTITIES}


def test_empty_document_list_makes_no_requests() -> None:
    extractor = GLiNERDocumentExtractor()
    with patch.object(extractor.text_extractor, "run") as run:
        assert extractor.run([], ["person"]) == {"documents": []}
    run.assert_not_called()


@pytest.mark.parametrize("labels", [[], [""], ["person", " "]])
def test_invalid_labels(labels: list[str]) -> None:
    with pytest.raises(ValueError, match="labels"):
        GLiNERTextExtractor().run("Alice", labels)
    with pytest.raises(ValueError, match="labels"):
        GLiNERDocumentExtractor().run([], labels)


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": {}}, {"data": {"entities": []}}])
def test_invalid_response(payload: dict) -> None:
    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        client_cls.return_value.post.return_value = response(payload)
        with pytest.raises(ValueError, match="pooling response"):
            GLiNERTextExtractor().run("Alice", ["person"])


def test_http_error_is_propagated() -> None:
    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        client_cls.return_value.post.return_value = response({}, status=500)
        with pytest.raises(httpx2.HTTPStatusError):
            GLiNERTextExtractor().run("Alice", ["person"])


def test_non_text_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="text document"):
        GLiNERDocumentExtractor().run([Document(content=None)], ["person"])


def test_invalid_concurrency() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_concurrency"):
            GLiNERDocumentExtractor(max_concurrency=value)


def test_components_work_in_haystack_pipeline() -> None:
    pipeline = Pipeline()
    extractor = GLiNERTextExtractor()
    pipeline.add_component("extractor", extractor)
    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        client_cls.return_value.post.return_value = response({"data": {"entities": ENTITIES}})
        assert pipeline.run({"extractor": {"text": "Alice", "labels": ["person"]}}) == {
            "extractor": {"entities": ENTITIES}
        }
        extractor.close()


def test_async_text_and_pipeline() -> None:
    async def scenario() -> None:
        async def handler(url: str, *, json: dict) -> httpx2.Response:
            assert url.endswith("/pooling")
            assert json["data"]["text"] == "Alice"
            return response({"data": {"entities": ENTITIES}})

        extractor = GLiNERTextExtractor()
        with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.AsyncClient") as client_cls:
            client = client_cls.return_value
            client.aclose = AsyncMock()
            client.post = AsyncMock(side_effect=handler)
            assert await extractor.run_async("Alice", ["person"]) == {"entities": ENTITIES}
            pipeline = Pipeline()
            pipeline.add_component("extractor", extractor)
            assert await pipeline.run_async({"extractor": {"text": "Alice", "labels": ["person"]}}) == {
                "extractor": {"entities": ENTITIES}
            }
            await extractor.close_async()
        client_cls.assert_called_once_with(timeout=30.0)
        assert client.post.await_count == 2

    asyncio.run(scenario())


def test_async_documents_are_bounded_and_ordered() -> None:
    async def scenario() -> None:
        active = 0
        peak = 0

        async def handler(url: str, *, json: dict) -> httpx2.Response:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return response({"data": {"entities": {"person": [{"text": json["data"]["text"]}]}}})

        docs = [Document(content=str(i)) for i in range(7)]
        extractor = GLiNERDocumentExtractor(max_concurrency=2)
        with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.AsyncClient") as client_cls:
            client_cls.return_value.post = AsyncMock(side_effect=handler)
            client_cls.return_value.aclose = AsyncMock()
            result = (await extractor.run_async(docs, ["person"]))["documents"]
            await extractor.close_async()
        assert peak == 2
        assert [doc.content for doc in result] == [str(i) for i in range(7)]
        assert [doc.meta["entities"]["person"][0]["text"] for doc in result] == [str(i) for i in range(7)]
        assert all(not doc.meta for doc in docs)

    asyncio.run(scenario())


def test_async_empty_documents() -> None:
    assert asyncio.run(GLiNERDocumentExtractor().run_async([], ["person"])) == {"documents": []}


def test_text_client_lifecycle_and_changing_labels() -> None:
    extractor = GLiNERTextExtractor(http_client_kwargs={"timeout": 11, "headers": {"x-test": "yes"}})
    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        client_cls.return_value.post.return_value = response({"data": {"entities": ENTITIES}})
        extractor.warm_up()
        extractor.warm_up()
        extractor.run("Alice", ["person"])
        extractor.run("Acme", ["organization"])
        assert client_cls.call_count == 1
        client_cls.assert_called_with(timeout=11, headers={"x-test": "yes"})
        assert [
            call.kwargs["json"]["data"]["schema"]["entities"] for call in client_cls.return_value.post.call_args_list
        ] == [
            {"person": ""},
            {"organization": ""},
        ]
        extractor.close()
        extractor.close()
        client_cls.return_value.close.assert_called_once_with()
        assert extractor.client is None
        extractor.warm_up()
        assert client_cls.call_count == 2
        extractor.close()


def test_document_client_lifecycle() -> None:
    extractor = GLiNERDocumentExtractor(http_client_kwargs={"timeout": 12})

    async def scenario() -> None:
        with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.AsyncClient") as client_cls:
            client_cls.return_value.aclose = AsyncMock()
            await extractor.warm_up_async()
            await extractor.warm_up_async()
            client_cls.assert_called_once_with(timeout=12)
            await extractor.close_async()
            await extractor.close_async()
            client_cls.return_value.aclose.assert_awaited_once_with()
            assert extractor.text_extractor.async_client is None

    with patch("haystack_integrations.components.extractors.gliner.extractors.httpx2.Client") as client_cls:
        extractor.warm_up()
        extractor.warm_up()
        client_cls.assert_called_once_with(timeout=12)
        extractor.close()
        client_cls.return_value.close.assert_called_once_with()
    asyncio.run(scenario())
