"""Fault-injected model lifecycle tests; no real weight downloads or GPU claims."""

import asyncio
import math
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, APITimeoutError

from app.core.config import Settings
from app.services import embedding_service as embeddings, llm_service as llms
from scripts import prepare_models
from model_fakes import client_stub


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "_create_directories", lambda self: None)
    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_MODEL="example/model",
                   LOCAL_EMBEDDING_REVISION="fixed-revision", LOCAL_EMBEDDING_DEVICE="cpu",
                   LOCAL_EMBEDDING_CACHE_DIR=str(tmp_path), LOCAL_EMBEDDING_BATCH_SIZE=8,
                   GENERATION_API_BASE_URL="https://generation.example/v1",
                   GENERATION_API_KEY="test-secret", GENERATION_MODEL="generation-model")
    monkeypatch.setattr(embeddings, "settings", cfg)
    monkeypatch.setattr(llms, "settings", cfg)
    monkeypatch.setattr(llms.LLMService, "_global_semaphore", None, raising=False)
    monkeypatch.setattr(llms.LLMService, "_failure_count", 0, raising=False)
    monkeypatch.setattr(llms.LLMService, "_circuit_opened_at", None, raising=False)
    monkeypatch.setattr(embeddings, "_snapshot_download", lambda **kw: str(tmp_path / "fixed-revision"))
    return cfg


class LocalModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.encode_kwargs = kwargs["encode_kwargs"]
        self._client = client_stub()

    def embed_documents(self, texts):
        return [[1.0, 0.5, 0.25] for _ in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_cache_is_loaded_without_network_and_reports_actual_revision(runtime, monkeypatch, tmp_path):
    requests, loads = [], []

    def snapshot(**kwargs):
        requests.append(kwargs)
        assert kwargs["local_files_only"] is True
        return str(tmp_path / "snapshots" / "actual-commit")

    def load(**kwargs):
        loads.append(kwargs)
        assert kwargs["model_kwargs"]["local_files_only"] is True
        return LocalModel(**kwargs)

    monkeypatch.setattr(embeddings, "_snapshot_download", snapshot)
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: load)
    service = embeddings.EmbeddingService()
    assert service.diagnostics()["status"] == "not_checked"
    result = service.preflight()
    assert result["status"] == "ready" and result["dimension"] == 3
    assert result["actual_revision"] == "actual-commit"
    assert len(requests) == len(loads) == 1
    assert requests[0]["repo_id"] == "example/model"
    assert requests[0]["revision"] == "fixed-revision"
    assert requests[0]["token"] is False


def test_first_download_and_later_offline_load(runtime, monkeypatch, tmp_path):
    downloaded, calls = False, []

    def snapshot(**kwargs):
        nonlocal downloaded
        calls.append(kwargs)
        if kwargs["local_files_only"] and not downloaded:
            raise FileNotFoundError("cache missing")
        downloaded = True
        return str(tmp_path / "fixed-revision")

    monkeypatch.setattr(embeddings, "_snapshot_download", snapshot)
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: LocalModel)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="prepare_models --download"):
        embeddings.EmbeddingService()
    assert not downloaded
    service = embeddings.EmbeddingService(allow_download=True)
    assert service.preflight()["status"] == "ready"
    offline = embeddings.EmbeddingService()
    assert offline.embed_query("private question") == [1, 0.5, 0.25]
    assert [c["local_files_only"] for c in calls] == [True, True, False, True]
    assert all(set(c) == {"repo_id", "revision", "cache_dir", "local_files_only", "token", "max_workers"}
               for c in calls)  # No documents, questions or generation credentials.


def test_partial_cache_is_repaired_only_with_download_permission(runtime, monkeypatch, tmp_path):
    complete, downloads = False, []

    def snapshot(**kwargs):
        nonlocal complete
        downloads.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            complete = True
        return str(tmp_path / "fixed-revision")

    def load(**kwargs):
        if not complete:
            raise OSError("tokenizer/weights missing")
        return LocalModel(**kwargs)

    monkeypatch.setattr(embeddings, "_snapshot_download", snapshot)
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: load)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="no hash/remote"):
        embeddings.EmbeddingService()
    assert downloads == [True]
    assert embeddings.EmbeddingService(allow_download=True).preflight()["status"] == "ready"
    assert downloads == [True, True, False]


def test_download_failure_does_not_load_another_model(runtime, monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("download-source-unreachable")

    monkeypatch.setattr(embeddings, "_snapshot_download", fail)
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: pytest.fail("Must not load another model"))
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="cache unavailable"):
        embeddings.EmbeddingService(allow_download=True)


def test_local_model_directory_uses_no_download_service(runtime, monkeypatch, tmp_path):
    runtime.LOCAL_EMBEDDING_MODEL = str(tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    monkeypatch.setattr(embeddings, "_snapshot_download", lambda **kw: pytest.fail("Unexpected download"))
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: LocalModel)
    assert embeddings.EmbeddingService().preflight()["status"] == "ready"


@pytest.mark.parametrize("recover_on_gpu", [True, False])
def test_gpu_oom_reduces_batch_then_uses_same_model_cpu(runtime, monkeypatch, recover_on_gpu):
    runtime.LOCAL_EMBEDDING_DEVICE = "auto"
    runtime.LOCAL_EMBEDDING_MAX_BATCH_RETRIES = 2
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    loads, attempts = [], []

    class MemoryLimited(LocalModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            loads.append(kwargs)

        def embed_documents(self, texts):
            device = self.kwargs["model_kwargs"]["device"]
            attempts.append((device, self.encode_kwargs["batch_size"]))
            if device == "cuda" and (not recover_on_gpu or self.encode_kwargs["batch_size"] > 2):
                raise RuntimeError("CUDA out of memory")
            return super().embed_documents(texts)

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: MemoryLimited)
    service = embeddings.EmbeddingService()
    result = service.embed_documents(["private text"] * 5)
    assert len(result) == 5
    assert attempts[:3] == [("cuda", 8), ("cuda", 4), ("cuda", 2)]
    assert service.device == ("cuda" if recover_on_gpu else "cpu")
    assert len(loads) == (1 if recover_on_gpu else 2)
    assert all(load["model_name"] == loads[0]["model_name"] for load in loads)
    assert all(load["model_kwargs"]["revision"] == "fixed-revision" for load in loads)
    assert all(load["encode_kwargs"]["normalize_embeddings"] is True for load in loads)
    if not recover_on_gpu:
        assert any(w["code"] == "cpu_fallback" for w in service.diagnostics()["warnings"])


def test_load_oom_releases_gpu_before_cpu_model_load(runtime, monkeypatch):
    runtime.LOCAL_EMBEDDING_DEVICE = "cuda:0"
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    devices = []

    def load(**kwargs):
        devices.append(kwargs["model_kwargs"]["device"])
        if devices[-1].startswith("cuda"):
            raise RuntimeError("CUDA out of memory")
        return LocalModel(**kwargs)

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: load)
    service = embeddings.EmbeddingService()
    assert devices == ["cuda:0", "cpu"]
    assert service.preflight()["device"] == "cpu"


def test_no_gpu_uses_cpu_and_cpu_failure_is_final(runtime, monkeypatch):
    runtime.LOCAL_EMBEDDING_DEVICE = "auto"
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: False)
    calls = []

    def load(**kwargs):
        calls.append(kwargs["model_kwargs"]["device"])
        raise RuntimeError("CPU cannot load model")

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: load)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="no hash/remote"):
        embeddings.EmbeddingService()
    assert calls == ["cpu"]


def test_cpu_encode_failure_after_gpu_oom_is_final(runtime, monkeypatch):
    runtime.LOCAL_EMBEDDING_DEVICE = "auto"
    runtime.LOCAL_EMBEDDING_MAX_BATCH_RETRIES = 0
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    calls = []

    class Broken(LocalModel):
        def embed_documents(self, texts):
            device = self.kwargs["model_kwargs"]["device"]
            calls.append(device)
            raise RuntimeError("CUDA out of memory" if device == "cuda" else "CPU failed")

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Broken)
    service = embeddings.EmbeddingService()
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="ingestion unavailable"):
        service.embed_documents(["text"])
    assert calls == ["cuda", "cpu"]
    assert not service.preflight_passed


@pytest.mark.parametrize("vectors", [[], [[1, 2]], [[1, 2], [1, 2, 3]],
                                      [[math.nan], [1]], [[math.inf], [1]],
                                      [[0, 0], [0, 0]], [[True], [1]], [["1"], [1]]])
def test_invalid_vectors_fail_preflight(runtime, monkeypatch, vectors):
    class Invalid(LocalModel):
        def embed_documents(self, texts):
            return vectors

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Invalid)
    service = embeddings.EmbeddingService()
    with pytest.raises(embeddings.LocalEmbeddingUnavailable):
        service.preflight()
    assert service.diagnostics()["status"] != "ready"


def test_dimension_cannot_change_between_query_and_documents(runtime, monkeypatch):
    class Invalid(LocalModel):
        def embed_query(self, text):
            return [1, 2]

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Invalid)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable):
        embeddings.EmbeddingService().preflight()


def test_cross_device_profile_vectors_and_ranking_are_consistent(runtime, monkeypatch):
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)

    class Consistent(LocalModel):
        def embed_documents(self, texts):
            epsilon = 1e-7 if self.kwargs["model_kwargs"]["device"] == "cuda" else 0
            return [[1 + epsilon, 0.1] if t == "waiting period" else [0.1, 1 + epsilon] for t in texts]

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Consistent)
    runtime.LOCAL_EMBEDDING_DEVICE = "cpu"
    cpu = embeddings.EmbeddingService()
    runtime.LOCAL_EMBEDDING_DEVICE = "cuda"
    gpu = embeddings.EmbeddingService()
    texts = ["waiting period", "exclusion"]
    cpu_vectors, gpu_vectors = cpu.embed_documents(texts), gpu.embed_documents(texts)
    for left, right in zip(cpu_vectors, gpu_vectors):
        assert left == pytest.approx(right, abs=1e-5)
    query = cpu.embed_query(texts[0])
    assert max(range(2), key=lambda i: sum(a*b for a, b in zip(query, cpu_vectors[i]))) == 0
    assert max(range(2), key=lambda i: sum(a*b for a, b in zip(query, gpu_vectors[i]))) == 0
    assert cpu.actual_revision == gpu.actual_revision


@pytest.mark.parametrize("status,body,code,calls", [
    (401, {}, "credentials", 1), (403, {}, "credentials", 1),
    (404, {}, "configuration", 1), (429, {"code": "insufficient_quota"}, "quota", 1),
    (429, {"code": "rate_limit_exceeded"}, "rate_limit", 3), (503, {}, "server", 3),
])
def test_generation_error_classification_and_retry_bound(runtime, monkeypatch, status, body, code, calls):
    seen, delays = [], []

    def handle(request):
        seen.append(request)
        return httpx.Response(status, json={"error": {"message": "test-secret private text", **body}})

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(llms.asyncio, "sleep", sleep)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: AsyncOpenAI(**kw, http_client=http))
            return await llms.LLMService().preflight()

    result = asyncio.run(run())
    assert result["status"] == "unavailable" and result["code"] == code
    assert len(seen) == calls and len(delays) == calls - 1
    assert all(0 <= delay <= runtime.LLM_BACKOFF_MAX for delay in delays)
    assert "test-secret" not in str(result)


def test_generation_timeout_and_recovery(runtime, monkeypatch):
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("private text", request=request)
        return httpx.Response(200, json={"id": "ok", "object": "chat.completion", "created": 0,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}], "model": "generation-model"})

    runtime.LLM_BACKOFF_BASE = 0

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: AsyncOpenAI(**kw, http_client=http))
            return await llms.LLMService().preflight()

    assert asyncio.run(run())["status"] == "available"
    assert len(attempts) == 3
    assert llms.classify_generation_error(APITimeoutError(request=attempts[0])).code == "timeout"


def test_missing_generation_config_does_not_call_network(runtime, monkeypatch):
    runtime.GENERATION_API_KEY = ""
    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: pytest.fail("Client must not be created"))
    assert asyncio.run(llms.LLMService().preflight())["code"] == "not_configured"


def test_invalid_generation_url_is_a_configuration_error_without_request(runtime, monkeypatch):
    runtime.GENERATION_API_BASE_URL = "invalid-url-with-test-secret"
    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: pytest.fail("Invalid URL must not create a client"))
    result = asyncio.run(llms.LLMService().preflight())
    assert result["code"] == "configuration"
    assert "test-secret" not in str(result)


def test_invalid_generation_response_is_not_retried(runtime, monkeypatch):
    attempts = []

    async def create(**kwargs):
        attempts.append(kwargs)
        return SimpleNamespace(choices=[])

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert asyncio.run(llms.LLMService().preflight())["code"] == "invalid_response"
    assert len(attempts) == 1


def test_prepare_checks_generation_independently_after_local_failure(runtime, monkeypatch):
    def fail(**kwargs):
        raise embeddings.LocalEmbeddingUnavailable("cache missing")

    monkeypatch.setattr(prepare_models, "EmbeddingService", fail)
    runtime.GENERATION_API_KEY = ""
    report = asyncio.run(prepare_models.check(generation=True))
    assert report["local_embedding"]["status"] == "unavailable"
    assert report["generation"]["code"] == "not_configured"
    assert report["model_services_ready"] is False


@pytest.mark.parametrize("entry", ["completion", "response", "stream", "structured", "events"])
def test_all_generation_entry_points_share_permanent_error_policy(runtime, monkeypatch, entry):
    attempts = []

    async def create(**kwargs):
        attempts.append(kwargs)
        raise llms.GenerationAPIError("credentials")

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    async def run():
        service = llms.LLMService()
        if entry == "response":
            result = await service.agenerate_response("question", [])
            assert not result["success"] and result["error"] == "credentials"
        elif entry == "events":
            events = [event async for event in service.stream_answer("question", [])]
            assert events[-1]["type"] == "error" and events[-1]["message"] == "credentials"
            assert not any(event["type"] == "end" for event in events)
        else:
            with pytest.raises(llms.GenerationAPIError, match="credentials"):
                if entry == "stream":
                    _ = [text async for text in service.agenerate_stream("question", [])]
                elif entry == "structured":
                    await service.agenerate_structured_decision([])
                else:
                    await service._chat_completion([])

    asyncio.run(run())
    assert len(attempts) == 1


def test_partial_stream_failure_is_closed_without_replay(runtime, monkeypatch):
    attempts, closed = [], []

    class Stream:
        async def __aiter__(self):
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
            raise APITimeoutError(request=httpx.Request("POST", "https://generation.example/v1"))

        async def close(self):
            closed.append(True)

    async def create(**kwargs):
        attempts.append(kwargs)
        return Stream()

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    async def run():
        return [event async for event in llms.LLMService().stream_answer("question", [])]

    events = asyncio.run(run())
    assert events[-1]["type"] == "error" and events[-1]["message"] == "timeout"
    assert any(event.get("content") == "partial" for event in events)
    assert not any(event["type"] == "end" for event in events)
    assert len(attempts) == len(closed) == 1


@pytest.mark.parametrize("entry", ["stream", "events"])
@pytest.mark.parametrize("payloads", [[], ["", " \n"], [None], [42]])
def test_unusable_stream_is_closed_and_never_reports_success(runtime, monkeypatch, entry, payloads):
    attempts, closed = [], []

    class Stream:
        async def __aiter__(self):
            for content in payloads:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])

        async def close(self):
            closed.append(True)

    async def create(**kwargs):
        attempts.append(kwargs)
        return Stream()

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    async def run():
        service = llms.LLMService()
        if entry == "stream":
            with pytest.raises(llms.GenerationAPIError, match="invalid_response"):
                _ = [text async for text in service.agenerate_stream("question", [])]
        else:
            events = [event async for event in service.stream_answer("question", [])]
            assert events[-1]["type"] == "error" and events[-1]["message"] == "invalid_response"
            assert not any(event["type"] == "end" for event in events)

    asyncio.run(run())
    assert len(attempts) == len(closed) == 1


def test_valid_stream_with_metadata_chunks_still_completes(runtime, monkeypatch):
    class Stream:
        async def __aiter__(self):
            yield SimpleNamespace(choices=[])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="OK"))])
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])

        async def close(self):
            pass

    async def create(**kwargs):
        return Stream()

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))

    async def run():
        return [event async for event in llms.LLMService().stream_answer("question", [])]

    events = asyncio.run(run())
    assert events[-1]["type"] == "end" and events[-1]["answer"] == "OK"
    assert events[-1]["success"]


@pytest.mark.parametrize("entry", ["stream", "events"])
@pytest.mark.parametrize("delta,valid", [(None, False), ({"role": "assistant"}, False), ({"content": "OK"}, True)])
def test_sdk_sse_transport_requires_actual_text(runtime, monkeypatch, entry, delta, valid):
    import json

    requests = []

    def handle(request):
        requests.append(request)
        payload = "data: [DONE]\n\n"
        if delta is not None:
            chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0,
                     "model": "generation-model", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            payload = "data: " + json.dumps(chunk) + "\n\n" + payload
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: AsyncOpenAI(**kw, http_client=http))
            service = llms.LLMService()
            if entry == "events":
                events = [event async for event in service.stream_answer("question", [])]
                assert events[-1]["type"] == ("end" if valid else "error")
                if not valid:
                    assert events[-1]["message"] == "invalid_response"
            elif valid:
                assert "".join([text async for text in service.agenerate_stream("question", [])]) == "OK"
            else:
                with pytest.raises(llms.GenerationAPIError, match="invalid_response"):
                    _ = [text async for text in service.agenerate_stream("question", [])]

    asyncio.run(run())
    assert len(requests) == 1


def test_cancelled_generation_releases_queue_without_retry(runtime, monkeypatch):
    attempts = []

    async def run():
        entered, blocked = asyncio.Event(), asyncio.Event()

        async def create(**kwargs):
            attempts.append(kwargs)
            entered.set()
            await blocked.wait()

        monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        service = llms.LLMService()
        request = asyncio.create_task(service.preflight())
        await entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert service.semaphore._value == runtime.LLM_MAX_CONCURRENCY

    asyncio.run(run())
    assert len(attempts) == 1


def test_timeout_is_bounded_and_reports_unavailable(runtime, monkeypatch):
    attempts = []
    runtime.LLM_BACKOFF_BASE = 0

    async def create(**kwargs):
        attempts.append(kwargs)
        raise APITimeoutError(request=httpx.Request("POST", "https://generation.example/v1"))

    monkeypatch.setattr(llms, "AsyncOpenAI", lambda **kw: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    result = asyncio.run(llms.LLMService().preflight())
    assert result["status"] == "unavailable" and result["code"] == "timeout"
    assert len(attempts) == runtime.LLM_MAX_RETRIES


def test_gpu_non_memory_failure_does_not_silently_retry(runtime, monkeypatch):
    runtime.LOCAL_EMBEDDING_DEVICE = "cuda"
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    loads = []

    def load(**kwargs):
        loads.append(kwargs)
        raise RuntimeError("wrong model configuration")

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: load)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable):
        embeddings.EmbeddingService()
    assert len(loads) == 1


def test_model_dependency_failure_does_not_download(runtime, monkeypatch):
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return "/cached/fixed-revision"

    def missing():
        raise ImportError("missing model dependencies")

    monkeypatch.setattr(embeddings, "_snapshot_download", snapshot)
    monkeypatch.setattr(embeddings, "_load_embedding_class", missing)
    with pytest.raises(embeddings.LocalEmbeddingUnavailable):
        embeddings.EmbeddingService(allow_download=True)
    assert all(call["local_files_only"] for call in calls)


def test_actual_huggingface_wrapper_passes_local_only_flags_to_sentence_transformer(runtime, monkeypatch):
    import numpy as np

    loaded, encoded = [], []

    class SentenceTransformerStub:
        def __init__(self, model_name, **kwargs):
            loaded.append((model_name, kwargs))
            self.tokenizer = client_stub().tokenizer
            self.max_seq_length = 512

        def encode(self, texts, **kwargs):
            encoded.append(kwargs)
            return np.array([[1.0, 0.5] for _ in texts])

    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        SimpleNamespace(SentenceTransformer=SentenceTransformerStub))
    service = embeddings.EmbeddingService()
    assert service.preflight()["status"] == "ready"
    assert loaded[0][1]["local_files_only"] is True
    assert loaded[0][1]["trust_remote_code"] is False
    assert loaded[0][1]["revision"] == "fixed-revision"
    assert all(call["normalize_embeddings"] for call in encoded)
