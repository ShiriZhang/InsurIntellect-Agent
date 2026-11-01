"""Offline local-embedding/generation contracts; no real model quality claims."""

import asyncio
import json
import logging
import io
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from app.core import app_logging
from app.core.config import Settings
from app.services import embedding_service, llm_service
from model_fakes import client_stub


@pytest.fixture
def config(monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "_create_directories", lambda self: None)
    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_MODEL="example/multilingual-model",
                   LOCAL_EMBEDDING_REVISION="fixed-model-revision",
                   LOCAL_EMBEDDING_DEVICE="cpu", LOCAL_EMBEDDING_BATCH_SIZE=7,
                   LOCAL_EMBEDDING_CACHE_DIR="./temp/test-model-cache")
    monkeypatch.setattr(embedding_service, "settings", cfg)
    monkeypatch.setattr(llm_service, "settings", cfg)
    monkeypatch.setattr(app_logging, "settings", cfg)
    monkeypatch.setattr(embedding_service, "_snapshot_download",
                        lambda **kw: "/test-cache/snapshots/fixed-model-revision")
    monkeypatch.setattr(llm_service.LLMService, "_global_semaphore", None, raising=False)
    monkeypatch.setattr(llm_service.LLMService, "_failure_count", 0, raising=False)
    monkeypatch.setattr(llm_service.LLMService, "_circuit_opened_at", None, raising=False)
    return cfg


@pytest.fixture
def local_model(monkeypatch):
    loaded = []
    encoded = []

    class LocalModel:
        def __init__(self, **kwargs):
            loaded.append(kwargs)
            self._client = client_stub()

        def embed_query(self, text):
            encoded.append(text)
            return [0.1, 0.2, 0.3]

        def embed_documents(self, texts):
            encoded.extend(texts)
            return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(embedding_service, "_load_embedding_class", lambda: LocalModel)
    return loaded, encoded


def configure_generation(cfg):
    cfg.GENERATION_API_BASE_URL = "https://generation.example/v1"
    cfg.GENERATION_API_KEY = "generation-test-secret"
    cfg.GENERATION_MODEL = "answer-model"


@pytest.mark.parametrize("generation_configured", [True, False])
def test_local_text_has_no_embedding_outbound_request(config, local_model, monkeypatch,
                                                      generation_configured):
    if generation_configured:
        configure_generation(config)
    outbound = []

    def handle(request):
        body = json.loads(request.content)
        outbound.append((request.url.host, request.url.path, body))
        assert request.url.host == "generation.example"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer generation-test-secret"
        assert body["model"] == "answer-model"
        assert body["messages"] == [{"role": "user", "content": "generation question"}]
        return httpx.Response(200, request=request, json={
            "id": "test-chat", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "test answer"}}],
        })

    # Every HTTPX call uses the recorder; requests has no allowed network path.
    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kw: handle(request))

    async def async_send(self, request, **kwargs):
        return handle(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    import requests
    monkeypatch.setattr(requests.Session, "request",
                        lambda *a, **kw: pytest.fail("Unexpected requests outbound call"))

    async def run():
        embedder = embedding_service.EmbeddingService()
        assert local_model[1] == []  # No text is used to load/probe the model.
        assert embedder.embed_query("私有问题 question") == [0.1, 0.2, 0.3]
        assert len(embedder.embed_documents(["私有保单正文", "private document"])) == 2
        assert await embedder.aembed_query("async private question") == [0.1, 0.2, 0.3]
        assert len(await embedder.aembed_documents(["async private document"])) == 1
        assert outbound == []
        assert embedder.embedding_function is embedder
        # Inject the transport explicitly: SDK versions may use different default HTTP clients.
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            monkeypatch.setattr(llm_service, "AsyncOpenAI",
                                lambda **kw: AsyncOpenAI(**kw, http_client=http))
            generator = llm_service.LLMService()
            if generation_configured:
                response = await generator._chat_completion(
                    [{"role": "user", "content": "generation question"}])
                assert response.choices[0].message.content == "test answer"
            else:
                assert not generator.has_credentials()

    asyncio.run(run())
    assert len(outbound) == int(generation_configured)
    loaded = local_model[0][0]
    assert loaded == {
        "model_name": "/test-cache/snapshots/fixed-model-revision", "cache_folder": "./temp/test-model-cache",
        "model_kwargs": {"revision": "fixed-model-revision", "device": "cpu", "trust_remote_code": False,
                         "local_files_only": True},
        "encode_kwargs": {"batch_size": 7, "normalize_embeddings": True, "prompt": ""},
    }
    assert "generation-test-secret" not in str(loaded)
    assert len(local_model[1]) == 5


def test_generation_changes_do_not_override_local_settings(config, local_model):
    embedding_service.EmbeddingService()
    before = local_model[0][0]
    configure_generation(config)
    config.OPENAI_EMBEDDING_MODEL = "https://remote-embedding.example/v1"
    config.EMBEDDING_MODEL = "local:hash"
    embedding_service.EmbeddingService()
    assert local_model[0][1] == before


@pytest.mark.parametrize("field", ["LOCAL_EMBEDDING_MODEL", "LOCAL_EMBEDDING_REVISION",
                                    "LOCAL_EMBEDDING_CACHE_DIR"])
def test_missing_local_config_never_uses_generation(config, local_model, field):
    configure_generation(config)
    setattr(config, field, "")
    with pytest.raises(embedding_service.LocalEmbeddingUnavailable):
        embedding_service.EmbeddingService()
    assert local_model[0] == []


@pytest.mark.parametrize("model", ["local:hash", "hash", "offline", "https://embed.example/v1",
                                    "openai:text-embedding-model", "remote:model"])
def test_remote_and_hash_configs_are_rejected(config, local_model, model):
    config.LOCAL_EMBEDDING_MODEL = model
    with pytest.raises(embedding_service.LocalEmbeddingUnavailable, match="remote embedding"):
        embedding_service.EmbeddingService()
    assert local_model[0] == []


@pytest.mark.parametrize("failure", [ImportError, RuntimeError])
def test_model_load_failure_is_explicit_without_fallback(config, monkeypatch, caplog, failure):
    configure_generation(config)

    def failed_load():
        raise failure("generation-test-secret private document body")

    monkeypatch.setattr(embedding_service, "_load_embedding_class", failed_load)
    with pytest.raises(embedding_service.LocalEmbeddingUnavailable, match="no hash/remote") as error:
        embedding_service.EmbeddingService()
    assert "private document body" not in str(error.value) + caplog.text
    assert "generation-test-secret" not in str(error.value) + caplog.text


def test_encode_failure_is_guarded_for_chroma_and_async(config, local_model, monkeypatch):
    service = embedding_service.EmbeddingService()

    def fail(*args):
        raise RuntimeError("private input in provider exception")

    monkeypatch.setattr(service._embedding, "embed_documents", fail)
    monkeypatch.setattr(service._embedding, "embed_query", fail)
    with pytest.raises(embedding_service.LocalEmbeddingUnavailable, match="ingestion unavailable") as error:
        service.embedding_function.embed_documents(["private input"])
    assert "private input" not in str(error.value)
    with pytest.raises(embedding_service.LocalEmbeddingUnavailable, match="semantic search unavailable"):
        asyncio.run(service.aembed_query("private question"))


@pytest.mark.parametrize("cuda_available,expected", [(False, "cpu"), (True, "cuda")])
def test_auto_device_selection(config, local_model, monkeypatch, cuda_available, expected):
    config.LOCAL_EMBEDDING_DEVICE = "auto"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available)))
    assert embedding_service.EmbeddingService().device == expected
    assert local_model[0][0]["model_kwargs"]["device"] == expected


@pytest.mark.parametrize("field", ["GENERATION_API_BASE_URL", "GENERATION_API_KEY", "GENERATION_MODEL"])
def test_missing_generation_does_not_disable_local_embedding(config, local_model, field):
    configure_generation(config)
    setattr(config, field, "")
    config.OPENAI_API_KEY = "old-key"
    config.OPENAI_BASE_URL = "https://old.example/v1"
    config.OPENAI_MODEL = "old-model"
    assert not llm_service.LLMService().has_credentials()
    assert embedding_service.EmbeddingService().embed_query("question") == [0.1, 0.2, 0.3]


def test_legacy_generation_alias_priority(config, monkeypatch):
    config.SILICONFLOW_API_KEY = "silicon-key"
    config.SILICONFLOW_BASE_URL = "https://silicon.example/v1"
    config.SILICONFLOW_MODEL = "silicon-model"
    assert (config.generation_api_key, config.generation_api_base_url, config.generation_model()) == (
        "silicon-key", "https://silicon.example/v1", "silicon-model")
    config.OPENAI_API_KEY = "openai-key"
    config.OPENAI_BASE_URL = "https://old.example/v1"
    config.OPENAI_MODEL = "old-model"
    assert (config.generation_api_key, config.generation_api_base_url, config.generation_model()) == (
        "openai-key", "https://old.example/v1", "old-model")
    config.OPENAI_MODEL_CORE = "old-core"
    config.OPENAI_MODEL_LIGHT = "old-light"
    assert config.generation_model() == "old-core"
    assert config.generation_model("light") == "old-light"
    configure_generation(config)
    captured = []
    monkeypatch.setattr(llm_service, "AsyncOpenAI", lambda **kw: captured.append(kw) or object())
    service = llm_service.LLMService()
    assert service.model == service.core_model_name == service.light_model_name == "answer-model"
    assert captured[0]["base_url"] == "https://generation.example/v1"
    assert captured[0]["api_key"] == "generation-test-secret"
    assert llm_service.LLMService.with_model("explicit-model").core_model_name == "explicit-model"


def test_env_priority_and_empty_override(config, monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("GENERATION_API_KEY=file-secret\nOPENAI_API_KEY=old-secret\n", encoding="utf-8")
    monkeypatch.setenv("GENERATION_API_KEY", "env-secret")
    assert Settings(_env_file=env_file).generation_api_key == "env-secret"
    assert Settings(_env_file=env_file, GENERATION_API_KEY="init-secret").generation_api_key == "init-secret"
    assert Settings(_env_file=env_file, GENERATION_API_KEY="").generation_api_key == ""
    with pytest.raises(ValueError):
        Settings(_env_file=None, LOCAL_EMBEDDING_BATCH_SIZE=0)


def test_application_logs_and_repr_redact_secrets(config, caplog):
    configure_generation(config)
    config.GENERATION_API_BASE_URL = "https://url-user:url-password@generation.example/v1?key=url-secret"
    logger = app_logging.get_logger("test.local.config")
    with caplog.at_level(logging.INFO):
        logger.info("settings=%r", config)
        try:
            raise RuntimeError("generation-test-secret " + config.GENERATION_API_BASE_URL)
        except RuntimeError:
            logger.exception("Provider failed")
    for secret in ("generation-test-secret", "url-user", "url-password", "url-secret"):
        assert secret not in caplog.text
        assert secret not in repr(config)
    assert "[REDACTED]" in caplog.text


def test_generation_error_logs_do_not_echo_provider_body(config, monkeypatch, caplog):
    configure_generation(config)
    config.LLM_MAX_RETRIES = 2
    config.LLM_BACKOFF_BASE = 0

    def handle(request):
        return httpx.Response(401, json={"error": {"message": "generation-test-secret", "type": "auth"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            monkeypatch.setattr(llm_service, "AsyncOpenAI", lambda **kw: AsyncOpenAI(**kw, http_client=http))
            return await llm_service.LLMService().agenerate_response("question", [])

    result = asyncio.run(run())
    assert result["success"] is False
    assert "generation-test-secret" not in caplog.text + str(result)


def test_actual_http_client_logs_are_redacted_before_root_handlers(config, caplog):
    configure_generation(config)
    config.GENERATION_API_BASE_URL = "https://url-user:url-password@generation.example/v1?token=url-secret"
    with caplog.at_level(logging.INFO, logger="httpx"):
        with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client:
            client.get(config.GENERATION_API_BASE_URL)
    assert "HTTP Request" in caplog.text
    for secret in ("url-user", "url-password", "url-secret"):
        assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_independent_third_party_handler_redacts_encoded_keys_and_exception(config):
    configure_generation(config)
    config.GENERATION_API_KEY = "review/key+secret"
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    logger = logging.getLogger("test.third_party.own_sink")
    before_handlers, before_level, before_propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        try:
            raise RuntimeError("review/key+secret")
        except RuntimeError:
            logger.exception("request=https://generation.example/?key=review%%2Fkey%%2Bsecret %s", "third-party")
        assert "RuntimeError" in output.getvalue()
        assert "review/key+secret" not in output.getvalue()
        assert "review%2Fkey%2Bsecret" not in output.getvalue()
        assert "[REDACTED]" in output.getvalue()
    finally:
        logger.handlers, logger.level, logger.propagate = before_handlers, before_level, before_propagate
        handler.close()
