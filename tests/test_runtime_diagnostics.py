"""Passive health and launcher checks, no models, network or live user library."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import health
from app.core.config import Settings


def test_passive_checks_never_instantiate_model_or_provider(monkeypatch, tmp_path):
    from app.core import config
    cfg = Settings(_env_file=None, GENERATION_API_BASE_URL="", GENERATION_API_KEY="",
                   GENERATION_MODEL="", LOCAL_EMBEDDING_REVISION="")
    monkeypatch.setattr(config, "settings", cfg)
    monkeypatch.setattr(health.db_manager, "health_check", AsyncMock(return_value=True))
    app = FastAPI()
    app.include_router(health.router, prefix="/health")
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        assert client.get("/health/").json()["llm_status"] is False
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["local_embedding"]["device"] is None
        assert response.json()["local_embedding"]["status"] == "unavailable"
        assert client.get("/health/model").json()["generation"]["status"] == "unavailable"


def test_readiness_requires_matching_profile_and_recent_generation(monkeypatch):
    import time
    from app.core import config
    cfg = Settings(_env_file=None, GENERATION_API_BASE_URL="https://example.invalid/v1",
                   GENERATION_API_KEY="test", GENERATION_MODEL="test", LOCAL_EMBEDDING_REVISION="fixed")
    monkeypatch.setattr(config, "settings", cfg)
    monkeypatch.setattr(health.db_manager, "health_check", AsyncMock(return_value=True))
    profile = {"fingerprint": "test"}
    import threading
    state = SimpleNamespace(lock=threading.RLock(), current=SimpleNamespace(id=3, profile=profile), serving=True)
    model = SimpleNamespace(diagnostics=lambda: {"status": "ready", "embedding_profile": profile, "device": "cpu"})
    app = FastAPI()
    app.include_router(health.router)
    app.state.index_coordinator = SimpleNamespace(state=state)
    app.state.document_ingestion = SimpleNamespace(embedding=model)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503  # Credentials alone prove nothing.
        app.state.generation_check = (time.monotonic() + 60, {"status": "available"})
        assert client.get("/ready").status_code == 200
        state.current.profile = {"fingerprint": "different"}
        assert client.get("/ready").status_code == 503
        state.current.profile = profile
        app.state.generation_check = (0, {"status": "available"})
        assert client.get("/ready").status_code == 503


def test_shared_launcher_preserves_settings(monkeypatch, tmp_path):
    from scripts import launch
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "environment/model")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.73")
    dotenv = tmp_path / ".env"
    dotenv.write_text("LOCAL_EMBEDDING_MODEL=dotenv/model\nSIMILARITY_THRESHOLD=0.11\nHOST=127.0.0.2\nPORT=9999\n")
    cfg = Settings(_env_file=dotenv)
    assert cfg.LOCAL_EMBEDDING_MODEL == "environment/model"
    assert cfg.SIMILARITY_THRESHOLD == 0.73
    monkeypatch.setattr(launch, "settings", cfg)
    calls = []
    import uvicorn
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    launch.main()
    assert calls[0][1]["host"] == "127.0.0.2" and calls[0][1]["port"] == 9999


def test_restart_existing_library_can_explicitly_prepare_query_model(monkeypatch):
    import threading
    import time
    from app.core import config
    from app.services.scoped_query import ScopedQueryProcessor
    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_REVISION="fixed",
                   GENERATION_API_BASE_URL="https://example.invalid/v1",
                   GENERATION_API_KEY="test", GENERATION_MODEL="test")
    monkeypatch.setattr(config, "settings", cfg)
    monkeypatch.setattr(health.db_manager, "health_check", AsyncMock(return_value=True))
    profile = {"fingerprint": "fixed"}
    class Model:
        checked = False
        def preflight(self):
            self.checked = True
            return self.diagnostics()
        def diagnostics(self):
            return {"status": "ready" if self.checked else "not_checked",
                    "embedding_profile": profile, "device": "cpu"}
    processor = ScopedQueryProcessor(embedding_factory=Model)
    app = FastAPI()
    app.include_router(health.router)
    app.state.query_processor = processor
    app.state.document_ingestion = SimpleNamespace(embedding=None)
    app.state.index_coordinator = SimpleNamespace(state=SimpleNamespace(
        lock=threading.RLock(), serving=True, current=SimpleNamespace(id=3, profile=profile)))
    app.state.generation_check = (time.monotonic() + 60, {"status": "available"})
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 503
        assert processor.embedding is None
        assert client.post("/check-local").json()["status"] == "ready"
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["local_embedding"]["device"] == "cpu"
