from fastapi.testclient import TestClient


def test_application_owns_lazy_worker_and_stops_it():
    from app.main import app
    with TestClient(app):
        service = app.state.document_ingestion
        assert service._thread.is_alive()
        assert service.embedding is None  # Liveness must not load or download models.
    assert not service._thread.is_alive()
