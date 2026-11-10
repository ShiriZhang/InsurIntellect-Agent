#!/usr/bin/env python
"""Isolated empty-library contract smoke. No models or generation requests."""
import os
import tempfile
from pathlib import Path


def main():
    # Run as a separate process so app/settings singletons cannot reference a live library.
    with tempfile.TemporaryDirectory(prefix="insur_empty_") as directory:
        root = Path(directory)
        for key, value in {
            "DATABASE_URL": "sqlite:///" + (root / "app.db").as_posix(),
            "DATABASE_URL_ASYNC": "sqlite+aiosqlite:///" + (root / "app.db").as_posix(),
            "CHROMA_PERSIST_DIRECTORY": str(root / "chroma"),
            "PROCESSED_DATA_PATH": str(root / "processed"),
            "PDF_STORAGE_PATH": str(root / "pdfs"),
            "LOCAL_EMBEDDING_CACHE_DIR": str(root / "models"),
            "LOCAL_EMBEDDING_REVISION": "", "GENERATION_API_KEY": "",
            "GENERATION_API_BASE_URL": "", "GENERATION_MODEL": "",
            "ENABLE_AUTO_RESTART": "false",
        }.items():
            os.environ[key] = value
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app) as client:
            assert client.get("/api/v1/health/live").json() == {"status": "alive"}
            assert client.get("/api/v1/health/ready").status_code == 503
            corpus = client.get("/api/v1/corpus").json()
            assert corpus["documents"] == []
            for stream in (False, True):
                for scope in ({}, {"document_ids": None}, {"document_ids": []},
                              {"document_ids": [999999]}):
                    response = client.post("/api/v1/queries/ask", json={
                        "question": "What is the waiting period?", "stream": stream, **scope})
                    assert response.status_code == 422, response.text
                    response = client.post("/api/v1/queries/batch", json={
                        "questions": ["What is the waiting period?"], **scope})
                    assert response.status_code == 422, response.text
        app.state.document_ingestion.engine.dispose()
    print("PASS empty library: alive, not ready, no sources, ask/batch scope errors 422")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
