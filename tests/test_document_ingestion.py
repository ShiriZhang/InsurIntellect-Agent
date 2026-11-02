from io import BytesIO
import json
from threading import Event
import time

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.models.database_models import Document, DocumentChunk, IngestionAttempt
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_storage import DocumentStorage
from app.services.embedding_service import LocalEmbeddingUnavailable
from ingestion_fakes import LocalEncoderFake
from test_pdf_quality import make_pdf


@pytest.fixture
def ingestion(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ingestion.db'}", connect_args={"timeout": 10})
    service = DocumentIngestionService(DocumentStorage(engine, tmp_path / 'originals'),
                                       auto_publish=False, embedding_factory=LocalEncoderFake, queue_size=1, batch_size=2)
    service.initialize()
    yield service
    service.stop()
    engine.dispose()


def submit_pdf(service, tmp_path, text="1. Conditions\nNo payment unless all conditions are met."):
    path = make_pdf(tmp_path / 'input.pdf', [text])
    with path.open('rb') as source:
        return service.submit(source, 'policy.pdf')


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    pytest.fail("Worker did not reach expected state")


def test_bounded_notifications_never_drop_sqlite_jobs_or_duplicate_chunks(ingestion, tmp_path):
    items = [submit_pdf(ingestion, tmp_path, f"Policy {i} excludes payment unless conditions hold.") for i in range(6)]
    assert ingestion.notifications.qsize() == 1
    assert ingestion.run_until_idle()
    assert all(ingestion.snapshot(item.document_id)["task_status"] == "staged" for item in items)
    assert all(not ingestion.snapshot(item.document_id)["ready"] for item in items)
    with Session(ingestion.engine) as db:
        chunks = db.scalars(select(DocumentChunk)).all()
        assert len(chunks) == 6 and len({c.vector_id for c in chunks}) == 6
        assert all(c.page_number == 1 and json.loads(c.embedding_json) for c in chunks)
        assert all(db.get(Document, item.document_id).product_name is None for item in items)
    assert ingestion.run_until_idle()
    with Session(ingestion.engine) as db:
        assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 6


def test_recovery_rebuilds_only_interrupted_attempt_and_preserves_identity(ingestion, tmp_path):
    item = submit_pdf(ingestion, tmp_path, "Waiting period is 90 days. " * 10)
    assert ingestion.run_until_idle()
    with Session(ingestion.engine) as db:
        ids = list(db.scalars(select(DocumentChunk.vector_id)))
        db.get(IngestionAttempt, (item.document_id, 1)).status = "indexing"
        db.get(Document, item.document_id).processing_status = "indexing"
        db.commit()
    restarted = DocumentIngestionService(ingestion.storage, auto_publish=False, embedding_factory=LocalEncoderFake)
    assert restarted.run_until_idle()
    with Session(ingestion.engine) as db:
        assert list(db.scalars(select(DocumentChunk.vector_id))) == ids
        assert db.get(IngestionAttempt, (item.document_id, 1)).status == "staged"


@pytest.mark.parametrize("failure", ["load", "encode", "invalid"])
def test_local_model_failure_is_distinct_and_never_ready(ingestion, tmp_path, failure):
    class Broken(LocalEncoderFake):
        def __init__(self):
            if failure == "load":
                raise LocalEmbeddingUnavailable("do not expose source or credentials")
            super().__init__()
        def embed_documents(self, texts):
            if failure == "encode":
                raise LocalEmbeddingUnavailable("local model failed")
            return [[float('nan')] * 3 for t in texts]
    ingestion.embedding_factory = Broken
    item = submit_pdf(ingestion, tmp_path)
    assert ingestion.run_until_idle()
    status = ingestion.snapshot(item.document_id)
    assert status["error_code"] == "local_embedding_unavailable" and not status["ready"]
    with Session(ingestion.engine) as db:
        assert not db.scalars(select(DocumentChunk)).all()
        assert db.get(Document, item.document_id).error_message == "local_embedding_unavailable"


def test_invalid_pdf_fails_before_loading_model(ingestion):
    ingestion.embedding_factory = lambda: pytest.fail("Invalid PDF must not load model")
    item = ingestion.submit(BytesIO(b"not PDF"), 'fake.pdf')
    ingestion.run_until_idle()
    assert ingestion.snapshot(item.document_id)["error_code"] == "not_pdf"


def test_entire_mixed_scan_rejected_without_partial_chunks(ingestion, tmp_path):
    path = make_pdf(tmp_path / 'mixed.pdf', ["Good policy text", "header"], scan_pages=[2])
    with path.open('rb') as source:
        item = ingestion.submit(source, path.name)
    ingestion.run_until_idle()
    status = ingestion.snapshot(item.document_id)
    assert status["error_code"] == "unreadable_pages" and status["error_pages"] == [2]
    with Session(ingestion.engine) as db:
        assert not db.scalars(select(DocumentChunk)).all()


@pytest.mark.parametrize("action", ["cancel", "delete", "shutdown"])
def test_late_encoder_results_never_write_after_invalidation(ingestion, tmp_path, monkeypatch, action):
    started, release = Event(), Event()
    class Slow(LocalEncoderFake):
        def embed_documents(self, texts):
            started.set()
            assert release.wait(5)
            return super().embed_documents(texts)
    ingestion.embedding_factory = Slow
    item = submit_pdf(ingestion, tmp_path)
    ingestion.start()
    try:
        assert started.wait(5)
        if action == "cancel":
            ingestion.cancel(item.document_id)
        elif action == "delete":
            with Session(ingestion.engine) as db:
                doc = db.get(Document, item.document_id)
                doc.deletion_status = "deleting"
                doc.deletion_epoch += 1
                db.commit()
        else:
            monkeypatch.setattr("app.services.document_ingestion.settings.INGEST_SHUTDOWN_TIMEOUT", .05)
            ingestion.stop()
            other = DocumentIngestionService(ingestion.storage)
            assert not other.worker_lock.acquire()  # Native execution still owns the lock.
        release.set()
        ingestion.stop()
        assert not ingestion._thread.is_alive()
        with Session(ingestion.engine) as db:
            assert not db.scalars(select(DocumentChunk)).all()
            task = db.get(IngestionAttempt, (item.document_id, 1))
            assert task.status != "staged"
        assert not ingestion.snapshot(item.document_id)["ready"]
    finally:
        release.set()
        ingestion.stop()
    if action == "shutdown":
        restarted = DocumentIngestionService(ingestion.storage, auto_publish=False, embedding_factory=LocalEncoderFake)
        restarted.run_until_idle()
        assert restarted.snapshot(item.document_id)["task_status"] == "staged"


def test_only_one_worker_and_uploads_continue_while_encoding(ingestion, tmp_path):
    started, release = Event(), Event()
    class Slow(LocalEncoderFake):
        def embed_documents(self, texts):
            started.set()
            assert release.wait(5)
            return super().embed_documents(texts)
    ingestion.embedding_factory = Slow
    first = submit_pdf(ingestion, tmp_path)
    assert ingestion.start()
    try:
        assert started.wait(5)
        another = DocumentIngestionService(ingestion.storage, auto_publish=False, embedding_factory=LocalEncoderFake)
        assert another.start() and not another.run_until_idle()
        assert another.embedding is None  # Standby must not recover or encode while CLI/app owns lock.
        # The worker does not hold SQLite's write lock during native encoding.
        second = submit_pdf(ingestion, tmp_path, "Another policy with different conditions.")
        assert first.document_id != second.document_id
        release.set()
        wait_for(lambda: ingestion.snapshot(second.document_id)["task_status"] == "staged")
    finally:
        release.set()
        if 'another' in locals():
            another.stop()
        ingestion.stop()


def test_standby_worker_automatically_takes_over_after_owner_exits(ingestion, tmp_path):
    other = DocumentIngestionService(ingestion.storage, auto_publish=False, embedding_factory=LocalEncoderFake)
    assert ingestion.start()
    wait_for(lambda: ingestion.worker_lock.handle is not None)
    try:
        assert other.start()
        assert other.worker_lock.handle is None
        ingestion.stop()  # The CLI/app that owned the queue exits.
        item = submit_pdf(other, tmp_path, "The next upload must be processed without restarting the app.")
        wait_for(lambda: other.snapshot(item.document_id)["task_status"] == "staged")
    finally:
        other.stop()
        ingestion.stop()


def test_recovery_never_requeues_deletion_or_superseded_attempt(ingestion, tmp_path):
    item = submit_pdf(ingestion, tmp_path)
    with Session(ingestion.engine) as db:
        doc = db.get(Document, item.document_id)
        doc.deletion_status = "deleting"
        db.get(IngestionAttempt, (doc.id, 1)).status = "indexing"
        db.commit()
    ingestion.run_until_idle()
    with Session(ingestion.engine) as db:
        assert db.get(IngestionAttempt, (item.document_id, 1)).status == "cancelled"
        assert db.get(Document, item.document_id).deletion_status == "deleting"
        assert not db.scalars(select(DocumentChunk)).all()


def test_real_embedding_service_device_fallback_is_preserved_by_worker(ingestion, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from app.core.config import Settings
    from app.services import embedding_service as embeddings
    from model_fakes import client_stub

    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_MODEL="example/model",
                   LOCAL_EMBEDDING_REVISION="fixed-revision", LOCAL_EMBEDDING_CACHE_DIR=str(tmp_path),
                   LOCAL_EMBEDDING_DEVICE="cuda", LOCAL_EMBEDDING_BATCH_SIZE=2,
                   LOCAL_EMBEDDING_MAX_BATCH_RETRIES=1)
    monkeypatch.setattr(embeddings, "settings", cfg)
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    monkeypatch.setattr(embeddings, "_snapshot_download", lambda **kwargs: str(tmp_path / "fixed-revision"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    loads = []
    class Encoder:
        def __init__(self, **kwargs):
            self.device = kwargs["model_kwargs"]["device"]
            self.encode_kwargs = kwargs["encode_kwargs"]
            self._client = client_stub()
            loads.append(self.device)
        def embed_documents(self, texts):
            if self.device == "cuda" and any("source conditions" in t for t in texts):
                raise RuntimeError("CUDA out of memory")
            return [[1.0, .5, .25] for t in texts]
        def embed_query(self, text):
            return self.embed_documents([text])[0]
    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Encoder)
    ingestion.embedding_factory = embeddings.EmbeddingService
    item = submit_pdf(ingestion, tmp_path, "The source conditions exclude illness unless stated otherwise.")
    ingestion.run_until_idle()
    status = ingestion.snapshot(item.document_id)
    assert status["task_status"] == "staged" and not status["ready"]
    assert loads == ["cuda", "cpu"]
    assert status["diagnostics"]["device"] == "cpu"
    assert {w["code"] for w in status["diagnostics"]["warnings"]} >= {"batch_reduced", "cpu_fallback"}
