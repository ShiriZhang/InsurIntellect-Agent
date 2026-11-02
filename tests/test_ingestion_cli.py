from io import BytesIO
import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models.database_models import Document, DocumentChunk, IngestionAttempt
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_storage import DocumentStorage
from ingestion_fakes import LocalEncoderFake
from scripts import simple_ingest
from test_pdf_quality import make_pdf


@pytest.fixture
def runtime(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    engine = create_engine(f"sqlite:///{tmp_path / 'cli.db'}")
    service = DocumentIngestionService(DocumentStorage(engine, tmp_path / "originals"),
                                       embedding_factory=LocalEncoderFake)
    service.initialize()
    yield service, source
    service.stop()
    engine.dispose()


def test_cli_and_upload_entry_share_identity_candidates_and_readiness(runtime, monkeypatch):
    service, source = runtime
    path = make_pdf(source / 'policy.pdf', ["1. Conditions\nA claim is excluded unless notified within 30 days."])
    uploaded = service.submit(BytesIO(path.read_bytes()), "same-content.pdf")
    # Ingestion never constructs a generation client, even when credentials exist.
    monkeypatch.setattr("app.services.llm_service.LLMService.__init__",
                        lambda *a, **kw: pytest.fail("No generation API in ingestion"))
    result = simple_ingest.ingest(source, service=service)
    assert result["prepared"] and not result["ready"]
    assert result["documents"][0]["document_id"] == uploaded.document_id
    assert result["documents"][0]["task_status"] == "staged"
    with Session(service.engine) as db:
        doc = db.scalars(select(Document)).one()
        assert doc.product_name is None and doc.effective_date is None
        task = db.scalars(select(IngestionAttempt)).one()
        assert task.status == "staged"
        profile = json.loads(task.embedding_profile_json)
        chunks = db.scalars(select(DocumentChunk)).all()
        assert all(c.embedding_fingerprint == profile["fingerprint"] for c in chunks)
        assert all(c.page_number == 1 for c in chunks)


def test_cli_invalid_pdf_has_same_failure_as_upload(runtime):
    service, source = runtime
    (source / 'broken.pdf').write_bytes(b"not PDF")
    result = simple_ingest.ingest(source, service=service)
    assert not result["prepared"] and not result["ready"]
    status = result["documents"][0]
    assert status["error_code"] == "not_pdf"
    assert service.snapshot(status["document_id"]) == status


def test_no_wait_jobs_remain_durable_for_application_worker(runtime):
    service, source = runtime
    make_pdf(source / 'queued.pdf', ["Waiting period applies except to accidental injury."])
    result = simple_ingest.ingest(source, service=service, wait=False)
    assert result["documents"][0]["task_status"] == "queued"
    assert not result["ready"] and service.embedding is None
    service.run_until_idle()
    assert service.snapshot(result["documents"][0]["document_id"])["task_status"] == "staged"


def test_reset_is_rejected_before_any_file_or_database_change(runtime, tmp_path):
    service, source = runtime
    sentinel = source / 'keep.pdf'
    sentinel.write_bytes(b"keep original")
    with pytest.raises(SystemExit) as error:
        simple_ingest.main(["--pdf-dir", str(source), "--reset"], service=service)
    assert error.value.code == 2
    assert sentinel.read_bytes() == b"keep original"
    with Session(service.engine) as db:
        assert not db.scalars(select(Document)).all()


def test_main_json_and_exit_status_are_honest_about_staging(runtime, capsys):
    service, source = runtime
    make_pdf(source / 'valid.pdf', ["Valid insurance policy conditions."])
    assert simple_ingest.main(["--pdf-dir", str(source)], service=service) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["prepared"] and not output["ready"]


def test_cli_timeout_preserves_queued_work(runtime, monkeypatch):
    service, source = runtime
    make_pdf(source / 'valid.pdf', ["Valid insurance policy conditions."])
    monkeypatch.setattr(service, "start", lambda: False)  # Another worker owns the queue.
    result = simple_ingest.ingest(source, service=service, timeout=.01)
    assert result["timed_out"] and not result["ready"]
    assert result["documents"][0]["task_status"] == "queued"
    service.run_until_idle()
    assert service.snapshot(result["documents"][0]["document_id"])["task_status"] == "staged"
