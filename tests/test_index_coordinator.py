from io import BytesIO
import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models.database_models import Document, DocumentChunk, IndexCommit, IndexPointer, IngestionAttempt
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_storage import DocumentStorage
from app.services.index_coordinator import IndexCoordinator, IndexUnavailable, PublicationCommittedButUnavailable
from app.services.index_resources import IndexIntegrityError
from index_fakes import VectorStoreFake
from ingestion_fakes import LocalEncoderFake
from test_pdf_quality import make_pdf


@pytest.fixture
def library(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}")
    ingestion = DocumentIngestionService(DocumentStorage(engine, tmp_path / 'pdfs'), auto_publish=False, embedding_factory=LocalEncoderFake)
    ingestion.initialize()
    vectors = VectorStoreFake()
    coordinator = IndexCoordinator(engine, vectors=vectors, vector_root=tmp_path / 'vectors', processed_root=tmp_path / 'processed')
    yield ingestion, coordinator, tmp_path
    ingestion.stop()
    engine.dispose()


def candidate(library, name="A", text=None):
    ingestion, _, root = library
    path = make_pdf(root / f'{name}.pdf', [text or f"Policy {name}: waiting period is 90 days except accidents."])
    result = ingestion.submit(BytesIO(path.read_bytes()), path.name)
    assert ingestion.run_until_idle()
    assert ingestion.snapshot(result.document_id)["task_status"] == "staged"
    return result


def publish(library, name="A"):
    item = candidate(library, name)
    return item, library[1].publish(item.document_id, item.attempt)


def test_publish_commits_complete_indexes_and_exact_source_identity(library):
    ingestion, coordinator, _ = library
    item, version = publish(library)
    status = ingestion.snapshot(item.document_id)
    assert status["ready"] and status["task_status"] == "ready"
    with Session(ingestion.engine) as db:
        assert db.get(IndexPointer, "current").commit_id == version
        row = db.get(IndexCommit, version)
        assert json.loads(row.document_attempts_json) == [[item.document_id, item.attempt]]
        snapshot = coordinator.snapshots.read(row.snapshot_path, row.snapshot_sha256)
        assert snapshot["document_attempts"] == [[item.document_id, item.attempt]]
        assert set(snapshot["ids"]) == set(db.scalars(select(DocumentChunk.vector_id)))
    assert coordinator.publish(item.document_id, item.attempt) == version


@pytest.mark.parametrize("failure", ["vectors", "snapshot", "before_commit"])
def test_partial_write_failure_keeps_old_pointer_and_previous_sources(library, monkeypatch, failure):
    ingestion, coordinator, _ = library
    first, version = publish(library)
    second = candidate(library, 'B')
    if failure == "vectors":
        coordinator.vectors.fail_write = True
    elif failure == "snapshot":
        monkeypatch.setattr(coordinator.snapshots, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("snapshot write failure")))
    else:
        coordinator.fault = lambda stage: (_ for _ in ()).throw(RuntimeError(stage)) if stage == failure else None
    with pytest.raises((OSError, RuntimeError)):
        coordinator.publish(second.document_id, second.attempt)
    assert not ingestion.snapshot(second.document_id)["ready"]
    assert ingestion.snapshot(first.document_id)["ready"]
    with Session(ingestion.engine) as db:
        assert db.get(IndexPointer, "current").commit_id == version
        assert len(list(db.scalars(select(IndexCommit)))) == 1
        profile = json.loads(db.get(IndexCommit, version).embedding_profile_json)
    assert all(r["metadata"]["document_id"] == first.document_id
               for r in coordinator.vectors.rows[profile["fingerprint"]].values())


@pytest.mark.parametrize("stage", ["after_commit", "switch"])
def test_post_commit_switch_failure_blocks_new_readers_until_pointer_recovery(library, stage):
    ingestion, coordinator, _ = library
    first, version = publish(library)
    second = candidate(library, 'B')
    coordinator.fault = lambda point: (_ for _ in ()).throw(RuntimeError(point)) if point == stage else None
    with pytest.raises(PublicationCommittedButUnavailable):
        coordinator.publish(second.document_id, second.attempt)
    with Session(ingestion.engine) as db:
        committed = db.get(IndexPointer, "current").commit_id
        assert committed != version and db.get(Document, second.document_id).is_queryable
        profile = json.loads(db.get(IndexCommit, committed).embedding_profile_json)
    with pytest.raises(IndexUnavailable):
        coordinator.open_view([first.document_id], profile)
    coordinator.fault = lambda point: None
    assert coordinator.recover() == committed
    assert coordinator.ready_ids() == [first.document_id, second.document_id]


def test_missing_candidate_chunk_or_changed_profile_never_publishes(library):
    ingestion, coordinator, _ = library
    first, _ = publish(library)
    second = candidate(library, 'B')
    with Session(ingestion.engine) as db:
        task = db.get(IngestionAttempt, (second.document_id, second.attempt))
        task.expected_chunks += 1
        db.commit()
    with pytest.raises(IndexIntegrityError, match="incomplete"):
        coordinator.publish(second.document_id, second.attempt)
    assert coordinator.ready_ids() == [first.document_id]


def test_directory_change_cannot_adopt_committed_resources(library):
    ingestion, coordinator, root = library
    publish(library)
    changed = IndexCoordinator(ingestion.engine, vectors=coordinator.vectors,
                               vector_root=root / 'other-vectors', processed_root=root / 'other-processed')
    with pytest.raises(IndexUnavailable, match="roots"):
        changed.recover()
