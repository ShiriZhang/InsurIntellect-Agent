from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from app.core.schema_migrations import migrate_schema
from app.models.database_models import (
    ActiveContentClaim, DeletionTask, Document, DocumentChunk, IngestionAttempt, QueryHistory,
)
from app.services.document_storage import DocumentStorage, SourceUnavailable


@pytest.fixture
def storage(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", connect_args={"timeout": 30})
    with engine.begin() as conn:
        migrate_schema(conn)
    service = DocumentStorage(engine, tmp_path / 'originals')
    yield service
    engine.dispose()


def accept(storage, content=b"%PDF-text", name="policy.pdf"):
    return storage.accept(BytesIO(content), name)


def test_same_name_different_bytes_and_untrusted_name(storage, tmp_path):
    first = accept(storage, b"one", "../../policy.pdf")
    second = accept(storage, b"two", "policy.pdf")
    assert first.document_id != second.document_id
    assert storage.original_path(first.document_id).read_bytes() == b"one"
    assert storage.original_path(second.document_id).read_bytes() == b"two"
    with Session(storage.engine) as db:
        assert db.get(Document, first.document_id).original_name == "policy.pdf"
        assert db.get(Document, first.document_id).product_name is None
    assert not (tmp_path / 'policy.pdf').exists()


def test_concurrent_duplicate_has_one_identity_and_one_task(storage):
    # Separate service instances/connections share only SQLite's write lock.
    def submit(_):
        return accept(DocumentStorage(storage.engine, storage.root))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(submit, range(16)))
    assert len({result.document_id for result in results}) == 1
    assert sum(result.created for result in results) == 1
    with Session(storage.engine) as db:
        assert db.scalar(select(func.count()).select_from(Document)) == 1
        assert db.scalar(select(func.count()).select_from(IngestionAttempt)) == 1
        assert db.scalar(select(func.count()).select_from(ActiveContentClaim)) == 1
    assert not list(storage.root.glob('.upload-*'))


@pytest.mark.parametrize("stage", ["queued", "parsing", "indexing", "ready"])
def test_live_duplicate_does_not_enqueue(storage, stage):
    first = accept(storage)
    with Session(storage.engine) as db:
        db.get(Document, first.document_id).processing_status = stage
        db.commit()
    second = accept(storage, name="renamed.pdf")
    assert not second.created and second.attempt == 1
    assert second.processing_status == stage
    assert second.document_id == first.document_id


def test_failed_retry_is_versioned_and_isolates_other_documents(storage):
    first = accept(storage)
    other = accept(storage, b"other")
    with Session(storage.engine) as db:
        doc = db.get(Document, first.document_id)
        doc.processing_status = "failed"
        doc.error_message = "failure"
        db.add_all([
            DocumentChunk(document_id=first.document_id, content="stale", attempt=1, vector_id="old-v"),
            DocumentChunk(document_id=other.document_id, content="keep", attempt=1),
        ])
        db.commit()
    retry = accept(storage)
    assert retry.document_id == first.document_id and retry.attempt == 2 and not retry.created
    with Session(storage.engine) as db:
        doc = db.get(Document, first.document_id)
        assert not doc.is_queryable and doc.error_message is None
        assert db.scalars(select(DocumentChunk)).one().document_id == other.document_id
        assert db.get(IngestionAttempt, (doc.id, 1)).status == "superseded"
        assert json.loads(db.get(IngestionAttempt, (doc.id, 2)).cleanup_json)["vector_ids"] == ["old-v"]
        doc.processing_status = "failed"
        db.commit()
    accept(storage)
    with Session(storage.engine) as db:
        assert json.loads(db.get(IngestionAttempt, (first.document_id, 3)).cleanup_json)["vector_ids"] == ["old-v"]


def test_deleted_reupload_does_not_revive_old_reference(storage):
    first = accept(storage)
    with Session(storage.engine) as db:
        doc = db.get(Document, first.document_id)
        storage.original_path(doc.id).unlink()
        doc.file_path = None
        doc.deletion_status = "deleted"
        doc.deletion_epoch = 1
        db.add(DeletionTask(document_id=doc.id, deletion_epoch=1, status="completed", stage="done"))
        db.add(QueryHistory(query="q", response="keep answer", source_status="deleted",
                            retrieved_chunks=json.dumps([{"document_id": doc.id, "source_status": "deleted"}])))
        db.commit()
    second = accept(storage)
    assert second.created and second.document_id > first.document_id
    with pytest.raises(SourceUnavailable):
        storage.original_path(first.document_id)
    with Session(storage.engine) as db:
        history = db.scalars(select(QueryHistory)).one()
        assert history.response == "keep answer" and history.source_status == "deleted"
        assert json.loads(history.retrieved_chunks)[0]["document_id"] == first.document_id
        assert db.get(DeletionTask, first.document_id).status == "completed"


@pytest.mark.parametrize("stage", ["deleting", "failed"])
def test_deleting_or_cleanup_failed_cannot_be_retried(storage, stage):
    first = accept(storage)
    with Session(storage.engine) as db:
        db.get(Document, first.document_id).deletion_status = stage
        db.commit()
    with pytest.raises(SourceUnavailable):
        accept(storage)
    with pytest.raises(SourceUnavailable):
        storage.original_path(first.document_id)


@pytest.mark.parametrize("failure", ["read", "replace", "commit", "size"])
def test_interruption_leaves_no_registered_or_partial_source(storage, monkeypatch, failure):
    class Interrupted(BytesIO):
        def read(self, size):
            if self.tell():
                raise OSError("interrupted read")
            return super().read(size)

    def fail(*args, **kwargs):
        raise OSError("interrupted write")

    source = Interrupted(b"partial") if failure == "read" else BytesIO(b"text")
    if failure == "replace":
        monkeypatch.setattr("app.services.document_storage.os.replace", fail)
    if failure == "commit":
        event.listen(storage.engine, "commit", fail)
    if failure == "size":
        storage.max_bytes = 1
    with pytest.raises((OSError, ValueError)):
        storage.accept(source, "policy.pdf")
    if failure == "commit":
        event.remove(storage.engine, "commit", fail)
    with Session(storage.engine) as db:
        assert db.scalar(select(func.count()).select_from(Document)) == 0
        assert db.scalar(select(func.count()).select_from(IngestionAttempt)) == 0
        assert db.scalar(select(func.count()).select_from(ActiveContentClaim)) == 0
    assert not list(storage.root.rglob('*.pdf'))
    assert not list(storage.root.glob('.upload-*'))


def test_orphan_and_external_path_never_used_as_original(storage, tmp_path):
    orphan = storage.root / 'orphan.pdf'
    orphan.write_bytes(b"orphan")
    with pytest.raises(SourceUnavailable):
        storage.original_path(999)
    first = accept(storage)
    external = tmp_path / 'outside.pdf'
    external.write_bytes(b"outside")
    with Session(storage.engine) as db:
        db.get(Document, first.document_id).file_path = str(external)
        db.commit()
    with pytest.raises(SourceUnavailable):
        storage.original_path(first.document_id)


@pytest.mark.parametrize("existing_failed,second_content", [
    (False, b"different source"), (False, b"first source"), (True, b"first source"),
])
def test_failed_cleanup_cannot_delete_next_accepted_upload(
    storage, monkeypatch, existing_failed, second_content,
):
    if existing_failed:
        previous = accept(storage, b"first source")
        storage.original_path(previous.document_id).unlink()
        with Session(storage.engine) as db:
            db.get(Document, previous.document_id).processing_status = "failed"
            db.commit()

    second_storage = DocumentStorage(storage.engine, storage.root)
    accepted = []
    unlink = Path.unlink

    def accept_next_before_cleanup(path, *args, **kwargs):
        if path.suffix == ".pdf" and not accepted:
            # The failing transaction has released its lock. Complete another
            # upload before its deferred file cleanup, without timing or sleeps.
            accepted.append(accept(second_storage, second_content))
        return unlink(path, *args, **kwargs)

    def fail_commit(connection):
        connection.rollback()
        raise OSError("injected commit failure")

    monkeypatch.setattr(storage, "_commit", fail_commit)
    monkeypatch.setattr(Path, "unlink", accept_next_before_cleanup)
    with pytest.raises(OSError, match="injected commit failure"):
        accept(storage, b"first source")

    assert len(accepted) == 1
    assert accepted[0].document_id == 1  # Rolled-back IDs/attempts really were reused.
    assert second_storage.original_path(accepted[0].document_id).read_bytes() == second_content
    with Session(storage.engine) as db:
        assert db.scalar(select(func.count()).select_from(Document)) == 1
        doc = db.get(Document, accepted[0].document_id)
        assert doc.processing_status == "queued"
        assert db.get(IngestionAttempt, (doc.id, accepted[0].attempt)).status == "queued"
    assert not list(storage.root.glob('.upload-*'))


@pytest.mark.parametrize("legacy_filename", [False, True])
@pytest.mark.parametrize("commit_fails", [False, True])
def test_failed_retry_reuses_original_without_replacing_or_cleaning_it(
    storage, monkeypatch, legacy_filename, commit_fails,
):
    previous = accept(storage, b"original bytes")
    path = storage.original_path(previous.document_id)
    if legacy_filename:
        legacy_path = path.with_name("original.pdf")
        path.rename(legacy_path)
        path = legacy_path
    with Session(storage.engine) as db:
        doc = db.get(Document, previous.document_id)
        doc.file_path = str(path)
        doc.processing_status = "failed"
        db.commit()

    def unexpected_replace(*args, **kwargs):
        pytest.fail("A retry must not replace an existing registered original")

    monkeypatch.setattr("app.services.document_storage.os.replace", unexpected_replace)
    if commit_fails:
        def fail_commit(connection):
            connection.rollback()
            raise OSError("injected retry failure")
        monkeypatch.setattr(storage, "_commit", fail_commit)
        with pytest.raises(OSError, match="injected retry failure"):
            accept(storage, b"original bytes")
    else:
        retry = accept(storage, b"original bytes")
        assert retry.document_id == previous.document_id and retry.attempt == 2

    assert storage.original_path(previous.document_id) == path
    assert path.read_bytes() == b"original bytes"
    assert list(storage.root.rglob('*.pdf')) == [path]
    with Session(storage.engine) as db:
        doc = db.get(Document, previous.document_id)
        assert doc.attempt == (1 if commit_fails else 2)
        assert doc.processing_status == ("failed" if commit_fails else "queued")
    assert not list(storage.root.glob('.upload-*'))
