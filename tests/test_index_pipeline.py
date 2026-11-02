import json
from pathlib import Path
import pytest

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database_models import Document, DocumentChunk, IngestionAttempt
from app.services.document_ingestion import DocumentIngestionService
from scripts import simple_ingest
from ingestion_fakes import LocalEncoderFake
from test_index_coordinator import candidate, library, publish  # noqa: F401
from test_pdf_quality import make_pdf


def publishing_service(library, factory=LocalEncoderFake):
    ingestion, coordinator, _ = library
    return DocumentIngestionService(ingestion.storage, coordinator=coordinator, embedding_factory=factory)


def test_cli_and_default_worker_reach_ready_using_both_committed_indexes(library):
    _, coordinator, root = library
    source = root / 'source'
    source.mkdir()
    make_pdf(source / 'A.pdf', ['Policy A waiting period is 90 days except accidents.'])
    make_pdf(source / 'B.pdf', ['Policy B waiting period is 180 days, illness is excluded unless stated.'])
    service = publishing_service(library)
    try:
        result = simple_ingest.ingest(source, service=service)
        assert result['ready'] and result['prepared'] and result['publication_enabled']
        assert {item['task_status'] for item in result['documents']} == {'ready'}
        assert len(coordinator.ready_ids()) == 2
        assert len(list(coordinator.snapshots.root.glob('*/v-*.json'))) == 1
    finally:
        service.stop()


def test_restart_requeues_interrupted_attempt_without_duplicating_chunks(library):
    ingestion, coordinator, _ = library
    first, _ = publish(library)
    second = candidate(library, 'B')
    with Session(ingestion.engine) as db:
        ids = list(db.scalars(select(DocumentChunk.vector_id).where(DocumentChunk.document_id == second.document_id)))
        db.get(IngestionAttempt, (second.document_id, 1)).status = 'indexing'
        db.commit()
    restarted = publishing_service(library)
    assert restarted.run_until_idle()
    assert coordinator.ready_ids() == [first.document_id, second.document_id]
    with Session(ingestion.engine) as db:
        assert list(db.scalars(select(DocumentChunk.vector_id).where(DocumentChunk.document_id == second.document_id))) == ids


def test_prepared_candidate_publishes_without_loading_embedding_or_generation(library):
    item = candidate(library)
    def no_model():
        raise AssertionError('Publication must use persisted vectors, not run a model')
    service = publishing_service(library, no_model)
    service.run_until_idle()
    assert service.snapshot(item.document_id)['ready'] and service.embedding is None


def test_publication_failure_is_honest_and_failed_retry_has_new_attempt(library):
    ingestion, coordinator, root = library
    first, _ = publish(library)
    second = candidate(library, 'B')
    coordinator.vectors.fail_write = True
    service = publishing_service(library)
    service.run_until_idle()
    assert service.snapshot(second.document_id)['error_code'] == 'index_publication_failed'
    assert coordinator.ready_ids() == [first.document_id]
    coordinator.vectors.fail_write = False
    with (root / 'B.pdf').open('rb') as source:
        retry = service.submit(source, 'B.pdf')
    assert retry.document_id == second.document_id and retry.attempt == 2
    service.run_until_idle()
    assert service.snapshot(second.document_id)['ready']
    assert all(row['metadata']['attempt'] == 2 for row in next(iter(coordinator.vectors.rows.values())).values()
               if row['metadata']['document_id'] == second.document_id)


def test_cancellation_between_candidate_write_and_pointer_commit_cannot_publish(library):
    _, coordinator, _ = library
    first, _ = publish(library)
    second = candidate(library, 'B')
    service = publishing_service(library)
    coordinator.fault = lambda stage: service.cancel(second.document_id) if stage == 'after_snapshot' else None
    service.run_until_idle()
    assert coordinator.ready_ids() == [first.document_id]
    assert service.snapshot(second.document_id)['task_status'] == 'cancelled'
    assert not service.snapshot(second.document_id)['ready']


def test_same_dimension_profile_change_requires_rebuild_not_a_mixed_library(library):
    ingestion, coordinator, _ = library
    first, _ = publish(library)
    second = candidate(library, 'B')
    from app.services.embedding_profile import EmbeddingProfile
    with Session(ingestion.engine) as db:
        task = db.get(IngestionAttempt, (second.document_id, 1))
        record = json.loads(task.embedding_profile_json)['profile']
        changed = EmbeddingProfile.create(**{**record, 'model': 'different-model-same-dimension'}).to_record()
        task.embedding_profile_json = json.dumps(changed)
        db.commit()
    service = publishing_service(library)
    service.run_until_idle()
    assert coordinator.ready_ids() == [first.document_id]
    assert not service.snapshot(second.document_id)['ready']


@pytest.mark.parametrize('failure', ['collection', 'snapshot'])
def test_retirement_failure_cannot_rewrite_a_committed_ready_document(library, monkeypatch, failure):
    ingestion, coordinator, _ = library
    first, original = publish(library)
    second = candidate(library, 'B')
    with Session(ingestion.engine) as db:
        from app.models.database_models import IndexCommit, IndexPointer
        old = db.get(IndexCommit, original)
        old_collection, old_path = old.vector_collection, coordinator.snapshots.path(old.snapshot_path)
    drop, unlink = coordinator.vectors.drop, Path.unlink
    def fail_drop(profile, collection):
        if collection == old_collection:
            raise OSError('retirement failed')
        return drop(profile, collection)
    def fail_unlink(path, *args, **kwargs):
        if path == old_path:
            raise OSError('retirement failed')
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as failures:
        if failure == 'collection':
            failures.setattr(coordinator.vectors, 'drop', fail_drop)
        else:
            failures.setattr(Path, 'unlink', fail_unlink)
        service = publishing_service(library)
        service.run_until_idle()
        assert service.snapshot(second.document_id)['ready']
        assert coordinator.state.gc_pending and coordinator.state.gc_error == 'OSError'
        with Session(ingestion.engine) as db:
            current = db.get(IndexPointer, 'current').commit_id
            assert current != original
            assert db.get(Document, second.document_id).is_queryable
            assert list(db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == second.document_id)))
            profile = json.loads(db.get(IndexCommit, current).embedding_profile_json)
        with coordinator.open_view([second.document_id], profile) as view:
            assert view.search('waiting period', [1., .5, .25])  # GC errors cannot mask valid evidence.
    coordinator.recover()
    assert coordinator.ready_ids() == [first.document_id, second.document_id]
    assert not coordinator.state.gc_pending and coordinator.state.gc_error is None


def test_failed_delete_index_cleanup_keeps_other_sources_queryable_and_reports_incomplete(library, monkeypatch):
    ingestion, coordinator, _ = library
    first, original = publish(library)
    second, _ = publish(library, 'B')
    with Session(ingestion.engine) as db:
        from app.models.database_models import IndexCommit
        old_collection = db.get(IndexCommit, original).vector_collection
        db.get(Document, first.document_id).deletion_status = 'deleting'
        db.commit()
    drop = coordinator.vectors.drop
    with monkeypatch.context() as failures:
        def fail_drop(profile, collection):
            if collection == old_collection:
                raise OSError('cleanup failed')
            return drop(profile, collection)
        failures.setattr(coordinator.vectors, 'drop', fail_drop)
        assert not coordinator.invalidate_source(first.document_id)
        assert coordinator.ready_ids() == [second.document_id]
        assert coordinator.state.serving
    assert coordinator.invalidate_source(first.document_id)


def test_unrelated_gc_cannot_clear_failed_orphan_cleanup_status(library, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from uuid import uuid4
    from app.models.database_models import IndexCommit

    ingestion, coordinator, _ = library
    _, version = publish(library)
    item = candidate(library, 'Uncommitted')
    with Session(ingestion.engine) as db:
        profile = json.loads(db.get(IndexCommit, version).embedding_profile_json)
        rows, _ = coordinator._rows(db, [(item.document_id, item.attempt)], profile)
        db.get(Document, item.document_id).deletion_status = 'deleting'
        db.commit()
    orphan = coordinator.vectors.collection_name(profile, uuid4().hex)
    coordinator.vectors.write(profile, rows, collection=orphan, owner=coordinator.library_id)
    cleanup, drop = coordinator._cleanup_best_effort, coordinator.vectors.drop
    def fail_orphan(profile, collection):
        if collection == orphan:
            raise OSError('orphan cleanup failed')
        return drop(profile, collection)
    def unrelated_collection_between_cleanup_and_result(*, sweep_orphans=False):
        cleanup(sweep_orphans=sweep_orphans)
        if sweep_orphans:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(coordinator.collect_garbage).result(timeout=5)
    with monkeypatch.context() as failures:
        failures.setattr(coordinator.vectors, 'drop', fail_orphan)
        failures.setattr(coordinator, '_cleanup_best_effort', unrelated_collection_between_cleanup_and_result)
        assert not coordinator.invalidate_source(item.document_id)
        assert orphan in coordinator.vectors.collections
        assert coordinator.state.gc_error == 'OSError' and coordinator.state.gc_pending
    assert coordinator.invalidate_source(item.document_id)
    assert orphan not in coordinator.vectors.collections
    assert coordinator.state.gc_error is None
