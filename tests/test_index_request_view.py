import gc
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from sqlalchemy.orm import Session

from app.models.database_models import Document, IndexCommit, QueryHistory
from app.services.index_coordinator import SourceInvalidated
from test_index_coordinator import candidate, library, publish  # noqa: F401


def committed(library, version):
    with Session(library[0].engine) as db:
        row = db.get(IndexCommit, version)
        return json.loads(row.embedding_profile_json), row.snapshot_path


def test_new_publication_visible_without_restart_does_not_expand_old_view(library):
    _, coordinator, _ = library
    first, version = publish(library)
    profile, old_path = committed(library, version)
    old = coordinator.open_view([first.document_id], profile)
    second, _ = publish(library, 'B')
    assert coordinator.snapshots.path(old_path).exists()  # Accepted reader has not retrieved yet.
    result = old.search("waiting period", [1., .5, .25])
    assert {r["document_id"] for r in result} == {first.document_id}
    assert not coordinator.state.readers  # Generation holds only selected evidence, no snapshot lease.
    assert not hasattr(old, 'snapshot')
    with coordinator.open_view([first.document_id, second.document_id], profile) as current:
        assert {r["document_id"] for r in current.search("waiting period", [1., .5, .25])} == {first.document_id, second.document_id}
    assert not coordinator.snapshots.path(old_path).exists()
    old.close()


def test_deletion_scrubs_related_evidence_and_old_snapshots_but_other_view_survives(library):
    ingestion, coordinator, _ = library
    first, _ = publish(library)
    second, version = publish(library, 'B')
    profile, old_path = committed(library, version)
    related = coordinator.open_view([first.document_id, second.document_id], profile)
    original_evidence = related.search("waiting period", [1., .5, .25])
    alias = original_evidence[0]
    unrelated = coordinator.open_view([second.document_id], profile)
    with Session(ingestion.engine) as db:
        doc = db.get(Document, first.document_id)
        doc.deletion_status = 'deleting'
        doc.deletion_epoch += 1
        db.commit()
    coordinator.recover()
    assert not original_evidence and not alias
    with pytest.raises(SourceInvalidated):
        related.assert_valid()
    assert unrelated.search("waiting period", [1., .5, .25])[0]["document_id"] == second.document_id
    unrelated.close()
    coordinator.collect_garbage()
    assert not coordinator.snapshots.path(old_path).exists()


def test_late_history_write_hook_rechecks_database_invalidation(library):
    ingestion, coordinator, _ = library
    item, version = publish(library)
    profile, _ = committed(library, version)
    with coordinator.open_view([item.document_id], profile) as view:
        view.search("waiting period", [1., .5, .25])
        view.guard_write(lambda db: db.add(QueryHistory(query="valid", response="preserved")))
        with Session(ingestion.engine) as db:
            db.get(Document, item.document_id).deletion_status = 'deleting'
            db.commit()
        with pytest.raises(SourceInvalidated):
            view.guard_write(lambda db: db.add(QueryHistory(query="late", response="must not save")))
    with Session(ingestion.engine) as db:
        from sqlalchemy import select
        assert list(db.scalars(select(QueryHistory.query))) == ['valid']


def test_unprepared_reader_is_released_when_request_is_abandoned(library):
    _, coordinator, _ = library
    first, version = publish(library)
    profile, _ = committed(library, version)
    view = coordinator.open_view([first.document_id], profile)
    assert coordinator.state.readers
    del view
    gc.collect()
    assert not coordinator.state.readers


def test_candidate_is_never_visible_before_sqlite_pointer_commit(library):
    _, coordinator, _ = library
    first, version = publish(library)
    profile, _ = committed(library, version)
    second = candidate(library, 'B')
    def inspect(point):
        if point == 'after_snapshot':
            view = coordinator.open_view([first.document_id], profile)
            try:
                assert {r["document_id"] for r in view.search("waiting period", [1., .5, .25])} == {first.document_id}
            finally:
                view.close()
            with pytest.raises(SourceInvalidated):
                coordinator.open_view([second.document_id], profile)
    coordinator.fault = inspect
    coordinator.publish(second.document_id, second.attempt)


def test_vector_filter_failure_can_only_use_in_scope_keyword_evidence(library):
    _, coordinator, _ = library
    first, version = publish(library)
    profile, _ = committed(library, version)
    publish(library, 'B')
    coordinator.vectors.fail_query = True
    with coordinator.open_view([first.document_id], profile) as view:
        result = view.search("waiting period", [1., .5, .25])
        assert result and {r["document_id"] for r in result} == {first.document_id}


def test_slow_candidate_construction_does_not_hold_query_or_upload_write_lock(library, monkeypatch):
    ingestion, coordinator, root = library
    first, version = publish(library)
    profile, _ = committed(library, version)
    second = candidate(library, 'B')
    started, release = Event(), Event()
    write = coordinator.vectors.write
    def slow_write(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return write(*args, **kwargs)
    monkeypatch.setattr(coordinator.vectors, 'write', slow_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(coordinator.publish, second.document_id, second.attempt)
        try:
            assert started.wait(5)
            with coordinator.open_view([first.document_id], profile) as view:
                assert view.search('waiting period', [1., .5, .25])
            from test_pdf_quality import make_pdf
            path = make_pdf(root / 'C.pdf', ['Another source must be accepted while indexing.'])
            with path.open('rb') as source:
                third = ingestion.submit(source, path.name)
            assert third.document_id not in {first.document_id, second.document_id}
        finally:
            release.set()
        future.result(timeout=5)


def test_cancelled_native_reader_retains_lease_until_late_result_is_discarded(library, monkeypatch):
    ingestion, coordinator, _ = library
    item, version = publish(library)
    profile, old_path = committed(library, version)
    view = coordinator.open_view([item.document_id], profile)
    started, release = Event(), Event()
    query = coordinator.vectors.query
    def slow_query(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return query(*args, **kwargs)
    monkeypatch.setattr(coordinator.vectors, 'query', slow_query)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(view.search, 'waiting period', [1., .5, .25])
        try:
            assert started.wait(5)
            with Session(ingestion.engine) as db:
                db.get(Document, item.document_id).deletion_status = 'deleting'
                db.commit()
            assert not coordinator.invalidate_source(item.document_id)
            assert coordinator.snapshots.path(old_path).exists()
            assert coordinator.state.readers
        finally:
            release.set()
        with pytest.raises(SourceInvalidated):
            future.result(timeout=5)
    assert not coordinator.state.readers
    assert not coordinator.snapshots.path(old_path).exists()
    assert coordinator.invalidate_source(item.document_id)
