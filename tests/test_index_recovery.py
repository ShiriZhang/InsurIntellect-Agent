import json

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database_models import DeletionTask, Document, IndexCommit, IndexPointer, IngestionAttempt
from app.services.index_coordinator import IndexCoordinator
from test_index_coordinator import candidate, library, publish  # noqa: F401


def test_missing_keyword_snapshot_rebuilds_without_embedding_or_generation(library):
    ingestion, coordinator, _ = library
    item, version = publish(library)
    with Session(ingestion.engine) as db:
        row = db.get(IndexCommit, version)
        path = coordinator.snapshots.path(row.snapshot_path)
        profile = json.loads(row.embedding_profile_json)
    path.unlink()
    restarted = IndexCoordinator(ingestion.engine, vector_root=coordinator.vector_root,
                                 processed_root=coordinator.processed_root, vectors=coordinator.vectors)
    assert restarted.recover() == version
    assert path.exists()
    with restarted.open_view([item.document_id], profile) as view:
        assert view.search("waiting period", [1., .5, .25])[0]["document_id"] == item.document_id


def test_orphan_files_and_legacy_manifest_cannot_restore_deleted_or_uncommitted_source(library):
    ingestion, coordinator, root = library
    first, version = publish(library)
    second = candidate(library, "unpublished")
    with Session(ingestion.engine) as db:
        profile = json.loads(db.get(IndexCommit, version).embedding_profile_json)
        db.add(DeletionTask(document_id=first.document_id, deletion_epoch=1, status="pending"))
        # Journal alone wins, even if an interrupted older writer left status=ready.
        db.commit()
    orphan, _ = coordinator.snapshots.write(coordinator.snapshots.build(profile, [], {}))
    temporary = coordinator.snapshots.path(orphan).parent / (coordinator.snapshots.temp_prefix + 'abandoned')
    temporary.write_text("source material", encoding="utf-8")
    (root / 'processed' / 'corpus_manifest.json').write_text(json.dumps({"documents": [{"id": first.document_id}]}))
    coordinator.recover()
    assert not coordinator.snapshots.path(orphan).exists() and not temporary.exists()
    assert coordinator.ready_ids() == []
    assert not ingestion.snapshot(first.document_id)["ready"]
    assert not ingestion.snapshot(second.document_id)["ready"]
    with Session(ingestion.engine) as db:
        assert db.get(Document, first.document_id).deletion_status == "deleting"
        assert db.get(IngestionAttempt, (first.document_id, 1)).status == "cancelled"
        pointer = db.get(IndexPointer, 'current')
        row = db.get(IndexCommit, pointer.commit_id)
        assert json.loads(row.document_attempts_json) == []
        assert not coordinator.vectors.rows[profile["fingerprint"]]


def test_deleted_source_is_removed_while_other_committed_source_recovers(library):
    ingestion, coordinator, _ = library
    first, _ = publish(library)
    second, version = publish(library, 'B')
    with Session(ingestion.engine) as db:
        profile = json.loads(db.get(IndexCommit, version).embedding_profile_json)
        doc = db.get(Document, first.document_id)
        doc.deletion_status = 'failed'
        doc.deletion_epoch += 1
        db.commit()
    coordinator.recover()
    assert coordinator.ready_ids() == [second.document_id]
    for value in coordinator.vectors.rows[profile["fingerprint"]].values():
        assert value["metadata"]["document_id"] == second.document_id
    for path in coordinator.snapshots.root.glob('*/v-*.json'):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        assert snapshot["document_attempts"] == [[second.document_id, 1]]
    with coordinator.open_view([second.document_id], profile) as view:
        assert view.search("waiting period", [1., .5, .25])


def test_missing_committed_vectors_cannot_be_faked_from_manifest_or_model(library):
    ingestion, coordinator, _ = library
    item, version = publish(library)
    with Session(ingestion.engine) as db:
        profile = json.loads(db.get(IndexCommit, version).embedding_profile_json)
    coordinator.vectors.rows[profile["fingerprint"]].clear()
    with pytest.raises(Exception):
        coordinator.recover()
    assert coordinator.ready_ids() == []


def test_completed_tombstone_also_wins_over_restored_ready_row(library):
    ingestion, coordinator, _ = library
    item, _ = publish(library)
    with Session(ingestion.engine) as db:
        db.add(DeletionTask(document_id=item.document_id, deletion_epoch=1, status='completed',
                            stage='done', completed_stages_json='["all"]'))
        db.commit()
    assert not coordinator.ready_ids()
    coordinator.recover()
    with Session(ingestion.engine) as db:
        assert not db.get(Document, item.document_id).is_queryable
        task = db.get(DeletionTask, item.document_id)
        assert task.status == 'pending' and task.completed_stages_json == '[]'
    assert not coordinator.ready_ids()


def test_recovery_cannot_collect_another_library_sharing_configured_index_directories(library):
    from sqlalchemy import create_engine
    from app.services.document_ingestion import DocumentIngestionService
    from app.services.document_storage import DocumentStorage
    from ingestion_fakes import LocalEncoderFake

    ingestion, coordinator, root = library
    first, version = publish(library)
    second_engine = create_engine(f"sqlite:///{root / 'separate.db'}")
    second_ingestion = DocumentIngestionService(DocumentStorage(second_engine, root / 'other-pdfs'),
                                                auto_publish=False, embedding_factory=LocalEncoderFake)
    second_ingestion.initialize()
    other = IndexCoordinator(second_engine, vectors=coordinator.vectors, vector_root=coordinator.vector_root,
                              processed_root=coordinator.processed_root)
    try:
        second = candidate((second_ingestion, other, root), 'OtherLibrary')
        second_version = other.publish(second.document_id, second.attempt)
        with Session(second_engine) as db:
            other_path = db.get(IndexCommit, second_version).snapshot_path
            other_collection = db.get(IndexCommit, second_version).vector_collection
        coordinator.recover()
        assert coordinator.snapshots.path(other_path).exists()
        assert other_collection in coordinator.vectors.collections
        assert coordinator.ready_ids() == [first.document_id]
        assert other.ready_ids() == [second.document_id]
        with Session(ingestion.engine) as db:
            old_path = db.get(IndexCommit, version).snapshot_path
        with Session(second_engine) as db:
            other_profile = json.loads(db.get(IndexCommit, second_version).embedding_profile_json)
        unrelated = other.open_view([second.document_id], other_profile)
        try:
            publish(library, 'B')
            assert not coordinator.snapshots.path(old_path).exists()
            assert unrelated.search('waiting period', [1., .5, .25])
        finally:
            unrelated.close()
    finally:
        second_engine.dispose()
