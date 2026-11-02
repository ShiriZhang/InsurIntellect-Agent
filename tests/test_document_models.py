"""Document lifecycle persistence contracts, using disposable SQLite databases."""

import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.database_models import (
    DeletionTask, Document, DocumentChunk, IndexCommit, IndexPointer, QueryHistory, SessionScope,
)


def test_new_database_lifecycle_roundtrip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        doc = Document(filename="policy.pdf", attempt=1)
        db.add(doc)
        db.flush()
        assert doc.product_name is None and doc.effective_date is None
        assert doc.status == "active" and not doc.is_queryable
        commit = IndexCommit(embedding_fingerprint="profile", document_attempts_json=f"[[{doc.id},1]]",
                             vector_collection="profile", snapshot_path="versions/1")
        db.add(commit)
        db.flush()
        db.add(IndexPointer(commit_id=commit.id))
        doc.processing_status = "ready"
        doc.published_attempt = 1
        doc.published_fingerprint = "profile"
        doc.index_commit_id = commit.id
        assert doc.is_queryable
        db.add(DocumentChunk(document_id=doc.id, attempt=1, chunk_index=0, page_number=3,
                             page_chunk_index=0, content="原文", content_hash="hash",
                             vector_id=f"{doc.id}:1:3:0:hash", section_title="等待期"))
        for epoch, scope in enumerate(([doc.id], [99], [doc.id]), 1):
            db.add(SessionScope(session_id="s", epoch=epoch, scope_fingerprint=str(scope),
                                document_ids_json=json.dumps(scope), invalidated=epoch < 3))
        db.add(QueryHistory(query="q", response="a", session_id="s", scope_status="known",
                            scope_epoch=1, document_ids_json=f"[{doc.id}]", source_status="deleted"))
        doc.deletion_status = "deleting"
        doc.deletion_epoch = 1
        db.add(DeletionTask(document_id=doc.id, deletion_epoch=1, stage="files"))
        assert not doc.is_queryable
        db.commit()
    with Session(engine) as db:
        chunk = db.scalars(select(DocumentChunk)).one()
        assert chunk.page_number == 3 and chunk.content == "原文"
        assert db.scalars(select(SessionScope).order_by(SessionScope.epoch)).all()[2].epoch == 3
        assert db.scalars(select(QueryHistory)).one().source_status == "deleted"
        assert db.scalars(select(DeletionTask)).one().stage == "files"
        db.add(DocumentChunk(document_id=chunk.document_id, attempt=1, page_number=3,
                             page_chunk_index=0, content="duplicate"))
        with pytest.raises(IntegrityError):
            db.commit()
    engine.dispose()
