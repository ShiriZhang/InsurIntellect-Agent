"""Controlled original-file registration; parsing/publication belongs to ingestion.

This blocking service must run in a thread when called from an async endpoint.
SQLite BEGIN IMMEDIATE serializes content ownership across processes, not just threads.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import BinaryIO
from uuid import uuid4

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import ASYNC_DB_URL
from app.models.database_models import (
    ActiveContentClaim, Document, DocumentChunk, IngestionAttempt,
)


class SourceUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class StoredDocument:
    document_id: int
    attempt: int
    processing_status: str
    created: bool


class DocumentStorage:
    def __init__(self, engine=None, root=None, max_bytes=None):
        self.engine = engine or create_engine(
            ASYNC_DB_URL.replace("sqlite+aiosqlite:", "sqlite:"),
            connect_args={"timeout": 30},
        )
        self.root = Path(root or settings.PDF_STORAGE_PATH).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = settings.UPLOAD_MAX_BYTES if max_bytes is None else max_bytes

    @staticmethod
    def _commit(connection):
        try:
            connection.commit()
        except BaseException:
            # A failing commit hook can invalidate SQLAlchemy's transaction before
            # SQLite is rolled back. Never return a dirty connection to the pool.
            connection.connection.rollback()
            raise

    def _controlled_path(self, document_id, filename):
        if not re.fullmatch(r"original(?:-[0-9a-f]{32})?\.pdf", filename):
            raise SourceUnavailable("Invalid internal original filename")
        path = self.root / str(document_id) / filename
        if not path.resolve().is_relative_to(self.root):
            raise SourceUnavailable("Original path escapes controlled storage")
        return path

    def _registered_path(self, doc):
        if not doc.file_path:
            raise SourceUnavailable("Original is not in controlled storage")
        stored = Path(doc.file_path)
        path = self._controlled_path(doc.id, stored.name)
        if stored.resolve() != path.resolve() or not path.is_file():
            raise SourceUnavailable("Original is not in controlled storage")
        return path

    def original_path(self, document_id):
        """Resolve from database identity; never accept a client-provided file path."""
        with Session(self.engine) as db:
            doc = db.get(Document, document_id)
            if (doc is None or doc.deletion_status != "none" or doc.status != "active"
                    or not doc.file_path):
                raise SourceUnavailable("Source is missing or deleted")
            return self._registered_path(doc)

    def accept(self, source: BinaryIO, original_name: str) -> StoredDocument:
        """Spool + fsync before claiming a hash; commit only after atomic replacement.

        A hard crash can leave an unregistered file, which is never a ready source.
        The ingestion/deletion workers must finish recorded retry cleanup before publication.
        """
        name = original_name.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name or len(name) > 512 or "\x00" in name:
            raise ValueError("Invalid original filename")
        temp_path = None
        new_path = None
        committed = False
        try:
            digest = sha256()
            size = 0
            with tempfile.NamedTemporaryFile(dir=self.root, prefix=".upload-", suffix=".tmp",
                                             delete=False) as output:
                temp_path = Path(output.name)
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > self.max_bytes:
                        raise ValueError("File exceeds upload size limit")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            file_hash = digest.hexdigest()
            with self.engine.connect() as conn:
                conn.exec_driver_sql("BEGIN IMMEDIATE")
                with Session(bind=conn, expire_on_commit=False) as db:
                    claim = db.get(ActiveContentClaim, file_hash)
                    doc = db.get(Document, claim.document_id) if claim else None
                    # Include legacy rows without claims without inventing readiness.
                    if doc is None:
                        doc = db.scalars(select(Document).where(
                            Document.file_hash == file_hash,
                            Document.deletion_status != "deleted",
                        ).order_by(Document.id)).first()
                    if doc is not None and doc.deletion_status == "deleted":
                        db.delete(claim)
                        db.flush()
                        doc = None
                        claim = None
                    if doc is not None and (doc.deletion_status != "none" or doc.status != "active"):
                        raise SourceUnavailable("Source is being deleted or is inactive")
                    if doc is not None and doc.processing_status in {"queued", "parsing", "indexing", "ready"}:
                        result = StoredDocument(doc.id, doc.attempt, doc.processing_status, False)
                        self._commit(conn)
                        committed = True
                        return result
                    created = doc is None
                    if created:
                        doc = Document(filename=name, original_name=name, file_hash=file_hash,
                                       file_size=size, processing_status="queued", attempt=1)
                        db.add(doc)
                        db.flush()
                    else:
                        doc.attempt += 1
                    if claim is None:
                        db.add(ActiveContentClaim(file_hash=file_hash, document_id=doc.id))
                    cleanup = []
                    for chunk in db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)):
                        if chunk.vector_id:
                            cleanup.append(chunk.vector_id)
                    # Old vector identities remain durable until the index worker removes them.
                    old_attempts = db.scalars(select(IngestionAttempt).where(
                        IngestionAttempt.document_id == doc.id,
                        IngestionAttempt.status != "superseded",
                    )).all()
                    for old in old_attempts:
                        cleanup.extend(json.loads(old.cleanup_json or "{}").get("vector_ids", []))
                        old.status = "superseded"
                    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
                    doc.processing_status = "queued"
                    doc.error_code = doc.error_message = doc.error_pages_json = None
                    doc.page_count = doc.language = None
                    doc.published_attempt = doc.published_fingerprint = doc.index_commit_id = None
                    try:
                        # A failed retry already owns identical bytes. Reuse the
                        # registered original without replacing it or owning its cleanup.
                        path = self._registered_path(doc)
                    except SourceUnavailable:
                        # Rolled-back IDs/attempts can be reused. Each write needs
                        # its own path so deferred cleanup cannot remove a later upload.
                        path = self._controlled_path(doc.id, f"original-{uuid4().hex}.pdf")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        new_path = path
                        os.replace(temp_path, path)
                    doc.file_path = str(path)
                    doc.file_size = size
                    db.add(IngestionAttempt(document_id=doc.id, attempt=doc.attempt,
                                             deletion_epoch=doc.deletion_epoch,
                                             cleanup_json=json.dumps({"vector_ids": sorted(set(cleanup))})))
                    db.flush()
                    self._commit(conn)
                    committed = True
                    return StoredDocument(doc.id, doc.attempt, doc.processing_status, created)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            if not committed and new_path is not None:
                new_path.unlink(missing_ok=True)
