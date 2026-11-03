"""Persistent, retryable source erasure. No model loading or generation calls."""
from datetime import datetime
import json
from pathlib import Path
import threading
from sqlalchemy import select, delete
from sqlalchemy.orm import Session
from app.models.database_models import (Document, DocumentChunk, DocumentMetadata, IngestionAttempt,
                                       DeletionTask, QueryHistory, SessionScope, ActiveContentClaim)
from app.services.ingestion_worker_lock import IngestionWorkerLock

class DeletionPending(RuntimeError):
    pass


def scrub_history(db, document_id, name):
    for row in db.scalars(select(QueryHistory)):
        chunks = json.loads(row.retrieved_chunks or "[]")
        scope = json.loads(row.document_ids_json or "[]")
        affected = document_id in scope
        cleaned = []
        for chunk in chunks:
            identity = chunk.get("document_id", (chunk.get("metadata") or {}).get("document_id"))
            matches = str(identity) == str(document_id) or (identity is None and chunk.get("document_name") == name)
            if matches:
                affected = True
                chunk = {key: chunk.get(key) for key in ("chunk_id", "document_id", "document_name", "page_number")}
                chunk["document_id"] = document_id
                chunk["source_status"] = "deleted"
            cleaned.append(chunk)
        if affected:
            row.retrieved_chunks = json.dumps(cleaned, ensure_ascii=False)
            row.source_status = "deleted"
            row.rewritten_query = None
            meta = json.loads(row.rewriting_metadata_json or "{}")
            row.rewriting_metadata_json = json.dumps({"answer_kind": meta.get("answer_kind")})
    for scope in db.scalars(select(SessionScope)):
        if document_id in json.loads(scope.document_ids_json):
            scope.invalidated = True


class DocumentDeletionService:
    stages = ("indexes", "original", "history", "chunks", "complete")

    def __init__(self, ingestion, *, fault=lambda stage: None):
        self.ingestion, self.coordinator = ingestion, ingestion.coordinator
        self.engine, self.storage = ingestion.engine, ingestion.storage
        self.fault = fault
        self.stop_event = threading.Event()
        self.thread = None
        self.lock = IngestionWorkerLock(Path(self.coordinator.database_key + ".deletion.lock"))

    def snapshot(self, document_id):
        with Session(self.engine) as db:
            doc = db.get(Document, document_id)
            if doc is None:
                raise KeyError(document_id)
            task = db.get(DeletionTask, document_id)
            return {"document_id": document_id, "deletion_status": doc.deletion_status,
                    "status": task.status if task else "none", "stage": task.stage if task else None,
                    "error": task.error_message if task else None,
                    "completed_stages": json.loads(task.completed_stages_json or "[]") if task else []}

    def begin(self, document_id):
        with self.coordinator.state.lock:
            with self.coordinator.transaction() as db:
                doc = db.get(Document, document_id)
                if doc is None:
                    raise KeyError(document_id)
                task = db.get(DeletionTask, document_id)
                if task is None:
                    doc.deletion_epoch += 1
                    doc.deletion_status = "deleting"
                    task = DeletionTask(document_id=document_id, deletion_epoch=doc.deletion_epoch)
                    db.add(task)
                elif task.status != "completed":
                    task.status, task.error_message = "pending", None
                # History must become unreadable in the same commit that
                # makes the source deleting, even when later cleanup is blocked.
                scrub_history(db, document_id, doc.original_name or doc.filename)
                vector_ids = list(db.scalars(select(DocumentChunk.vector_id).where(DocumentChunk.document_id == document_id)))
                for attempt in db.scalars(select(IngestionAttempt).where(IngestionAttempt.document_id == document_id)):
                    vector_ids.extend(json.loads(attempt.cleanup_json or "{}").get("vector_ids", []))
                    attempt.status = "cancelled"
                db.execute(delete(DocumentMetadata).where(DocumentMetadata.chunk_id.in_(vector_ids)))
            for request in list(self.coordinator.state.requests):
                if document_id in request.document_ids:
                    request.cancel()
            for view in list(self.coordinator.state.views):
                if document_id in view.document_ids:
                    view.cancel("source_deleted")
        self.ingestion.cancel(document_id)
        return self.snapshot(document_id)

    def recover(self):
        # A process may be restoring old files while another deletion worker owns
        # cleanup. Serialize replay with that same OS lock.
        if not self.lock.acquire():
            return
        try:
            with Session(self.engine) as db:
                ids = set(db.scalars(select(DeletionTask.document_id))) | set(db.scalars(
                    select(Document.id).where(Document.deletion_status != "none")))
            for document_id in ids:
                self.begin(document_id)
                with self.coordinator.transaction() as db:
                    doc, task = db.get(Document, document_id), db.get(DeletionTask, document_id)
                    doc.deletion_status = "deleting"
                    task.status, task.stage, task.completed_stages_json = "pending", "invalidate", "[]"
        finally:
            self.lock.release()
        self.drain()

    def _original(self, document_id):
        # Only the controlled identity directory is owned by the application.
        root = self.storage.root.resolve()
        with Session(self.engine) as db:
            doc = db.get(Document, document_id)
            if doc.file_path:
                registered = Path(doc.file_path)
                expected = self.storage._controlled_path(document_id, registered.name)
                if registered.resolve() != expected.resolve():
                    raise RuntimeError("original_storage_configuration_changed")
        directory = root / str(document_id)
        if directory.resolve() != directory or not directory.is_relative_to(root):
            raise RuntimeError("original_directory_outside_controlled_storage")
        if directory.exists():
            paths = list(directory.rglob("*"))
            if any(not p.resolve().is_relative_to(directory) or p.is_symlink() for p in paths):
                raise RuntimeError("original_path_outside_controlled_storage")
            for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
                path.rmdir() if path.is_dir() else path.unlink(missing_ok=True)
            directory.rmdir()

    def _request_readers(self, document_id):
        database = Path(self.coordinator.database_key)
        pending = False
        for path in database.parent.glob(database.name + f".request-{document_id}-*.lock"):
            lease = IngestionWorkerLock(path)
            if not lease.acquire():
                pending = True
            else:
                lease.release()
                path.unlink(missing_ok=True)
        return pending

    def _step(self, document_id, stage):
        if stage == "indexes":
            if not self.coordinator.invalidate_source(document_id):
                raise DeletionPending("index_readers_or_cleanup_pending")
        elif stage == "original":
            self._original(document_id)
        elif stage == "history":
            with self.coordinator.state.lock, self.coordinator.transaction() as db:
                doc = db.get(Document, document_id)
                scrub_history(db, document_id, doc.original_name or doc.filename)
        elif stage == "chunks":
            with self.coordinator.transaction() as db:
                ids = list(db.scalars(select(DocumentChunk.vector_id).where(DocumentChunk.document_id == document_id)))
                db.execute(delete(DocumentMetadata).where(DocumentMetadata.chunk_id.in_(ids)))
                db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
                for task in db.scalars(select(IngestionAttempt).where(IngestionAttempt.document_id == document_id)):
                    task.status = "cancelled"
                    task.cleanup_json = task.embedding_profile_json = task.diagnostics_json = task.error_message = None
        elif stage == "complete":
            if self.ingestion.is_processing(document_id) or self._request_readers(document_id):
                raise DeletionPending("source_readers_still_releasing")
            with self.coordinator.state.lock, self.coordinator.transaction() as db:
                doc = db.get(Document, document_id)
                scrub_history(db, document_id, doc.original_name or doc.filename)
                doc.file_path = doc.published_fingerprint = doc.index_commit_id = doc.published_attempt = None
                doc.error_code = doc.error_message = doc.error_pages_json = None
                doc.product_name = doc.effective_date = doc.document_type = None
                doc.deletion_status, doc.status = "deleted", "deleted"
                doc.deleted_time = datetime.utcnow()
                db.execute(delete(ActiveContentClaim).where(ActiveContentClaim.document_id == document_id))
                task = db.get(DeletionTask, document_id)
                task.status, task.stage, task.error_message = "completed", "complete", None

    def run_once(self):
        if not self.lock.acquire():
            return False
        try:
            with Session(self.engine) as db:
                ids = list(db.scalars(select(DeletionTask.document_id).where(DeletionTask.status != "completed")))
            for document_id in ids:
                try:
                    if self.ingestion.is_processing(document_id):
                        raise DeletionPending("native_ingestion_still_releasing_source")
                    # Idempotently rerun stages; no old checkpoint can bypass a
                    # late cleanup requirement after a process crash.
                    for stage in self.stages:
                        with self.coordinator.transaction() as db:
                            db.get(DeletionTask, document_id).stage = stage
                        self._step(document_id, stage)
                        self.fault(stage)
                        with self.coordinator.transaction() as db:
                            task = db.get(DeletionTask, document_id)
                            completed = json.loads(task.completed_stages_json or "[]")
                            if stage not in completed:
                                completed.append(stage)
                            task.completed_stages_json = json.dumps(completed)
                except Exception as exc:
                    with self.coordinator.transaction() as db:
                        db.get(Document, document_id).deletion_status = "deleting"
                        task = db.get(DeletionTask, document_id)
                        task.status = "pending" if isinstance(exc, DeletionPending) else "failed"
                        task.error_message = str(exc) if isinstance(exc, DeletionPending) else type(exc).__name__
            return bool(ids)
        finally:
            self.lock.release()

    def drain(self):
        self.run_once()

    def start(self):
        self.thread = threading.Thread(target=self._run, name="document-deletion", daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            self.run_once()
            self.stop_event.wait(.25)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
