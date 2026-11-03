"""Durable single-worker PDF -> source chunks -> local vector candidate preparation.

Candidates remain unpublished until IndexCoordinator (batch 4) commits both indexes.
No model is loaded at startup and no generation API is called by ingestion.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
import threading

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.app_logging import get_logger
from app.core.config import settings
from app.core.schema_migrations import migrate_schema
from app.models.database_models import DeletionTask, Document, DocumentChunk, IngestionAttempt
from app.services.document_chunking import ChunkBudgetError, chunk_document
from app.services.document_storage import DocumentStorage, SourceUnavailable
from app.services.embedding_profile import EmbeddingCompatibilityError, file_digest
from app.services.embedding_service import EmbeddingService, LocalEmbeddingUnavailable
from app.services.ingestion_worker_lock import IngestionWorkerLock
from app.services.pdf_quality import PDFQualityError, parse_pdf

logger = get_logger(__name__)
TERMINAL_ATTEMPTS = {"staged", "ready", "failed", "cancelled", "superseded"}


class IngestionCancelled(RuntimeError):
    pass


class IngestionInterrupted(RuntimeError):
    pass


class DocumentIngestionService:
    def __init__(self, storage=None, *, embedding_factory=EmbeddingService, parser=parse_pdf,
                 queue_size=None, batch_size=None, auto_publish=True, coordinator=None):
        self.storage = storage or DocumentStorage()
        self._owns_storage = storage is None
        self.engine = self.storage.engine
        from app.services.index_coordinator import IndexCoordinator
        self.coordinator = (coordinator or IndexCoordinator(self.engine)) if auto_publish else None
        self.embedding_factory, self.parser = embedding_factory, parser
        self.embedding = None
        self.batch_size = batch_size or settings.INGEST_BATCH_SIZE
        self.notifications = Queue(maxsize=queue_size or settings.INGEST_QUEUE_SIZE)
        database = self.engine.url.database
        lock_path = (Path(database).resolve().with_name(Path(database).name + ".ingestion.lock")
                     if database and database != ":memory:" else self.storage.root / ".ingestion.lock")
        self.worker_lock = IngestionWorkerLock(lock_path)
        self._stop = threading.Event()
        self._cancelled = set()
        self._active_documents = set()
        self._state_lock = threading.RLock()
        self._thread = None
        self.last_error = None

    def initialize(self):
        with self.engine.begin() as connection:
            migrate_schema(connection)

    @contextmanager
    def _transaction(self):
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            with Session(bind=connection, expire_on_commit=False) as db:
                yield db
                db.flush()
                self.storage._commit(connection)

    def submit(self, source, original_name):
        result = self.storage.accept(source, original_name)
        try:
            self.notifications.put_nowait((result.document_id, result.attempt))
        except Full:
            pass  # SQLite is the queue; this bounded queue only provides wakeup hints.
        return result

    def snapshot(self, document_id):
        with Session(self.engine) as db:
            doc = db.get(Document, document_id)
            if doc is None:
                raise SourceUnavailable("Unknown document")
            task = db.get(IngestionAttempt, (doc.id, doc.attempt))
            return {"document_id": doc.id, "attempt": doc.attempt,
                    "processing_status": doc.processing_status,
                    "task_status": task.status if task else None,
                    "ready": doc.is_queryable and db.get(DeletionTask, doc.id) is None
                    and (self.coordinator is None or self.coordinator.state.serving),
                    "page_count": doc.page_count, "language": doc.language,
                    "error_code": doc.error_code, "error_pages": json.loads(doc.error_pages_json or "[]"),
                    "diagnostics": json.loads(task.diagnostics_json or "{}") if task else {}}

    def _current(self, db, identity):
        document_id, attempt, epoch = identity
        doc = db.get(Document, document_id)
        task = db.get(IngestionAttempt, (document_id, attempt))
        with self._state_lock:
            cancelled = (document_id, attempt) in self._cancelled
        if (cancelled or doc is None or task is None or doc.attempt != attempt
                or doc.status != "active" or doc.deletion_status != "none"
                or doc.deletion_epoch != epoch or task.deletion_epoch != epoch
                or task.status in {"cancelled", "superseded"}):
            raise IngestionCancelled()
        return doc, task

    def _check(self, identity):
        if self._stop.is_set():
            raise IngestionInterrupted()
        with Session(self.engine) as db:
            self._current(db, identity)

    def source_lock(self, document_id):
        return IngestionWorkerLock(self.worker_lock.path.with_name(self.worker_lock.path.name + f".source-{document_id}"))

    def is_processing(self, document_id):
        with self._state_lock:
            if document_id in self._active_documents:
                return True
        lock = self.source_lock(document_id)
        if not lock.acquire():
            return True
        lock.release()
        return False

    def cancel(self, document_id):
        """Persist cancellation before a late parser/encoder can write its result."""
        with self._transaction() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                return
            task = db.get(IngestionAttempt, (doc.id, doc.attempt))
            if task is None or task.status in {"ready", "failed", "superseded"}:
                return
            with self._state_lock:
                self._cancelled.add((doc.id, doc.attempt))
            task.status = "cancelled"
            if doc.deletion_status == "none":
                doc.processing_status = "failed"
                doc.error_code = "cancelled"
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id,
                                                    DocumentChunk.attempt == doc.attempt))

    def recover(self):
        """Only the OS-lock owner may requeue interrupted work; deletion always wins."""
        with self._transaction() as db:
            for task in db.scalars(select(IngestionAttempt).where(
                    IngestionAttempt.status.in_(["queued", "parsing", "indexing", "staged"]))):
                doc = db.get(Document, task.document_id)
                if (doc is None or doc.attempt != task.attempt or doc.status != "active"
                        or doc.deletion_status != "none" or doc.deletion_epoch != task.deletion_epoch):
                    task.status = "cancelled"
                elif task.status in {"parsing", "indexing"}:
                    task.status = doc.processing_status = "queued"

    def _claim(self):
        while True:
            with self._transaction() as db:
                task = db.scalars(select(IngestionAttempt).where(IngestionAttempt.status == "queued")
                                  .order_by(IngestionAttempt.created_time, IngestionAttempt.document_id)).first()
                if task is None:
                    return None
                identity = task.document_id, task.attempt, task.deletion_epoch
                try:
                    doc, task = self._current(db, identity)
                except IngestionCancelled:
                    task.status = "cancelled"
                    continue
                doc.processing_status = task.status = "parsing"
                return identity

    @staticmethod
    def _validate_vectors(vectors, count, dimension):
        if len(vectors) != count:
            raise LocalEmbeddingUnavailable("embedding_count_mismatch")
        for vector in vectors:
            if (len(vector) != dimension or not vector
                    or any(isinstance(v, (bool, str, bytes)) or not math.isfinite(float(v)) for v in vector)
                    or not any(vector) or not math.isfinite(math.hypot(*vector))):
                raise LocalEmbeddingUnavailable("embedding_invalid_vector")

    def _prepare(self, identity):
        self._check(identity)
        path = self.storage.original_path(identity[0])
        with Session(self.engine) as db:
            doc, _ = self._current(db, identity)
            expected_hash = doc.file_hash
        if file_digest(path) != expected_hash:
            raise PDFQualityError("original_hash_mismatch")
        parsed = self.parser(path, check_cancelled=lambda: self._check(identity))
        self._check(identity)
        if self.embedding is None:
            self.embedding = self.embedding_factory()
            self.embedding.preflight()
        self._check(identity)
        embedding = self.embedding
        chunks = chunk_document(parsed, identity[0], identity[1], embedding,
                                check_cancelled=lambda: self._check(identity))
        if not chunks:
            raise PDFQualityError("no_readable_text")
        profile = embedding.embedding_profile().to_record()
        with self._transaction() as db:
            doc, task = self._current(db, identity)
            doc.page_count, doc.language = parsed.page_count, parsed.language
            doc.processing_status = task.status = "indexing"
            task.expected_chunks = len(chunks)
            task.embedding_profile_json = json.dumps(profile, ensure_ascii=False)
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == identity[0],
                                                    DocumentChunk.attempt == identity[1]))
        for offset in range(0, len(chunks), self.batch_size):
            self._check(identity)
            batch = chunks[offset:offset + self.batch_size]
            vectors = embedding.embed_documents([chunk.embedding_text for chunk in batch])
            self._validate_vectors(vectors, len(batch), embedding.dimension)
            embedding.require_compatible(profile)
            self._check(identity)
            with self._transaction() as db:
                self._current(db, identity)
                for chunk, vector in zip(batch, vectors):
                    db.add(DocumentChunk(
                        document_id=identity[0], attempt=identity[1], chunk_index=chunk.chunk_index,
                        content=chunk.content, page_number=chunk.page_number, content_hash=chunk.content_hash,
                        vector_id=chunk.vector_id, section_title=chunk.section_title,
                        page_chunk_index=chunk.page_chunk_index,
                        normalized_content=embedding.input_policy.prepare(chunk.embedding_text, query=False),
                        context_before=chunk.context_before, source_start=chunk.source_start,
                        source_end=chunk.source_end, embedding_fingerprint=profile["fingerprint"],
                        embedding_json=json.dumps(vector, allow_nan=False),
                    ))
        self._check(identity)
        with self._transaction() as db:
            _, task = self._current(db, identity)
            count = db.scalar(select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == identity[0], DocumentChunk.attempt == identity[1]))
            if count != task.expected_chunks:
                raise RuntimeError("candidate_incomplete")
            task.status = "staged"  # Only the coordinator can publish this prepared candidate.
            task.diagnostics_json = json.dumps(embedding.diagnostics(), ensure_ascii=False)

    def _record_error(self, identity, code, pages=()):
        with self._transaction() as db:
            try:
                doc, task = self._current(db, identity)
            except IngestionCancelled:
                return
            doc.processing_status = task.status = "failed"
            doc.error_code = code
            doc.error_message = task.error_message = code  # No source text or credentials in errors.
            doc.error_pages_json = json.dumps(list(pages))
            db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == identity[0],
                                                    DocumentChunk.attempt == identity[1]))

    def process_next(self):
        """Worker-internal: caller must hold the service's OS lock."""
        if self.coordinator is not None and self._publish_next():
            return True
        try:
            while True:
                self.notifications.get_nowait()
        except Empty:
            pass
        identity = self._claim()
        if identity is None:
            return False
        source_lock = self.source_lock(identity[0])
        if not source_lock.acquire():
            return False
        with self._state_lock:
            self._active_documents.add(identity[0])
        try:
            self._prepare(identity)
        except IngestionInterrupted:
            with self._transaction() as db:
                try:
                    doc, task = self._current(db, identity)
                    doc.processing_status = task.status = "queued"
                except IngestionCancelled:
                    pass
        except IngestionCancelled:
            pass
        except PDFQualityError as exc:
            self._record_error(identity, exc.code, exc.pages)
        except (LocalEmbeddingUnavailable, EmbeddingCompatibilityError):
            self.embedding = None
            self._record_error(identity, "local_embedding_unavailable")
        except ChunkBudgetError:
            self._record_error(identity, "input_budget_exceeded")
        except Exception as exc:
            logger.error("Ingestion failed (%s)", type(exc).__name__)
            self._record_error(identity, "ingestion_failed")
        finally:
            with self._state_lock:
                self._active_documents.discard(identity[0])
            source_lock.release()
        return True

    def _publish_next(self):
        from app.services.index_coordinator import PublicationCommittedButUnavailable

        if self._stop.is_set():
            return False
        with Session(self.engine) as db:
            task = db.scalars(select(IngestionAttempt).where(IngestionAttempt.status == "staged")
                              .order_by(IngestionAttempt.created_time, IngestionAttempt.document_id)).first()
            if task is None:
                return False
            identity = task.document_id, task.attempt, task.deletion_epoch
        try:
            self._check(identity)
            self.coordinator.publish(identity[0], identity[1], check_cancelled=lambda: self._check(identity))
        except IngestionInterrupted:
            return False  # Prepared candidate remains durable for the next worker.
        except IngestionCancelled:
            with self._transaction() as db:
                task = db.get(IngestionAttempt, identity[:2])
                if task is not None and task.status == "staged":
                    task.status = "cancelled"
        except PublicationCommittedButUnavailable:
            raise  # Never rewrite a committed ready source as a failed attempt.
        except Exception as exc:
            logger.error("Index publication failed (%s)", type(exc).__name__)
            self._record_error(identity, "index_publication_failed")
        return True

    def run_until_idle(self):
        """CLI synchronous drain. An active application worker retains ownership."""
        if self._thread is not None and self._thread.is_alive():
            return False
        if not self.worker_lock.acquire():
            return False
        try:
            if self.coordinator is not None:
                from app.services.document_deletion import DocumentDeletionService
                DocumentDeletionService(self).recover()
                self.coordinator.recover()
            self.recover()
            while not self._stop.is_set() and self.process_next():
                pass
            return True
        finally:
            self.worker_lock.release()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self.last_error = None
        self._thread = threading.Thread(target=self._run, name="document-ingestion", daemon=True)
        self._thread.start()
        return True

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self.worker_lock.acquire():
                    self._stop.wait(settings.INGEST_POLL_INTERVAL)
                    continue
                try:
                    if self.coordinator is not None:
                        from app.services.document_deletion import DocumentDeletionService
                        DocumentDeletionService(self).recover()
                        self.coordinator.recover()
                    self.recover()
                    while not self._stop.is_set():
                        if not self.process_next():
                            self._stop.wait(settings.INGEST_POLL_INTERVAL)
                finally:
                    self.worker_lock.release()
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.error("Ingestion worker stopped (%s)", self.last_error)
        finally:
            self.worker_lock.release()
            if self._owns_storage:
                self.engine.dispose()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=settings.INGEST_SHUTDOWN_TIMEOUT)
        # A native encoder cannot be killed safely. Keep the OS lock until it
        # returns; cancellation checks discard its late output and requeue work.
