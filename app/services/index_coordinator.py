"""SQLite-committed index publication, recovery and source-aware resource lifetime."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import threading
import time
import weakref
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.app_logging import get_logger
from app.core.config import settings
from app.models.database_models import DeletionTask, Document, DocumentChunk, IndexCommit, IndexPointer, IngestionAttempt
from app.services.document_storage import DocumentStorage
from app.services.embedding_profile import digest
from app.services.index_resources import IndexIntegrityError, IndexSnapshots, ProfileVectorStore, checked_profile
from app.services.ingestion_worker_lock import IngestionWorkerLock

logger = get_logger(__name__)


class IndexUnavailable(RuntimeError):
    pass


class SourceInvalidated(RuntimeError):
    pass


class PublicationCommittedButUnavailable(IndexUnavailable):
    """SQLite committed; never mark this candidate failed or roll back the pointer."""


@dataclass(frozen=True)
class PublishedIndex:
    id: int
    profile: dict
    pairs: tuple
    snapshot_path: str
    snapshot_sha256: str
    vector_collection: str


class _SharedState:
    def __init__(self):
        self.lock = threading.RLock()
        self.writer = threading.RLock()
        self.readers = {}
        self.views = weakref.WeakSet()
        self.current = None
        self.serving = False
        self.writer_owner = None
        self.retirement_pending = False
        self.retirement_error = None
        self.orphan_pending = False
        self.orphan_error = None

    @property
    def gc_pending(self):
        return self.retirement_pending or self.orphan_pending

    @property
    def gc_error(self):
        return self.orphan_error or self.retirement_error


_states = weakref.WeakValueDictionary()
_states_lock = threading.Lock()


class IndexCoordinator:
    def __init__(self, engine, *, vector_root=None, processed_root=None, vectors=None, fault=lambda stage: None):
        self.engine = engine
        self.vector_root = Path(vector_root or settings.CHROMA_PERSIST_DIRECTORY).resolve()
        self.processed_root = Path(processed_root or settings.PROCESSED_DATA_PATH).resolve()
        self.vectors = vectors or ProfileVectorStore(self.vector_root)
        self.fault = fault
        database = engine.url.database
        self.database_key = str(Path(database).resolve()) if database and database != ":memory:" else str(id(engine))
        self.library_id = digest([self.database_key, str(self.vector_root), str(self.processed_root)])
        self.snapshots = IndexSnapshots(self.processed_root, library_id=self.library_id)
        self.lock_path = (Path(self.database_key + ".publish.lock") if database and database != ":memory:"
                          else self.processed_root / ".publish.lock")
        key = self.database_key, str(self.vector_root), str(self.processed_root)
        with _states_lock:
            self.state = _states.setdefault(key, _SharedState())

    @contextmanager
    def _writer(self):
        with self.state.writer:
            lock = IngestionWorkerLock(self.lock_path)
            deadline = time.monotonic() + 10
            while not lock.acquire():
                if time.monotonic() >= deadline:
                    raise IndexUnavailable("index_publisher_busy")
                time.sleep(.01)
            self.state.writer_owner = threading.get_ident()
            try:
                yield
            finally:
                lock.release()
                self.state.writer_owner = None

    @contextmanager
    def transaction(self):
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            with Session(bind=connection, expire_on_commit=False) as db:
                yield db
                db.flush()
                DocumentStorage._commit(connection)

    def _record(self, db):
        pointer = db.get(IndexPointer, "current")
        if pointer is None:
            return None
        row = db.get(IndexCommit, pointer.commit_id)
        if (row is None or row.retired or row.vector_root != str(self.vector_root)
                or row.processed_root != str(self.processed_root) or row.library_id != self.library_id):
            raise IndexUnavailable("committed_index_roots_missing_or_changed")
        profile = json.loads(row.embedding_profile_json or "null")
        checked_profile(profile)
        if row.embedding_fingerprint != profile["fingerprint"] or not row.snapshot_sha256:
            raise IndexIntegrityError("committed_profile_missing")
        self.snapshots.path(row.snapshot_path)
        return PublishedIndex(row.id, profile, tuple(tuple(p) for p in json.loads(row.document_attempts_json)),
                              row.snapshot_path, row.snapshot_sha256, row.vector_collection)

    @staticmethod
    def _valid(doc, attempt, fingerprint=None):
        return (doc is not None and doc.is_queryable and doc.published_attempt == attempt
                and (fingerprint is None or doc.published_fingerprint == fingerprint))

    def _rows(self, db, pairs, profile):
        rows, epochs = [], {}
        dimension = profile["profile"].get("dimension")
        for document_id, attempt in pairs:
            doc = db.get(Document, document_id)
            task = db.get(IngestionAttempt, (document_id, attempt))
            if doc is None or task is None or not task.expected_chunks:
                raise IndexIntegrityError("source_checkpoint_missing")
            if json.loads(task.embedding_profile_json or "null") != profile:
                raise IndexIntegrityError("source_profile_mismatch_rebuild_required")
            chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document_id,
                                     DocumentChunk.attempt == attempt).order_by(DocumentChunk.chunk_index)).all()
            if len(chunks) != task.expected_chunks or [c.chunk_index for c in chunks] != list(range(len(chunks))):
                raise IndexIntegrityError("source_chunks_incomplete")
            epochs[str(document_id)] = doc.deletion_epoch
            for chunk in chunks:
                content_hash = sha256(chunk.content.encode("utf-8")).hexdigest()
                identity = digest([document_id, attempt, chunk.page_number, chunk.page_chunk_index, content_hash])
                if (chunk.content_hash != content_hash or chunk.vector_id != identity
                        or chunk.embedding_fingerprint != profile["fingerprint"]
                        or chunk.page_number is None or not 1 <= chunk.page_number <= (doc.page_count or 0)):
                    raise IndexIntegrityError("chunk_identity_or_page_invalid")
                vector = json.loads(chunk.embedding_json or "null")
                if (not isinstance(vector, list) or not vector or len(vector) != dimension
                        or any(isinstance(v, (bool, str)) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in vector)
                        or not any(vector) or not math.isfinite(math.hypot(*vector))):
                    raise IndexIntegrityError("candidate_vector_invalid")
                metadata = {"chunk_id": identity, "document_id": document_id, "attempt": attempt,
                            "source_key": f"{document_id}:{attempt}", "deletion_epoch": doc.deletion_epoch,
                            "filename": doc.original_name or doc.filename, "document_title": doc.original_name or doc.filename,
                            "page_number": chunk.page_number, "content_hash": content_hash,
                            "embedding_fingerprint": profile["fingerprint"]}
                for key in ("product_name", "effective_date", "document_type"):
                    if getattr(doc, key):
                        metadata[key] = getattr(doc, key)
                rows.append({"id": identity, "content": chunk.content, "metadata": metadata, "embedding": vector})
        if len({r["id"] for r in rows}) != len(rows):
            raise IndexIntegrityError("duplicate_vector_identity")
        return rows, epochs

    def _check_sources(self, db, pairs, epochs, candidate=None):
        for document_id, attempt in pairs:
            doc = db.get(Document, document_id)
            task = db.get(IngestionAttempt, (document_id, attempt))
            if (doc is None or task is None or doc.status != "active" or doc.deletion_status != "none"
                    or doc.attempt != attempt or doc.deletion_epoch != epochs[str(document_id)]
                    or task.deletion_epoch != doc.deletion_epoch or db.get(DeletionTask, document_id) is not None):
                raise SourceInvalidated("source_changed_during_publication")
            if (document_id, attempt) == candidate:
                if task.status != "staged" or doc.processing_status != "indexing":
                    raise SourceInvalidated("candidate_not_staged")
            elif not self._valid(doc, attempt):
                raise SourceInvalidated("published_source_invalidated")

    def _activate(self, committed):
        self.fault("switch")
        self.state.current = committed.id if committed else None
        self.state.serving = True

    def _commit_version(self, previous, profile, pairs, epochs, relative, checksum, collection, candidate=None):
        with self.state.lock:
            with self.transaction() as db:
                current = self._record(db)
                if (current.id if current else None) != (previous.id if previous else None):
                    raise IndexUnavailable("publication_pointer_changed")
                self._check_sources(db, pairs, epochs, candidate)
                row = IndexCommit(embedding_fingerprint=profile["fingerprint"],
                                  embedding_profile_json=json.dumps(profile, ensure_ascii=False),
                                  document_attempts_json=json.dumps(pairs),
                                  vector_collection=collection, snapshot_path=relative,
                                  snapshot_sha256=checksum, vector_root=str(self.vector_root),
                                  processed_root=str(self.processed_root))
                row.library_id = self.library_id
                db.add(row)
                db.flush()
                pointer = db.get(IndexPointer, "current")
                if pointer is None:
                    db.add(IndexPointer(name="current", commit_id=row.id))
                else:
                    pointer.commit_id = row.id
                for document_id, attempt in pairs:
                    doc = db.get(Document, document_id)
                    doc.index_commit_id = row.id
                    if (document_id, attempt) == candidate:
                        doc.published_attempt = attempt
                        doc.published_fingerprint = profile["fingerprint"]
                        doc.processing_status = "ready"
                        db.get(IngestionAttempt, candidate).status = "ready"
                result = PublishedIndex(row.id, profile, tuple(map(tuple, pairs)), relative, checksum, collection)
                self.fault("before_commit")
            # DB commit wins over process state. An exception here must block new readers.
            try:
                self.fault("after_commit")
                self._activate(result)
            except BaseException as exc:
                self.state.serving = False
                raise PublicationCommittedButUnavailable("recover_committed_pointer_before_serving") from exc
            return result

    def publish(self, document_id, attempt, *, check_cancelled=lambda: None):
        candidate = document_id, attempt
        with self._writer():
            with Session(self.engine) as db:
                previous = self._record(db)
                task = db.get(IngestionAttempt, candidate)
                doc = db.get(Document, document_id)
                if task is None or doc is None:
                    raise IndexIntegrityError("candidate_missing")
                profile = json.loads(task.embedding_profile_json or "null")
                checked_profile(profile)
                if previous and previous.profile != profile:
                    raise IndexIntegrityError("embedding_profile_changed_rebuild_required")
                pairs = list(previous.pairs) if previous else []
                if candidate in pairs:
                    return previous.id  # Restart/repeated publication cannot duplicate a source.
                pairs.append(candidate)
                rows, epochs = self._rows(db, pairs, profile)
                self._check_sources(db, pairs, epochs, candidate)
            old_rows = [r for r in rows if r["metadata"]["document_id"] != document_id]
            relative = None
            committed = False
            collection = self.vectors.collection_name(profile, uuid4().hex)
            try:
                check_cancelled()
                if old_rows:
                    self.vectors.verify(profile, old_rows, collection=previous.vector_collection)
                self.vectors.write(profile, rows, collection=collection, owner=self.library_id)
                self.vectors.verify(profile, rows, collection=collection)
                self.fault("after_vectors")
                snapshot = self.snapshots.build(profile, rows, epochs)
                snapshot['vector_collection'] = collection
                relative, checksum = self.snapshots.write(snapshot)
                self.snapshots.read(relative, checksum)
                self.fault("after_snapshot")
                check_cancelled()
                result = self._commit_version(previous, profile, pairs, epochs, relative, checksum, collection, candidate)
                committed = True
            except PublicationCommittedButUnavailable:
                committed = True
                raise
            finally:
                if not committed:
                    # This unique unpublished collection is never an existing ready version.
                    try:
                        self.vectors.drop(profile, collection)
                    finally:
                        if relative is not None:
                            self.snapshots.path(relative).unlink(missing_ok=True)
            self._cleanup_best_effort()
            return result.id

    def _replay_deletions(self):
        with self.transaction() as db:
            for task in db.scalars(select(DeletionTask)):
                doc = db.get(Document, task.document_id)
                if doc is not None:
                    if doc.deletion_status == "none":
                        doc.deletion_status = "deleting"
                        if task.status == "completed":
                            # A restored older row must replay the retained tombstone,
                            # including cleanup, rather than resurrect a finished deletion.
                            task.status = "pending"
                            task.stage = "invalidate"
                            task.completed_stages_json = "[]"
                    doc.deletion_epoch = max(doc.deletion_epoch, task.deletion_epoch)
            invalid = set(db.scalars(select(Document.id).where(
                (Document.deletion_status != "none") | (Document.status != "active"))))
            for task in db.scalars(select(IngestionAttempt).where(IngestionAttempt.document_id.in_(invalid))):
                if task.status != "superseded":
                    task.status = "cancelled"
        with self.state.lock:
            for view in list(self.state.views):
                if invalid.intersection(view.document_ids):
                    view.cancel("source_deleted")
        return invalid

    def recover(self):
        """Only SQLite's committed pointer is authoritative; never scan/import PDFs or manifests."""
        with self._writer():
            try:
                invalid = self._replay_deletions()
                with Session(self.engine) as db:
                    current = self._record(db)
                    if current:
                        pairs = [p for p in current.pairs if self._valid(db.get(Document, p[0]), p[1], current.profile["fingerprint"])]
                        rows, epochs = self._rows(db, pairs, current.profile)
                    else:
                        pairs, rows, epochs = [], [], {}
                if current:
                    if rows:
                        self.vectors.verify(current.profile, rows, collection=current.vector_collection)
                    if tuple(pairs) != current.pairs:
                        collection = self.vectors.collection_name(current.profile, uuid4().hex)
                        self.vectors.write(current.profile, rows, collection=collection, owner=self.library_id)
                        self.vectors.verify(current.profile, rows, collection=collection)
                        snapshot = self.snapshots.build(current.profile, rows, epochs)
                        snapshot['vector_collection'] = collection
                        relative, checksum = self.snapshots.write(snapshot)
                        current = self._commit_version(current, current.profile, pairs, epochs, relative, checksum, collection)
                    else:
                        try:
                            snapshot = self.snapshots.read(current.snapshot_path, current.snapshot_sha256)
                            if snapshot["document_attempts"] != list(map(list, current.pairs)):
                                raise IndexIntegrityError("snapshot_sources_mismatch")
                        except (OSError, ValueError, KeyError, IndexIntegrityError):
                            # Rebuild keywords from valid committed SQLite chunks; no model/API needed.
                            snapshot = self.snapshots.build(current.profile, rows, epochs)
                            snapshot['vector_collection'] = current.vector_collection
                            _, checksum = self.snapshots.write(snapshot, current.snapshot_path)
                            with self.transaction() as db:
                                if self._record(db).id != current.id:
                                    raise IndexUnavailable("recovery_pointer_changed")
                                db.get(IndexCommit, current.id).snapshot_sha256 = checksum
                            current = PublishedIndex(current.id, current.profile, current.pairs, current.snapshot_path, checksum, current.vector_collection)
                with self.state.lock:
                    self._activate(current)
                self._cleanup_best_effort(sweep_orphans=True)
                return current.id if current else None
            except BaseException:
                self.state.serving = False
                raise

    def _collect_locked(self):
        pending = False
        error = None
        with self.state.lock:
            with self.transaction() as db:
                pointer = db.get(IndexPointer, "current")
                current_id = pointer.commit_id if pointer else None
                for row in db.scalars(select(IndexCommit).where(IndexCommit.vector_root == str(self.vector_root),
                                                        IndexCommit.processed_root == str(self.processed_root),
                                                        IndexCommit.library_id == self.library_id)):
                    if row.id == current_id:
                        continue
                    if self.state.readers.get(row.id, 0) or self._has_process_readers(row.id):
                        pending = True
                        continue
                    try:
                        self.vectors.drop(json.loads(row.embedding_profile_json), row.vector_collection)
                        self.snapshots.path(row.snapshot_path).unlink(missing_ok=True)
                        row.retired = True
                    except PermissionError:
                        pending, error = True, 'PermissionError'
        with self.state.lock:
            self.state.retirement_pending, self.state.retirement_error = pending, error

    def _cleanup_best_effort(self, *, sweep_orphans=False):
        try:
            self._collect_locked()
        except Exception as exc:
            # Retirement cannot undo a committed publication or a valid query.
            # Deletion callers still see incomplete cleanup and must retry.
            with self.state.lock:
                self.state.retirement_pending = True
                self.state.retirement_error = type(exc).__name__
            logger.warning('Index retirement pending (%s)', type(exc).__name__)
        if sweep_orphans:
            try:
                self._sweep_orphans_locked()
            except Exception as exc:
                with self.state.lock:
                    self.state.orphan_pending = True
                    self.state.orphan_error = type(exc).__name__
                logger.warning('Orphan index cleanup pending (%s)', type(exc).__name__)
            else:
                # Only a real successful sweep can clear an earlier orphan error.
                # Ordinary request-time retirement cannot prove orphans were removed.
                with self.state.lock:
                    self.state.orphan_pending = False
                    self.state.orphan_error = None

    def _has_process_readers(self, commit_id):
        active = False
        for path in self.snapshots.root.glob(f"r-{self.library_id[:16]}-{commit_id}-*.lock"):
            lock = IngestionWorkerLock(path)
            if not lock.acquire():
                active = True
            else:
                lock.release()
                path.unlink(missing_ok=True)
        return active

    def reader_lease(self, commit_id):
        lock = IngestionWorkerLock(self.snapshots.root / f"r-{self.library_id[:16]}-{commit_id}-{uuid4().hex}.lock")
        if not lock.acquire():
            raise IndexUnavailable("reader_lease_unavailable")
        return lock

    def _sweep_orphans_locked(self):
        with Session(self.engine) as db:
            known = {r.snapshot_path for r in db.scalars(select(IndexCommit).where(
                IndexCommit.vector_root == str(self.vector_root), IndexCommit.processed_root == str(self.processed_root),
                IndexCommit.library_id == self.library_id))}
            known_collections = {r.vector_collection for r in db.scalars(select(IndexCommit).where(
                IndexCommit.library_id == self.library_id))}
        for name, profile in self.vectors.managed_versions(self.library_id):
            if name not in known_collections:
                self.vectors.drop(profile, name)
        for path in self.snapshots.root.glob("*/v-*.json"):
            relative = path.relative_to(self.snapshots.root).as_posix()
            if relative not in known:
                try:
                    owner = json.loads(path.read_text(encoding='utf-8')).get('library_id')
                except (ValueError, OSError):
                    continue  # Unidentified legacy/corrupt files are never imported.
                if owner == self.library_id:
                    self.snapshots.path(relative).unlink(missing_ok=True)
        for path in self.snapshots.root.glob(f"*/{self.snapshots.temp_prefix}*"):
            if path.resolve().is_relative_to(self.snapshots.root):
                path.unlink(missing_ok=True)

    def collect_garbage(self):
        # Current-pointer checks and retirement share one short SQLite transaction.
        # No publisher lock is needed: unpublished unique files are not considered.
        if self.state.writer_owner != threading.get_ident():
            self._cleanup_best_effort()

    def ready_ids(self):
        with Session(self.engine) as db:
            current = self._record(db)
            if current is None or not self.state.serving:
                return []
            return [p[0] for p in current.pairs if self._valid(db.get(Document, p[0]), p[1], current.profile["fingerprint"])
                    and db.get(DeletionTask, p[0]) is None]

    def open_view(self, document_ids, profile):
        from app.services.index_request_view import IndexRequestView
        return IndexRequestView(self, document_ids, profile)

    def open_current_view(self, document_ids):
        """Freeze committed identity before loading/encoding the question model."""
        with self.state.lock, Session(self.engine) as db:
            current = self._record(db)
            if current is None:
                raise IndexUnavailable("no_committed_index")
            return self.open_view(document_ids, current.profile)

    def invalidate_source(self, document_id):
        """Deletion hook: block/cancel first, rebuild, then report index cleanup completeness.

        The caller must already have persisted deletion. This never marks the
        overall deletion complete: PDF/history/SQLite cleanup belongs to task 7.
        """
        with Session(self.engine) as db:
            doc = db.get(Document, document_id)
            if doc is not None and doc.deletion_status == 'none' and db.get(DeletionTask, document_id) is None:
                raise SourceInvalidated('persist_deletion_before_index_invalidation')
        with self.state.lock:
            for view in list(self.state.views):
                if document_id in view.document_ids:
                    view.cancel('source_deleted')
        self.recover()
        with Session(self.engine) as db:
            return self.state.gc_error is None and not any(document_id in {p[0] for p in json.loads(row.document_attempts_json)}
                           for row in db.scalars(select(IndexCommit).where(
                               IndexCommit.library_id == self.library_id, IndexCommit.retired.is_(False))))


def default_coordinator():
    storage = DocumentStorage()
    return IndexCoordinator(storage.engine)
