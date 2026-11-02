"""Scoped evidence views: short snapshot readers, cancellable small evidence sets."""

import math

from sqlalchemy.orm import Session

from app.core.fusion import reciprocal_rank_fusion
from app.services.evidence_scores import cosine_distance, cosine_distance_to_score
from app.services.index_coordinator import IndexUnavailable, SourceInvalidated
from app.services.index_resources import checked_profile
from app.services.retrieval_text import rank_lexical_candidates


class IndexRequestView:
    def __init__(self, coordinator, document_ids, profile):
        self.coordinator = coordinator
        self.document_ids = tuple(sorted(set(document_ids)))
        if not self.document_ids or any(type(i) is not int or i <= 0 for i in self.document_ids):
            raise SourceInvalidated("explicit_nonempty_document_ids_required")
        self._evidence = []
        self.cancel_reason = None
        self._handle = None
        self._process_lease = None
        self._preparing = False
        self._released = False
        self._closed = False
        checked_profile(profile)
        with coordinator.state.lock:
            if not coordinator.state.serving:
                raise IndexUnavailable("recover_committed_pointer_before_serving")
            # An external publisher may retire a file between the pointer read and
            # opening it. Retry before accepting/fixing this request's version.
            for _ in range(3):
                with coordinator.transaction() as db:
                    current = coordinator._record(db)
                    if current is None:
                        raise IndexUnavailable("no_committed_index")
                    checked_profile(profile).require_compatible(current.profile)
                    self.commit = current
                    self.pairs, self.epochs = [], {}
                    for document_id in self.document_ids:
                        from app.models.database_models import DeletionTask, Document
                        doc = db.get(Document, document_id)
                        if (not coordinator._valid(doc, doc.published_attempt if doc else None, profile["fingerprint"])
                                or (document_id, doc.published_attempt) not in current.pairs
                                or db.get(DeletionTask, document_id) is not None):
                            raise SourceInvalidated("document_not_ready_in_committed_index")
                        self.pairs.append((document_id, doc.published_attempt))
                        self.epochs[document_id] = doc.deletion_epoch
                    try:
                        self._handle = coordinator.snapshots.path(current.snapshot_path).open("rb")
                        self._process_lease = coordinator.reader_lease(current.id)
                        break
                    except FileNotFoundError:
                        continue
            if self._handle is None:
                raise IndexUnavailable("committed_snapshot_missing_recover_required")
            self.pairs = tuple(self.pairs)
            self.commit_id = current.id
            coordinator.state.readers[current.id] = coordinator.state.readers.get(current.id, 0) + 1
            coordinator.state.views.add(self)

    def _validate(self, db):
        from app.models.database_models import DeletionTask, Document

        if self.cancel_reason or self._closed:
            raise SourceInvalidated(self.cancel_reason or "request_view_closed")
        for document_id, attempt in self.pairs:
            doc = db.get(Document, document_id)
            if (not self.coordinator._valid(doc, attempt, self.commit.profile["fingerprint"])
                    or doc.deletion_epoch != self.epochs[document_id] or db.get(DeletionTask, document_id) is not None):
                self.cancel("source_invalidated")
                raise SourceInvalidated("source_invalidated")

    def assert_valid(self):
        with self.coordinator.state.lock, Session(self.coordinator.engine) as db:
            self._validate(db)

    @property
    def evidence(self):
        self.assert_valid()
        return self._evidence

    def guard_write(self, write):
        """Run a synchronous final save within the same SQLite invalidation boundary."""
        with self.coordinator.state.lock, self.coordinator.transaction() as db:
            self._validate(db)
            return write(db)

    def _release_reader(self):
        with self.coordinator.state.lock:
            if self._released:
                return
            self._released = True
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            if self._process_lease is not None:
                lease, self._process_lease = self._process_lease, None
                lease.release()
                try:
                    lease.path.unlink(missing_ok=True)
                except PermissionError:
                    pass  # A collector owns the now-unlocked marker and will remove it.
            key = self.commit_id
            count = self.coordinator.state.readers.get(key, 0) - 1
            if count > 0:
                self.coordinator.state.readers[key] = count
            else:
                self.coordinator.state.readers.pop(key, None)

    def cancel(self, reason="cancelled"):
        with self.coordinator.state.lock:
            self.cancel_reason = reason
            for chunk in self._evidence:
                chunk.clear()  # Scrub aliases held by downstream evidence preparation.
            self._evidence.clear()
            if not self._preparing:
                self._release_reader()

    def search(self, question, vector, *, top_k=8, per_document=False):
        with self.coordinator.state.lock:
            self.assert_valid()
            if self._released:
                raise IndexUnavailable("request_view_already_prepared")
        dimension = self.commit.profile["profile"].get("dimension")
        if (len(vector) != dimension or not all(math.isfinite(v) for v in vector)
                or not any(vector) or not math.isfinite(math.hypot(*vector))):
            raise IndexUnavailable("query_vector_invalid")
        with self.coordinator.state.lock:
            self.assert_valid()
            self._preparing = True
        snapshot = None
        try:
            snapshot = self.coordinator.snapshots.decode(self._handle.read(), self.commit.snapshot_sha256)
            if snapshot.get('vector_collection') != self.commit.vector_collection:
                raise IndexUnavailable('snapshot_vector_version_mismatch')
            allowed = {f"{doc}:{attempt}" for doc, attempt in self.pairs}
            ids = [key for key in snapshot["ids"] if snapshot["chunks"][key]["metadata"]["source_key"] in allowed]
            mapping = {key: snapshot["chunks"][key] for key in ids}
            positions = {key: i for i, key in enumerate(snapshot["ids"])}
            tokens = [snapshot["tokens"][positions[key]] for key in ids]
            snapshot.clear()  # Drop all out-of-scope source material before native retrieval.
            self.assert_valid()
            groups = [{f"{doc}:{attempt}"} for doc, attempt in self.pairs] if per_document else [allowed]
            selected, distances, rankings, lexical_scores = [], {}, {}, {}
            for source_keys in groups:
                scoped_ids = [key for key in ids if mapping[key]["metadata"]["source_key"] in source_keys]
                scoped_set = set(scoped_ids)
                scoped_tokens = [tokens[i] for i, key in enumerate(ids) if key in scoped_set]
                chosen, semantic, fused_scores, lexical = self._rank(
                    question, vector, source_keys, scoped_ids, scoped_tokens, top_k)
                selected.extend(chosen)
                distances.update(semantic)
                rankings.update(fused_scores)
                lexical_scores.update(lexical)
            stored = self.coordinator.vectors.get(self.commit.profile, selected, collection=self.commit.vector_collection)
            result = []
            for key in selected:
                source = mapping[key]
                row = stored.get(key)
                if row is None or row["content"] != source["content"] or row["metadata"] != source["metadata"]:
                    raise IndexUnavailable("committed_evidence_missing_or_changed")
                distance = distances.get(key)
                if distance is None:
                    distance = cosine_distance(vector, row["embedding"])
                md = source["metadata"]
                score = cosine_distance_to_score(distance)
                result.append({"chunk_id": key, "document_id": md["document_id"],
                               "document_name": md["filename"], "page_number": md["page_number"],
                               "content": source["content"], "similarity_score": score,
                               "metadata": {**md, "index_commit_id": self.commit_id,
                                            "raw_distance": distance, "evidence_score": score,
                                            "score_metric": "cosine", "bm25_score": lexical_scores.get(key),
                                            "rrf_score": rankings.get(key)}})
            with self.coordinator.state.lock:
                self.assert_valid()
                self._evidence[:] = result
            return self._evidence
        finally:
            if snapshot is not None:
                snapshot.clear()
            self._preparing = False
            self._release_reader()  # Generation retains only the selected evidence above.
            self.coordinator.collect_garbage()

    def _rank(self, question, vector, source_keys, ids, tokens, top_k):
        from rank_bm25 import BM25Plus

        lexical_ids, lexical_scores = [], {}
        vector_ids, distances = [], {}
        lexical_failed = vector_failed = False
        try:
            if tokens and any(tokens):
                lexical_ids, lexical_scores = rank_lexical_candidates(BM25Plus(tokens), ids, question, limit=50)
        except Exception:
            lexical_failed = True
        try:
            vector_ids, distances = self.coordinator.vectors.query(
                self.commit.profile, vector, source_keys, 50, collection=self.commit.vector_collection)
            if not set(vector_ids).issubset(ids):
                raise IndexUnavailable("vector_scope_filter_failed")
        except Exception:
            vector_ids, distances = [], {}
            vector_failed = True
        if lexical_failed and vector_failed:
            raise IndexUnavailable("all_scoped_retrieval_paths_failed")
        fused, rankings = reciprocal_rank_fusion([vector_ids, lexical_ids], top_n=None)
        return fused[:top_k], distances, rankings, lexical_scores

    def close(self):
        self.cancel("request_view_closed")
        self._closed = True
        with self.coordinator.state.lock:
            self.coordinator.state.views.discard(self)
        self.coordinator.collect_garbage()

    def __enter__(self):
        self.assert_valid()
        return self

    def __exit__(self, *args):
        self.close()
        self.coordinator.collect_garbage()

    def __del__(self):
        # Also close an accepted but never-prepared request's file handle.
        if getattr(self, "_handle", None) is not None:
            if hasattr(self, "commit_id"):
                self._release_reader()
            else:
                self._handle.close()
