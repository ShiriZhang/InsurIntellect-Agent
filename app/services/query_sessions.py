"""SQLite-serialized scope epochs and intent-only previous user questions."""
from dataclasses import dataclass, field
import threading
import json
import re
from uuid import uuid4
from sqlalchemy import select
from app.models.database_models import SessionScope, QueryHistory, Document, DeletionTask
from app.services.embedding_profile import digest

@dataclass(frozen=True)
class PreparedSession:
    session_id: str
    epoch: int
    fingerprint: str
    previous_mode: str
    previous_question: str
    cancelled: threading.Event = field(default_factory=threading.Event)


def prepare_session(coordinator, request):
    session_id = request.session_id or uuid4().hex
    fingerprint = digest(request.document_ids)
    with coordinator.transaction() as db:
        requested_epoch = getattr(request, "scope_epoch", None)
        if requested_epoch is not None:
            current = db.get(SessionScope, (session_id, requested_epoch))
            if current is None or current.invalidated or current.scope_fingerprint != fingerprint:
                from fastapi import HTTPException
                raise HTTPException(422, "The submitted scope epoch is invalid or no longer available")
        else:
            current = db.scalars(select(SessionScope).where(SessionScope.session_id == session_id)
                                 .order_by(SessionScope.epoch.desc())).first()
            if current is None or current.invalidated or current.scope_fingerprint != fingerprint:
                current = SessionScope(session_id=session_id, epoch=1 if current is None else current.epoch + 1,
                                       scope_fingerprint=fingerprint, document_ids_json=json.dumps(request.document_ids))
                db.add(current)
        previous = db.scalars(select(QueryHistory).where(
            QueryHistory.session_id == session_id, QueryHistory.scope_epoch == current.epoch,
            QueryHistory.scope_status == "known", QueryHistory.source_status == "active")
            .order_by(QueryHistory.id.desc()).limit(1)).first()
        return PreparedSession(session_id, current.epoch, fingerprint, previous.query_mode if previous else "qa", (previous.rewritten_query or previous.query) if previous else "")


def resolve_followup(question, session, ids):
    followup = re.search(r"^(那|那么|它|这个|这项)|呢[？?]?$|^(what about|how about|and |does it|is it|it\b)", question.strip(), re.I)
    if not followup:
        return question, False
    bare = re.fullmatch(r"(它|这个|这项)(呢|怎么样)?[？?]?|(?:what about )?it[?]?", question.strip(), re.I)
    multiple_topics = re.search(r"和|以及|、|\band\b|\bor\b", session.previous_question, re.I)
    if bare and (len(ids) > 1 or multiple_topics):
        return question, True
    # Bare pronouns or several possible policies require a named referent.
    if not session.previous_question or (len(ids) > 1 and re.search(r"它|\bit\b", question, re.I)):
        return question, True
    return session.previous_question[-1000:] + "\n" + question, False


def save_terminal(coordinator, view, request, prepared, response, intent_question=None):
    with coordinator.state.lock, coordinator.transaction() as db:
        invalid = any((doc := db.get(Document, i)) is None or not doc.is_queryable
                      or db.get(DeletionTask, i) is not None for i in request.document_ids)
        if view and not invalid:
            try:
                view._validate(db)
            except Exception:
                invalid = True
        scope = db.get(SessionScope, (prepared.session_id, prepared.epoch))
        if invalid or prepared.cancelled.is_set():
            if scope and invalid:
                scope.invalidated = True
            from app.services.language_service import terminal
            response.answer_kind = "cancelled"
            response.answer = terminal("cancelled", response.answer_language)
            response.source_status = "deleted" if invalid else "active"
            response.confidence_score = 0
            intent_question = None
            response.retrieved_chunks = []
            response.chunks_used = 0
        row = QueryHistory(query=request.question, rewritten_query=intent_question, response=response.answer, model_used=request.query_type,
            response_time=response.response_time, session_id=prepared.session_id,
            scope_status="known", document_ids_json=json.dumps(request.document_ids),
            scope_fingerprint=prepared.fingerprint, scope_epoch=prepared.epoch,
            answer_language=response.answer_language, query_mode=response.query_mode,
            index_commit_id=response.index_commit_id, source_status=response.source_status,
            rewriting_metadata_json=json.dumps({"answer_kind": response.answer_kind,
                "document_coverage": response.document_coverage}),
            retrieved_chunks=json.dumps([c.model_dump() for c in response.retrieved_chunks], ensure_ascii=False))
        db.add(row)
        db.flush()
        response.query_id = row.id


def history_fields(row):
    meta = json.loads(row.rewriting_metadata_json or "{}")
    return {"document_ids": json.loads(row.document_ids_json) if row.document_ids_json else None,
            "session_id": row.session_id, "scope_epoch": row.scope_epoch,
            "scope_fingerprint": row.scope_fingerprint, "answer_language": row.answer_language,
            "query_mode": row.query_mode, "index_commit_id": row.index_commit_id,
            "source_status": row.source_status, "answer_kind": meta.get("answer_kind")}
