"""One scoped evidence preparation for ordinary QA and per-document comparison."""

import asyncio
import json
from time import monotonic
import threading

from fastapi import HTTPException

from app.core.config import settings
from app.models.database_models import QueryHistory
from app.models.schemas import QueryResponse
from app.services import citation_policy, response_shaping, refusal_policy
from app.services.embedding_service import EmbeddingService, LocalEmbeddingUnavailable
from app.services.embedding_profile import EmbeddingCompatibilityError
from app.services.index_coordinator import IndexUnavailable, SourceInvalidated
from app.services.index_resources import IndexIntegrityError
from app.services.llm_service import LLMService
from app.services.query_mode import resolve_mode
from app.services.query_scope import validate_scope


class ScopedQueryProcessor:
    def __init__(self, embedding_factory=EmbeddingService, llm_factory=LLMService):
        self.embedding_factory = embedding_factory
        self.llm_factory = llm_factory
        self.embedding = None
        self._model_lock = threading.Lock()

    def _encode(self, question, profile):
        with self._model_lock:
            if self.embedding is None:
                self.embedding = self.embedding_factory()
            vector = self.embedding.embed_query(question)
            self.embedding.require_compatible(profile)
            return vector

    async def process(self, db, request, *, coordinator, stream=False):
        started = monotonic()
        names = await validate_scope(db, request)
        mode = resolve_mode(request)
        view = None
        chunks, coverage = [], []
        answer, kind = "", "answer"
        if mode == "clarification":
            answer, kind = "请明确是普通条款解释，还是逐份文档比较。", "clarification"
        else:
            try:
                view = await asyncio.to_thread(coordinator.open_current_view, request.document_ids)
                if refusal_policy.is_advice_or_guarantee_question(request.question):
                    answer, kind = refusal_policy.refusal_answer("advice"), "advice"
                elif (any(marker in request.question for marker in refusal_policy.CHITCHAT_OR_OFFTOPIC_MARKERS)
                      and not any(word in request.question for word in refusal_policy.INSURANCE_KEYWORDS)):
                    # Preserve the existing explicit off-topic boundary without
                    # carrying forward the old short-English false rejection.
                    answer, kind = refusal_policy.refusal_answer(), "refusal"
                else:
                    vector = await asyncio.to_thread(self._encode, request.question, view.commit.profile)
                    raw = await asyncio.to_thread(view.search, request.question, vector,
                                                  top_k=settings.MAX_RETRIEVED_CHUNKS,
                                                  per_document=mode == "compare")
                    eligible = citation_policy.eligible_evidence(raw, settings.SIMILARITY_THRESHOLD)
                    if mode == "compare":
                        for doc_id in request.document_ids:
                            selected = citation_policy.curate_citations(
                                [c for c in eligible if c["document_id"] == doc_id],
                                request.question, limit=settings.MAX_RETRIEVED_CHUNKS)
                            chunks.extend(selected)
                            coverage.append({"document_id": doc_id, "document_name": names[doc_id],
                                             "status": "evidence" if selected else "insufficient_evidence"})
                    else:
                        chunks = citation_policy.curate_citations(
                            eligible, request.question, limit=settings.MAX_RETRIEVED_CHUNKS)
                    view.assert_valid()
                    if not chunks:
                        answer, kind = refusal_policy.refusal_answer(), "refusal"
            except SourceInvalidated:
                if view:
                    view.close()
                raise HTTPException(422, "Document scope changed or is no longer ready") from None
            except (LocalEmbeddingUnavailable, EmbeddingCompatibilityError, IndexUnavailable, IndexIntegrityError):
                if view:
                    view.close()
                raise HTTPException(503, "Local model/index unavailable; check profile and rebuild readiness") from None
            except BaseException:
                if view:
                    view.close()
                raise

        # Finalize evidence once. Generation, API, SSE and history use this order.
        for index, chunk in enumerate(chunks, 1):
            chunk["metadata"] = {**chunk.get("metadata", {}), "citation_number": index}
        generation_question = request.question
        if mode == "compare":
            generation_question += (
                "\n逐份比较以下文档；分别说明事实和条件，不合并成同一合同。"
                "每份必须出现；insufficient_evidence 表示未找到充分依据，不得编造。\n"
                + json.dumps(coverage, ensure_ascii=False))
        llm = None
        if not answer:
            llm = self.llm_factory()
            # UTF-8 bytes bound tokens conservatively without a remote tokenizer.
            # Reserve room per document; never drop a document to fit global top-k.
            available = (settings.GENERATION_CONTEXT_TOKENS - settings.GENERATION_PROMPT_RESERVE_TOKENS
                         - settings.OPENAI_MAX_TOKENS - len(generation_question.encode("utf-8")))
            groups = [[c for c in chunks if c["document_id"] == i] for i in request.document_ids] if mode == "compare" else [chunks]
            per_document_budget = available // len(groups)
            if any(sum(len(c["content"].encode("utf-8")) + len(c["document_name"].encode("utf-8")) + 128
                       for c in group) > per_document_budget for group in groups):
                answer, kind = "证据超过安全上下文预算，请缩小问题或文档范围后重试。", "clarification"
                chunks = []
            elif not llm.has_credentials():
                answer, kind = "生成 API 不可用；以下仅为检索原文，尚未完成解释。", "llm_unavailable"

        def result(text, terminal_kind):
            public_chunks = chunks if terminal_kind in {"answer", "llm_unavailable"} else []
            return QueryResponse(
                question=request.question, answer=text, query_type=request.query_type,
                document_ids=request.document_ids, query_mode=mode, document_coverage=coverage,
                retrieved_chunks=response_shaping.to_retrieved_chunks(public_chunks), chunks_used=len(public_chunks),
                confidence_score=citation_policy.best_similarity(chunks), answer_kind=terminal_kind,
                response_time=monotonic() - started,
                embedding_provider=getattr(self.embedding, "provider", None))

        def check():
            if view:
                view.assert_valid()

        def save(response):
            if not view:
                return
            def write(session):
                row = QueryHistory(query=request.question, response=response.answer,
                                   model_used=request.query_type, response_time=response.response_time,
                                   session_id=request.session_id, document_ids_json=json.dumps(request.document_ids),
                                   query_mode=mode, index_commit_id=view.commit_id,
                                   retrieved_chunks=json.dumps([c.model_dump() for c in response.retrieved_chunks], ensure_ascii=False))
                session.add(row)
                session.flush()
                return row.id
            response.query_id = view.guard_write(write)

        def missing_evidence_suffix():
            missing = [c for c in coverage if c['status'] == 'insufficient_evidence']
            return "".join(f"\n文档 {c['document_id']} ({c['document_name']})：未找到充分依据。" for c in missing)

        if not stream:
            try:
                check()
                if not answer:
                    generated = await llm.agenerate_response(generation_question, chunks)
                    answer = generated.get("answer", "")
                    kind = "answer" if generated.get("success", False) else "degraded"
                check()
                response = result(answer + missing_evidence_suffix(), kind)
                await asyncio.to_thread(save, response)
                return response
            finally:
                if view:
                    view.close()

        async def events():
            def event(name, payload):
                return f"event: {name}\ndata: {json.dumps({'type': name, **payload}, ensure_ascii=False)}\n\n"
            try:
                check()
                yield event("start", {"question": request.question, "document_ids": request.document_ids, "query_mode": mode})
                text = answer
                if not answer:
                    yield event("context", {"retrieved_chunks": [c.model_dump() for c in response_shaping.to_retrieved_chunks(chunks)],
                                            "document_coverage": coverage, "query_mode": mode})
                    upstream = llm.agenerate_stream(generation_question, chunks)
                    try:
                        async for token in upstream:
                            check()
                            text += token
                            yield event("token", {"content": token})
                    finally:
                        await upstream.aclose()
                check()
                response = result(text + missing_evidence_suffix(), kind)
                await asyncio.to_thread(save, response)
                yield event("end", {**response.model_dump(), "success": True})
            except Exception as exc:
                yield event("error", {"message": "回答中止，来源或服务不可用。", "error": type(exc).__name__})
            finally:
                if view:
                    view.close()
        return events()
