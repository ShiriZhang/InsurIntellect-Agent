"""One scoped evidence preparation for ordinary QA and per-document comparison."""

import asyncio
import json
from time import monotonic
import threading


from app.core.config import settings
from app.models.schemas import QueryResponse
from app.services import citation_policy, response_shaping, refusal_policy
from app.services.embedding_service import EmbeddingService, LocalEmbeddingUnavailable
from app.services.embedding_profile import EmbeddingCompatibilityError
from app.services.index_coordinator import IndexUnavailable, SourceInvalidated
from app.services.index_resources import IndexIntegrityError
from app.services.llm_service import LLMService
from app.services.query_mode import resolve_mode
from app.services.query_scope import validate_scope
from app.services.language_service import resolve_language, terminal
from app.services.query_sessions import prepare_session, resolve_followup, save_terminal
from app.services.llm_service import GenerationAPIError


async def record_cancellation(coordinator, view, request, prepared, language, mode, intent):
    prepared.cancelled.set()
    response = QueryResponse(question=request.question, answer=terminal("cancelled", language),
        query_type=request.query_type, answer_kind="cancelled", answer_language=language,
        session_id=prepared.session_id, document_ids=request.document_ids, query_mode=mode,
        scope_epoch=prepared.epoch, scope_fingerprint=prepared.fingerprint,
        index_commit_id=view.commit_id if view else None)
    await asyncio.shield(asyncio.to_thread(save_terminal, coordinator, view, request, prepared, response, intent))


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
        from app.services.source_requests import SourceRequest
        await validate_scope(db, request)
        try:
            lease = SourceRequest(coordinator, request.document_ids)
        except SourceInvalidated:
            from fastapi import HTTPException
            raise HTTPException(422, "Source is no longer queryable") from None
        try:
            response = await self._process(db, request, coordinator=coordinator, stream=stream, lease=lease)
        except BaseException:
            lease.close()
            raise
        if not stream:
            lease.close()
            return response
        async def leased_events():
            try:
                async for event in response:
                    yield event
            finally:
                await response.aclose()
                lease.close()
        return leased_events()

    async def _process(self, db, request, *, coordinator, stream=False, lease):
        started = monotonic()
        names = await validate_scope(db, request)
        language = resolve_language(request.question, request.answer_language)
        prepared = await asyncio.to_thread(prepare_session, coordinator, request)
        effective_question, ambiguous = resolve_followup(request.question, prepared, request.document_ids)
        mode = resolve_mode(request.model_copy(update={"question": effective_question}))
        if effective_question != request.question and request.query_mode == "auto" and prepared.previous_mode == "compare":
            mode = "compare"
            if len(request.document_ids) > 5:
                from fastapi import HTTPException
                raise HTTPException(422, "At most five documents can be compared")
        view = None
        chunks, coverage = [], []
        answer, kind = "", "answer"
        if (mode == "clarification" or ambiguous) and not (refusal_policy.is_advice_or_guarantee_question(request.question) or refusal_policy.is_off_topic(request.question, [])):
            answer, kind = terminal("clarification", language), "clarification"
        else:
            try:
                view = await asyncio.to_thread(coordinator.open_current_view, request.document_ids)
                if refusal_policy.is_advice_or_guarantee_question(request.question):
                    answer, kind = terminal("advice", language), "advice"
                elif refusal_policy.is_off_topic(request.question, []):
                    answer, kind = terminal("refusal", language), "refusal"
                else:
                    vector = await lease.wait(asyncio.wait_for(asyncio.to_thread(self._encode, effective_question, view.commit.profile), settings.CONTEXT_BUILD_TIMEOUT_SECS))
                    raw = await lease.wait(asyncio.to_thread(view.search, effective_question, vector,
                                                  top_k=settings.MAX_RETRIEVED_CHUNKS,
                                                  per_document=mode == "compare"))
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
                        answer, kind = terminal("refusal", language), "refusal"
            except TimeoutError:
                answer, kind = terminal("timeout", language), "timeout"
            except SourceInvalidated:
                answer, kind = terminal("cancelled", language), "cancelled"
            except (LocalEmbeddingUnavailable, EmbeddingCompatibilityError, IndexUnavailable, IndexIntegrityError):
                answer, kind = terminal("local_unavailable", language), "local_unavailable"
            except asyncio.CancelledError:
                await record_cancellation(coordinator, view, request, prepared, language, mode, effective_question)
                if view:
                    view.close()
                raise
            except BaseException:
                if view:
                    view.close()
                raise

        if chunks:
            try:
                lease.retain(chunks)
            except SourceInvalidated:
                chunks = []
                answer, kind = terminal("cancelled", language), "cancelled"

        # Finalize evidence once. Generation, API, SSE and history use this order.
        for index, chunk in enumerate(chunks, 1):
            chunk["metadata"] = {**chunk.get("metadata", {}), "citation_number": index}
        generation_question = effective_question
        if mode == "compare":
            generation_question += (
                "\n逐份比较以下文档；分别说明事实和条件，不合并成同一合同。"
                "每份必须出现；insufficient_evidence 表示未找到充分依据，不得编造。\n"
                + json.dumps(coverage, ensure_ascii=False))
        llm = None
        if not answer:
            llm = self.llm_factory()
            llm.answer_language = language
            # UTF-8 bytes bound tokens conservatively without a remote tokenizer.
            # Reserve room per document; never drop a document to fit global top-k.
            available = (settings.GENERATION_CONTEXT_TOKENS - settings.GENERATION_PROMPT_RESERVE_TOKENS
                         - settings.OPENAI_MAX_TOKENS - len(generation_question.encode("utf-8")))
            groups = [[c for c in chunks if c["document_id"] == i] for i in request.document_ids] if mode == "compare" else [chunks]
            per_document_budget = available // len(groups)
            if any(sum(len(c["content"].encode("utf-8")) + len(c["document_name"].encode("utf-8")) + 128
                       for c in group) > per_document_budget for group in groups):
                answer, kind = terminal("budget", language), "clarification"
                chunks = []
            elif not llm.has_credentials():
                answer, kind = terminal("llm_unavailable", language), "llm_unavailable"

        def result(text, terminal_kind):
            public_chunks = chunks if terminal_kind in {"answer", "llm_unavailable"} else []
            return QueryResponse(
                question=request.question, answer=text, query_type=request.query_type,
                session_id=prepared.session_id, scope_epoch=prepared.epoch, scope_fingerprint=prepared.fingerprint,
                answer_language=language, index_commit_id=view.commit_id if view else None,
                document_ids=request.document_ids, query_mode=mode, document_coverage=coverage,
                retrieved_chunks=response_shaping.to_retrieved_chunks(public_chunks), chunks_used=len(public_chunks),
                confidence_score=citation_policy.best_similarity(chunks), answer_kind=terminal_kind,
                response_time=monotonic() - started,
                embedding_provider=getattr(self.embedding, "provider", None))

        def check():
            lease.check()
            if view:
                view.assert_valid()

        def save(response):
            save_terminal(coordinator, view, request, prepared, response, effective_question)

        def missing_evidence_suffix():
            missing = [c for c in coverage if c['status'] == 'insufficient_evidence']
            return "".join((f"\nDocument {c['document_id']} ({c['document_name']}): insufficient evidence."
                            if language == "en" else f"\n文档 {c['document_id']} ({c['document_name']})：未找到充分依据。") for c in missing)

        if not stream:
            try:
                check()
                if not answer:
                    generated = await lease.wait(asyncio.wait_for(llm.agenerate_response(generation_question, chunks), settings.OPENAI_TIMEOUT_SECS))
                    kind = "answer" if generated.get("success", False) else ("timeout" if generated.get("error") == "timeout" else "llm_unavailable")
                    answer = generated.get("answer", "") if kind == "answer" else terminal(kind, language)
                check()
                response = result(answer + missing_evidence_suffix(), kind)
                await asyncio.to_thread(save, response)
                return response
            except asyncio.CancelledError:
                await record_cancellation(coordinator, view, request, prepared, language, mode, effective_question)
                raise
            except Exception as exc:
                kind = "cancelled" if isinstance(exc, SourceInvalidated) else "timeout" if isinstance(exc, TimeoutError) or getattr(exc, "code", "") == "timeout" else "llm_unavailable"
                response = result(terminal(kind, language), kind)
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
                yield event("start", {"question": request.question, "document_ids": request.document_ids, "query_mode": mode, "answer_language": language, "session_id": prepared.session_id, "scope_epoch": prepared.epoch})
                text = answer
                if not answer:
                    yield event("context", {"retrieved_chunks": [c.model_dump() for c in response_shaping.to_retrieved_chunks(chunks)],
                                            "document_coverage": coverage, "query_mode": mode, "answer_language": language, "session_id": prepared.session_id, "scope_epoch": prepared.epoch})
                    upstream = llm.agenerate_stream(generation_question, chunks)
                    try:
                        async with asyncio.timeout(settings.OPENAI_TIMEOUT_SECS):
                            while True:
                                try:
                                    token = await lease.wait(anext(upstream))
                                except StopAsyncIteration:
                                    break
                                check()
                                text += token
                                yield event("token", {"content": token})
                    finally:
                        await upstream.aclose()
                check()
                response = result(text + missing_evidence_suffix(), kind)
                await asyncio.to_thread(save, response)
                yield event("end", {**response.model_dump(), "success": response.answer_kind in {"answer", "refusal", "advice", "clarification"}})
            except (asyncio.CancelledError, GeneratorExit):
                await record_cancellation(coordinator, view, request, prepared, language, mode, effective_question)
                raise
            except Exception as exc:
                terminal_kind = "cancelled" if isinstance(exc, SourceInvalidated) else "timeout" if isinstance(exc, TimeoutError) or getattr(exc, "code", "") == "timeout" else "llm_unavailable"
                response = result(terminal(terminal_kind, language), terminal_kind)
                await asyncio.to_thread(save, response)
                yield event("end", {**response.model_dump(), "success": False})
            finally:
                if view:
                    view.close()
        return events()
