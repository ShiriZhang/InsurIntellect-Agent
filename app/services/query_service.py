"""
查询服务
整合向量搜索和LLM生成,提供完整的问答功能
"""

from datetime import datetime
import asyncio
from typing import List, Dict, Any, Optional
import json
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.app_logging import get_logger
from app.models.database_models import QueryHistory
from app.models.schemas import QueryRequest, QueryResponse, RetrievedChunk, QueryStatistics
from app.core.rag_workflow import get_agent

from app.services.llm_service import LLMService
from app.services.query_rewriter_service import QueryRewriterService
from app.services.query_router_service import QueryRouterService
from app.services.text_to_sql_service import TextToSQLService
from app.services.kg_service import KGService
from app.services import citation_policy, refusal_policy, response_shaping, query_history_service

logger = get_logger(__name__)


class QueryService:
    """查询服务"""
    
    def __init__(self):
        """初始化查询服务,使用InsurIntellectAgent RAG工作流"""
        # 延迟初始化agent,避免在服务启动时就创建
        self._agent = None
        # 保留对后台 query 历史保存任务的强引用，防 GC 静默回收 task
        # （裸 asyncio.create_task 未持引用则可能被 GC，Python 官方文档明确警告）
        self._bg_tasks: set = set()

    def _spawn_bg(self, coro):
        """启动后台任务并持有强引用；任务结束后自动从集合退出。"""
        try:
            task = asyncio.create_task(coro)
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
            return task
        except Exception as exc:
            # create_task 失败（如无 running loop）时手动关闭 coro 防
            # "coroutine was never awaited" 警告与资源泄漏
            logger.debug(f"后台任务启动失败：{exc}")
            try:
                coro.close()
            except Exception:
                pass
            return None

    @property
    def agent(self):
        """获取agent实例,使用延迟初始化"""
        if self._agent is None:
            logger.info("初始化agent实例...")
            try:
                self._agent = get_agent()
                logger.info(f"Agent实例初始化成功: {type(self._agent)}")
            except Exception as e:
                logger.error(f"Agent实例初始化失败: {e}", exc_info=True)
                raise
        return self._agent

    @property
    def embedding_service(self):
        """Expose agent embedding service for provider metadata on responses."""
        try:
            return getattr(self.agent, "embedding_service", None)
        except Exception:
            return None

    def _chunk_text(self, c: Dict[str, Any]) -> str:
        return citation_policy.chunk_text(c)

    def _is_weak_citation_chunk(self, c: Dict[str, Any]) -> bool:
        return citation_policy.is_weak_citation_chunk(c)

    def _doc_page_key(self, c: Dict[str, Any]) -> str:
        return citation_policy.doc_page_key(c)

    def _relevance_bonus(self, question: str, c: Dict[str, Any]) -> float:
        return citation_policy.relevance_bonus(question, c)

    def curate_citations(
        self,
        chunks: List[Dict[str, Any]],
        question: str = "",
        *,
        limit: int = 4,
    ) -> List[Dict[str, Any]]:
        return citation_policy.curate_citations(chunks, question, limit=limit)

    def public_citations(
        self,
        chunks: List[Dict[str, Any]],
        *,
        for_ui: bool = True,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        return citation_policy.public_citations(chunks, min_score=min_score)

    def citations_for_kind(
        self,
        kind: str,
        chunks: List[Dict[str, Any]],
        *,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        return citation_policy.citations_for_kind(kind, chunks, min_score=min_score)

    def _to_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> List[RetrievedChunk]:
        return response_shaping.to_retrieved_chunks(chunks)

    async def _get_chat_history(
        self,
        db: AsyncSession,
        session_id: Optional[str],
        limit: int = 10,
    ) -> str:
        """按会话获取最近若干轮用户/助理对话，拼接为简洁文本。

        - 仅在 session_id 存在时生效；否则返回空字符串。
        - 返回按时间升序的最近 N 条，格式："用户: ...\n助理: ..."，以便提示词引用。
        """
        try:
            if not session_id:
                return ""
            stmt = (
                select(QueryHistory)
                .where(QueryHistory.session_id == session_id)
                .order_by(QueryHistory.created_time.desc())
                .limit(limit)
            )
            result = await db.execute(stmt)
            recent = result.scalars().all() or []
            # 反转为时间升序，便于阅读
            recent = list(reversed(recent))
            lines: List[str] = []
            for r in recent:
                q = (getattr(r, "query", None) or "").strip()
                a = (getattr(r, "response", None) or "").strip()
                if q:
                    lines.append(f"用户: {q}")
                if a:
                    lines.append(f"助理: {a}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"获取会话历史失败：{e}")
            return ""
    

    def _normalize_sim(self, score: Any) -> float:
        return citation_policy.normalize_sim(score)

    def _chunk_gate_score(self, c: Dict[str, Any]) -> float:
        return citation_policy.chunk_gate_score(c)

    def _best_similarity(self, chunks: List[Dict[str, Any]]) -> float:
        return citation_policy.best_similarity(chunks)

    def _refusal_answer(self, reason: str = "insufficient_evidence") -> str:
        return refusal_policy.refusal_answer(reason)

    def _is_advice_or_guarantee_question(self, question: str) -> bool:
        return refusal_policy.is_advice_or_guarantee_question(question)

    def _has_llm_credentials(self) -> bool:
        return settings.generation_configured

    def _is_off_topic(self, question: str, chunks: List[Dict[str, Any]]) -> bool:
        return refusal_policy.is_off_topic(question, chunks)


    async def process_query(
        self,
        db: AsyncSession,
        request: QueryRequest,
        stream: bool = False,
    ) -> Any:
        """处理查询请求，支持非流与SSE流（统一入口）。"""
        start_time = datetime.utcnow()

        # 统一：准备 Agent、LLM（轻量/核心）与可选查询重写
        agent = self.agent
        llm_core = LLMService.with_model(settings.generation_model())
        llm_light = LLMService.with_model(settings.generation_model("light"))
        rewriter = QueryRewriterService(llm_service=llm_light)
        simple_mode = bool(getattr(settings, "SIMPLE_RAG_MODE", True))

        async def _rewrite_if_enabled(user_q: str) -> Dict[str, Any]:
            if not settings.ENABLE_QUERY_REWRITING:
                return {}
            try:
                chat_history = await self._get_chat_history(db, getattr(request, "session_id", None), limit=10)
                return await rewriter.rewrite_query(user_query=user_q, chat_history=chat_history)
            except Exception as e:
                logger.warning(f"查询重写失败，使用原始查询: {e}")
                return {}

        if not stream:
            try:
                # 强边界：购买/保证获赔类问题直接拒答（仍可检索展示）
                if self._is_advice_or_guarantee_question(request.question):
                    try:
                        ctx = await asyncio.wait_for(
                            agent.abuild_context(request.question),
                            timeout=getattr(settings, "CONTEXT_BUILD_TIMEOUT_SECS", 20),
                        )
                        retrieved_chunks = ctx.get("retrieved_chunks", []) or []
                        retrieved_chunks = self.curate_citations(
                            retrieved_chunks,
                            getattr(request, "question", "") or "",
                            limit=4,
                        )
                    except Exception:
                        retrieved_chunks = []
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    # advice: never attach sources (UI sticky / empty pane)
                    return QueryResponse(
                        question=request.question,
                        answer=self._refusal_answer("advice"),
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=0.0,
                        query_id=None,
                        answer_kind="advice",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )

                rewrite_result = await _rewrite_if_enabled(request.question)
                # 使用独立查询作为检索输入
                question_for_retrieval = rewrite_result.get("independent_query") or request.question

                # 在构建上下文前调用路由器（R4.2）；SIMPLE 模式强制 RAG
                route_result = {"route": "RAG"}
                if (not simple_mode) and settings.ENABLE_QUERY_ROUTING:
                    router = QueryRouterService(llm_service=llm_light)
                    try:
                        route_result = await router.route_query(request.question)
                    except Exception as re:
                        logger.warning(f"查询路由失败，回退RAG: {re}")

                if (not simple_mode) and route_result.get("route") == "SQL":
                    # SQL 路径：执行只读查询并用 LLM 总结（R4.3）
                    t2s = TextToSQLService()
                    sql_query = route_result.get("query") or ""
                    try:
                        sql_rows = await t2s.aexecute_sql_query(db, sql_query)
                    except Exception as e:
                        logger.error(f"SQL 执行失败: {e}")
                        sql_rows = []

                    # 将结构化结果格式化为上下文字符串
                    try:
                        results_text = json.dumps(sql_rows, ensure_ascii=False)
                    except Exception:
                        results_text = str(sql_rows)

                    sql_context = [
                        {
                            "document_name": "SQL结果",
                            "page_number": "N/A",
                            "content": f"SQL: {sql_query}\n结果: {results_text}",
                        }
                    ]

                    gen_res = await llm_core.agenerate_response(
                        query=request.question,
                        context_chunks=sql_context,
                    )

                    answer_text = gen_res.get("answer", "")
                    response_time = (datetime.utcnow() - start_time).total_seconds()

                    response = QueryResponse(
                        question=request.question,
                        answer=answer_text,
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=0.9,
                        query_id=None,
                        answer_kind="answer",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )

                    # 后台保存查询历史，避免阻塞主流程
                    try:
                        meta_payload = {
                            "route": "SQL",
                            "sql_query": sql_query,
                            "sql_result_count": len(sql_rows),
                        }
                        try:
                            if rewrite_result:
                                meta_payload.update({
                                    "primary_search_intent": rewrite_result.get("primary_search_intent"),
                                    "query_vectors": rewrite_result.get("query_vectors"),
                                    "micro_ontology": rewrite_result.get("micro_ontology"),
                                })
                        except Exception:
                            pass
                        rewriting_metadata_json = json.dumps(meta_payload, ensure_ascii=False)
                        self._spawn_bg(
                            self._save_query_history_bg(
                                question=request.question,
                                answer=answer_text,
                                query_type=request.query_type,
                                response_time=response_time,
                                chunks_used=0,
                                session_id=getattr(request, "session_id", None),
                                rewritten_query=(rewrite_result.get("independent_query") if rewrite_result else None),
                                rewriting_metadata_json=rewriting_metadata_json,
                                retrieved_chunks_json=None,
                            )
                        )
                    except Exception as save_error:
                        logger.debug(f"查询历史后台保存启动失败(SQL路径): {save_error}")

                    return response

                # 异步构建上下文（T3: asyncio.to_thread 包装同步I/O）
                try:
                    ctx = await asyncio.wait_for(
                        agent.abuild_context(question_for_retrieval),
                        timeout=getattr(settings, "CONTEXT_BUILD_TIMEOUT_SECS", 5),
                    )
                except asyncio.TimeoutError:
                    logger.warning("上下文构建超时，返回降级响应")
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    fallback = QueryResponse(
                        question=request.question,
                        answer="系统当前繁忙，检索上下文构建超时，请稍后重试。",
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=0.0,
                        query_id=None,
                        answer_kind="degraded",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )
                    # 后台保存，不阻塞
                    try:
                        self._spawn_bg(
                            self._save_query_history_bg(
                                question=request.question,
                                answer=fallback.answer,
                                query_type=request.query_type,
                                response_time=response_time,
                                chunks_used=0,
                                session_id=getattr(request, "session_id", None),
                                rewritten_query=None,
                                rewriting_metadata_json=json.dumps({"route": "RAG", "degraded": True}, ensure_ascii=False),
                                retrieved_chunks_json=None,
                            )
                        )
                    except Exception:
                        pass
                    return fallback
                retrieved_chunks = ctx.get("retrieved_chunks", []) or []
                retrieved_chunks = citation_policy.eligible_evidence(
                    retrieved_chunks, float(settings.SIMILARITY_THRESHOLD))
                retrieved_chunks = self.curate_citations(
                    retrieved_chunks,
                    getattr(request, "question", "") or "",
                    limit=4,
                )
                rewritten_query = (
                    ctx.get("rewritten_query")
                    or (rewrite_result.get("independent_query") if rewrite_result else None)
                    or (question_for_retrieval if question_for_retrieval != request.question else None)
                )

                # 注入 KG 事实作为优先片段（仅 advanced）
                if not simple_mode:
                    try:
                        kg_service = KGService()
                        kg_facts = await kg_service.aget_facts(db, question_for_retrieval)
                        if kg_facts:
                            kg_chunk = {
                                "chunk_id": None,
                                "document_id": None,
                                "document_name": "知识图谱事实",
                                "content": "\n".join([f"- {f}" for f in kg_facts]),
                                "page_number": "N/A",
                                "similarity_score": 1.0,
                                "metadata": {"is_kg": True, "facts_count": len(kg_facts)},
                            }
                            retrieved_chunks = [kg_chunk] + retrieved_chunks
                    except Exception as kg_err:
                        logger.debug(f"KG事实注入失败：{kg_err}")

                # 拒答门闩：无检索 / 低分 / 明显离题
                best_sim = self._best_similarity(retrieved_chunks)
                off_topic = self._is_off_topic(request.question, retrieved_chunks)
                # fallback 对齐 config.py 默认值 0.32，与 README hero "BGE threshold 0.32" 公开承诺一致
                if off_topic or (not retrieved_chunks) or best_sim < float(getattr(settings, "SIMILARITY_THRESHOLD", 0.32)):
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    # Public API: no low-score filler citations on refusal
                    response = QueryResponse(
                        question=request.question,
                        answer=self._refusal_answer("insufficient_evidence"),
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=round(best_sim, 4),
                        query_id=None,
                        answer_kind="refusal",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )
                    try:
                        self._spawn_bg(
                            self._save_query_history_bg(
                                question=request.question,
                                answer=response.answer,
                                query_type=request.query_type,
                                response_time=response_time,
                                chunks_used=0,
                                session_id=getattr(request, "session_id", None),
                                rewritten_query=rewritten_query,
                                rewriting_metadata_json=json.dumps(
                                    {
                                        "route": "RAG",
                                        "refused": True,
                                        "best_similarity": best_sim,
                                        "debug_chunk_count": len(retrieved_chunks or []),
                                    },
                                    ensure_ascii=False,
                                ),
                                # keep debug-only payload off public response; optional history
                                retrieved_chunks_json=None,
                            )
                        )
                    except Exception:
                        pass
                    return response

                # 无 API key：检索结果诚实降级
                if not self._has_llm_credentials():
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    preview_lines = []
                    for i, c in enumerate(retrieved_chunks[:3], 1):
                        name = c.get("document_name") or "未知文档"
                        page = c.get("page_number")
                        snippet = (c.get("content") or "")[:180].replace("\n", " ")
                        preview_lines.append(f"[{i}] {name} p.{page}: {snippet}")
                    answer_text = (
                        "【结论】\n"
                        "LLM 不可用（未配置 API Key），已返回检索片段摘要，未生成完整条款解释。\n\n"
                        "【条款依据】\n"
                        + ("\n".join(preview_lines) if preview_lines else "（无片段）")
                        + "\n\n【不确定/边界】\n本系统不构成保险销售或理赔承诺。"
                    )
                    return QueryResponse(
                        question=request.question,
                        answer=answer_text,
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=len(retrieved_chunks),
                        retrieved_chunks=self._to_retrieved_chunks(self.public_citations(retrieved_chunks)),
                        confidence_score=round(best_sim, 4),
                        query_id=None,
                        answer_kind="llm_unavailable",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )

                # 非流式：调用异步非流 LLM（T2）
                try:
                    gen_res = await asyncio.wait_for(
                        llm_core.agenerate_response(
                            query=request.question,
                            context_chunks=retrieved_chunks,
                        ),
                        timeout=getattr(settings, "LLM_ANSWER_TIMEOUT_SECS", 60),
                    )
                    if not gen_res.get("success", True):
                        preview = []
                        for i, c in enumerate(retrieved_chunks[:3], 1):
                            preview.append(
                                f"[{i}] {c.get('document_name')} p.{c.get('page_number')}: "
                                f"{(c.get('content') or '')[:120]}"
                            )
                        answer_text = (
                            "【结论】\nLLM 调用失败，以下为检索到的条款片段摘要。\n\n"
                            "【条款依据】\n"
                            + "\n".join(preview)
                            + "\n\n【不确定/边界】\n本系统不构成保险销售或理赔承诺。"
                        )
                        response_time = (datetime.utcnow() - start_time).total_seconds()
                        ui_chunks = self.citations_for_kind("llm_unavailable", retrieved_chunks)
                        return QueryResponse(
                            question=request.question,
                            answer=answer_text,
                            query_type=request.query_type,
                            response_time=response_time,
                            chunks_used=len(ui_chunks),
                            retrieved_chunks=self._to_retrieved_chunks(ui_chunks),
                            confidence_score=round(best_sim, 4),
                            query_id=None,
                            answer_kind="llm_unavailable",
                            embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                        )
                except asyncio.TimeoutError:

                    logger.warning("LLM生成超时，返回降级响应")
                    answer_text = "系统当前繁忙，生成答案超时，请稍后重试。"
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    response = QueryResponse(
                        question=request.question,
                        answer=answer_text,
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=0.0,
                        query_id=None,
                        answer_kind="degraded",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )
                    # 后台保存，不阻塞
                    try:
                        rewriting_metadata_json = json.dumps({"route": "RAG", "degraded": True}, ensure_ascii=False)
                        self._spawn_bg(
                            self._save_query_history_bg(
                                question=request.question,
                                answer=answer_text,
                                query_type=request.query_type,
                                response_time=response_time,
                                chunks_used=len(retrieved_chunks),
                                session_id=getattr(request, "session_id", None),
                                rewritten_query=None,
                                rewriting_metadata_json=rewriting_metadata_json,
                                retrieved_chunks_json=json.dumps(retrieved_chunks, ensure_ascii=False) if retrieved_chunks else None,
                            )
                        )
                    except Exception:
                        pass
                    return response
                except TimeoutError:
                    logger.warning("LLM并发队列等待超时，返回降级响应")
                    answer_text = "系统当前繁忙，请稍后重试。"
                    response_time = (datetime.utcnow() - start_time).total_seconds()
                    response = QueryResponse(
                        question=request.question,
                        answer=answer_text,
                        query_type=request.query_type,
                        response_time=response_time,
                        chunks_used=0,
                        retrieved_chunks=[],
                        confidence_score=0.0,
                        query_id=None,
                        answer_kind="degraded",
                        embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                    )
                    # 后台保存，不阻塞
                    try:
                        rewriting_metadata_json = json.dumps({"route": "RAG", "degraded": True, "reason": "queue_timeout"}, ensure_ascii=False)
                        self._spawn_bg(
                            self._save_query_history_bg(
                                question=request.question,
                                answer=answer_text,
                                query_type=request.query_type,
                                response_time=response_time,
                                chunks_used=len(retrieved_chunks),
                                session_id=getattr(request, "session_id", None),
                                rewritten_query=None,
                                rewriting_metadata_json=rewriting_metadata_json,
                                retrieved_chunks_json=json.dumps(retrieved_chunks, ensure_ascii=False) if retrieved_chunks else None,
                            )
                        )
                    except Exception:
                        pass
                    return response

                answer_text = gen_res.get("answer", "")
                response_time = (datetime.utcnow() - start_time).total_seconds()

                ui_chunks = self.citations_for_kind("answer", retrieved_chunks)
                response = QueryResponse(
                    question=request.question,
                    answer=answer_text,
                    query_type=request.query_type,
                    response_time=response_time,
                    chunks_used=len(ui_chunks),
                    retrieved_chunks=self._to_retrieved_chunks(ui_chunks) if ui_chunks else [],
                    confidence_score=round(best_sim, 4),
                    query_id=None,
                    answer_kind="answer",
                    embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                )

                # 保存查询历史
                # 后台保存，不阻塞
                try:
                    meta_payload = {"route": "RAG"}
                    try:
                        if rewrite_result:
                            meta_payload.update({
                                "primary_search_intent": rewrite_result.get("primary_search_intent"),
                                "query_vectors": rewrite_result.get("query_vectors"),
                                "micro_ontology": rewrite_result.get("micro_ontology"),
                            })
                    except Exception:
                        pass
                    rewriting_metadata_json = json.dumps(meta_payload, ensure_ascii=False)
                    self._spawn_bg(
                        self._save_query_history_bg(
                            question=request.question,
                            answer=answer_text,
                            query_type=request.query_type,
                            response_time=response_time,
                            chunks_used=len(retrieved_chunks),
                            session_id=getattr(request, "session_id", None),
                            rewritten_query=rewritten_query,
                            rewriting_metadata_json=rewriting_metadata_json,
                            retrieved_chunks_json=json.dumps(retrieved_chunks, ensure_ascii=False) if retrieved_chunks else None,
                        )
                    )
                except Exception as save_error:
                    logger.debug(f"查询历史后台保存启动失败: {save_error}")

                return response
            except Exception as e:
                logger.error(f"非流式查询处理异常: {e}")
                response_time = (datetime.utcnow() - start_time).total_seconds()
                error_response = QueryResponse(
                    question=request.question,
                    answer="抱歉,处理您的查询时出现了错误,请稍后重试",
                    query_type=request.query_type,
                    response_time=response_time,
                    chunks_used=0,
                    retrieved_chunks=[],
                    confidence_score=0.0,
                    query_id=None,
                    answer_kind="degraded",
                    embedding_provider=getattr(getattr(self, "embedding_service", None), "provider", None),
                )
                try:
                    self._spawn_bg(
                        self._save_query_history_bg(
                            question=request.question,
                            answer=error_response.answer,
                            query_type=request.query_type,
                            response_time=response_time,
                            chunks_used=0,
                            session_id=getattr(request, "session_id", None),
                            rewritten_query=None,
                            rewriting_metadata_json=json.dumps({"route": "RAG"}, ensure_ascii=False),
                            retrieved_chunks_json=None,
                        )
                    )
                except Exception:
                    pass
                return error_response

        async def _sse_gen():
            try:
                rewrite_result = await _rewrite_if_enabled(request.question)
                question_for_retrieval = rewrite_result.get("independent_query") or request.question

                # 起始事件
                import json as _json
                start_evt = {
                    "type": "start",
                    "question": request.question,
                    "query_type": request.query_type,
                    "timestamp": start_time.timestamp(),
                }
                yield f"event: start\n"
                yield f"data: {_json.dumps(start_evt, ensure_ascii=False)}\n\n"

                # Early advice gate: no sources on purchase/guarantee questions
                if self._is_advice_or_guarantee_question(request.question):
                    advice_ans = self._refusal_answer("advice")
                    end_evt = {
                        "type": "end",
                        "answer": advice_ans,
                        "response_time": (datetime.utcnow() - start_time).total_seconds(),
                        "query_id": None,
                        "success": True,
                        "retrieved_chunks": [],
                        "chunks_used": 0,
                        "answer_kind": "advice",
                        "embedding_provider": getattr(getattr(self, "embedding_service", None), "provider", None),
                    }
                    yield f"event: end\n"
                    yield f"data: {_json.dumps(end_evt, ensure_ascii=False)}\n\n"
                    return

                # 在构建上下文前调用路由器（R4.2 - 流式路径），受开关控制
                # SIMPLE 模式强制 RAG，跳过 advanced 路由（与非流式路径一致）
                route_result = {"route": "RAG"}
                if (not simple_mode) and settings.ENABLE_QUERY_ROUTING:
                    router = QueryRouterService(llm_service=llm_light)
                    try:
                        route_result = await router.route_query(request.question)
                    except Exception as re:
                        logger.warning(f"流式查询路由失败，回退RAG: {re}")
                if route_result.get("route") == "SQL":
                    # SQL 路径：直接执行并总结，随后发送 end 事件
                    t2s = TextToSQLService()
                    sql_query = route_result.get("query") or ""
                    try:
                        sql_rows = await t2s.aexecute_sql_query(db, sql_query)
                    except Exception as e:
                        logger.error(f"SQL 执行失败(流式): {e}")
                        sql_rows = []

                    # 发送上下文事件，包含SQL元信息
                    ctx_evt = {
                        "type": "context",
                        "rewritten_query": (rewrite_result.get("independent_query") if rewrite_result else None),
                        "retrieved_chunks": [],
                        "route": "SQL",
                        "sql_query": sql_query,
                        "sql_result_count": len(sql_rows),
                    }
                    yield f"event: context\n"
                    yield f"data: {_json.dumps(ctx_evt, ensure_ascii=False)}\n\n"

                    # 总结结果
                    try:
                        results_text = _json.dumps(sql_rows, ensure_ascii=False)
                    except Exception:
                        results_text = str(sql_rows)
                    sql_context = [{"document_name": "SQL结果", "page_number": "N/A", "content": f"SQL: {sql_query}\n结果: {results_text}"}]
                    gen_res = await llm_core.agenerate_response(query=request.question, context_chunks=sql_context)
                    final_text = gen_res.get("answer", "")

                    # 保存历史
                    try:
                        meta_payload = {
                            "route": "SQL",
                            "sql_query": sql_query,
                            "sql_result_count": len(sql_rows),
                        }
                        if rewrite_result:
                            meta_payload.update({
                                "primary_search_intent": rewrite_result.get("primary_search_intent"),
                                "query_vectors": rewrite_result.get("query_vectors"),
                                "micro_ontology": rewrite_result.get("micro_ontology"),
                            })
                        rewriting_metadata_json = _json.dumps(meta_payload, ensure_ascii=False)
                        query_id = await self._save_query_history(
                            db,
                            request.question,
                            final_text.strip(),
                            request.query_type,
                            (datetime.utcnow() - start_time).total_seconds(),
                            0,
                            getattr(request, "session_id", None),
                            rewritten_query=(rewrite_result.get("independent_query") if rewrite_result else None),
                            rewriting_metadata_json=rewriting_metadata_json,
                            retrieved_chunks_json=None,
                        )
                    except Exception as save_error:
                        logger.warning(f"流式(SQL)查询历史保存失败: {save_error}")
                        query_id = None

                    end_evt = {
                        "type": "end",
                        "answer": final_text.strip(),
                        "response_time": (datetime.utcnow() - start_time).total_seconds(),
                        "query_id": query_id,
                        "success": True,
                    }
                    yield f"event: end\n"
                    yield f"data: {_json.dumps(end_evt, ensure_ascii=False)}\n\n"
                    return

                # 异步构建上下文
                ctx = await agent.abuild_context(question_for_retrieval)
                retrieved_chunks = ctx.get("retrieved_chunks", []) or []
                retrieved_chunks = citation_policy.eligible_evidence(
                    retrieved_chunks, float(settings.SIMILARITY_THRESHOLD))
                retrieved_chunks = self.curate_citations(
                    retrieved_chunks,
                    getattr(request, "question", "") or "",
                    limit=4,
                )
                rewritten_query = (
                    ctx.get("rewritten_query")
                    or (rewrite_result.get("independent_query") if rewrite_result else None)
                    or (question_for_retrieval if question_for_retrieval != request.question else None)
                )

                # 注入 KG 事实作为优先片段（仅 advanced / SSE 路径）
                if not simple_mode:
                    try:
                        kg_service = KGService()
                        kg_facts = await kg_service.aget_facts(db, question_for_retrieval)
                        if kg_facts:
                            kg_chunk = {
                                "chunk_id": None,
                                "document_id": None,
                                "document_name": "知识图谱事实",
                                "content": "\n".join([f"- {f}" for f in kg_facts]),
                                "page_number": "N/A",
                                "similarity_score": 1.0,
                                "metadata": {"is_kg": True, "facts_count": len(kg_facts)},
                            }
                            retrieved_chunks = [kg_chunk] + retrieved_chunks
                    except Exception as kg_err:
                        logger.debug(f"KG事实注入失败(流式)：{kg_err}")

                # Refusal gate for SSE path (public chunks empty)
                best_sim = self._best_similarity(retrieved_chunks)
                off_topic = self._is_off_topic(request.question, retrieved_chunks)
                # fallback 对齐 config.py 默认值 0.32，与 README hero "BGE threshold 0.32" 公开承诺一致
                thr = float(getattr(settings, "SIMILARITY_THRESHOLD", 0.32))
                if off_topic or (not retrieved_chunks) or best_sim < thr:
                    refuse_ans = self._refusal_answer("insufficient_evidence")
                    end_evt = {
                        "type": "end",
                        "answer": refuse_ans,
                        "response_time": (datetime.utcnow() - start_time).total_seconds(),
                        "query_id": None,
                        "success": True,
                        "retrieved_chunks": [],
                        "chunks_used": 0,
                        "answer_kind": "refusal",
                        "confidence_score": round(best_sim, 4),
                        "embedding_provider": getattr(getattr(self, "embedding_service", None), "provider", None),
                    }
                    yield f"event: end\n"
                    yield f"data: {_json.dumps(end_evt, ensure_ascii=False)}\n\n"
                    return

                context_evt = {
                    "type": "context",
                    "rewritten_query": rewritten_query,
                    "retrieved_chunks": retrieved_chunks,
                    "route": "RAG",
                }
                yield f"event: context\n"
                yield f"data: {_json.dumps(context_evt, ensure_ascii=False)}\n\n"

                if not retrieved_chunks:
                    # 无上下文，直接返回错误并持久化
                    err_msg = "未检索到与问题相关的文档片段"
                    try:
                        await self._save_query_history(
                            db,
                            request.question,
                            err_msg,
                            request.query_type,
                            (datetime.utcnow() - start_time).total_seconds(),
                            0,
                            getattr(request, "session_id", None),
                            rewritten_query=rewritten_query,
                            rewriting_metadata_json=_json.dumps({"route": "RAG"}, ensure_ascii=False),
                            retrieved_chunks_json=None,
                        )
                    except Exception:
                        pass
                    err_evt = {"type": "error", "message": err_msg}
                    yield f"event: error\n"
                    yield f"data: {_json.dumps(err_evt, ensure_ascii=False)}\n\n"
                    return

                # 调用LLM流式生成：agenerate_stream 直接 yield 文本增量
                final_text = ""
                async for content in llm_core.agenerate_stream(
                    query=request.question,
                    context_chunks=retrieved_chunks,
                ):
                    try:
                        final_text += content
                        tok_evt = {"type": "token", "content": content}
                        yield f"event: token\n"
                        yield f"data: {_json.dumps(tok_evt, ensure_ascii=False)}\n\n"
                    except Exception:
                        # 忽略单次解析错误，继续流式
                        pass

                # 结束：保存历史并发送end事件
                try:
                    # 记录路由为 RAG，并合并重写元信息（若存在）
                    meta_payload = {"route": "RAG"}
                    try:
                        if rewrite_result:
                            meta_payload.update({
                                "primary_search_intent": rewrite_result.get("primary_search_intent"),
                                "query_vectors": rewrite_result.get("query_vectors"),
                                "micro_ontology": rewrite_result.get("micro_ontology"),
                            })
                    except Exception:
                        pass
                    rewriting_metadata_json = json.dumps(meta_payload, ensure_ascii=False)

                    query_id = await self._save_query_history(
                        db,
                        request.question,
                        final_text.strip() if final_text else "",
                        request.query_type,
                        (datetime.utcnow() - start_time).total_seconds(),
                        len(retrieved_chunks),
                        getattr(request, "session_id", None),
                        rewritten_query=rewritten_query,
                        rewriting_metadata_json=rewriting_metadata_json,
                        retrieved_chunks_json=json.dumps(retrieved_chunks, ensure_ascii=False),
                    )
                except Exception as save_error:
                    logger.warning(f"流式查询历史保存失败: {save_error}")
                    query_id = None

                _ans = final_text.strip() if final_text else ""
                _kind = "answer"
                _q = request.question or ""
                if "未在已入库条款中找到充分依据" in _ans:
                    _kind = "refusal"
                elif self._is_advice_or_guarantee_question(_q):
                    _kind = "advice"
                elif "LLM 不可用" in _ans:
                    _kind = "llm_unavailable"
                elif "系统当前繁忙" in _ans or "出现了错误" in _ans:
                    _kind = "degraded"
                ui_chunks = self.citations_for_kind(_kind, retrieved_chunks)
                try:
                    chunk_payload = [
                        rc.model_dump() if hasattr(rc, "model_dump") else dict(rc)
                        for rc in self._to_retrieved_chunks(ui_chunks)
                    ]
                except Exception:
                    chunk_payload = ui_chunks
                end_evt = {
                    "type": "end",
                    "answer": _ans,
                    "response_time": (datetime.utcnow() - start_time).total_seconds(),
                    "query_id": query_id,
                    "success": True,
                    "retrieved_chunks": chunk_payload,
                    "chunks_used": len(chunk_payload),
                    "confidence_score": round(best_sim, 4),
                    "answer_kind": _kind,
                    "embedding_provider": getattr(getattr(self, "embedding_service", None), "provider", None),
                }
                yield f"event: end\n"
                yield f"data: {_json.dumps(end_evt, ensure_ascii=False)}\n\n"

            except Exception as e:
                import json as _json3
                err_evt = {"type": "error", "message": str(e)}
                yield f"event: error\n"
                yield f"data: {_json3.dumps(err_evt, ensure_ascii=False)}\n\n"

        # 返回异步生成器（SSE字符串）
        return _sse_gen()

    async def _save_query_history(
        self,
        db: AsyncSession,
        question: str,
        answer: str,
        query_type: str,
        response_time: float,
        chunks_used: int,
        session_id: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        rewriting_metadata_json: Optional[str] = None,
        retrieved_chunks_json: Optional[str] = None,
    ) -> Optional[int]:
        return await query_history_service.save_query_history(
            db=db,
            question=question,
            answer=answer,
            query_type=query_type,
            response_time=response_time,
            chunks_used=chunks_used,
            session_id=session_id,
            rewritten_query=rewritten_query,
            rewriting_metadata_json=rewriting_metadata_json,
            retrieved_chunks_json=retrieved_chunks_json,
        )

    async def _save_query_history_bg(
        self,
        question: str,
        answer: str,
        query_type: str,
        response_time: float,
        chunks_used: int,
        session_id: Optional[str] = None,
        rewritten_query: Optional[str] = None,
        rewriting_metadata_json: Optional[str] = None,
        retrieved_chunks_json: Optional[str] = None,
    ) -> None:
        await query_history_service.save_query_history_bg(
            question=question,
            answer=answer,
            query_type=query_type,
            response_time=response_time,
            chunks_used=chunks_used,
            session_id=session_id,
            rewritten_query=rewritten_query,
            rewriting_metadata_json=rewriting_metadata_json,
            retrieved_chunks_json=retrieved_chunks_json,
        )

    async def get_query_history(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50,
        query_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[QueryHistory]:
        return await query_history_service.get_query_history(
            db=db,
            skip=skip,
            limit=limit,
            query_type=query_type,
            start_date=start_date,
            end_date=end_date,
        )

    async def update_query_feedback(
        self,
        db: AsyncSession,
        query_id: int,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        return await query_history_service.update_query_feedback(
            db=db,
            query_id=query_id,
            rating=rating,
            comment=comment,
        )

    async def get_query_statistics(self, db: AsyncSession, days: int = 30) -> QueryStatistics:
        return await query_history_service.get_query_statistics(db, days=days)


@lru_cache(maxsize=1)
def _load_ontology() -> dict:
    """加载保险术语本体库JSON，使用缓存减少I/O。"""
    try:
        with open(settings.ONTOLOGY_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载本体库失败，使用空本体：{e}")
        return {
            "synonyms": {},
            "product_types": [],
            "coverage_terms": [],
            "claim_terms": [],
            "exclusions": [],
        }
