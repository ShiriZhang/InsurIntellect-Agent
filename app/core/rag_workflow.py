#!/usr/bin/env python3
"""
InsurIntellect Agent - RAG Workflow Engine
=========================================

This module implements the core RAG (Retrieval-Augmented Generation) workflow
for the InsurIntellect Agent. It orchestrates a multi-stage AI collaboration
process from receiving user queries to generating final answers.

The workflow consists of:
1. Query Architect: Rewrites and optimizes user queries
2. Document Retrieval: Searches the vector database for relevant content
3. Lead Reviewer: Filters and selects the most relevant document chunks
4. Report Author: Generates comprehensive answers based on selected context

Author: InsurIntellect Agent Development Team
Version: 1.0.0
"""

import json
import asyncio
from typing import Any, Dict, List, Tuple

from app.core.app_logging import get_logger as app_get_logger
from app.core.config import settings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.models.schemas import QueryIntent
from app.prompts import QUERY_ARCHITECT_PROMPT
from app.core.fusion import reciprocal_rank_fusion
from app.services.embedding_service import EmbeddingService
from app.services.query_intent_service import QueryIntentService
from app.services.retrieval_text import lexical_profile, rank_lexical_candidates
from app.services.evidence_scores import cosine_distance, cosine_distance_to_score, finite_number

logger = app_get_logger(__name__)



def _normalize_vector_score(raw_distance: float) -> float:
    """Compatibility helper: input is explicitly a cosine distance."""
    return cosine_distance_to_score(raw_distance)


class InsurIntellectAgent:
    """
    InsurIntellect智能代理核心类
    
    封装从接收用户问题到生成最终答案的多阶段 AI 协作工作流。
    通过查询架构师、文档检索、首席评审员和报告撰写人的协作，
    提供高质量的保险领域智能问答服务。
    """
    
    def __init__(self):
        """
        初始化InsurIntellect智能代理
        
        初始化组件:
        - ChatOpenAI LLM实例
        - 本地 HuggingFace 嵌入模型
        - ChromaDB向量数据库检索器
        """
        logger.info("正在初始化InsurIntellect智能代理...")
        
        try:
            # 初始化 ChatOpenAI（OpenAI-compatible；兼容 siliconflow 回退）
            _model = settings.generation_model()
            _api_key = settings.generation_api_key
            _base_url = settings.generation_api_base_url
            self.llm = None
            if settings.generation_configured:
                self.llm = ChatOpenAI(
                    model=_model,
                    temperature=settings.OPENAI_TEMPERATURE,
                    max_tokens=settings.OPENAI_MAX_TOKENS,
                    api_key=_api_key,
                    base_url=_base_url,
                )
                logger.info("ChatOpenAI LLM初始化成功")
            else:
                logger.warning("未配置 API Key：跳过 ChatOpenAI 初始化（检索仍可用，生成走 llm_unavailable）")
            
            # 初始化本地嵌入服务
            self.embedding_service = EmbeddingService()
            self.embeddings = self.embedding_service.embedding_function
            logger.info("本地嵌入服务初始化成功")
            # 额外记录到应用主日志记录器，确保写入 logs/app.log
            try:
                app_get_logger("insurintellect").info("本地嵌入服务初始化成功")
            except Exception:
                pass
            # 意图分类服务（KCP-FIX-4）
            try:
                self.intent_service = QueryIntentService()
            except Exception:
                self.intent_service = None
            
            # Only committed, fingerprinted resources may enter the normal path.
            from app.services.index_coordinator import default_coordinator
            self.embedding_service.preflight()
            self.index_coordinator = default_coordinator()
            self.index_coordinator.recover()
            self.bm25_index = None
            self.bm25_chunk_map = {}
            self.bm25_available = False
            self.regulatory_docs = {}

            # 最近一次运行的检索与排序详情（用于审计与持久化）
            self.last_run: Dict[str, Any] = {}

        except Exception as e:
            logger.exception("初始化InsurIntellect智能代理失败")
            raise
    
    def run_query_architect(self, user_query: str) -> Dict[str, Any]:
        """
        运行查询架构师：重写和优化用户查询
        
        Args:
            user_query (str): 用户原始查询
            
        Returns:
            Dict[str, Any]: 包含重写后查询的字典
        """
        try:
            logger.info("查询架构师开始工作...")
            
            # 使用QUERY_ARCHITECT_PROMPT模板
            prompt = QUERY_ARCHITECT_PROMPT.format(user_query=user_query)
            
            # 调用LLM
            response = self.llm.invoke(prompt)
            
            # 解析JSON响应（更稳健）：支持去除代码块、提取首尾花括号、宽松容错
            try:
                content = (response.content or "").strip()

                def _strip_code_fences(text: str) -> str:
                    t = text.strip()
                    if "```json" in t:
                        s = t.find("```json") + 7
                        e = t.find("```", s)
                        if e != -1:
                            return t[s:e].strip()
                    if "```" in t:
                        s = t.find("```") + 3
                        e = t.find("```", s)
                        if e != -1:
                            return t[s:e].strip()
                    return t

                def _extract_json_object(text: str) -> str:
                    # 尝试定位首个 '{' 与最后一个 '}' 并做括号配对
                    import re
                    t = _strip_code_fences(text)
                    # 如果已经是纯 JSON 尝试直接解析
                    try:
                        json.loads(t)
                        return t
                    except Exception:
                        pass
                    # 正则粗略截取最外层对象
                    first = t.find('{')
                    last = t.rfind('}')
                    if first != -1 and last != -1 and last > first:
                        candidate = t[first:last+1]
                        # 简单括号计数校验
                        stack = 0
                        for ch in candidate:
                            if ch == '{':
                                stack += 1
                            elif ch == '}':
                                stack -= 1
                        if stack == 0:
                            return candidate.strip()
                    return t

                raw = _extract_json_object(content)
                parsed = json.loads(raw)

                # 归一化与回退：保障下游使用 rewritten_query
                normalized: Dict[str, Any] = dict(parsed)
                # 提取独立查询作为 rewritten_query 的首选
                rewritten = (
                    parsed.get("independent_query")
                    or parsed.get("rewritten_query")
                    or parsed.get("original_query")
                    or user_query
                )
                normalized["rewritten_query"] = rewritten

                # 规范 query_vectors：允许字符串列表或对象列表
                qv = parsed.get("query_vectors")
                if isinstance(qv, list):
                    fixed = []
                    for item in qv:
                        if isinstance(item, str):
                            fixed.append({"title": "变体", "query": item})
                        elif isinstance(item, dict):
                            # 统一字段名
                            q = item.get("query") or item.get("text") or ""
                            t = item.get("title") or item.get("vector_type") or "变体"
                            fixed.append({"title": t, "query": q})
                    normalized["query_vectors"] = fixed

                logger.info(f"查询架构师完成工作,重写查询: {normalized.get('rewritten_query', '解析失败')}")
                return normalized
                
            except json.JSONDecodeError as e:
                logger.exception("查询架构师返回的不是合法JSON")
                logger.debug(f"原始响应内容: {response.content}")
                
                # 返回默认结构
                return {
                    "rewritten_query": user_query,
                    "search_strategy": "默认搜索",
                    "key_concepts": [user_query],
                    "error": f"JSON解析错误: {str(e)}"
                }
                
        except Exception as e:
            logger.exception("查询架构师执行失败")
            return {
                "rewritten_query": user_query,
                "search_strategy": "默认搜索",
                "key_concepts": [user_query],
                "error": f"执行错误: {str(e)}"
            }
            
    async def abuild_context(self, user_query: str, document_ids=None) -> Dict[str, Any]:
        """
        异步版本：构建 RAG 上下文（不生成最终答案）。

        混合检索：并发执行向量检索与BM25检索，RRF融合后重取Top-K。

        返回：{
          "rewritten_query": str,
          "context": str,
          "retrieved_chunks": list,
        }
        """
        if hasattr(self, "index_coordinator"):
            return await self._published_context(user_query, document_ids)
        try:
            # 步骤1：查询架构师重写查询（遵循配置，必要时启用微超时）
            if not settings.ENABLE_QUERY_REWRITING:
                rewritten_query = user_query
            else:
                try:
                    # 微超时：避免在高并发下线程池被长时间占用
                    micro_timeout = max(1.0, float(settings.CONTEXT_BUILD_TIMEOUT_SECS) - 1.0)
                    architect_result = await asyncio.wait_for(
                        asyncio.to_thread(self.run_query_architect, user_query),
                        timeout=micro_timeout,
                    )
                    rewritten_query = architect_result.get("rewritten_query", user_query)
                except Exception:
                    # 超时或错误则直接回退使用原始查询
                    rewritten_query = user_query
            standalone_query = rewritten_query

            # 意图分类：SIMPLE_RAG_MODE 下跳过，避免主路径额外 LLM 调用
            metadata_filter: Dict[str, Any] | None = None
            try:
                if (
                    not getattr(settings, "SIMPLE_RAG_MODE", True)
                    and hasattr(self, "intent_service")
                    and self.intent_service is not None
                ):
                    intent: QueryIntent = await self.intent_service.classify_intent(rewritten_query)
                    metadata_filter = intent.metadata_filter or None
                    logger.info(f"意图分类：{intent.intent}，应用过滤器：{metadata_filter}")
            except Exception as e:
                logger.warning(f"意图分类失败，继续无过滤：{e}")

            # 预先异步获取查询嵌入（双模式服务）（I2.1）
            try:
                query_embedding = await self.embedding_service.aembed_query(rewritten_query)
            except Exception:
                query_embedding = None

            # 步骤2：并发检索（向量检索 n=50 与 BM25 n=50）
            logger.info(f"(async) 混合检索：向量与BM25并发，查询: {rewritten_query}")

            def _matches_metadata_filter(md: Dict[str, Any], flt: Dict[str, Any] | None) -> bool:
                if not flt:
                    return True
                try:
                    for k, v in flt.items():
                        if isinstance(v, dict) and "$in" in v:
                            if md.get(k) not in v["$in"]:
                                return False
                        else:
                            if str(md.get(k)).strip() != str(v).strip():
                                return False
                    return True
                except Exception:
                    return False

            async def _vector_search_k50() -> Tuple[List[Tuple[Document, float | None]], Dict[str, float | None], List[str]]:
                try:
                    # 传入动态元数据过滤（如底层不支持，则由回退路径处理）
                    results = await asyncio.to_thread(
                        self.vectorstore.similarity_search_with_score,
                        rewritten_query,
                        k=50,
                        filter=metadata_filter,
                    )
                    ids: List[str] = []
                    distance_map: Dict[str, float | None] = {}
                    for doc, score in results:
                        cid = getattr(doc, "id", None) or (doc.metadata or {}).get("chunk_id")
                        if cid:
                            ids.append(cid)
                            distance_map[cid] = finite_number(score)
                    return results, distance_map, ids
                except Exception:
                    # 回退：使用底层 Chroma collection 查询距离
                    try:
                        collection = self.vectorstore._collection  # type: ignore[attr-defined]
                        # 使用预先计算的查询向量（若不可用则实时计算）
                        emb = query_embedding if query_embedding is not None else await self.embedding_service.aembed_query(rewritten_query)
                        payload = await asyncio.to_thread(
                            collection.query,
                            query_embeddings=[emb],
                            n_results=50,
                            include=["distances", "documents", "metadatas", "embeddings"],
                            where=metadata_filter,
                        )
                        # 对齐到 ids 列表长度，逐项越界保护：collection.query 在
                        # 个别 Chroma 版本/边界情况下可能返回 documents 为 None 或短 list，
                        # 旧实现 payload["documents"][0][i] 直接索引会 IndexError 吞掉回退路径。
                        ids_arr = (payload.get("ids", [[]]) or [[]])
                        docs_arr = (payload.get("documents", [[]]) or [[]])
                        metas_arr = (payload.get("metadatas", [[]]) or [[]])
                        dists_arr = (payload.get("distances", [[]]) or [[]])
                        ids_inner = ids_arr[0] if ids_arr else []
                        docs_inner = docs_arr[0] if docs_arr else []
                        metas_inner = metas_arr[0] if metas_arr else []
                        dists_inner = dists_arr[0] if dists_arr else []
                        docs: List[Tuple[Document, float | None]] = []
                        distance_map: Dict[str, float | None] = {}
                        ids: List[str] = []
                        for i in range(len(ids_inner)):
                            cid = ids_inner[i]
                            text = docs_inner[i] if i < len(docs_inner) else ""
                            md = (metas_inner[i] if i < len(metas_inner) else {}) or {}
                            dist = dists_inner[i] if i < len(dists_inner) else None
                            raw_distance = finite_number(dist)
                            d = Document(page_content=text, metadata=md)
                            docs.append((d, raw_distance))
                            ids.append(cid)
                            distance_map[cid] = raw_distance
                        return docs, distance_map, ids
                    except Exception:
                        return [], {}, []

            async def _bm25_search_k50() -> Tuple[List[str], Dict[str, float]]:
                if not self.bm25_available or not self.bm25_index:
                    return [], {}
                try:
                    model = self.bm25_index.get("bm25")
                    ids = self.bm25_index.get("ids") or []
                    bm25_ids, bm25_scores = rank_lexical_candidates(
                        model, ids, rewritten_query, chunk_map=self.bm25_chunk_map, limit=50)

                    # 应用动态元数据过滤（通过底层元数据获取）
                    if metadata_filter:
                        try:
                            collection = self.vectorstore._collection  # type: ignore[attr-defined]
                            payload = await asyncio.to_thread(collection.get, ids=bm25_ids, include=["metadatas"])
                            ids_list = payload.get("ids", []) or []
                            metas_list = payload.get("metadatas", []) or []
                            filtered_ids: List[str] = []
                            filtered_scores: Dict[str, float] = {}
                            for i in range(len(ids_list)):
                                cid = ids_list[i]
                                md = metas_list[i] or {}
                                if _matches_metadata_filter(md, metadata_filter):
                                    filtered_ids.append(cid)
                                    filtered_scores[cid] = bm25_scores.get(cid, 0.0)
                            bm25_ids = filtered_ids
                            bm25_scores = filtered_scores
                        except Exception as e:
                            logger.warning(f"BM25 过滤应用失败，禁用本次关键词路径：{type(e).__name__}")
                            return [], {}

                    return bm25_ids, bm25_scores
                except Exception:
                    return [], {}

            (_vector_docs_distances, vector_distance_map, vector_ids), (bm25_ids, bm25_score_map) = await asyncio.gather(
                _vector_search_k50(),
                _bm25_search_k50(),
            )

            if not vector_ids and not bm25_ids:
                logger.warning("(async) 未检索到任何结果")
                self.last_run = {"rewritten_query": rewritten_query, "retrieved_chunks": []}
                return {"rewritten_query": rewritten_query, "context": "", "retrieved_chunks": []}

            # 步骤3：RRF融合（k=60），支持纯向量回退
            lists_to_fuse: List[List[str]] = []
            if vector_ids:
                lists_to_fuse.append(vector_ids)
            if bm25_ids:
                lists_to_fuse.append(bm25_ids)
            fused_ids, rrf_scores = reciprocal_rank_fusion(lists_to_fuse, k=60, top_n=None)

            # 步骤4：重取 Top-K 文档块
            top_k = settings.MAX_RETRIEVED_CHUNKS
            top_ids = fused_ids[: top_k]
            top_docs = await asyncio.to_thread(self._refetch_chunks_by_ids, top_ids)

            # 构建上下文字符串（纯CPU操作，直接在协程内执行）
            context_parts: List[str] = []
            for i, doc in enumerate(top_docs, 1):
                metadata_info = []
                if doc.metadata.get('document_title'):
                    metadata_info.append(f"文档标题: {doc.metadata['document_title']}")
                if doc.metadata.get('product_name'):
                    metadata_info.append(f"产品名称: {doc.metadata['product_name']}")
                if doc.metadata.get('effective_date'):
                    metadata_info.append(f"生效日期: {doc.metadata['effective_date']}")
                if doc.metadata.get('document_type'):
                    metadata_info.append(f"文档类型: {doc.metadata['document_type']}")
                metadata_str = " | ".join(metadata_info) if metadata_info else "无元数据"
                context_parts.append(f"【相关文档{i}】\n{metadata_str}\n内容: {doc.page_content}\n")
            final_context = "\n".join(context_parts)

            # 审计输出：结构化片段（添加 vector_score / bm25_score / rrf_score）
            retrieved_chunks: List[Dict[str, Any]] = []
            try:
                for doc in top_docs:
                    md = doc.metadata or {}
                    cid = md.get("chunk_id")
                    raw_distance = vector_distance_map.get(cid)
                    if raw_distance is None:
                        # BM25-only candidates need evidence in the same embedding space.
                        raw_distance = cosine_distance(query_embedding, md.get("embedding"))
                    sim = cosine_distance_to_score(raw_distance)
                    md["raw_distance"] = raw_distance
                    md["vector_score"] = sim
                    md["bm25_score"] = bm25_score_map.get(cid)
                    md["rrf_score"] = rrf_scores.get(cid)
                    rd = md.get("ranking_details", {})
                    retrieved_chunks.append({
                        "chunk_id": md.get("chunk_id"),
                        "document_id": md.get("document_id"),
                        "document_name": md.get("document_title") or md.get("filename") or md.get("source"),
                        "content": doc.page_content,
                        "page_number": md.get("page_number"),
                        "similarity_score": sim,
                        "metadata": {
                            "ranking_details": rd,
                            "document_type": md.get("document_type"),
                            "product_name": md.get("product_name"),
                            "effective_date": md.get("effective_date"),
                            "expiry_date": md.get("expiry_date"),
                            "abolition_date": md.get("abolition_date"),
                            "keywords_json": md.get("keywords_json"),
                            "raw_distance": raw_distance,
                            "evidence_score": sim,
                            "score_metric": "cosine",
                            "evidence_status": "available" if raw_distance is not None else "unavailable",
                            "vector_score": sim,
                            "bm25_score": md.get("bm25_score"),
                            "rrf_score": md.get("rrf_score"),
                        },
                    })
                self.last_run = {
                    "rewritten_query": rewritten_query,
                    "retrieved_chunks": retrieved_chunks,
                }
            except Exception:
                self.last_run = {
                    "rewritten_query": rewritten_query,
                    "retrieved_chunks": [],
                }

            return {
                "rewritten_query": rewritten_query,
                "context": final_context,
                "retrieved_chunks": retrieved_chunks,
            }
        except Exception:
            logger.exception("(async) 构建上下文失败")
            self.last_run = {"rewritten_query": user_query, "retrieved_chunks": []}
            return {"rewritten_query": user_query, "context": "", "retrieved_chunks": []}

    async def _published_context(self, user_query, document_ids):
        coordinator = self.index_coordinator
        # API-level explicit-scope validation is task 5. Capture this compatibility
        # scope once; publication while encoding cannot enlarge an accepted request.
        ids = document_ids if document_ids is not None else await asyncio.to_thread(coordinator.ready_ids)
        if not ids:
            return {"rewritten_query": user_query, "context": "", "retrieved_chunks": []}
        profile = self.embedding_service.embedding_profile().to_record()
        view = await asyncio.to_thread(coordinator.open_view, ids, profile)
        try:
            vector = await self.embedding_service.aembed_query(user_query)
            chunks = await asyncio.to_thread(view.search, user_query, vector, top_k=settings.MAX_RETRIEVED_CHUNKS)
            view.assert_valid()
            self.last_run = {"index_commit_id": view.commit_id, "document_ids": view.document_ids,
                             "chunk_ids": [chunk["chunk_id"] for chunk in chunks]}
            return {"rewritten_query": user_query,
                    "context": "\n\n".join(f"[{i}] {c['document_name']} p.{c['page_number']}\n{c['content']}"
                                             for i, c in enumerate(chunks, 1)),
                    "retrieved_chunks": chunks, "index_commit_id": view.commit_id, "_index_view": view}
        except BaseException:
            view.close()
            raise

    def set_bm25_resources(self, index_payload: Any, chunk_map: Dict[str, str]) -> None:
        try:
            self.bm25_index = index_payload
            self.bm25_chunk_map = chunk_map or {}
            self.bm25_available = bool(
                isinstance(index_payload, dict)
                and index_payload.get("lexical_profile") == lexical_profile())
            if index_payload and not self.bm25_available:
                logger.warning("BM25 lexical profile missing/incompatible; rebuild keyword resources")
            logger.info("BM25 资源已注入到 RAGWorkflow")
        except Exception as e:
            logger.warning(f"BM25 资源注入失败: {e}")

    def _refetch_chunks_by_ids(self, chunk_ids: List[str]) -> List[Document]:
        try:
            collection = self.vectorstore._collection  # type: ignore[attr-defined]
            # (FIX-1.2) 获取预存嵌入向量，避免在排序阶段重复计算
            payload = collection.get(ids=chunk_ids, include=["documents", "metadatas", "embeddings"])
            # 对齐到 ids 长度，逐项越界保护：Chroma 在 chunk_id 被删除或返回子集时
            # 可能给出 documents=None 或短于 ids 的列表，导致 [i] 越界 IndexError
            # 进而让整个 abuild_context 进 except 返回空 context（误拒答）。
            ids_list = payload.get("ids", []) or []
            docs_list = payload.get("documents", []) or []
            metas_list = payload.get("metadatas", []) or []
            embs_list = payload.get("embeddings")
            if embs_list is None:
                embs_list = []
            out: List[Document] = []
            for i in range(len(ids_list)):
                text = docs_list[i] if i < len(docs_list) else None
                md = (metas_list[i] if i < len(metas_list) else {}) or {}
                try:
                    emb = embs_list[i] if i < len(embs_list) else None
                except Exception:
                    emb = None
                # (FIX-1.3) 可选：将嵌入向量附加到元数据，供排序使用
                if emb is not None:
                    try:
                        md["embedding"] = emb
                    except Exception:
                        pass
                md["chunk_id"] = ids_list[i]
                out.append(Document(page_content=text or "", metadata=md))
            return out
        except Exception:
            return []
    


# 创建全局智能代理实例
def get_agent() -> InsurIntellectAgent:
    """
    获取InsurIntellect智能代理实例
    
    Returns:
        InsurIntellectAgent: 智能代理实例
    """
    if not hasattr(get_agent, '_instance') or get_agent._instance is None:
        try:
            get_agent._instance = InsurIntellectAgent()
        except Exception as e:
            logger.exception("InsurIntellectAgent初始化失败")
            # 不缓存失败的实例,下次调用时重新尝试
            get_agent._instance = None
            raise
    return get_agent._instance


def reset_agent():
    """重置智能代理实例, 强制重新初始化"""
    if hasattr(get_agent, '_instance'):
        get_agent._instance = None


def set_bm25_resources(index_payload: Any, chunk_map: Dict[str, str]) -> None:
    agent = get_agent()
    agent.set_bm25_resources(index_payload, chunk_map)


if __name__ == "__main__":
    # 手工冒烟入口：直接检索 + RRF 融合（不跑 long-running agent 链）
    import asyncio

    agent = InsurIntellectAgent()
    test_query = "什么是车险的免赔额？"
    try:
        result = asyncio.run(agent.abuild_context(test_query))
    except Exception as exc:  # pragma: no cover - 仅本地手动跑
        logger.error("abuild_context failed: %s", exc, exc_info=True)
        raise
    logger.info("用户问题: %s", test_query)
    logger.info("检索结果 chunks: %d", len(result.get("retrieved_chunks", [])) if isinstance(result, dict) else 0)


