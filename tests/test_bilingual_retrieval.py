"""Shared lexical retrieval and independent cosine gates; all sources are synthetic."""

import asyncio
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.documents import Document
from rank_bm25 import BM25Plus

from app.core import rag_workflow
from app.core.fusion import reciprocal_rank_fusion
from app.services import citation_policy, response_shaping
from app.services.evidence_scores import (
    chunk_evidence_score, cosine_distance, cosine_distance_to_score, normalize_evidence_score,
)
from app.services.retrieval_text import (
    lexical_profile, normalize_retrieval_text, rank_lexical_candidates, tokenize_retrieval,
)


def test_bilingual_normalization_preserves_original_and_case_insensitive_identifiers():
    original = "ＡlphaCare-X2 的 Waiting Period 为９０天。NOT covered unless specified."
    normalized = normalize_retrieval_text(original)
    assert "alphacare-x2" in normalized and "90天" in normalized
    assert "９０" in original  # Only a separate retrieval copy is normalized.
    tokens = tokenize_retrieval(original)
    assert {"alphacare-x2", "waiting", "period", "90", "not", "unless", "term:waiting_period"} <= set(tokens)
    assert tokenize_retrieval("Waiting Period") == tokenize_retrieval("WAITING PERIOD")


def test_numeric_dates_units_product_names_and_negation_survive_tokenization():
    text = "AlphaCare-X2在2026-10-25生效，非意外事故不予赔付，保额10,000.50元，等待期90天。USD -5.25 80% $100 1/2"
    tokens = set(tokenize_retrieval(text))
    assert {"alphacare-x2", "2026-10-25", "非", "不予", "10,000.50", "元", "90", "天",
            "usd", "-5.25", "80%", "$", "100", "1/2"} <= tokens
    assert "not" in tokenize_retrieval("not covered")
    assert "non-accidental" in tokenize_retrieval("non-accidental death")


@pytest.mark.parametrize("zh,en,canonical", [
    ("等待期", "waiting period", "waiting_period"),
    ("责任免除", "exclusions", "exclusion"),
    ("免赔额", "deductible", "deductible"),
    ("重大疾病", "critical illness", "critical_illness"),
])
def test_terms_are_additive_recall_hints(zh, en, canonical):
    marker = "term:" + canonical
    assert marker in tokenize_retrieval(zh)
    assert marker in tokenize_retrieval(en)
    assert zh in tokenize_retrieval(zh)
    assert set(en.split()) <= set(tokenize_retrieval(en))


def test_english_term_does_not_match_inside_product_identifier():
    assert "term:claim" not in tokenize_retrieval("SuperClaimsPlan-X2")
    assert "term:waiting_period" not in tokenize_retrieval("waiting periodical")


def test_real_bm25plus_baseline_cannot_make_nonmatching_chunks_candidates():
    texts = ["本合同等待期为90天，其他合同条件仍须满足。", "Premium payments are due annually."]
    model = BM25Plus([tokenize_retrieval(text) for text in texts])
    query = "waiting period"
    assert model.get_scores(tokenize_retrieval(query))[1] > 0  # BM25Plus baseline is not a match.
    ids, scores = rank_lexical_candidates(model, ["waiting", "premium"], query)
    assert ids == ["waiting"] and set(scores) == {"waiting"}
    assert rank_lexical_candidates(model, ["waiting", "premium"], "unrelatedword")[0] == []


def test_bm25_keyword_profile_is_required_and_stable():
    agent = rag_workflow.InsurIntellectAgent.__new__(rag_workflow.InsurIntellectAgent)
    model = BM25Plus([tokenize_retrieval("等待期"), tokenize_retrieval("保费")])
    payload = {"bm25": model, "ids": ["a", "b"]}
    agent.set_bm25_resources(payload, {})
    assert agent.bm25_available is False
    agent.set_bm25_resources({**payload, "lexical_profile": lexical_profile()}, {})
    assert agent.bm25_available is True
    profile = lexical_profile()
    profile["fingerprint"] = "different"
    assert profile != lexical_profile()
    agent.set_bm25_resources({**payload, "lexical_profile": profile}, {})
    assert agent.bm25_available is False


def test_cosine_distance_mapping_is_monotonic_and_not_similarity_guessing():
    distances = [0, 0.05, 0.2, 0.7, 1, 1.5, 2]
    scores = [cosine_distance_to_score(value) for value in distances]
    assert scores == sorted(scores, reverse=True)
    assert scores == pytest.approx([1, 0.95, 0.8, 0.3, 0, 0, 0])
    assert rag_workflow._normalize_vector_score(0.2) == pytest.approx(0.8)
    assert normalize_evidence_score(0.2) == 0.2
    assert normalize_evidence_score(1000) == 0


@pytest.mark.parametrize("distance", [None, math.nan, math.inf, -math.inf, -5, True, "bad"])
def test_invalid_distance_never_becomes_perfect_evidence(distance):
    assert cosine_distance_to_score(distance) == 0


@pytest.mark.parametrize("vector", [None, [], [0, 0], [math.nan, 1], [1], [True, 1]])
def test_missing_or_invalid_vectors_have_no_semantic_distance(vector):
    assert cosine_distance([1, 0], vector) is None


def test_cosine_vectors_match_distance_score_contract():
    assert cosine_distance([1, 0], [1, 0]) == pytest.approx(0)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1)
    assert cosine_distance([1, 0], [-1, 0]) == pytest.approx(2)
    assert cosine_distance_to_score(cosine_distance([1, 0], [3, 4])) == pytest.approx(0.6)


@pytest.mark.parametrize("metadata", [
    {"rrf_score": 1000, "bm25_score": 99999},
    {"bm25_score": 1000, "ranking_details": {"original_similarity": 1, "final_score": 1}},
    {"raw_distance": 1.2, "vector_score": 1, "rrf_score": 1000},
    {"vector_score": 0, "bm25_score": 1000},
    {"evidence_score": 0, "vector_score": 1, "bm25_score": 1000},
])
def test_ranks_cannot_override_zero_unknown_or_insufficient_evidence(metadata):
    chunk = {"chunk_id": 1, "document_id": 1, "document_name": "test.pdf", "page_number": 1,
             "content": "本合同等待期为90天，不满足约定条件时不承担保险责任。",
             "similarity_score": 0.99, "metadata": metadata}
    assert chunk_evidence_score(chunk) == 0
    assert citation_policy.eligible_evidence([chunk], 0.32) == []
    assert citation_policy.public_citations([chunk]) == []
    assert response_shaping.to_retrieved_chunks([chunk])[0].similarity_score == 0
    assert citation_policy.relevance_bonus("等待期", chunk) > 0  # Ranking hints still do not open the gate.


def test_raw_distance_is_authoritative_for_public_and_gate_scores():
    chunk = {"chunk_id": 1, "document_id": 1, "document_name": "test.pdf", "page_number": 1,
             "content": "本合同等待期为90天，不满足约定条件时不承担保险责任。",
             "similarity_score": 1, "metadata": {"raw_distance": 0.8, "bm25_score": 10000}}
    assert chunk_evidence_score(chunk) == pytest.approx(0.2)
    assert response_shaping.to_retrieved_chunks([chunk])[0].similarity_score == pytest.approx(0.2)
    assert citation_policy.eligible_evidence([chunk], 0.32) == []
    chunk["metadata"]["raw_distance"] = 0
    assert chunk_evidence_score(chunk) == 1


def make_agent(*, primary_distances=None, fallback=False, bm25=False, query_vector=None, vectors=None):
    ids = ["a", "b"]
    text = "本合同等待期为90天，非意外事故未满足合同约定条件时，不承担保险责任。"
    sources = {cid: {"content": text + cid, "metadata": {"document_id": i + 1,
                "document_title": cid + ".pdf", "page_number": 2}} for i, cid in enumerate(ids)}
    vectors = vectors or {"a": [1, 0], "b": [0.1, 1]}
    primary_distances = primary_distances or {}

    class Collection:
        def get(self, ids, include):
            return {"ids": ids, "documents": [sources[c]["content"] for c in ids],
                    "metadatas": [dict(sources[c]["metadata"]) for c in ids],
                    "embeddings": np.array([vectors[c] for c in ids])}

        def query(self, **kwargs):
            selected = list(primary_distances)
            return {"ids": [selected], "documents": [[sources[c]["content"] for c in selected]],
                    "metadatas": [[dict(sources[c]["metadata"]) for c in selected]],
                    "distances": [[primary_distances[c] for c in selected]]}

    class Store:
        _collection = Collection()

        def similarity_search_with_score(self, query, **kwargs):
            if fallback:
                raise RuntimeError("Injected primary vector path failure")
            return [(Document(page_content=sources[c]["content"], metadata={"chunk_id": c}), d)
                    for c, d in primary_distances.items()]

    async def embed_query(query):
        return [1, 0] if query_vector is None else query_vector

    agent = rag_workflow.InsurIntellectAgent.__new__(rag_workflow.InsurIntellectAgent)
    agent.embedding_service = SimpleNamespace(aembed_query=embed_query, provider="test_injected")
    agent.vectorstore = Store()
    if bm25:
        class KeywordModel:
            doc_freqs = [dict.fromkeys(tokenize_retrieval(sources[c]["content"]), 1) for c in ids]

            def get_scores(self, tokens):
                return [1000, 1000000]

        agent.set_bm25_resources({"bm25": KeywordModel(), "ids": ids,
                                 "lexical_profile": lexical_profile()}, sources)
    else:
        agent.bm25_available, agent.bm25_index, agent.bm25_chunk_map = False, None, {}
    return agent, sources


@pytest.mark.parametrize("fallback", [False, True])
def test_primary_and_fallback_vector_paths_apply_the_same_distance_conversion(monkeypatch, fallback):
    monkeypatch.setattr(rag_workflow.settings, "ENABLE_QUERY_REWRITING", False)
    agent, sources = make_agent(primary_distances={"a": 0.15, "b": 0.85}, fallback=fallback)
    context = asyncio.run(agent.abuild_context("等待期多久？"))
    chunks = {c["chunk_id"]: c for c in context["retrieved_chunks"]}
    assert chunks["a"]["similarity_score"] == pytest.approx(0.85)
    assert chunks["b"]["similarity_score"] == pytest.approx(0.15)
    assert chunks["a"]["metadata"]["raw_distance"] == 0.15
    assert chunks["a"]["content"] == sources["a"]["content"]


def test_bm25_only_candidates_use_same_space_vectors_instead_of_keyword_scores(monkeypatch):
    monkeypatch.setattr(rag_workflow.settings, "ENABLE_QUERY_REWRITING", False)
    agent, _ = make_agent(bm25=True)
    chunks = asyncio.run(agent.abuild_context("等待期多久？"))["retrieved_chunks"]
    by_id = {c["chunk_id"]: c for c in chunks}
    assert by_id["b"]["metadata"]["bm25_score"] == 1000000
    assert by_id["b"]["similarity_score"] == pytest.approx(0.1 / math.hypot(0.1, 1))
    assert by_id["a"]["similarity_score"] == 1
    assert [c["chunk_id"] for c in citation_policy.eligible_evidence(chunks, 0.32)] == ["a"]


def test_keyword_result_with_missing_vector_is_not_semantic_evidence(monkeypatch):
    monkeypatch.setattr(rag_workflow.settings, "ENABLE_QUERY_REWRITING", False)
    agent, _ = make_agent(bm25=True, vectors={"a": [0, 0], "b": [0, 0]})
    chunks = asyncio.run(agent.abuild_context("等待期多久？"))["retrieved_chunks"]
    assert len(chunks) == 2
    assert all(c["metadata"]["evidence_status"] == "unavailable" for c in chunks)
    assert citation_policy.eligible_evidence(chunks, 0.32) == []


@pytest.mark.parametrize("stream", [False, True])
def test_high_bm25_low_evidence_is_refused_before_either_generation_transport(monkeypatch, stream):
    from app.services.query_service import QueryService
    from app.models.schemas import QueryRequest

    agent, _ = make_agent(bm25=True, vectors={"a": [0.1, 1], "b": [0.1, 1]})
    service = QueryService()
    service._agent = agent
    monkeypatch.setattr(service, "_spawn_bg", lambda coro: coro.close())
    monkeypatch.setattr(rag_workflow.settings, "ENABLE_QUERY_REWRITING", False)
    monkeypatch.setattr(rag_workflow.settings, "SIMPLE_RAG_MODE", True)

    async def run():
        response = await service.process_query(None, QueryRequest(question="等待期多久？"), stream=stream)
        if not stream:
            assert response.answer_kind == "refusal"
            assert response.retrieved_chunks == []
        else:
            messages = [line async for line in response]
            events = [json.loads(line[6:].strip()) for line in messages if line.startswith("data: ")]
            end = next(event for event in events if event.get("type") == "end")
            assert end["answer_kind"] == "refusal" and end["retrieved_chunks"] == []
            assert not any(event.get("type") == "token" for event in events)

    asyncio.run(run())


def test_rrf_changes_only_ranking_and_does_not_create_an_evidence_score():
    ids, ranks = reciprocal_rank_fusion([["b", "a"], ["b", "a"]])
    assert ids == ["b", "a"] and ranks["b"] > ranks["a"]
    chunk = {"similarity_score": ranks["b"], "metadata": {"rrf_score": ranks["b"]}}
    assert chunk_evidence_score(chunk) == 0


@pytest.mark.parametrize("stream", [False, True])
def test_insufficient_keyword_candidate_never_enters_generation_context(monkeypatch, stream):
    from app.services import query_service
    from app.models.schemas import QueryRequest

    agent, _ = make_agent(bm25=True)
    service = query_service.QueryService()
    service._agent = agent
    contexts = []

    class Generator:
        async def agenerate_response(self, query, context_chunks):
            contexts.append(context_chunks)
            return {"answer": "【结论】等待期为90天。", "success": True}

        async def agenerate_stream(self, query, context_chunks):
            contexts.append(context_chunks)
            yield "【结论】等待期为90天。"

    async def save(*args, **kwargs):
        return 1

    monkeypatch.setattr(service, "_has_llm_credentials", lambda: True)
    monkeypatch.setattr(service, "_spawn_bg", lambda coro: coro.close())
    monkeypatch.setattr(service, "_save_query_history", save)
    monkeypatch.setattr(query_service.LLMService, "with_model", lambda *a, **kw: Generator())
    monkeypatch.setattr(rag_workflow.settings, "ENABLE_QUERY_REWRITING", False)
    monkeypatch.setattr(rag_workflow.settings, "SIMPLE_RAG_MODE", True)

    async def run():
        response = await service.process_query(None, QueryRequest(question="等待期多久？"), stream=stream)
        if stream:
            messages = [line async for line in response]
            events = [json.loads(line[6:].strip()) for line in messages if line.startswith("data: ")]
            response = next(event for event in events if event.get("type") == "end")
            assert response["confidence_score"] == 1
        else:
            assert response.answer_kind == "answer" and response.confidence_score == 1

    asyncio.run(run())
    assert len(contexts) == 1
    assert [c["chunk_id"] for c in contexts[0]] == ["a"]


def test_cli_ingestion_prepares_shared_source_text_without_legacy_publication(tmp_path):
    from scripts import simple_ingest
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    from app.models.database_models import DocumentChunk
    from app.services.document_ingestion import DocumentIngestionService
    from app.services.document_storage import DocumentStorage
    from ingestion_fakes import LocalEncoderFake
    from test_pdf_quality import make_pdf

    source = tmp_path / "source"
    source.mkdir()
    original = "AlphaCare-X2 等待期90天；生效日2026-10-25，非意外事故不予赔付。"
    make_pdf(source / "sample.pdf", ["", original])
    engine = create_engine(f"sqlite:///{tmp_path / 'cli.db'}")
    service = DocumentIngestionService(DocumentStorage(engine, tmp_path / "pdfs"), embedding_factory=LocalEncoderFake)
    try:
        result = simple_ingest.ingest(source, service=service)
        assert result["prepared"] and not result["ready"]
        with Session(engine) as db:
            chunk = db.scalars(select(DocumentChunk)).one()
            assert chunk.page_number == 2 and original in chunk.content
            assert "term:waiting_period" in tokenize_retrieval(chunk.content)
    finally:
        service.stop()
        engine.dispose()
