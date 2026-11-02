from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.document_chunking import ChunkBudgetError, chunk_document
from app.services.embedding_profile import InputPolicy
from app.services.pdf_quality import PDFPage, ParsedPDF


def budget_model(**overrides):
    config = Settings(_env_file=None, CHUNK_SIZE=120, CHUNK_OVERLAP=20, **overrides)
    return SimpleNamespace(input_policy=InputPolicy.from_settings(config), max_input_tokens=100,
                           count_tokens=lambda text: len(text) + 10)  # Prefix/special-token overhead.


def test_long_english_keeps_negations_conditions_and_every_character():
    text = "1. Waiting period\n" + ("Payment is excluded unless the accident occurs after inception. " * 25)
    parsed = ParsedPDF((PDFPage(7, text),), "en")
    model = budget_model()
    chunks = chunk_document(parsed, 12, 3, model)
    assert len(chunks) > 5
    assert "".join(c.content for c in chunks) == text
    assert all(model.count_tokens(c.embedding_text) <= 100 for c in chunks)
    assert all(c.page_number == 7 for c in chunks)
    assert all(c.section_title == "1. Waiting period" for c in chunks)
    assert all(c.context_before for c in chunks[1:])
    for c in chunks:
        assert text[c.source_start:c.source_end] == c.content


def test_chinese_clauses_cross_page_blank_page_and_deterministic_identity():
    pages = (PDFPage(1, "第一条 保险责任\n非意外事故等待期九十日。\n第二条 除外责任\n酒后驾驶不予赔付。"),
             PDFPage(2, ""), PDFPage(3, "续页条件：除合同另有约定，不予赔付。"))
    parsed = ParsedPDF(pages, "zh")
    model = budget_model()
    chunks = chunk_document(parsed, 8, 1, model)
    assert chunks == chunk_document(parsed, 8, 1, model)
    assert {c.vector_id for c in chunks}.isdisjoint(c.vector_id for c in chunk_document(parsed, 8, 2, model))
    assert {c.vector_id for c in chunks}.isdisjoint(c.vector_id for c in chunk_document(parsed, 9, 1, model))
    assert [c.page_number for c in chunks] == [1, 1, 3]
    assert chunks[-1].section_title == "第二条 除外责任"
    assert "不予赔付" in chunks[-1].content
    for p in pages:
        assert "".join(c.content for c in chunks if c.page_number == p.number) == p.text


def test_generation_byte_budget_and_profile_track_chunk_policy():
    model = budget_model(GENERATION_CONTEXT_TOKENS=100, GENERATION_PROMPT_RESERVE_TOKENS=50,
                         GENERATION_CHUNK_MAX_BYTES=80)
    parsed = ParsedPDF((PDFPage(1, "等待期不适用意外事故。" * 20),), "zh")
    chunks = chunk_document(parsed, 1, 1, model)
    assert "".join(c.content for c in chunks) == parsed.pages[0].text
    assert all(len(c.embedding_text.encode("utf-8")) <= 50 for c in chunks)
    profile = model.input_policy.profile_fields(100)
    assert profile["chunking"]["version"] == "page-clause-token-v1"
    assert profile["chunking"]["generation_chunk_bytes"] == 50


def test_impossible_budget_fails_instead_of_truncating():
    model = budget_model(GENERATION_CONTEXT_TOKENS=10, GENERATION_PROMPT_RESERVE_TOKENS=10)
    with pytest.raises(ChunkBudgetError, match="reserve_exhausted"):
        chunk_document(ParsedPDF((PDFPage(1, "Valid text"),), "en"), 1, 1, model)
