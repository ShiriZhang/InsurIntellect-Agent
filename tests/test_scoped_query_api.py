"""Temporary-library contract regressions; no real model/API quality claim."""
import json

import pytest
from sqlalchemy.orm import Session

from app.api.endpoints import queries
from app.core.config import settings
from app.models.database_models import Document, QueryHistory
from app.services.scoped_query import ScopedQueryProcessor
from ingestion_fakes import LocalEncoderFake
from test_document_api import api  # noqa: F401
from test_index_coordinator import library  # noqa: F401
from test_pdf_quality import make_pdf


class QueryEncoder(LocalEncoderFake):
    query_calls = 0

    def embed_query(self, question):
        self.query_calls += 1
        return [1., .5, .25]


class Generator:
    contexts = []

    def has_credentials(self):
        return True

    async def agenerate_response(self, query, chunks):
        self.contexts.append((query, chunks))
        return {'answer': 'Facts from the selected documents [1].', 'success': True}

    async def agenerate_stream(self, query, chunks):
        self.contexts.append((query, chunks))
        yield 'Facts from the selected documents [1].'


@pytest.fixture
def scoped(api, monkeypatch):
    processor = ScopedQueryProcessor(QueryEncoder, Generator)
    monkeypatch.setattr(queries.query_service, 'scoped', processor)
    Generator.contexts = []
    api.processor = processor
    return api


def ready(api, name='policy.pdf', text='The waiting period is 90 days except accidents.', pages=None):
    path = make_pdf(api.root / name, pages or [text])
    response = api.client.post('/api/v1/documents', files={'file': (name, path.read_bytes(), 'application/pdf')})
    assert response.status_code == 202
    assert api.ingestion.run_until_idle()
    doc_id = response.json()['document_id']
    assert api.client.get(f'/api/v1/documents/{doc_id}').json()['ready']
    return doc_id


def ask(api, ids, *, question='What is the waiting period?', stream=False, mode='qa'):
    return api.client.post('/api/v1/queries/ask', json={
        'question': question, 'document_ids': ids, 'stream': stream, 'query_mode': mode})


@pytest.mark.parametrize('endpoint', ['ask', 'batch'])
@pytest.mark.parametrize('scope', ['missing', None, [], [0], [True], ['1']])
def test_scope_is_required_and_strict_before_model_or_stream(scoped, endpoint, scope):
    body = {'question': '等待期多久', 'questions': ['等待期多久'], 'stream': True}
    if scope != 'missing':
        body['document_ids'] = scope
    response = scoped.client.post(f'/api/v1/queries/{endpoint}', json=body)
    assert response.status_code == 422
    assert scoped.processor.embedding is None and not Generator.contexts


@pytest.mark.parametrize('state', ['unknown', 'queued', 'failed', 'deleting', 'deleted'])
@pytest.mark.parametrize('endpoint', ['ask', 'batch'])
def test_one_invalid_source_rejects_entire_request(scoped, state, endpoint):
    first = ready(scoped)
    other = 999
    if state != 'unknown':
        other = ready(scoped, 'other.pdf', 'The waiting period is 180 days.')
        with Session(scoped.ingestion.engine) as db:
            doc = db.get(Document, other)
            if state in {'deleting', 'deleted'}:
                doc.deletion_status = state
            else:
                doc.processing_status = state
            db.commit()
    response = scoped.client.post(f'/api/v1/queries/{endpoint}', json={
        'question': '等待期多久', 'questions': ['等待期多久'], 'document_ids': [first, other], 'stream': True})
    assert response.status_code == 422
    assert other in response.json()['detail']['document_ids']
    assert not Generator.contexts and scoped.processor.embedding is None


def test_deduplicated_scope_and_structured_batch(scoped):
    doc = ready(scoped)
    response = ask(scoped, [doc, doc])
    assert response.status_code == 200
    assert response.json()['document_ids'] == [doc]
    response = scoped.client.post('/api/v1/queries/batch', json={
        'questions': ['What is the waiting period?', '等待期多久？'], 'document_ids': [doc, doc], 'query_mode': 'qa'})
    assert response.status_code == 200
    assert all(r['document_ids'] == [doc] for r in response.json())


def test_new_ready_visible_and_unknown_scope_never_uses_disk(scoped):
    a = ready(scoped)
    assert ask(scoped, [a]).status_code == 200
    b = ready(scoped, 'other.pdf', 'The waiting period is 180 days.')
    response = ask(scoped, [b]).json()
    assert {c['document_id'] for c in response['retrieved_chunks']} == {b}


@pytest.mark.parametrize('stream', [False, True])
def test_modes_override_ambiguity_and_comparison_limit_before_sse(scoped, stream):
    ids = [ready(scoped, f'{i}.pdf', f'Policy {i}: waiting period is {90+i} days.') for i in range(6)]
    assert ask(scoped, ids, question='Compare waiting periods', mode='auto', stream=stream).status_code == 422
    response = ask(scoped, ids, question='Compare waiting periods', mode='qa', stream=stream)
    assert response.status_code == 200
    ambiguous = ask(scoped, ids[:2], question='这些等待期呢？', mode='auto', stream=stream)
    if stream:
        assert '"answer_kind": "clarification"' in ambiguous.text
    else:
        assert ambiguous.json()['answer_kind'] == 'clarification'
    assert ask(scoped, ids[:2], question='这些等待期呢？', mode='compare', stream=stream).status_code == 200


def test_comparison_checks_every_document_and_embeds_question_once(scoped):
    a = ready(scoped, 'a.pdf', pages=[f'Waiting period is 90 days. Condition {i}.' for i in range(12)])
    b = ready(scoped, 'b.pdf', 'Waiting period is 180 days, except accidents.')
    c = ready(scoped, 'c.pdf', 'Premium payments are due annually.')
    vectors = scoped.ingestion.coordinator.vectors
    # Third document has no semantic or lexical support for this question.
    for bucket in vectors.collections.values():
        for row in bucket.values():
            if row['metadata']['document_id'] == c:
                row['embedding'] = [-1., -.5, -.25]
    response = ask(scoped, [a, b, c], mode='compare').json()
    assert response['answer_kind'] == 'answer'
    assert scoped.processor.embedding.query_calls == 1
    assert [r['status'] for r in response['document_coverage']] == ['evidence', 'evidence', 'insufficient_evidence']
    assert {chunk['document_id'] for chunk in response['retrieved_chunks']} == {a, b}
    assert '180 days' in ' '.join(c['content'] for c in response['retrieved_chunks'])
    assert f'文档 {c}' in response['answer'] and '未找到充分依据' in response['answer']
    assert 'insufficient_evidence' in Generator.contexts[0][0]


@pytest.mark.parametrize('stream', [False, True])
def test_budget_does_not_silently_drop_document(scoped, monkeypatch, stream):
    a = ready(scoped)
    b = ready(scoped, 'b.pdf', 'Waiting period is 180 days.')
    monkeypatch.setattr(settings, 'GENERATION_CONTEXT_TOKENS', 10)
    response = ask(scoped, [a, b], mode='compare', stream=stream)
    assert response.status_code == 200
    if stream:
        assert '"answer_kind": "clarification"' in response.text
    else:
        assert response.json()['answer_kind'] == 'clarification'
    assert not Generator.contexts


@pytest.mark.parametrize('failure', ['vector', 'leaky_filter', 'lexical'])
def test_scope_survives_many_distractors_and_single_path_failure(scoped, monkeypatch, failure):
    a = ready(scoped)
    b = ready(scoped, 'distractors.pdf', pages=[f'Waiting period is 180 days. Clause {i}.' for i in range(60)])
    coordinator = scoped.ingestion.coordinator
    if failure == 'vector':
        coordinator.vectors.fail_query = True
    elif failure == 'leaky_filter':
        original = coordinator.vectors.query
        def leaking(profile, vector, keys, limit, **kw):
            return original(profile, vector, {f'{a}:1', f'{b}:1'}, limit, **kw)
        monkeypatch.setattr(coordinator.vectors, 'query', leaking)
    else:
        monkeypatch.setattr('app.services.index_request_view.rank_lexical_candidates',
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('BM25 failed')))
    response = ask(scoped, [a]).json()
    assert response['answer_kind'] == 'answer'
    assert {c['document_id'] for c in response['retrieved_chunks']} == {a}
    assert {c['document_id'] for c in Generator.contexts[0][1]} == {a}


@pytest.mark.parametrize('stream', [False, True])
def test_final_citation_order_matches_generation_and_history(scoped, monkeypatch, stream):
    doc = ready(scoped, pages=['Waiting period is 90 days.', 'Accidents are not subject to this period.'])
    vectors = scoped.ingestion.coordinator.vectors
    original = vectors.get
    monkeypatch.setattr(vectors, 'get', lambda *a, **kw: dict(reversed(list(original(*a, **kw).items()))))
    response = ask(scoped, [doc], stream=stream)
    if stream:
        payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        data = next(p for p in payloads if p['type'] == 'end')
    else:
        data = response.json()
    chunks = data['retrieved_chunks']
    assert [c['chunk_id'] for c in chunks] == [c['chunk_id'] for c in Generator.contexts[0][1]]
    assert [c['metadata']['citation_number'] for c in chunks] == list(range(1, len(chunks)+1))
    with Session(scoped.ingestion.engine) as db:
        history = db.get(QueryHistory, data['query_id'])
        assert json.loads(history.retrieved_chunks) == chunks


@pytest.mark.parametrize('question,mode', [
    ('这几份的等待期分别多久', 'compare'), ('Compare the waiting periods', 'compare'),
    ('How does A differ from B?', 'compare'), ('解释等待期是什么意思', 'qa'),
    ('What is a waiting period?', 'qa'), ('这些期限呢？', 'clarification'),
])
def test_bilingual_mode_resolution(question, mode):
    from app.models.schemas import QueryRequest
    from app.services.query_mode import resolve_mode
    assert resolve_mode(QueryRequest(question=question, document_ids=[1, 2])) == mode


def test_all_retrieval_paths_failed_is_unavailable_not_no_evidence(scoped, monkeypatch):
    doc = ready(scoped)
    scoped.ingestion.coordinator.vectors.fail_query = True
    monkeypatch.setattr('app.services.index_request_view.rank_lexical_candidates',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('failed')))
    assert ask(scoped, [doc]).status_code == 503
    assert not Generator.contexts


def test_scope_invalidated_during_retrieval_cannot_generate(scoped, monkeypatch):
    doc_id = ready(scoped)
    vectors = scoped.ingestion.coordinator.vectors
    original = vectors.query
    def invalidating(*a, **kw):
        rows = original(*a, **kw)
        with Session(scoped.ingestion.engine) as db:
            db.get(Document, doc_id).deletion_status = 'deleting'
            db.commit()
        return rows
    monkeypatch.setattr(vectors, 'query', invalidating)
    assert ask(scoped, [doc_id]).status_code == 422
    assert not Generator.contexts


def test_exact_paired_source_keys_and_reversed_refetch(scoped, monkeypatch):
    a = ready(scoped)
    b = ready(scoped, 'b.pdf', 'Waiting period is 180 days.')
    vectors = scoped.ingestion.coordinator.vectors
    original = vectors.query
    observed = []
    def recording(profile, vector, source_keys, limit, **kw):
        observed.append(set(source_keys))
        return original(profile, vector, source_keys, limit, **kw)
    monkeypatch.setattr(vectors, 'query', recording)
    assert ask(scoped, [a, b], mode='compare').status_code == 200
    assert observed == [{f'{a}:1'}, {f'{b}:1'}]


def test_batch_checks_late_comparison_limit_before_first_question(scoped):
    ids = [ready(scoped, f'{i}.pdf', f'Policy {i}: waiting period is 90 days.') for i in range(6)]
    response = scoped.client.post('/api/v1/queries/batch', json={
        'questions': ['Explain waiting periods', 'Compare waiting periods'], 'document_ids': ids})
    assert response.status_code == 422 and not Generator.contexts


@pytest.mark.parametrize('stream', [False, True])
def test_existing_explicit_off_topic_boundary_keeps_sources_empty(scoped, stream):
    doc = ready(scoped)
    response = ask(scoped, [doc], question='今天北京天气怎么样？', stream=stream)
    if stream:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        data = next(event for event in events if event['type'] == 'end')
    else:
        data = response.json()
    assert data['answer_kind'] == 'refusal' and not data['retrieved_chunks']
    assert not Generator.contexts


@pytest.mark.parametrize('question', [
    'Explain what each policy year means.', 'Explain the term comparison rate.',
    'Which exclusion clause applies to death by suicide?',
    'Explain the difference between a deductible and a co-payment.',
    'What is the difference between a deductible and a co-payment?',
    '解释免赔额和共付额的区别。',
])
def test_ordinary_explanations_do_not_trigger_comparison_limit(question):
    from app.models.schemas import QueryRequest
    from app.services.query_mode import resolve_mode
    assert resolve_mode(QueryRequest(question=question, document_ids=list(range(1, 7)))) == 'qa'


@pytest.mark.parametrize('question', [
    'Explain the differences between these policies.',
    'Explain the waiting period in each selected policy.',
    '解释这几份文档各自的等待期。',
])
def test_document_directed_comparison_takes_priority_over_explanation(question):
    from app.models.schemas import QueryRequest
    from app.services.query_mode import resolve_mode
    assert resolve_mode(QueryRequest(question=question, document_ids=[1, 2])) == 'compare'
