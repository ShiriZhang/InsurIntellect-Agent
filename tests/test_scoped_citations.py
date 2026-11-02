from types import SimpleNamespace

from app.core.rag_workflow import InsurIntellectAgent
from app.services.citation_policy import curate_citations, public_citations
from app.services.llm_service import LLMService


def test_same_name_same_page_and_shared_prefix_preserve_distinct_conditions():
    prefix = 'The waiting period does not apply to accidents except '
    chunks = [
        {'document_id': 1, 'chunk_id': 'a', 'document_name': 'same.pdf', 'page_number': 1,
         'content': prefix + 'specified injuries.', 'similarity_score': .8},
        {'document_id': 1, 'chunk_id': 'b', 'document_name': 'same.pdf', 'page_number': 1,
         'content': prefix + 'excluded activities.', 'similarity_score': .9},
        {'document_id': 2, 'chunk_id': 'c', 'document_name': 'same.pdf', 'page_number': 1,
         'content': prefix + 'specified injuries.', 'similarity_score': .99},
        {'document_id': 1, 'chunk_id': 'd', 'document_name': 'same.pdf', 'page_number': 1,
         'content': 'No cover for war.', 'similarity_score': .8},
    ]
    selected = curate_citations(chunks + [chunks[0]], limit=10)
    assert [c['chunk_id'] for c in selected] == ['a', 'b', 'c', 'd']
    assert public_citations(selected) == selected
    prompt = LLMService()._build_user_prompt('Compare exclusions', selected)
    assert '[1]' in prompt and '[4]' in prompt
    assert '文档 ID: 1' in prompt and '文档 ID: 2' in prompt


def test_legacy_refetch_restores_requested_fusion_order():
    class Collection:
        def get(self, **kwargs):
            return {'ids': ['b', 'a'], 'documents': ['B', 'A'], 'metadatas': [{}, {}], 'embeddings': [[1, 0], [0, 1]]}
    agent = InsurIntellectAgent.__new__(InsurIntellectAgent)
    agent.vectorstore = SimpleNamespace(_collection=Collection())
    result = agent._refetch_chunks_by_ids(['a', 'b'])
    assert [doc.page_content for doc in result] == ['A', 'B']
    assert [doc.metadata['chunk_id'] for doc in result] == ['a', 'b']
