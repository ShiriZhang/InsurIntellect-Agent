import asyncio

from app.core.rag_workflow import InsurIntellectAgent
from ingestion_fakes import LocalEncoderFake
from test_index_coordinator import library, publish  # noqa: F401


def test_live_rag_reads_new_ready_sources_without_restart_and_releases_full_snapshots(library):
    _, coordinator, _ = library
    first, _ = publish(library)
    class QueryModel(LocalEncoderFake):
        async def aembed_query(self, text):
            return [1., .5, .25]
    agent = InsurIntellectAgent.__new__(InsurIntellectAgent)
    agent.embedding_service = QueryModel()
    agent.index_coordinator = coordinator
    async def scenario():
        initial = await agent.abuild_context('waiting period', [first.document_id])
        second, _ = publish(library, 'B')
        current = await agent.abuild_context('waiting period', [second.document_id])
        assert {c['document_id'] for c in initial['retrieved_chunks']} == {first.document_id}
        assert {c['document_id'] for c in current['retrieved_chunks']} == {second.document_id}
        assert not coordinator.state.readers
        assert 'retrieved_chunks' not in agent.last_run  # No global full-source request cache.
        initial['_index_view'].close()
        current['_index_view'].close()
    asyncio.run(scenario())
