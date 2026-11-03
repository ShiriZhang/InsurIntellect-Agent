"""Deletion contracts and fault recovery use only temporary source libraries."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import threading
import time

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.database_models import Document, DocumentChunk, QueryHistory, SessionScope, DeletionTask
from app.services.document_deletion import DocumentDeletionService
from app.services.index_coordinator import IndexCoordinator
from app.services.document_storage import SourceUnavailable
from test_scoped_query_api import scoped, ready, ask, Generator  # noqa: F401
from test_document_api import api  # noqa: F401
from test_index_coordinator import library, publish, candidate  # noqa: F401


def deletion(api):
    service = DocumentDeletionService(api.ingestion)
    api.app.state.document_deletion = service
    return service


def finish(service, document_id):
    for _ in range(30):
        service.drain()
        if service.snapshot(document_id)['status']=='completed':
            return
        time.sleep(.05)
    assert service.snapshot(document_id)['status']=='completed', service.snapshot(document_id)


def test_delete_contract_identity_history_and_reupload(scoped):
    a=ready(scoped); b=ready(scoped,'b.pdf','Waiting period is 180 days.')
    original=scoped.ingestion.storage.original_path(a).read_bytes()
    result=ask(scoped,[a,b],mode='compare').json()
    before=scoped.client.get(f"/api/v1/queries/history/{result['query_id']}").json()
    service=deletion(scoped)
    response=scoped.client.delete(f'/api/v1/documents/{a}')
    assert response.status_code==202 and response.json()['deletion_status']=='deleting'
    assert ask(scoped,[a]).status_code==422
    pending_history=scoped.client.get(f"/api/v1/queries/history/{result['query_id']}").json()
    assert all('content' not in c for c in pending_history['metadata']['retrieved_chunks'] if c['document_id']==a)
    assert scoped.client.get(f'/api/v1/documents/{a}/file').status_code==410
    assert ask(scoped,[b]).status_code==200
    finish(service,a)
    assert scoped.client.delete(f'/api/v1/documents/{a}').status_code==200
    after=scoped.client.get(f"/api/v1/queries/history/{result['query_id']}").json()
    assert after['answer']==before['answer'] and after['source_status']=='deleted'
    chunks=after['metadata']['retrieved_chunks']
    invalid=[c for c in chunks if c['document_id']==a]
    assert invalid and all(c['source_status']=='deleted' and 'content' not in c and 'metadata' not in c for c in invalid)
    assert any(c.get('content') for c in chunks if c['document_id']==b)
    with Session(scoped.ingestion.engine) as db:
        assert not list(db.scalars(select(DocumentChunk).where(DocumentChunk.document_id==a)))
        assert db.get(Document,a).file_path is None
        assert db.get(QueryHistory,result['query_id']).rewritten_query is None
    for bucket in scoped.ingestion.coordinator.vectors.collections.values():
        assert all(r['metadata']['document_id']!=a for r in bucket.values())
    assert not (scoped.ingestion.storage.root / str(a)).exists()
    upload=scoped.client.post('/api/v1/documents',files={'file':('policy.pdf',original,'application/pdf')})
    assert upload.status_code==202 and upload.json()['document_id']!=a
    assert scoped.client.delete('/api/v1/documents/999999').status_code==404


@pytest.mark.parametrize('stage',DocumentDeletionService.stages)
def test_failure_at_each_cleanup_stage_never_reports_success_and_recovers(scoped,stage):
    a=ready(scoped); b=ready(scoped,'b.pdf','Waiting period is 180 days.')
    service=deletion(scoped)
    service.begin(a)
    def fail(current):
        if current==stage:
            raise OSError('simulated crash with private text that must not be stored')
    service.fault=fail
    service.drain()
    status=service.snapshot(a)
    assert status['status']=='failed' and status['deletion_status']=='deleting'
    assert status['error']=='OSError'
    assert ask(scoped,[a]).status_code==422
    restarted=DocumentDeletionService(scoped.ingestion)
    restarted.recover()
    assert restarted.snapshot(a)['status']=='completed'
    assert ask(scoped,[b]).status_code==200


@pytest.mark.parametrize("native_stage", ["parser", "encoder"])
def test_native_ingestion_late_result_cannot_restore_source(library, native_stage):
    ingestion,coordinator,root=library
    ingestion.coordinator=coordinator
    started,release=threading.Event(),threading.Event()
    original_parser=ingestion.parser
    def blocked(*args,**kwargs):
        parsed=original_parser(*args,**kwargs)
        started.set(); release.wait(5)
        return parsed
    if native_stage == 'parser':
        ingestion.parser=blocked
    else:
        from ingestion_fakes import LocalEncoderFake
        class BlockedEncoder(LocalEncoderFake):
            def embed_documents(self, texts):
                started.set(); release.wait(5)
                return super().embed_documents(texts)
        ingestion.embedding_factory=BlockedEncoder
    from test_pdf_quality import make_pdf
    path=make_pdf(root/'queued.pdf',['Waiting period is 90 days.'])
    item=ingestion.submit(BytesIO(path.read_bytes()),path.name)
    service=DocumentDeletionService(ingestion)
    with ThreadPoolExecutor() as pool:
        worker=pool.submit(ingestion.run_until_idle)
        assert started.wait(3)
        service.begin(item.document_id)
        service.drain()
        assert service.snapshot(item.document_id)['status']=='pending'
        release.set(); worker.result(5)
    finish(service,item.document_id)
    assert ingestion.snapshot(item.document_id)['task_status']=='cancelled'
    with Session(ingestion.engine) as db:
        assert not list(db.scalars(select(DocumentChunk)))


@pytest.mark.parametrize('stream',[False,True])
def test_delete_interrupts_blocked_generation_and_keeps_unrelated_queries(scoped,stream):
    a=ready(scoped); b=ready(scoped,'b.pdf','Waiting period is 180 days.')
    service=deletion(scoped)
    started=threading.Event(); closed=threading.Event()
    class Blocked(Generator):
        async def agenerate_response(self,q,chunks):
            import asyncio
            started.set()
            try: await asyncio.sleep(30)
            finally: closed.set()
        async def agenerate_stream(self,q,chunks):
            import asyncio
            started.set()
            try:
                await asyncio.sleep(30)
                yield 'late stale token'
            finally: closed.set()
    scoped.processor.llm_factory=Blocked
    with ThreadPoolExecutor() as pool:
        future=pool.submit(ask,scoped,[a,b],mode='compare',stream=stream)
        assert started.wait(3)
        service.begin(a)
        response=future.result(3)
    assert closed.is_set()
    if stream:
        data=next(json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ') and json.loads(line[6:])['type']=='end')
    else:
        data=response.json()
    assert data['answer_kind']=='cancelled' and data['source_status']=='deleted'
    assert data['document_ids']==[a,b] and not data['retrieved_chunks']
    assert 'late stale' not in data['answer']
    finish(service,a)
    scoped.processor.llm_factory=Generator
    assert ask(scoped,[b]).status_code==200


def test_restored_completed_tombstone_replays_original_and_history_cleanup(scoped):
    a=ready(scoped)
    response=ask(scoped,[a]).json()
    path=scoped.ingestion.storage.original_path(a)
    original=path.read_bytes()
    service=deletion(scoped); service.begin(a); finish(service,a)
    path.parent.mkdir(exist_ok=True); path.write_bytes(original)
    with Session(scoped.ingestion.engine) as db:
        doc=db.get(Document,a)
        doc.status='active'; doc.deletion_status='none'; doc.file_path=str(path)
        row=db.get(QueryHistory,response['query_id'])
        row.retrieved_chunks=json.dumps([{'document_id':a,'content':'restored old source','metadata':{'embedding':[1,2]}}])
        db.commit()
    restarted=DocumentDeletionService(scoped.ingestion)
    restarted.recover()
    assert restarted.snapshot(a)['status']=='completed' and not path.exists()
    assert scoped.client.get(f'/api/v1/documents/{a}/file').status_code==410
    with Session(scoped.ingestion.engine) as db:
        assert 'restored old source' not in db.get(QueryHistory,response['query_id']).retrieved_chunks


def test_old_snapshot_reader_defers_completion_and_clears_related_memory(scoped):
    a=ready(scoped); b=ready(scoped,'b.pdf','Waiting period is 180 days.')
    coordinator=scoped.ingestion.coordinator
    view=coordinator.open_current_view([b])
    service=deletion(scoped); service.begin(a); service.drain()
    assert service.snapshot(a)['status']=='pending'
    assert view.search('waiting period',[1.,.5,.25])
    view.close(); finish(service,a)
    for path in coordinator.snapshots.root.glob('*/v-*.json'):
        assert str(a) not in {str(pair[0]) for pair in json.loads(path.read_text())['document_attempts']}



def test_cross_process_reader_marker_blocks_success_until_released(scoped):
    from app.services.ingestion_worker_lock import IngestionWorkerLock
    from pathlib import Path
    a=ready(scoped)
    coordinator=scoped.ingestion.coordinator
    marker=IngestionWorkerLock(Path(coordinator.database_key+f'.request-{a}-external.lock'))
    assert marker.acquire()
    service=deletion(scoped)
    try:
        service.begin(a); service.drain()
        assert service.snapshot(a)['status']=='pending'
    finally:
        marker.release()
    finish(service,a)


def test_recovered_old_index_and_owned_temporary_files_cannot_revive_source(scoped):
    from copy import deepcopy
    from app.models.database_models import IndexCommit, IndexPointer
    a=ready(scoped)
    coordinator=scoped.ingestion.coordinator
    with Session(scoped.ingestion.engine) as db:
        version=db.get(IndexPointer,'current').commit_id
        row=db.get(IndexCommit,version)
        profile=json.loads(row.embedding_profile_json)
        snapshot_path=row.snapshot_path
        collection=row.vector_collection
    snapshot=coordinator.snapshots.path(snapshot_path).read_bytes()
    rows=deepcopy(list(coordinator.vectors.collections[collection].values()))
    temporary=coordinator.snapshots.path(snapshot_path).parent/(coordinator.snapshots.temp_prefix+'failure')
    temporary.write_text('private source temporary copy')
    service=deletion(scoped); service.begin(a); finish(service,a)
    assert not temporary.exists()
    restored=coordinator.snapshots.path(snapshot_path)
    restored.parent.mkdir(parents=True,exist_ok=True); restored.write_bytes(snapshot)
    coordinator.vectors.write(profile,rows,collection=collection,owner=coordinator.library_id)
    with Session(scoped.ingestion.engine) as db:
        db.get(IndexCommit,version).retired=False
        db.get(IndexPointer,'current').commit_id=version
        doc=db.get(Document,a)
        doc.status='active'; doc.deletion_status='none'; doc.processing_status='ready'
        doc.published_attempt=1; doc.published_fingerprint=profile['fingerprint']; doc.index_commit_id=version
        db.commit()
    service.recover()
    assert service.snapshot(a)['status']=='completed'
    assert coordinator.ready_ids()==[]
    assert all(not bucket for bucket in coordinator.vectors.collections.values())
    for path in coordinator.snapshots.root.glob('*/v-*.json'):
        assert not json.loads(path.read_text())['chunks']


def test_cleanup_never_needs_local_model_or_generation(scoped):
    a=ready(scoped)
    def unavailable():
        raise AssertionError('deletion must not load a model or call generation')
    scoped.ingestion.embedding_factory=unavailable
    scoped.processor.embedding_factory=unavailable
    scoped.processor.llm_factory=unavailable
    service=deletion(scoped); service.begin(a); finish(service,a)


def test_changed_original_root_does_not_report_erase_or_touch_external_files(scoped):
    a=ready(scoped)
    external=scoped.root/'external.pdf'; external.write_bytes(b'keep external bytes')
    with Session(scoped.ingestion.engine) as db:
        db.get(Document,a).file_path=str(external); db.commit()
    service=deletion(scoped); service.begin(a); service.drain()
    assert service.snapshot(a)['status']=='failed'
    assert external.read_bytes()==b'keep external bytes'
    assert scoped.client.get(f'/api/v1/documents/{a}/file').status_code==410
