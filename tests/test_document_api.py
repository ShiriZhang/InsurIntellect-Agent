"""API contracts use a temporary library and an offline encoder stand-in."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import Session

from app.api.endpoints import corpus, documents, queries
from app.core.database import get_db
from app.models.database_models import Document
from test_index_coordinator import library  # noqa: F401
from test_pdf_quality import make_pdf


@pytest.fixture
def api(library):
    ingestion, coordinator, root = library
    ingestion.coordinator = coordinator
    engine = create_async_engine(str(ingestion.engine.url).replace('sqlite:', 'sqlite+aiosqlite:'))
    sessions = async_sessionmaker(engine)
    async def db():
        async with sessions() as session:
            yield session
    app = FastAPI()
    app.state.document_ingestion = ingestion
    app.state.index_coordinator = coordinator
    app.dependency_overrides[get_db] = db
    app.include_router(documents.router, prefix='/api/v1/documents')
    app.include_router(corpus.router, prefix='/api/v1/corpus')
    app.include_router(queries.router, prefix='/api/v1/queries')
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, ingestion=ingestion, root=root)
    asyncio.run(engine.dispose())


def test_upload_identity_preview_duplicates_and_orphans(api):
    c = api.client
    (api.ingestion.storage.root / 'orphan.pdf').write_bytes(b'%PDF-orphan')
    assert c.get('/api/v1/corpus').json()['documents'] == []
    originals = [make_pdf(api.root / f'{i}.pdf', [f'Waiting period is {i + 90} days.']).read_bytes()
                 for i in range(2)]
    ids = []
    for content in originals:
        result = c.post('/api/v1/documents', files={'file': ('same.pdf', content, 'application/pdf')})
        assert result.status_code == 202
        doc = result.json()
        assert not doc['ready']
        ids.append(doc['document_id'])
        assert c.get(doc['file_url']).content == content
        assert c.get(doc['file_url']).headers['cache-control'] == 'no-store'
    assert ids[0] != ids[1]
    duplicate = c.post('/api/v1/documents', files={'file': ('other.pdf', originals[0], 'application/pdf')})
    assert duplicate.status_code == 200
    assert duplicate.json()['document_id'] == ids[0]
    assert len(c.get('/api/v1/corpus').json()['documents']) == 2
    assert c.get('/api/v1/documents/9999').status_code == 404
    assert c.get('/api/v1/documents/9999/file').status_code == 404
    with Session(api.ingestion.engine) as db:
        db.get(Document, ids[0]).deletion_status = 'deleting'
        db.commit()
    assert c.get(f'/api/v1/documents/{ids[0]}/file').status_code == 410


def test_invalid_uploads_and_public_limits(api):
    c = api.client
    assert c.post('/api/v1/documents', files={'file': ('a.txt', b'text', 'text/plain')}).status_code == 415
    assert c.post('/api/v1/documents', files={'file': ('a.pdf', b'broken', 'application/pdf')}).status_code == 422
    api.ingestion.storage.max_bytes = 8
    assert c.post('/api/v1/documents', files={'file': ('a.pdf', b'%PDF-oversized', 'application/pdf')}).status_code == 413
    limits = c.get('/api/v1/documents').json()
    assert limits['max_pages'] == 200 and limits['design_capacity'] == 50
    assert limits['upload_max_bytes'] > 0


def test_failed_retry_is_accepted_and_over_capacity_is_advisory(api):
    c = api.client
    content = make_pdf(api.root / 'retry.pdf', ['The waiting period is 90 days.']).read_bytes()
    first = c.post('/api/v1/documents', files={'file': ('retry.pdf', content, 'application/pdf')}).json()
    with Session(api.ingestion.engine) as db:
        db.get(Document, first['document_id']).processing_status = 'failed'
        for i in range(50):
            db.add(Document(filename=f'{i}.pdf', processing_status='pending'))
        db.commit()
    retried = c.post('/api/v1/documents', files={'file': ('retry.pdf', content, 'application/pdf')})
    assert retried.status_code == 202 and retried.json()['attempt'] == 2
    assert retried.json()['document_id'] == first['document_id']
    assert c.get('/api/v1/documents').json()['capacity_warning']
