"""Loopback browser harness with a fresh managed library, real services by default."""
from contextlib import contextmanager
import socket
import threading
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
import uvicorn
from app.api.endpoints import documents, queries, corpus
from app.core.database import get_db
from app.services.document_storage import DocumentStorage
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_deletion import DocumentDeletionService
from app.services.index_coordinator import IndexCoordinator
from app.services.embedding_service import EmbeddingService
from app.services.scoped_query import ScopedQueryProcessor

ROOT = Path(__file__).resolve().parents[1]

@contextmanager
def serve_library(root, *, embedding_factory=EmbeddingService, processor=None, vectors=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{(root/'library.db').as_posix()}")
    coordinator = IndexCoordinator(engine, vector_root=root/'vectors', processed_root=root/'processed', vectors=vectors)
    ingestion = DocumentIngestionService(DocumentStorage(engine,root/'pdfs'),embedding_factory=embedding_factory,coordinator=coordinator)
    ingestion.initialize()
    coordinator.recover()
    deletion = DocumentDeletionService(ingestion)
    deletion.recover()
    async_engine = create_async_engine(f"sqlite+aiosqlite:///{(root/'library.db').as_posix()}")
    sessions = async_sessionmaker(async_engine)
    async def db():
        async with sessions() as session: yield session
    app = FastAPI()
    app.state.document_ingestion = ingestion
    app.state.document_deletion = deletion
    app.state.index_coordinator = coordinator
    app.dependency_overrides[get_db] = db
    app.include_router(documents.router,prefix='/api/v1/documents')
    app.include_router(queries.router,prefix='/api/v1/queries')
    app.include_router(corpus.router,prefix='/api/v1/corpus')
    @app.get('/')
    async def index(): return FileResponse(ROOT/'static/index.html')
    @app.get('/api/v1/health/')
    async def health(): return {'status':'healthy'}
    app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
    previous = queries.query_service.scoped
    queries.query_service.scoped = processor or ScopedQueryProcessor()
    sock = socket.socket(); sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app,log_level='error',lifespan='off'))
    thread = threading.Thread(target=lambda:server.run(sockets=[sock]),daemon=True)
    ingestion.start(); deletion.start(); thread.start()
    try:
        deadline=time.monotonic()+10
        while not server.started and time.monotonic()<deadline: time.sleep(.02)
        if not server.started: raise RuntimeError('browser harness failed to start')
        yield f'http://127.0.0.1:{port}', ingestion, deletion
    finally:
        server.should_exit=True; thread.join(10)
        deletion.stop(); ingestion.stop(); sock.close()
        import asyncio
        asyncio.run(async_engine.dispose())
        queries.query_service.scoped=previous
        engine.dispose()
