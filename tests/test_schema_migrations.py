import asyncio

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from app.core.schema_migrations import migrate_schema
from app.models.database_models import Document, QueryHistory


def test_legacy_schema_migrates_without_inventing_readiness_or_scope(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT NOT NULL, "
                             "status TEXT, product_name TEXT)")
        conn.exec_driver_sql("INSERT INTO documents VALUES (7, 'old.pdf', 'active', NULL)")
        conn.exec_driver_sql("INSERT INTO documents VALUES (8, 'archived.pdf', 'archived', NULL)")
        conn.exec_driver_sql("CREATE TABLE query_history (id INTEGER PRIMARY KEY, query TEXT NOT NULL, "
                             "response TEXT)")
        conn.exec_driver_sql("INSERT INTO query_history VALUES (1, 'q', 'keep answer')")
        conn.exec_driver_sql("CREATE TABLE document_chunks (id INTEGER PRIMARY KEY, "
                             "document_id INTEGER NOT NULL, content TEXT NOT NULL)")
        conn.exec_driver_sql("INSERT INTO document_chunks VALUES (1, 7, 'keep source')")
        migrate_schema(conn)
        migrate_schema(conn)
        assert conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar() == 3
        assert "deletion_tasks" in inspect(conn).get_table_names()
    with Session(engine) as db:
        doc = db.get(Document, 7)
        assert doc.status == "active" and doc.processing_status == "pending"
        assert not doc.is_queryable and doc.published_attempt is None
        assert doc.product_name is None and doc.effective_date is None
        assert db.get(Document, 8).status == "archived"
        history = db.get(QueryHistory, 1)
        assert history.response == "keep answer"
        assert history.scope_status == "unknown" and history.document_ids_json is None
        assert history.scope_epoch is None and history.scope_fingerprint is None
    engine.dispose()


def test_startup_and_history_compatibility_entry_points(tmp_path, monkeypatch):
    async def scenario():
        from app.core import database
        from app.services.query_history_service import ensure_query_history_columns

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'init.db'}")
        monkeypatch.setattr(database, "async_engine", engine)
        await database.init_db()
        await database.init_db()
        async with async_sessionmaker(engine)() as db:
            await ensure_query_history_columns(db)
            assert (await db.execute(text("SELECT count(*) FROM schema_migrations"))).scalar() == 3
        await engine.dispose()
    asyncio.run(scenario())


def test_unknown_history_excluded_from_followups(tmp_path):
    async def scenario():
        from app.services.query_service import QueryService

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'history.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(migrate_schema)
        async with async_sessionmaker(engine)() as db:
            db.add_all([
                QueryHistory(query="legacy secret", session_id="s"),
                QueryHistory(query="deleted secret", session_id="s", scope_status="known",
                             scope_fingerprint="fp", scope_epoch=1, source_status="deleted"),
                QueryHistory(query="other scope", session_id="s", scope_status="known",
                             scope_fingerprint="other", scope_epoch=2, source_status="active"),
                QueryHistory(query="valid", session_id="s", scope_status="known",
                             scope_fingerprint="fp", scope_epoch=1, source_status="active"),
            ])
            await db.commit()
            service = QueryService.__new__(QueryService)
            assert await service._get_chat_history(db, "s") == ""
            assert await service._get_chat_history(db, "s", scope_fingerprint="fp", scope_epoch=1) == "用户: valid"
            assert len((await db.scalars(select(QueryHistory))).all()) == 4
        await engine.dispose()
    asyncio.run(scenario())


def test_version_one_database_receives_candidate_checkpoint_columns(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'version-one.db'}")
    with engine.begin() as conn:
        migrate_schema(conn)
        conn.exec_driver_sql("DELETE FROM schema_migrations WHERE version >= 2")
        for name in ("embedding_json", "context_before", "source_start", "source_end"):
            conn.exec_driver_sql(f"ALTER TABLE document_chunks DROP COLUMN {name}")
        for name in ("embedding_profile_json", "diagnostics_json", "expected_chunks"):
            conn.exec_driver_sql(f"ALTER TABLE ingestion_attempts DROP COLUMN {name}")
        conn.exec_driver_sql("INSERT INTO documents(id, filename, status) VALUES (10, 'keep.pdf', 'active')")
        migrate_schema(conn)
        migrate_schema(conn)
        assert "embedding_json" in {c["name"] for c in inspect(conn).get_columns("document_chunks")}
        assert "expected_chunks" in {c["name"] for c in inspect(conn).get_columns("ingestion_attempts")}
    with Session(engine) as db:
        assert db.get(Document, 10).filename == 'keep.pdf'
        assert not db.get(Document, 10).is_queryable
    engine.dispose()


def test_version_two_commit_without_fingerprint_resources_is_not_adopted(tmp_path):
    from app.models.database_models import IndexCommit, IndexPointer
    from app.services.index_coordinator import IndexCoordinator, IndexUnavailable
    import pytest

    engine = create_engine(f"sqlite:///{tmp_path / 'version-two.db'}")
    with engine.begin() as conn:
        migrate_schema(conn)
        conn.exec_driver_sql("DELETE FROM schema_migrations WHERE version = 3")
        for name in ('snapshot_sha256', 'embedding_profile_json', 'vector_root', 'processed_root', 'library_id', 'retired'):
            conn.exec_driver_sql(f"ALTER TABLE index_commits DROP COLUMN {name}")
        conn.exec_driver_sql("INSERT INTO index_commits(id, embedding_fingerprint, document_attempts_json, vector_collection, snapshot_path) "
                             "VALUES (1, 'unverified', '[]', 'insurance_documents', 'legacy-manifest.json')")
        conn.exec_driver_sql("INSERT INTO index_pointer(name, commit_id) VALUES ('current', 1)")
        migrate_schema(conn)
        migrate_schema(conn)
    with Session(engine) as db:
        assert db.get(IndexPointer, 'current').commit_id == 1
        assert db.get(IndexCommit, 1).embedding_profile_json is None
        assert db.get(IndexCommit, 1).retired is False
    coordinator = IndexCoordinator(engine, vector_root=tmp_path / 'vectors', processed_root=tmp_path / 'processed')
    with pytest.raises(IndexUnavailable):
        coordinator.recover()
    assert not coordinator.state.serving
    engine.dispose()


def test_concurrent_application_and_cli_upgrade_is_serialized(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, BrokenBarrierError
    from sqlalchemy import event

    engine = create_engine(f"sqlite:///{tmp_path / 'concurrent.db'}", connect_args={'timeout': 5})
    with engine.begin() as conn:
        migrate_schema(conn)
        conn.exec_driver_sql('DELETE FROM schema_migrations WHERE version=3')
        for name in ('snapshot_sha256', 'embedding_profile_json', 'vector_root', 'processed_root', 'library_id', 'retired'):
            conn.exec_driver_sql(f'ALTER TABLE index_commits DROP COLUMN {name}')
    # Old migration allows both callers to inspect missing columns before either
    # ALTER. With a write reservation only one can reach this barrier, then the
    # other inspects the fully upgraded schema and never attempts the same DDL.
    both_altering = Barrier(2)
    def interleave(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('ALTER TABLE "index_commits" ADD COLUMN snapshot_sha256'):
            try:
                both_altering.wait(timeout=.2)
            except BrokenBarrierError:
                pass
    event.listen(engine, 'before_cursor_execute', interleave)
    def upgrade(_):
        with engine.begin() as conn:
            migrate_schema(conn)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(upgrade, range(2)))
        with engine.connect() as conn:
            assert list(conn.execute(text('SELECT version FROM schema_migrations ORDER BY version')).scalars()) == [1, 2, 3]
    finally:
        event.remove(engine, 'before_cursor_execute', interleave)
        engine.dispose()
