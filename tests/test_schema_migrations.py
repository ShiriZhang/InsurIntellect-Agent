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
        assert conn.execute(text("SELECT count(*) FROM schema_migrations")).scalar() == 1
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
            assert (await db.execute(text("SELECT count(*) FROM schema_migrations"))).scalar() == 1
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
