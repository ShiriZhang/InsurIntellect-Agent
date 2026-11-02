"""Additive, versioned SQLite migrations; called inside the caller's transaction."""

from sqlalchemy import inspect, text
from sqlalchemy.schema import CreateColumn

from app.core.database import Base


def migrate_schema(connection):
    # Register models for standalone callers as well as application startup.
    import app.models.database_models  # noqa: F401

    if connection.dialect.name != "sqlite":
        raise RuntimeError("Document-library schema migration requires SQLite")
    connection.execute(text(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    ))
    # Force SQLite to acquire its write reservation before deciding which DDL is
    # missing. SQLAlchemy's begin() alone does not start a SQLite legacy-mode
    # transaction for SELECT/DDL, so concurrent CLI/app starts could race ALTERs.
    connection.execute(text("UPDATE schema_migrations SET version = version WHERE 0"))
    applied = set(connection.execute(text("SELECT version FROM schema_migrations")).scalars())
    if 3 in applied:
        return
    Base.metadata.create_all(connection)
    for table in Base.metadata.sorted_tables:
        existing = {column["name"] for column in inspect(connection).get_columns(table.name)}
        for column in table.columns:
            if column.name not in existing:
                ddl = str(CreateColumn(column).compile(dialect=connection.dialect))
                connection.exec_driver_sql(f'ALTER TABLE "{table.name}" ADD COLUMN {ddl}')
        for index in table.indexes:
            index.create(connection, checkfirst=True)
    # A nullable attempt keeps unversioned legacy rows out of this identity constraint.
    connection.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_chunk_position ON document_chunks "
        "(document_id, attempt, page_number, page_chunk_index)"
    )
    if 1 not in applied:
        connection.execute(text("INSERT INTO schema_migrations(version) VALUES (1)"))
    if 2 not in applied:
        connection.execute(text("INSERT INTO schema_migrations(version) VALUES (2)"))
    connection.execute(text("INSERT INTO schema_migrations(version) VALUES (3)"))
