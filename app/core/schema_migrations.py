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
    if connection.execute(text("SELECT 1 FROM schema_migrations WHERE version = 1")).first():
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
    connection.execute(text("INSERT INTO schema_migrations(version) VALUES (1)"))
