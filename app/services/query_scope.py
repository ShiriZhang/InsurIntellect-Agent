"""Validate the whole explicit selection before any model or stream work."""

from fastapi import HTTPException
from sqlalchemy import select

from app.models.database_models import DeletionTask, Document


async def validate_scope(db, request):
    ids = request.document_ids
    docs = {d.id: d for d in (await db.execute(select(Document).where(Document.id.in_(ids)))).scalars()}
    deleting = set((await db.execute(select(DeletionTask.document_id).where(
        DeletionTask.document_id.in_(ids)))).scalars())
    invalid = [i for i in ids if i not in docs or not docs[i].is_queryable or i in deleting]
    if invalid:
        raise HTTPException(422, {"error": "invalid_document_scope", "document_ids": invalid})
    return {i: docs[i].original_name or docs[i].filename for i in ids}
