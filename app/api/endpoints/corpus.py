"""Compatibility projection of the authoritative document library."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.api.endpoints.documents import list_documents

router = APIRouter()


@router.get("")
@router.get("/", include_in_schema=False)
async def list_corpus(request: Request, db: AsyncSession = Depends(get_db)):
    library = await list_documents(request, db)
    return {
        **library.model_dump(),
        "documents": [
            {**doc.model_dump(), "id": doc.document_id, "display_name": doc.name,
             "pages": doc.page_count, "url": doc.file_url}
            for doc in library.documents
        ],
    }
