"""Identity-based document library; disk files never establish membership/readiness."""

import asyncio
import json

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.database_models import Document
from app.models.schemas import DocumentListResponse, DocumentResponse
from app.services.document_storage import SourceUnavailable

router = APIRouter()


def ingestion_service(request):
    return request.app.state.document_ingestion


def project_document(doc, ready_ids):
    return DocumentResponse(
        document_id=doc.id, name=doc.original_name or doc.filename,
        processing_status=doc.processing_status, deletion_status=doc.deletion_status,
        ready=doc.id in ready_ids, page_count=doc.page_count, language=doc.language,
        file_size=doc.file_size, attempt=doc.attempt, error_code=doc.error_code,
        error_message=doc.error_message, error_pages=json.loads(doc.error_pages_json or "[]"),
        file_url=(f"/api/v1/documents/{doc.id}/file"
                  if doc.deletion_status == "none" and doc.status == "active" else None),
    )


@router.get("", response_model=DocumentListResponse)
@router.get("/", response_model=DocumentListResponse, include_in_schema=False)
async def list_documents(request: Request, db: AsyncSession = Depends(get_db)):
    service = ingestion_service(request)
    ready_ids = set(await asyncio.to_thread(service.coordinator.ready_ids))
    docs = (await db.execute(select(Document).order_by(Document.id))).scalars().all()
    count = sum(d.deletion_status != "deleted" for d in docs)
    return DocumentListResponse(
        documents=[project_document(d, ready_ids) for d in docs],
        upload_max_bytes=settings.UPLOAD_MAX_BYTES, max_pages=200, design_capacity=50,
        capacity_warning="超过 50 份设计规模，仍可继续上传。" if count > 50 else None,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def document_detail(document_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Unknown document ID")
    ready_ids = set(await asyncio.to_thread(ingestion_service(request).coordinator.ready_ids))
    return project_document(doc, ready_ids)


@router.post("", response_model=DocumentResponse, status_code=202)
async def upload_document(request: Request, response: Response, file: UploadFile = File(...),
                          db: AsyncSession = Depends(get_db)):
    service = ingestion_service(request)
    try:
        if file.content_type not in {"application/pdf", "application/octet-stream"}:
            raise HTTPException(415, "Only PDF uploads are supported")
        if file.size is not None and file.size > service.storage.max_bytes:
            raise HTTPException(413, "File exceeds upload size limit")
        header = await file.read(5)
        await file.seek(0)
        if header != b"%PDF-":
            raise HTTPException(422, "Invalid PDF signature")
        item = await asyncio.to_thread(service.submit, file.file, file.filename or "document.pdf")
        # Failed retries have an existing ID but create a new queued attempt.
        response.status_code = 202 if item.accepted else 200
        return await document_detail(item.document_id, request, db)
    except SourceUnavailable as exc:
        raise HTTPException(422, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(413 if "size limit" in str(exc) else 422, str(exc)) from None
    finally:
        await file.close()


@router.get("/{document_id}/file")
async def document_file(document_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "Unknown document ID")
    if doc.deletion_status != "none" or doc.status != "active":
        raise HTTPException(410, "Source is deleted or unavailable")
    try:
        path = await asyncio.to_thread(ingestion_service(request).storage.original_path, document_id)
    except SourceUnavailable:
        raise HTTPException(410, "Original source unavailable") from None
    return FileResponse(path, media_type="application/pdf", filename=doc.original_name or doc.filename,
                        headers={"Cache-Control": "no-store"}, content_disposition_type="inline")
