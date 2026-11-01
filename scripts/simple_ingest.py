#!/usr/bin/env python3
"""Lightweight sample ingest: PDF -> chunks -> Chroma + BM25 (no OCR/LLM metadata)."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Plus

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_text import tokenize_retrieval, lexical_profile

ROOT = Path(__file__).resolve().parents[1]


def _extract_pages(pdf_path: Path) -> List[tuple[int, str]]:
    import fitz

    doc = fitz.open(pdf_path)
    pages: List[tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        text = (page.get_text("text") or "").strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def _chunk_pages(pdf_path: Path, pages: List[tuple[int, str]]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(180, int(settings.CHUNK_SIZE) if settings.CHUNK_SIZE else 400),
        chunk_overlap=max(20, int(settings.CHUNK_OVERLAP) if settings.CHUNK_OVERLAP else 40),
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    docs: List[Document] = []
    for page_number, text in pages:
        for idx, piece in enumerate(splitter.split_text(text)):
            piece = piece.strip()
            if not piece:
                continue
            key = f"{pdf_path.name}|{page_number}|{idx}|{piece[:80]}"
            chunk_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "chunk_id": chunk_id,
                        "document_id": pdf_path.stem,
                        "document_title": pdf_path.name,
                        "filename": pdf_path.name,
                        "source": str(pdf_path),
                        "page_number": page_number,
                        "document_type": "条款",
                        "product_name": "演示样本",
                        "effective_date": "2024-01-01",
                    },
                )
            )
    return docs


def ingest(pdf_dir: Path, persist_dir: Path, collection: str, reset: bool) -> Dict[str, Any]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {pdf_dir}")

    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    all_docs: List[Document] = []
    for pdf in pdfs:
        pages = _extract_pages(pdf)
        chunks = _chunk_pages(pdf, pages)
        print(f"{pdf.name}: pages={len(pages)} chunks={len(chunks)}")
        all_docs.extend(chunks)

    if not all_docs:
        raise SystemExit("No extractable text chunks")

    emb = EmbeddingService()
    vs = Chroma(
        collection_name=collection,
        collection_metadata={"hnsw:space": settings.EMBEDDING_DISTANCE},
        embedding_function=emb.embedding_function,
        persist_directory=str(persist_dir),
    )
    # batch add
    batch = 32
    for i in range(0, len(all_docs), batch):
        part = all_docs[i : i + batch]
        ids = [d.metadata["chunk_id"] for d in part]
        vs.add_documents(part, ids=ids)
        print(f"embedded {min(i + batch, len(all_docs))}/{len(all_docs)}")

    # BM25
    tokenized = [tokenize_retrieval(d.page_content) for d in all_docs]
    bm25 = BM25Plus(tokenized)
    chunk_map = {
        d.metadata["chunk_id"]: {
            "content": d.page_content,
            "metadata": d.metadata,
        }
        for d in all_docs
    }
    processed = Path(settings.PROCESSED_DATA_PATH)
    processed.mkdir(parents=True, exist_ok=True)
    with (processed / "bm25_index.pkl").open("wb") as f:
        pickle.dump({"bm25": bm25, "ids": [d.metadata["chunk_id"] for d in all_docs],
                     "lexical_profile": lexical_profile()}, f)
    with (processed / "bm25_chunk_map.json").open("w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)

    # corpus manifest for UI (Chinese display names for sample files)
    display_names = {
        "sample_term_life.pdf": "示例终身寿险条款",
        "sample_critical_illness.pdf": "示例重大疾病保险条款",
    }
    manifest = {
        "documents": [
            {
                "name": p.name,
                "path": str(p),
                "pages": len(_extract_pages(p)),
                "display_name": display_names.get(p.name, p.stem),
            }
            for p in pdfs
        ],
        "chunk_count": len(all_docs),
        "collection": collection,
    }
    with (processed / "corpus_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {"docs": len(pdfs), "chunks": len(all_docs), "persist": str(persist_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple local ingest for demo PDFs")
    parser.add_argument("--pdf-dir", default=str(ROOT / "data" / "documents" / "pdfs"))
    parser.add_argument("--persist-dir", default=settings.CHROMA_PERSIST_DIRECTORY)
    parser.add_argument("--collection", default="insurance_documents")
    parser.add_argument("--reset", action="store_true", help="Wipe existing chroma dir first")
    args = parser.parse_args()

    result = ingest(Path(args.pdf_dir), Path(args.persist_dir), args.collection, args.reset)
    print("ingest-ok", result)


if __name__ == "__main__":
    main()
