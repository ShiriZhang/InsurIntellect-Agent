#!/usr/bin/env python3
"""Register PDFs through the shared durable ingestion service; never reset a library."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.document_ingestion import DocumentIngestionService, TERMINAL_ATTEMPTS
from app.services.pdf_quality import parse_pdf


def _extract_pages(pdf_path):
    """Compatibility projection for offline corpus/annotation inspection."""
    return [(page.number, page.text.strip()) for page in parse_pdf(pdf_path).pages if page.text.strip()]


def ingest(pdf_dir, *, service=None, wait=True, timeout=300):
    pdfs = sorted(path for path in Path(pdf_dir).glob("*.pdf") if path.is_file())
    if not pdfs:
        raise ValueError(f"No PDFs in {pdf_dir}")
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    own_service = service is None
    service = service or DocumentIngestionService()
    started_here = False
    try:
        service.initialize()
        submitted = []
        for path in pdfs:
            with path.open("rb") as source:
                result = service.submit(source, path.name)
            submitted.append(result)
        timed_out = False
        if wait:
            already_running = service._thread is not None and service._thread.is_alive()
            started_here = service.start() and not already_running
            deadline = time.monotonic() + timeout
            while True:
                statuses = [service.snapshot(item.document_id) for item in submitted]
                if all(item["task_status"] in TERMINAL_ATTEMPTS for item in statuses):
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                if service.last_error:
                    break
                time.sleep(.05)
        statuses = [service.snapshot(item.document_id) for item in submitted]
        return {"documents": statuses, "timed_out": timed_out,
                "prepared": all(item["task_status"] in {"staged", "ready"} for item in statuses),
                "ready": all(item["ready"] for item in statuses), "waiting": wait}
    finally:
        if started_here:
            service.stop()
        if own_service and (service._thread is None or not service._thread.is_alive()):
            service.engine.dispose()


def main(argv=None, *, service=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "samples")
    parser.add_argument("--no-wait", action="store_true", help="Persist jobs; let the application worker process them")
    parser.add_argument("--timeout", type=float, default=300, help="Maximum CLI wait in seconds; not a performance promise")
    parser.add_argument("--reset", action="store_true", help="Removed: always rejected without deleting files")
    args = parser.parse_args(argv)
    if args.reset:
        parser.error("--reset was removed: it can erase other documents; use the document lifecycle and profile rebuild flow")
    try:
        result = ingest(args.pdf_dir, service=service, wait=not args.no_wait, timeout=args.timeout)
    except (ValueError, OSError) as exc:
        parser.exit(1, "ingest failed: " + type(exc).__name__ + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if args.no_wait or result["prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
