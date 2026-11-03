"""Small cancellable request leases; cancellation also interrupts blocked API awaits."""
import asyncio
from pathlib import Path
from uuid import uuid4
from app.services.ingestion_worker_lock import IngestionWorkerLock
from sqlalchemy.orm import Session
from app.models.database_models import Document, DeletionTask
from app.services.index_coordinator import SourceInvalidated

class SourceRequest:
    def __init__(self, coordinator, ids):
        self.coordinator, self.document_ids = coordinator, tuple(ids)
        self.cancelled = False
        self.payloads = []
        self.locks = []
        with coordinator.state.lock, Session(coordinator.engine) as db:
            self.epochs = {i: db.get(Document, i).deletion_epoch for i in ids}
            self.check()
            for document_id in ids:
                lock = IngestionWorkerLock(Path(coordinator.database_key + f".request-{document_id}-{uuid4().hex}.lock"))
                if not lock.acquire():
                    raise RuntimeError("request_lease_unavailable")
                self.locks.append(lock)
            coordinator.state.requests.add(self)

    def check(self):
        with Session(self.coordinator.engine) as db:
            if self.cancelled or any((doc := db.get(Document, i)) is None or
                doc.deletion_status != "none" or doc.deletion_epoch != epoch or
                db.get(DeletionTask, i) is not None for i, epoch in self.epochs.items()):
                self.cancel()
                raise SourceInvalidated("source_deleted")

    def retain(self, payload):
        self.check()
        self.payloads.append(payload)

    def cancel(self):
        self.cancelled = True
        for payload in self.payloads:
            for chunk in payload:
                if isinstance(chunk, dict):
                    chunk.clear()
            payload.clear()
        self.payloads.clear()

    async def wait(self, awaitable):
        task = asyncio.ensure_future(awaitable)
        try:
            while True:
                self.check()
                done, _ = await asyncio.wait([task], timeout=.05)
                self.check()
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
            # Async API generators close in their finally path. Native threads
            # may finish later; ingestion/index write guards reject late results.
            try:
                await task
            except BaseException:
                pass

    def close(self):
        with self.coordinator.state.lock:
            self.coordinator.state.requests.discard(self)
        for lock in self.locks:
            lock.release()
            lock.path.unlink(missing_ok=True)
        self.locks.clear()
        self.payloads.clear()
