import json
import os
import subprocess
import sys

import pytest
from sqlalchemy.orm import Session

from app.models.database_models import Document, IndexCommit, IndexPointer
from app.services.index_resources import ProfileVectorStore
from test_index_coordinator import candidate, library  # noqa: F401


@pytest.mark.parametrize('crash_at', ['before_commit', 'after_commit'])
def test_hard_process_exit_recovers_sqlite_pointer_with_real_local_chroma(library, crash_at):
    ingestion, coordinator, _ = library
    coordinator.vectors = ProfileVectorStore(coordinator.vector_root)
    first = candidate(library)
    original = coordinator.publish(first.document_id, first.attempt)
    second = candidate(library, 'B')
    code = '''
import os,sys
from sqlalchemy import create_engine
from app.services.index_coordinator import IndexCoordinator
engine=create_engine(sys.argv[1])
coordinator=IndexCoordinator(engine,vector_root=sys.argv[2],processed_root=sys.argv[3],
    fault=lambda stage: os._exit(17) if stage==sys.argv[6] else None)
coordinator.publish(int(sys.argv[4]),int(sys.argv[5]))
'''
    child = subprocess.run([sys.executable, '-c', code, str(ingestion.engine.url), str(coordinator.vector_root),
                            str(coordinator.processed_root), str(second.document_id), str(second.attempt), crash_at],
                           timeout=45, capture_output=True, text=True, env={**os.environ, 'HF_HUB_OFFLINE': '1'})
    assert child.returncode == 17, child.stderr
    recovered = coordinator.recover()
    with Session(ingestion.engine) as db:
        pointer = db.get(IndexPointer, 'current').commit_id
        assert recovered == pointer
        assert db.get(Document, first.document_id).is_queryable
        assert db.get(Document, second.document_id).is_queryable == (crash_at == 'after_commit')
        assert (pointer != original) == (crash_at == 'after_commit')
        profile = json.loads(db.get(IndexCommit, pointer).embedding_profile_json)
    ids = [first.document_id, second.document_id] if crash_at == 'after_commit' else [first.document_id]
    with coordinator.open_view(ids, profile) as view:
        result = view.search('waiting period', [1., .5, .25])
        assert {r['document_id'] for r in result} == set(ids)


def test_external_publication_preserves_live_old_reader_and_new_queries_without_restart(library):
    ingestion, coordinator, _ = library
    coordinator.vectors = ProfileVectorStore(coordinator.vector_root)
    first = candidate(library)
    original = coordinator.publish(first.document_id, first.attempt)
    with Session(ingestion.engine) as db:
        row = db.get(IndexCommit, original)
        profile = json.loads(row.embedding_profile_json)
        old_path = row.snapshot_path
    view = coordinator.open_view([first.document_id], profile)
    second = candidate(library, 'B')
    code = '''
import sys
from sqlalchemy import create_engine
from app.services.index_coordinator import IndexCoordinator
c=IndexCoordinator(create_engine(sys.argv[1]),vector_root=sys.argv[2],processed_root=sys.argv[3])
c.publish(int(sys.argv[4]),1)
'''
    child = subprocess.run([sys.executable, '-c', code, str(ingestion.engine.url), str(coordinator.vector_root),
                            str(coordinator.processed_root), str(second.document_id)],
                           timeout=45, capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    assert coordinator.snapshots.path(old_path).exists()
    assert {r['document_id'] for r in view.search('waiting period', [1., .5, .25])} == {first.document_id}
    assert not coordinator.snapshots.path(old_path).exists()
    with coordinator.open_view([second.document_id], profile) as current:
        assert {r['document_id'] for r in current.search('waiting period', [1., .5, .25])} == {second.document_id}
    view.close()
