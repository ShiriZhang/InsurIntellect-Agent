from copy import deepcopy

import pytest

from app.services.embedding_profile import EmbeddingProfile
from app.services.index_resources import IndexIntegrityError, IndexSnapshots, ProfileVectorStore
from app.services.retrieval_text import lexical_profile, tokenize_retrieval


def profile(name="local-a"):
    return EmbeddingProfile.create(model=name, dimension=3, distance="cosine").to_record()


def row(key="chunk-a", text="等待期为90天，not covered unless specified.", document_id=1):
    return {"id": key, "content": text, "embedding": [1., .5, .25],
            "metadata": {"document_id": document_id, "attempt": 1, "source_key": f"{document_id}:1",
                         "chunk_id": key, "page_number": 3, "content_hash": "test-hash"}}


def test_actual_local_chroma_profile_and_directory_isolation(tmp_path):
    store = ProfileVectorStore(tmp_path / "vectors-a")
    a, b = profile(), profile("same-dimension-different-model")
    store.write(a, [row()])
    store.verify(a, [row()])
    store.write(b, [row("chunk-b", document_id=2)])
    assert set(store.get(a, ["chunk-a", "chunk-b"])) == {"chunk-a"}
    assert set(store.get(b, ["chunk-a", "chunk-b"])) == {"chunk-b"}
    other = ProfileVectorStore(tmp_path / "vectors-b")
    other.write(a, [row("elsewhere")])
    assert not other.get(a, ["chunk-a"])
    assert store.collection(a).metadata["hnsw:space"] == "cosine"
    ids, distances = store.query(a, [1., .5, .25], ["1:1"], 5)
    assert ids == ["chunk-a"] and abs(distances["chunk-a"]) < 1e-5


def test_unfingerprinted_legacy_collection_is_not_reused(tmp_path):
    store = ProfileVectorStore(tmp_path / "vectors")
    store.client.create_collection("insurance_documents", embedding_function=None)
    a = profile()
    store.client.create_collection(store.collection_name(a), metadata={"hnsw:space": "cosine"}, embedding_function=None)
    with pytest.raises(IndexIntegrityError, match="profile_mismatch"):
        store.write(a, [row()])
    with pytest.raises(IndexIntegrityError):
        store.collection({})


def test_atomic_versioned_snapshot_has_exact_identity_and_shared_tokens(tmp_path):
    snapshots = IndexSnapshots(tmp_path / "processed")
    rows = [row(), row("second-clause-same-page", "The waiting period does not apply to accidents.")]
    snapshot = snapshots.build(profile(), rows, {"1": 0})
    first_path, first_hash = snapshots.write(snapshot)
    second_path, _ = snapshots.write(snapshot)
    assert first_path != second_path
    recovered = snapshots.read(first_path, first_hash)
    assert recovered["document_attempts"] == [[1, 1]]
    assert recovered["lexical_profile"] == lexical_profile()
    assert recovered["tokens"] == [tokenize_retrieval(r["content"]) for r in rows]
    assert recovered["chunks"]["chunk-a"]["metadata"]["page_number"] == 3
    snapshots.path(first_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(IndexIntegrityError, match="checksum"):
        snapshots.read(first_path, first_hash)
    assert not list(snapshots.root.rglob('.tmp-*'))
    with pytest.raises(IndexIntegrityError, match="outside"):
        snapshots.path("../old-manifest.json")


def test_vector_completeness_and_source_provenance_are_checked(tmp_path):
    store = ProfileVectorStore(tmp_path / "vectors")
    a = profile()
    store.write(a, [row()])
    wrong = deepcopy(row())
    wrong["metadata"]["page_number"] = 9
    with pytest.raises(IndexIntegrityError, match="provenance"):
        store.verify(a, [wrong])
    with pytest.raises(IndexIntegrityError, match="incomplete"):
        store.verify(a, [row("missing")])
