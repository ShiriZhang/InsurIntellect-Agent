"""Profile-isolated local vectors and atomic, versioned keyword/source snapshots."""

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from uuid import uuid4

from app.services.embedding_profile import EmbeddingProfile, canonical_json
from app.services.retrieval_text import lexical_profile, tokenize_retrieval


class IndexIntegrityError(RuntimeError):
    pass


def checked_profile(record):
    try:
        profile = EmbeddingProfile(canonical_json(record["profile"]))
        profile.require_compatible(record)
        if not re.fullmatch(r"[0-9a-f]{64}", profile.fingerprint):
            raise ValueError()
        return profile
    except (KeyError, TypeError, ValueError) as exc:
        raise IndexIntegrityError("missing_or_invalid_embedding_profile") from exc


class ProfileVectorStore:
    """No default collection, remote backend, embedding function or implicit downloads."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self.root.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(str(self.root), settings=Settings(anonymized_telemetry=False))
        return self._client

    @staticmethod
    def collection_name(profile, version=None):
        base = "clauses_" + checked_profile(profile).fingerprint
        if version is not None and not re.fullmatch(r"[0-9a-f]{32}", version):
            raise IndexIntegrityError("invalid_vector_version")
        return base + ("_v_" + version if version is not None else "")

    def collection(self, profile, *, create=False, collection=None, owner=None):
        base = self.collection_name(profile)
        name = collection or base
        if name != base and not re.fullmatch(re.escape(base) + r"_v_[0-9a-f]{32}", name):
            raise IndexIntegrityError("vector_collection_profile_mismatch")
        expected = {"embedding_profile": canonical_json(profile),
                    "embedding_fingerprint": profile["fingerprint"], "hnsw:space": "cosine"}
        if owner is not None:
            expected['library_id'] = owner
        if create:
            collection = self.client.get_or_create_collection(name, metadata=expected, embedding_function=None)
        else:
            collection = self.client.get_collection(name, embedding_function=None)
        if any((collection.metadata or {}).get(k) != v for k, v in expected.items()):
            raise IndexIntegrityError("vector_profile_mismatch_rebuild_required")
        return collection

    def write(self, profile, rows, *, collection=None, owner=None):
        collection = self.collection(profile, create=True, collection=collection, owner=owner)
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            collection.upsert(ids=[r["id"] for r in batch], embeddings=[r["embedding"] for r in batch],
                              documents=[r["content"] for r in batch], metadatas=[r["metadata"] for r in batch])

    def get(self, profile, ids, *, collection=None):
        if not ids:
            return {}
        collection = self.collection(profile, collection=collection)
        result = {}
        for start in range(0, len(ids), 256):
            # Resolve existing identities first. Some Chroma versions fail an
            # embeddings read for a mixed existing/missing ID list instead of
            # returning the documented subset; never refetch different IDs.
            payload = collection.get(ids=ids[start:start + 256], include=["documents", "metadatas"])
            if not payload['ids']:
                continue
            vectors = collection.get(ids=payload['ids'], include=['embeddings'])
            embeddings = dict(zip(vectors['ids'], vectors['embeddings']))
            for i, key in enumerate(payload["ids"]):
                if key not in embeddings:
                    raise IndexIntegrityError('stored_vector_missing')
                result[key] = {"id": key, "content": payload["documents"][i],
                               "metadata": payload["metadatas"][i],
                               "embedding": list(map(float, embeddings[key]))}
        return result

    def verify(self, profile, rows, *, collection=None):
        actual = self.get(profile, [r["id"] for r in rows], collection=collection)
        if set(actual) != {r["id"] for r in rows}:
            raise IndexIntegrityError("vector_ids_incomplete")
        for row in rows:
            stored = actual[row["id"]]
            if stored["content"] != row["content"] or stored["metadata"] != row["metadata"]:
                raise IndexIntegrityError("vector_provenance_mismatch")
            a, b = stored["embedding"], row["embedding"]
            if (len(a) != len(b) or not all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6) for x, y in zip(a, b))):
                raise IndexIntegrityError("vector_values_mismatch")

    def query(self, profile, vector, source_keys, limit, *, collection=None):
        if not source_keys:
            return [], {}
        payload = self.collection(profile, collection=collection).query(
            query_embeddings=[vector], where={"source_key": {"$in": list(source_keys)}},
            n_results=limit, include=["distances"],
        )
        ids = payload["ids"][0]
        return ids, dict(zip(ids, payload["distances"][0]))

    def delete_ids(self, profile, ids, *, collection=None):
        if ids:
            self.collection(profile, collection=collection).delete(ids=ids)

    def drop(self, profile, collection):
        try:
            self.collection(profile, collection=collection)
            self.client.delete_collection(collection)
        except Exception as exc:
            if type(exc).__name__ not in {"NotFoundError", "InvalidCollectionException"}:
                raise

    def managed_versions(self, owner):
        if self._client is None and not self.root.exists():
            return []
        versions = []
        for value in self.client.list_collections():
            name = getattr(value, 'name', value)
            if not re.fullmatch(r'clauses_[0-9a-f]{64}_v_[0-9a-f]{32}', name):
                continue
            metadata = self.client.get_collection(name, embedding_function=None).metadata or {}
            if metadata.get('library_id') == owner:
                profile = json.loads(metadata['embedding_profile'])
                checked_profile(profile)
                versions.append((name, profile))
        return versions

    def purge_document(self, profile, document_id):
        try:
            collection = self.collection(profile)
        except Exception as exc:
            # A missing collection is only safe if it contains no retained sources;
            # the coordinator verifies every retained vector before serving.
            if type(exc).__name__ in {"NotFoundError", "InvalidCollectionException"}:
                return
            raise
        collection.delete(where={"document_id": document_id})


class IndexSnapshots:
    def __init__(self, root, *, library_id=None):
        self.root = Path(root).resolve() / "indexes"
        self.library_id = library_id
        self.temp_prefix = '.tmp-' + (library_id[:12] + '-' if library_id else '')

    def path(self, relative):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root) or not re.fullmatch(
                r"[0-9a-f]{16}/v-[0-9a-f]{32}\.json", Path(relative).as_posix()):
            raise IndexIntegrityError("snapshot_path_outside_managed_directory")
        return path

    def build(self, profile, rows, epochs):
        checked_profile(profile)
        ids = [r["id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise IndexIntegrityError("duplicate_chunk_identity")
        mapping = {r["id"]: {"content": r["content"], "metadata": r["metadata"]} for r in rows}
        pairs = sorted({(r["metadata"]["document_id"], r["metadata"]["attempt"]) for r in rows})
        return {"schema_version": 1, "embedding_profile": profile, "library_id": self.library_id,
                "document_attempts": [list(pair) for pair in pairs], "deletion_epochs": epochs,
                "lexical_profile": lexical_profile(), "ids": ids, "chunks": mapping,
                "tokens": [tokenize_retrieval(mapping[key]["content"]) for key in ids]}

    def write(self, snapshot, relative=None):
        fingerprint = checked_profile(snapshot["embedding_profile"]).fingerprint
        # The full fingerprint lives in SQLite and the snapshot. A short directory
        # namespace plus a unique version name avoids Windows' path-length ceiling.
        relative = relative or f"{fingerprint[:16]}/v-{uuid4().hex}.json"
        path = self.path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / (self.temp_prefix + uuid4().hex[:16])
        content = canonical_json(snapshot).encode("utf-8")
        try:
            with temporary.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return relative, sha256(content).hexdigest()

    def read(self, relative, expected_hash):
        content = self.path(relative).read_bytes()
        return self.decode(content, expected_hash)

    def decode(self, content, expected_hash):
        if sha256(content).hexdigest() != expected_hash:
            raise IndexIntegrityError("snapshot_checksum_mismatch")
        snapshot = json.loads(content)
        if snapshot.get('library_id') != self.library_id:
            raise IndexIntegrityError('snapshot_library_mismatch')
        checked_profile(snapshot["embedding_profile"])
        if snapshot.get("lexical_profile") != lexical_profile():
            raise IndexIntegrityError("keyword_profile_mismatch")
        ids, mapping = snapshot["ids"], snapshot["chunks"]
        if len(set(ids)) != len(ids) or set(ids) != set(mapping) or len(ids) != len(snapshot["tokens"]):
            raise IndexIntegrityError("snapshot_mapping_incomplete")
        return snapshot
