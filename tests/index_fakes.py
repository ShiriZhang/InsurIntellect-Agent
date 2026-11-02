"""Fault-injectable vector persistence stand-in; real Chroma is tested separately."""

from copy import deepcopy

from app.services.evidence_scores import cosine_distance
from app.services.index_resources import IndexIntegrityError, ProfileVectorStore


class VectorStoreFake:
    collection_name = staticmethod(ProfileVectorStore.collection_name)

    def __init__(self):
        self.rows = {}
        self.collections = {}
        self.owners = {}
        self.fail_write = False
        self.fail_query = False

    def write(self, profile, rows, *, collection=None, owner=None):
        collection = collection or profile["fingerprint"]
        bucket = self.collections.setdefault(collection, {})
        self.owners[collection] = (owner, profile)
        self.rows[profile["fingerprint"]] = bucket
        for row in rows:
            bucket[row["id"]] = deepcopy(row)
            if self.fail_write:
                raise OSError("injected partial vector failure")

    def get(self, profile, ids, *, collection=None):
        bucket = self.collections.get(collection or profile["fingerprint"], {})
        return {key: deepcopy(bucket[key]) for key in ids if key in bucket}

    def verify(self, profile, rows, *, collection=None):
        if self.get(profile, [r["id"] for r in rows], collection=collection) != {r["id"]: r for r in rows}:
            raise IndexIntegrityError("incomplete vector data")

    def delete_ids(self, profile, ids, *, collection=None):
        for key in ids:
            self.collections.get(collection or profile["fingerprint"], {}).pop(key, None)

    def drop(self, profile, collection):
        bucket = self.collections.pop(collection, None)
        if self.rows.get(profile['fingerprint']) is bucket:
            # Preserve the still-live committed version in this diagnostic alias.
            self.rows[profile['fingerprint']] = next(iter(self.collections.values()), {})
        self.owners.pop(collection, None)

    def managed_versions(self, owner):
        return [(name, profile) for name, (library, profile) in self.owners.items() if library == owner]

    def purge_document(self, profile, document_id):
        bucket = self.rows.get(profile["fingerprint"], {})
        for key in list(bucket):
            if bucket[key]["metadata"]["document_id"] == document_id:
                del bucket[key]

    def query(self, profile, vector, source_keys, limit, *, collection=None):
        if self.fail_query:
            raise RuntimeError("filter failed")
        rows = [r for r in self.collections.get(collection or profile["fingerprint"], {}).values()
                if r["metadata"]["source_key"] in source_keys]
        scored = sorted((cosine_distance(vector, r["embedding"]), r["id"]) for r in rows)
        chosen = scored[:limit]
        return [key for _, key in chosen], {key: distance for distance, key in chosen}
