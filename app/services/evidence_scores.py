"""Cosine evidence and retrieval ranks have distinct, explicit responsibilities."""

import math


def finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def normalize_evidence_score(value):
    """Validate a similarity; never guess a distance/rank from its numeric magnitude."""
    number = finite_number(value)
    return number if number is not None and 0 <= number <= 1 else 0.0


def cosine_distance_to_score(raw_distance):
    distance = finite_number(raw_distance)
    if distance is None or distance < -1e-6:
        return 0.0
    return max(0.0, min(1.0, 1.0 - distance))


def cosine_distance(query_vector, document_vector):
    """Return the same-space cosine distance, or None for absent/invalid vectors."""
    try:
        if len(query_vector) != len(document_vector) or not len(query_vector):
            return None
        query = [finite_number(v) for v in query_vector]
        document = [finite_number(v) for v in document_vector]
        if any(v is None for v in query + document):
            return None
        qnorm, dnorm = math.hypot(*query), math.hypot(*document)
        if not qnorm or not dnorm or not math.isfinite(qnorm * dnorm):
            return None
        cosine = math.fsum((q / qnorm) * (d / dnorm) for q, d in zip(query, document))
        return 1.0 - max(-1.0, min(1.0, cosine))
    except (TypeError, ValueError, OverflowError):
        return None


def chunk_evidence_score(chunk):
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if chunk.get("score_metric", metadata.get("score_metric", "cosine")) != "cosine":
        return 0.0
    for key in ("raw_distance", "evidence_score", "vector_score"):
        # Explicit zero/unknown evidence is authoritative, never replaced by a rank.
        for source in (chunk, metadata):
            if key in source:
                return (cosine_distance_to_score(source[key]) if key == "raw_distance"
                        else normalize_evidence_score(source[key]))
    ranking_keys = ("bm25_score", "rrf_score", "final_score", "ranking_details")
    if any(key in source for source in (chunk, metadata) for key in ranking_keys):
        return 0.0
    # Existing public schemas define this field as similarity; untyped ranking fields cannot use it.
    return normalize_evidence_score(chunk.get("similarity_score"))
