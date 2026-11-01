"""Shared bilingual lexical retrieval; original source/evidence text is never rewritten."""

from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import re
import unicodedata


# Recall hints only: these terms never establish factual or jurisdictional equivalence.
TERMS = {
    "waiting_period": ("等待期", "观察期", "waiting period", "waiting periods"),
    "cooling_off_period": ("犹豫期", "cooling-off period", "cooling off period"),
    "exclusion": ("责任免除", "除外责任", "exclusion", "exclusions"),
    "deductible": ("免赔额", "deductible", "deductibles"),
    "critical_illness": ("重大疾病", "重疾", "critical illness"),
    "pre_existing_condition": ("既往症", "既往病史", "pre-existing condition", "pre-existing conditions"),
    "claim": ("理赔", "claim", "claims"),
    "death_benefit": ("身故保险金", "death benefit", "death benefits"),
    "grace_period": ("宽限期", "grace period"),
    "policyholder": ("投保人", "policyholder", "policyholders"),
    "insured": ("被保险人", "insured"),
}
LEXICAL_VERSION = "bilingual-lexical-v1"
_WORDS = re.compile(
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|[a-z0-9]*[a-z][a-z0-9]*(?:[-_'][a-z0-9]+)*"
    r"|[+-]?\d+(?:[,.]\d+)*(?:/\d+(?:[,.]\d+)*)?(?:%|‰)?"
    r"|[¥$€£]"
    r"|[\u3400-\u9fff]+"
)


def normalize_retrieval_text(text):
    return " ".join(unicodedata.normalize("NFKC", text or "").casefold().split())


@lru_cache(maxsize=1)
def _segmenter():
    import jieba

    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for aliases in TERMS.values():
        for alias in aliases:
            if re.search(r"[\u3400-\u9fff]", alias):
                tokenizer.add_word(alias)
    return tokenizer


def tokenize_retrieval(text):
    normalized = normalize_retrieval_text(text)
    tokens = []
    for match in _WORDS.finditer(normalized):
        word = match.group()
        if re.fullmatch(r"[\u3400-\u9fff]+", word):
            tokens.extend(t for t in _segmenter().lcut(word, HMM=False) if t.strip())
        else:
            tokens.append(word)
    for term, aliases in TERMS.items():
        for alias in aliases:
            # Match Latin phrases at word boundaries, never inside a product identifier.
            matches = (alias in normalized if re.search(r"[\u3400-\u9fff]", alias)
                       else re.search(r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])", normalized))
            if matches:
                tokens.append("term:" + term)
                break
    return tokens


@lru_cache(maxsize=1)
def _lexical_fingerprint():
    with _segmenter().get_dict_file() as dictionary:
        hasher = hashlib.sha256()
        for block in iter(lambda: dictionary.read(1024 * 1024), b""):
            hasher.update(block)
    description = {"version": LEXICAL_VERSION, "terms": list(TERMS.items()), "pattern": _WORDS.pattern,
                   "normalization": "NFKC-casefold-whitespace", "unicode": unicodedata.unidata_version, "hmm": False,
                   "jieba": version("jieba"), "dictionary_sha256": hasher.hexdigest()}
    return hashlib.sha256(json.dumps(description, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def lexical_profile():
    return {"version": LEXICAL_VERSION, "fingerprint": _lexical_fingerprint()}


def rank_lexical_candidates(model, ids, query, *, chunk_map=None, limit=50):
    """BM25 ranking requires actual term overlap, including BM25Plus' baseline scores."""
    tokens = tokenize_retrieval(query)
    if not tokens:
        return [], {}
    scores = model.get_scores(tokens)
    if len(scores) != len(ids):
        raise ValueError("bm25_mapping_length_mismatch")
    doc_freqs = getattr(model, "doc_freqs", None)
    scored = []
    for i, cid in enumerate(ids):
        if doc_freqs is not None:
            vocabulary = doc_freqs[i]
        else:
            source = (chunk_map or {}).get(cid, {})
            content = source.get("content", "") if isinstance(source, dict) else str(source)
            vocabulary = set(tokenize_retrieval(content))
        if not any(token in vocabulary for token in tokens):
            continue
        score = float(scores[i])
        if math.isfinite(score):
            scored.append((str(cid), score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    selected = scored[:limit]
    return [cid for cid, _ in selected], dict(selected)
