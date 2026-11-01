"""Local embedding identity and exact tokenizer budgets, independent of generation."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import unicodedata


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def local_assets_digest(root):
    """A mutable local directory needs content identity in addition to a revision label."""
    root = Path(root)
    assets = {p.relative_to(root).as_posix(): file_digest(p)
              for p in sorted(root.rglob("*")) if p.is_file()
              and p.suffix.lower() in {".json", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".model", ".txt"}}
    if not assets:
        raise ValueError("local_model_assets_missing")
    return digest(assets)


def _token_metadata(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _token_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        result = [_token_metadata(v) for v in value]
        return sorted(result, key=canonical_json) if isinstance(value, (set, frozenset)) else result
    return {"content": str(value), **{k: getattr(value, k, None)
            for k in ("single_word", "lstrip", "rstrip", "normalized", "special")}}


def tokenizer_digest(tokenizer):
    """Include merge/normalization rules, not just vocabulary size or a model name."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        state = json.loads(backend.to_str())
        # These are mutable encoding state, controlled by the explicit input budget.
        state.pop("truncation", None)
        state.pop("padding", None)
    else:
        state = {"vocabulary": tokenizer.get_vocab()}
        vocab_file = getattr(tokenizer, "vocab_file", None)
        if vocab_file:
            state["vocabulary_file_sha256"] = file_digest(vocab_file)
        sp_model = getattr(tokenizer, "sp_model", None)
        if sp_model is not None:
            state["sentencepiece_sha256"] = hashlib.sha256(sp_model.serialized_model_proto()).hexdigest()
        # Slow tokenizers without serialized rules are limited to known vocab-based families.
        elif not vocab_file and "BertTokenizer" not in type(tokenizer).__name__:
            raise ValueError("tokenizer_rules_unavailable")
        basic = getattr(tokenizer, "basic_tokenizer", None)
        state["basic_tokenizer"] = {k: _token_metadata(getattr(basic, k, None))
                                    for k in ("do_lower_case", "never_split", "tokenize_chinese_chars",
                                              "strip_accents", "do_split_on_punc")}
        state["max_input_chars_per_word"] = getattr(
            getattr(tokenizer, "wordpiece_tokenizer", None), "max_input_chars_per_word", None)
    flags = ("do_lower_case", "strip_accents", "tokenize_chinese_chars", "add_prefix_space",
             "legacy", "do_basic_tokenize", "never_split")
    return digest({"class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
                   "rules": state,
                   "special_tokens": _token_metadata(getattr(tokenizer, "special_tokens_map", {})),
                   "added_tokens": _token_metadata(getattr(tokenizer, "added_tokens_decoder", {})),
                   "options": {k: _token_metadata(getattr(tokenizer, "init_kwargs", {}).get(k)) for k in flags}})


def encoder_versions():
    result = {}
    for package in ("langchain-huggingface", "sentence-transformers", "transformers", "tokenizers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


class EmbeddingCompatibilityError(ValueError):
    def __init__(self, reason, changed_fields=()):
        self.changed_fields = tuple(changed_fields)
        super().__init__(reason + "; rebuild from retained source PDFs into a new index")


class EmbeddingInputTooLong(ValueError):
    def __init__(self, tokens, limit, role):
        self.tokens, self.limit, self.role = tokens, limit, role
        super().__init__(f"embedding_input_budget_exceeded: {role} has {tokens} tokens; limit {limit}; no truncation")


@dataclass(frozen=True)
class EmbeddingProfile:
    """Immutable canonical data; runtime device, batch size and credentials are excluded."""

    canonical: str

    @classmethod
    def create(cls, **data):
        return cls(canonical_json({"schema_version": 1, **data}))

    @property
    def fingerprint(self):
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def to_record(self):
        return {"fingerprint": self.fingerprint, "profile": json.loads(self.canonical)}

    def require_compatible(self, stored):
        if not isinstance(stored, dict) or not isinstance(stored.get("profile"), dict):
            raise EmbeddingCompatibilityError("embedding_profile_missing")
        data = stored["profile"]
        try:
            stored_digest = digest(data)
        except (TypeError, ValueError):
            raise EmbeddingCompatibilityError("embedding_profile_invalid") from None
        if stored.get("fingerprint") != stored_digest:
            raise EmbeddingCompatibilityError("embedding_profile_corrupt")
        if self.fingerprint != stored_digest:
            current = json.loads(self.canonical)
            changed = sorted(k for k in current.keys() | data.keys() if current.get(k) != data.get(k))
            raise EmbeddingCompatibilityError("embedding_profile_mismatch", changed)


@dataclass(frozen=True)
class InputPolicy:
    max_tokens: int
    normalization: str
    query_prefix: str
    document_prefix: str
    normalize_vectors: bool
    distance: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.LOCAL_EMBEDDING_MAX_INPUT_TOKENS,
                   settings.LOCAL_EMBEDDING_TEXT_NORMALIZATION,
                   settings.LOCAL_EMBEDDING_QUERY_PREFIX, settings.LOCAL_EMBEDDING_DOCUMENT_PREFIX,
                   settings.LOCAL_EMBEDDING_NORMALIZE, settings.EMBEDDING_DISTANCE,
                   max(180, settings.CHUNK_SIZE or 400), max(20, settings.CHUNK_OVERLAP or 40))

    def prepare(self, text, *, query):
        if self.normalization == "NFKC":
            text = unicodedata.normalize("NFKC", text)
        # Match HuggingFaceEmbeddings' newline replacement explicitly.
        text = text.replace("\n", " ")
        prefix = self.query_prefix if query else self.document_prefix
        return prefix.replace("\n", " ") + text

    def profile_fields(self, effective_limit):
        return {"text_preprocessing": {"version": "local-text-v1", "normalization": self.normalization,
                                       "newlines": "space"},
                "prefixes": {"query_sha256": digest(self.query_prefix),
                             "document_sha256": digest(self.document_prefix)},
                "normalize_vectors": self.normalize_vectors, "distance": self.distance,
                "input_budget": {"max_tokens": effective_limit, "special_tokens": "included",
                                 "overflow": "reject", "implicit_model_prompt": "disabled"},
                "chunking": {"version": "legacy-character-v1", "size": self.chunk_size,
                             "overlap": self.chunk_overlap, "separators": ["\n\n", "\n", "。", "；", " ", ""]}}


def effective_token_limit(client, tokenizer, configured):
    limits = [configured]
    for value in (getattr(client, "max_seq_length", None), getattr(tokenizer, "model_max_length", None)):
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 1_000_000:
            limits.append(value)
    if len(limits) == 1:
        raise ValueError("model_token_limit_unknown")
    return min(limits)


def check_budget(tokenizer, text, limit, *, query):
    tokens = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
    if tokens > limit:
        raise EmbeddingInputTooLong(tokens, limit, "query" if query else "document")
    return tokens
