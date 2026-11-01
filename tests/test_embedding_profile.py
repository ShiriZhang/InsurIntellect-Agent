"""Semantic compatibility and exact budgets, without weights or network access."""

import json
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services import embedding_service as embeddings
from app.services.embedding_profile import (
    EmbeddingCompatibilityError, EmbeddingInputTooLong, EmbeddingProfile,
    InputPolicy, digest, tokenizer_digest,
)
from model_fakes import TokenizerStub, client_stub


@pytest.fixture
def environment(monkeypatch, tmp_path):
    for name in Settings.model_fields:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "_create_directories", lambda self: None)
    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_MODEL="example/model-a",
                   LOCAL_EMBEDDING_REVISION="fixed-revision", LOCAL_EMBEDDING_DEVICE="cpu",
                   LOCAL_EMBEDDING_CACHE_DIR=str(tmp_path))
    monkeypatch.setattr(embeddings, "settings", cfg)
    monkeypatch.setattr(embeddings, "_snapshot_download", lambda **kw: str(tmp_path / "commit-a"))
    loaded = []

    class Encoder:
        def __init__(self, **kwargs):
            self._client = client_stub()
            self.kwargs = kwargs
            self.encode_kwargs = kwargs["encode_kwargs"]
            self.calls = []
            loaded.append(self)

        def embed_documents(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 0.5, 0.25] for _ in texts]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    monkeypatch.setattr(embeddings, "_load_embedding_class", lambda: Encoder)
    return cfg, loaded, tmp_path


def prepared_service():
    service = embeddings.EmbeddingService()
    service.preflight()
    return service


@pytest.mark.parametrize("field,new_value,changed", [
    ("LOCAL_EMBEDDING_MODEL", "example/model-b", "model_id"),
    ("LOCAL_EMBEDDING_REVISION", "different-revision", "requested_revision"),
    ("LOCAL_EMBEDDING_QUERY_PREFIX", "query: ", "prefixes"),
    ("LOCAL_EMBEDDING_DOCUMENT_PREFIX", "passage: ", "prefixes"),
    ("LOCAL_EMBEDDING_TEXT_NORMALIZATION", "NFKC", "text_preprocessing"),
    ("LOCAL_EMBEDDING_NORMALIZE", False, "normalize_vectors"),
    ("LOCAL_EMBEDDING_MAX_INPUT_TOKENS", 64, "input_budget"),
    ("CHUNK_SIZE", 500, "chunking"),
    ("CHUNK_OVERLAP", 50, "chunking"),
])
def test_semantic_changes_require_rebuild_even_at_equal_dimensions(environment, field, new_value, changed):
    cfg, _, _ = environment
    first = prepared_service()
    record = first.embedding_profile().to_record()
    setattr(cfg, field, new_value)
    second = prepared_service()
    assert first.dimension == second.dimension == 3
    with pytest.raises(EmbeddingCompatibilityError, match="rebuild") as error:
        second.require_compatible(record)
    assert changed in error.value.changed_fields


def test_actual_revision_change_is_detected(environment, monkeypatch):
    _, _, root = environment
    record = prepared_service().embedding_profile().to_record()
    monkeypatch.setattr(embeddings, "_snapshot_download", lambda **kw: str(root / "commit-b"))
    with pytest.raises(EmbeddingCompatibilityError) as error:
        prepared_service().require_compatible(record)
    assert "actual_revision" in error.value.changed_fields


def test_tokenizer_rules_change_without_vocab_size_change(environment):
    _, loaded, _ = environment
    first = prepared_service().embedding_profile()
    second = embeddings.EmbeddingService()
    loaded[-1]._client.tokenizer = TokenizerStub(rule="changed-merge-order")
    second.preflight()
    with pytest.raises(EmbeddingCompatibilityError) as error:
        second.require_compatible(first.to_record())
    assert "tokenizer_sha256" in error.value.changed_fields


def test_generation_credentials_models_and_runtime_locations_do_not_change_identity(environment):
    cfg, _, _ = environment
    profile = prepared_service().embedding_profile()
    cfg.GENERATION_API_BASE_URL = "https://user:password@generation.example/v1?key=secret"
    cfg.GENERATION_API_KEY = "generation-secret"
    cfg.GENERATION_MODEL = "another-generation-model"
    cfg.OPENAI_API_KEY = "old-secret"
    cfg.OPENAI_MODEL_LIGHT = "new-light-model"
    cfg.LOCAL_EMBEDDING_CACHE_DIR = "another-cache-location"
    cfg.LOCAL_EMBEDDING_BATCH_SIZE = 1
    same = prepared_service().embedding_profile()
    same.require_compatible(json.loads(json.dumps(profile.to_record())))
    assert same.fingerprint == profile.fingerprint
    assert not any(word in same.canonical for word in ("secret", "password", "generation.example"))


def test_device_change_preserves_profile(environment, monkeypatch):
    cfg, _, _ = environment
    record = prepared_service().embedding_profile().to_record()
    cfg.LOCAL_EMBEDDING_DEVICE = "cuda"
    monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
    prepared_service().require_compatible(record)


@pytest.mark.parametrize("stored", [None, {}, {"fingerprint": "old-no-profile"},
                                   {"fingerprint": "tampered", "profile": {}}])
def test_unknown_or_corrupt_profile_is_not_accepted(environment, stored):
    with pytest.raises(EmbeddingCompatibilityError, match="rebuild"):
        prepared_service().require_compatible(stored)


def test_serialization_is_canonical_and_schema_change_is_incompatible(environment):
    profile = prepared_service().embedding_profile()
    record = profile.to_record()
    reordered = {"profile": dict(reversed(list(record["profile"].items()))),
                 "fingerprint": record["fingerprint"]}
    profile.require_compatible(reordered)
    new = dict(record["profile"], schema_version=2)
    with pytest.raises(EmbeddingCompatibilityError):
        profile.require_compatible({"fingerprint": digest(new), "profile": new})


def test_changed_local_weight_content_is_detected_even_with_same_revision(environment):
    cfg, _, root = environment
    cfg.LOCAL_EMBEDDING_MODEL = str(root)
    weights = root / "model.safetensors"
    weights.write_bytes(b"first fake weights")
    record = prepared_service().embedding_profile().to_record()
    weights.write_bytes(b"different fake weights")
    with pytest.raises(EmbeddingCompatibilityError) as error:
        prepared_service().require_compatible(record)
    assert "local_assets_sha256" in error.value.changed_fields


def test_effective_budget_honors_model_and_tokenizer_limits(environment):
    cfg, loaded, _ = environment
    cfg.LOCAL_EMBEDDING_MAX_INPUT_TOKENS = 100
    service = embeddings.EmbeddingService()
    loaded[-1]._client.max_seq_length = 32
    loaded[-1]._client.tokenizer.model_max_length = 64
    assert service.max_input_tokens == 32
    loaded[-1].calls.clear()
    assert service.count_tokens("a" * 30) == 32
    assert service.embed_query("a" * 30)
    with pytest.raises(EmbeddingInputTooLong) as error:
        service.embed_query("a" * 31)
    assert error.value.tokens == 33 and error.value.limit == 32
    assert len(loaded[-1].calls) == 1


@pytest.mark.parametrize("query", [True, False])
def test_prefix_and_special_tokens_are_counted_before_encoding(environment, query):
    cfg, loaded, _ = environment
    cfg.LOCAL_EMBEDDING_MAX_INPUT_TOKENS = 8
    cfg.LOCAL_EMBEDDING_QUERY_PREFIX = "Q: "
    cfg.LOCAL_EMBEDDING_DOCUMENT_PREFIX = "D: "
    service = embeddings.EmbeddingService()
    text = "中文英"  # 3 chars + 3 prefix chars + 2 special tokens = 8.
    assert service.count_tokens(text, query=query) == 8
    encode = service.embed_query if query else lambda value: service.embed_documents([value])
    encode(text)
    assert loaded[-1].calls == [[("Q: " if query else "D: ") + text]]
    assert loaded[-1].encode_kwargs["prompt"] == ""  # Disable hidden model prompts.
    with pytest.raises(EmbeddingInputTooLong, match="no truncation") as error:
        encode(text + "x")
    assert error.value.role == ("query" if query else "document")
    assert text not in str(error.value)
    assert len(loaded[-1].calls) == 1


def test_nfkc_and_newline_processing_matches_counted_input(environment):
    cfg, loaded, _ = environment
    cfg.LOCAL_EMBEDDING_TEXT_NORMALIZATION = "NFKC"
    service = embeddings.EmbeddingService()
    service.embed_query("Ａ\n等待期９０天")
    assert loaded[-1].calls == [["A 等待期90天"]]
    assert service.count_tokens("Ａ\n等待期９０天", query=True) == len("A 等待期90天") + 2


def test_special_token_changes_affect_identity_and_ephemeral_padding_does_not():
    tokenizer = TokenizerStub()
    original = tokenizer_digest(tokenizer)
    tokenizer.to_str = lambda: json.dumps({"model": {"type": "characters"}, "normalizer": None,
                                            "padding": {"length": 10}, "truncation": {"max_length": 10}})
    assert tokenizer_digest(tokenizer) == original
    tokenizer.special_tokens_map = {"cls_token": "<changed>"}
    assert tokenizer_digest(tokenizer) != original


def test_unknown_tokenizer_or_model_limit_fails_closed(environment):
    _, loaded, _ = environment
    service = embeddings.EmbeddingService()
    loaded[-1]._client = SimpleNamespace(tokenizer=TokenizerStub())
    loaded[-1]._client.tokenizer.model_max_length = 10**30
    with pytest.raises(embeddings.LocalEmbeddingUnavailable, match="input limit unavailable"):
        service.embed_query("question")
    assert loaded[-1].calls == []


def test_distance_configuration_is_explicit_and_only_cosine_is_supported(environment):
    cfg, _, _ = environment
    profile = prepared_service().embedding_profile().to_record()["profile"]
    assert cfg.EMBEDDING_DISTANCE == profile["distance"] == "cosine"
    with pytest.raises(ValueError):
        Settings(_env_file=None, EMBEDDING_DISTANCE="l2")


def test_changed_preprocessing_version_requires_rebuild(environment):
    profile = prepared_service().embedding_profile()
    values = profile.to_record()["profile"]
    values.pop("schema_version")
    values["text_preprocessing"]["version"] = "future-version"
    future = EmbeddingProfile.create(**values)
    with pytest.raises(EmbeddingCompatibilityError):
        future.require_compatible(profile.to_record())


def test_prefix_whitespace_is_not_stripped_by_settings(environment):
    cfg = Settings(_env_file=None, LOCAL_EMBEDDING_QUERY_PREFIX="query: ")
    assert cfg.LOCAL_EMBEDDING_QUERY_PREFIX.endswith(" ")
    assert InputPolicy.from_settings(cfg).prepare("condition", query=True) == "query: condition"


def test_slow_wordpiece_special_token_flags_and_rules_are_part_of_identity():
    class AddedToken:
        lstrip = False

        def __str__(self):
            return "[SPECIAL]"

    class BertTokenizerStub:
        init_kwargs = {"never_split": ["[SPECIAL]"]}
        special_tokens_map = {"cls_token": "[CLS]"}

        def get_vocab(self):
            return {"word": 1, "[SPECIAL]": 2}

    tokenizer = BertTokenizerStub()
    token = AddedToken()
    tokenizer.added_tokens_decoder = {2: token}
    original = tokenizer_digest(tokenizer)
    token.lstrip = True
    assert tokenizer_digest(tokenizer) != original
    original = tokenizer_digest(tokenizer)
    tokenizer.init_kwargs = {"never_split": ["another-token"]}
    assert tokenizer_digest(tokenizer) != original
