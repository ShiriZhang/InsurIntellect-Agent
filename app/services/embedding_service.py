"""Local Python embeddings; document and query text never use an embedding API."""

from __future__ import annotations

import asyncio
import re
import gc
import math
import threading
from pathlib import Path
from typing import List, Optional, Sequence

from app.core.app_logging import get_logger
from app.core.config import settings
from app.services.embedding_profile import (
    EmbeddingProfile, InputPolicy, check_budget, effective_token_limit,
    encoder_versions, local_assets_digest, tokenizer_digest,
)

logger = get_logger(__name__)


class LocalEmbeddingUnavailable(RuntimeError):
    """Local configuration, dependencies or model failed; ingestion/search is unavailable."""


def _snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _cuda_available():
    import torch

    return torch.cuda.is_available()


def _load_embedding_class():
    # Lazy import keeps application liveness independent of model dependencies/downloads.
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings


class EmbeddingService:
    """Use HuggingFaceEmbeddings/SentenceTransformers in this process only."""

    def __init__(self, model_name: Optional[str] = None, *, allow_download: bool = False) -> None:
        self.model_name = (model_name if model_name is not None
                           else settings.LOCAL_EMBEDDING_MODEL).strip()
        if self.model_name.startswith(("hf:", "local:")):
            self.model_name = self.model_name.split(":", 1)[1].strip()
        if (not self.model_name or "://" in self.model_name
                or self.model_name.lower() in {"hash", "local-hash", "offline"}
                or self.model_name.lower().startswith(("hash", "openai:", "remote:", "api:"))):
            raise LocalEmbeddingUnavailable(
                "Configure LOCAL_EMBEDDING_MODEL with a local model ID/directory; "
                "remote embedding and hash fallback are disabled"
            )
        self.revision = settings.LOCAL_EMBEDDING_REVISION
        self.cache_dir = settings.LOCAL_EMBEDDING_CACHE_DIR
        self.batch_size = settings.LOCAL_EMBEDDING_BATCH_SIZE
        if not self.revision or not self.cache_dir:
            raise LocalEmbeddingUnavailable(
                "Local model not configured: set LOCAL_EMBEDDING_REVISION and "
                "LOCAL_EMBEDDING_CACHE_DIR; ingestion/semantic search unavailable"
            )
        self.device = settings.LOCAL_EMBEDDING_DEVICE
        if self.device != "auto" and not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise LocalEmbeddingUnavailable("LOCAL_EMBEDDING_DEVICE must be auto/cpu/cuda/cuda:N")
        self._lock = threading.RLock()
        self._embedding = None
        self.dimension = None
        self.preflight_passed = False
        self.warnings = []
        self._batch_retries = 0
        self.input_policy = InputPolicy.from_settings(settings)
        self._encoder_versions = encoder_versions()
        self._tokenizer_details = None
        self._frozen_profile = None
        self._local_assets_sha256 = None
        try:
            if self.device != "cpu":
                available = _cuda_available()
                self.device = ("cuda" if self.device == "auto" else self.device) if available else "cpu"
            if self.device == "cpu":
                self._warn("cpu_execution", "CPU encoding is active; processing may be slower")
            self.asset_path = self._resolve_assets(allow_download=False)
        except LocalEmbeddingUnavailable:
            if not allow_download:
                raise
            self.asset_path = self._resolve_assets(allow_download=True)
        except Exception as exc:
            raise LocalEmbeddingUnavailable("Local device check failed (" + type(exc).__name__ + ")") from None
        self.actual_revision = Path(self.asset_path).name if not Path(self.model_name).is_dir() else self.revision
        # All model/tokenizer loads now use an on-disk snapshot with network disabled.
        failure = self._try_load()
        if failure and failure[0] and self.device.startswith("cuda"):
            self._switch_to_cpu()
            failure = self._try_load()
        elif (failure and failure[1] in {"OSError", "FileNotFoundError"}
              and allow_download and not Path(self.model_name).is_dir()):
            # A cached snapshot can exist while a weight/tokenizer file is missing.
            self.asset_path = self._resolve_assets(allow_download=True)
            self.actual_revision = Path(self.asset_path).name
            failure = self._try_load()
            if failure and failure[0] and self.device.startswith("cuda"):
                self._switch_to_cpu()
                failure = self._try_load()
        if failure:
            raise LocalEmbeddingUnavailable(
                "Local embedding model unavailable (" + failure[1] + "); check dependencies and "
                "complete cache with python -m scripts.prepare_models --download; no hash/remote fallback"
            ) from None
        self.provider = "huggingface"
        logger.info("Local embedding initialized: device=%s batch_size=%d", self.device, self.batch_size)

    def _warn(self, code, message):
        self.warnings.append({"code": code, "message": message})
        logger.warning("%s: %s", code, message)

    def _resolve_assets(self, *, allow_download):
        path = Path(self.model_name).expanduser()
        if path.is_dir():
            return str(path.resolve())
        try:
            return _snapshot_download(
                repo_id=self.model_name, revision=self.revision, cache_dir=self.cache_dir,
                local_files_only=not allow_download, token=False, max_workers=2,
            )
        except Exception as exc:
            raise LocalEmbeddingUnavailable(
                "Model cache unavailable (" + type(exc).__name__ + "); prepare weights/tokenizer "
                "with python -m scripts.prepare_models --download; ingestion/semantic search unavailable"
            ) from None

    @staticmethod
    def _is_oom(exc):
        return (type(exc).__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}
                or "cuda out of memory" in str(exc).lower())

    def _try_load(self):
        try:
            self._embedding = _load_embedding_class()(
                model_name=self.asset_path, cache_folder=self.cache_dir,
                model_kwargs={"revision": self.revision, "device": self.device,
                              "trust_remote_code": False, "local_files_only": True},
                encode_kwargs={"batch_size": self.batch_size,
                               "normalize_embeddings": self.input_policy.normalize_vectors,
                               "prompt": ""},
            )
            self._tokenizer_details = None
            if Path(self.model_name).is_dir():
                self._local_assets_sha256 = local_assets_digest(self.asset_path)
            return None
        except Exception as exc:
            # Return only diagnostics: do not retain a traceback holding GPU tensors.
            return self._is_oom(exc), type(exc).__name__

    def _switch_to_cpu(self):
        self._embedding = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        self.device = "cpu"
        self._warn("cpu_fallback", "GPU memory exhausted; using the same model on CPU, which may be slower")

    def _recover_oom(self):
        if (self.batch_size > 1
                and self._batch_retries < settings.LOCAL_EMBEDDING_MAX_BATCH_RETRIES):
            self.batch_size = max(1, self.batch_size // 2)
            self._batch_retries += 1
            self._embedding.encode_kwargs["batch_size"] = self.batch_size
            self._warn("batch_reduced", "GPU memory exhausted; batch size reduced to " + str(self.batch_size))
            return
        self._switch_to_cpu()
        failure = self._try_load()
        if failure:
            raise LocalEmbeddingUnavailable("Same-model CPU load failed (" + failure[1] + ")")

    def _validate(self, vectors, count):
        if len(vectors) != count:
            raise ValueError("embedding_count_mismatch")
        result = []
        dimension = self.dimension
        for row in vectors:
            if not len(row) or (dimension is not None and len(row) != dimension):
                raise ValueError("embedding_dimension_mismatch")
            converted = []
            for number in row:
                if isinstance(number, (bool, str, bytes)):
                    raise ValueError("embedding_invalid_number")
                number = float(number)
                if not math.isfinite(number):
                    raise ValueError("embedding_nonfinite_number")
                converted.append(number)
            if not math.isfinite(math.hypot(*converted)) or not any(converted):
                raise ValueError("embedding_invalid_norm")
            dimension = len(row)
            result.append(converted)
        self.dimension = dimension
        return result

    def _input_details(self):
        if self._tokenizer_details is None:
            try:
                client = self._embedding._client
                tokenizer = client.tokenizer
                identity = tokenizer_digest(tokenizer)
                limit = effective_token_limit(client, tokenizer, self.input_policy.max_tokens)
                self._tokenizer_details = tokenizer, identity, limit
            except Exception as exc:
                raise LocalEmbeddingUnavailable(
                    "Local tokenizer identity/input limit unavailable (" + type(exc).__name__ + ")"
                ) from None
        return self._tokenizer_details

    @property
    def max_input_tokens(self):
        return self._input_details()[2]

    def count_tokens(self, text, *, query=False):
        """Count the exact prepared input, with prefix and special tokens, without truncation."""
        tokenizer, _, _ = self._input_details()
        prepared = self.input_policy.prepare(text, query=query)
        return len(tokenizer.encode(prepared, add_special_tokens=True, truncation=False))

    def embedding_profile(self):
        if self.dimension is None:
            raise LocalEmbeddingUnavailable("Embedding dimension unknown; run model preflight first")
        _, tokenizer_sha256, limit = self._input_details()
        return EmbeddingProfile.create(
            model_id=self.model_name, requested_revision=self.revision,
            actual_revision=self.actual_revision, local_assets_sha256=self._local_assets_sha256,
            dimension=self.dimension, tokenizer_sha256=tokenizer_sha256,
            encoder_versions=self._encoder_versions,
            **self.input_policy.profile_fields(limit),
        )

    def require_compatible(self, stored_profile):
        self.embedding_profile().require_compatible(stored_profile)

    def _encode(self, texts, *, query=False):
        with self._lock:
            if not texts:
                return []
            prepared = [self.input_policy.prepare(text, query=query) for text in texts]
            while True:
                tokenizer, _, limit = self._input_details()
                for text in prepared:
                    check_budget(tokenizer, text, limit, query=query)
                if self._frozen_profile is not None:
                    self.require_compatible(self._frozen_profile.to_record())
                failure = None
                try:
                    vectors = ([self._embedding.embed_query(prepared[0])] if query
                               else self._embedding.embed_documents(prepared))
                    result = self._validate(vectors, len(texts))
                    if self._frozen_profile is None:
                        self._frozen_profile = self.embedding_profile()
                    return result
                except Exception as exc:
                    failure = self._is_oom(exc), type(exc).__name__
                # Outside the exception block, release all traceback-held tensors first.
                self.preflight_passed = False
                if self.device.startswith("cuda") and failure[0]:
                    self._recover_oom()
                    continue
                kind = "semantic search" if query else "ingestion"
                raise LocalEmbeddingUnavailable(
                    "Local embedding failed (" + failure[1] + "); " + kind + " unavailable"
                ) from None

    def preflight(self):
        """Check both interfaces with public fixed probes, never user data."""
        with self._lock:
            self.preflight_passed = False
            self.embed_documents(["模型预检 model check", "waiting period 等待期"])
            self.embed_query("模型预检 model check")
            self.preflight_passed = True
            return self.diagnostics()

    def diagnostics(self):
        return {"status": "ready" if self.preflight_passed else "not_checked",
                "model": self.model_name, "revision": self.revision,
                "actual_revision": self.actual_revision, "device": self.device,
                "batch_size": self.batch_size, "dimension": self.dimension,
                "distance": self.input_policy.distance,
                "max_input_tokens": self.max_input_tokens if self._tokenizer_details is not None else None,
                "embedding_profile": self.embedding_profile().to_record() if self._frozen_profile else None,
                "warnings": list(self.warnings)}

    @property
    def embedding_function(self):
        # Chroma calls this object directly, so expose the same guarded local path.
        return self

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], query=True)[0]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        with self._lock:
            items = list(texts)
            vectors = []
            offset = 0
            while offset < len(items):
                batch = items[offset:offset + self.batch_size]
                vectors.extend(self._encode(batch))
                offset += len(batch)
            return vectors

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, list(texts))
