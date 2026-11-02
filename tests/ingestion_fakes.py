"""Offline local encoder stand-in; does not establish real model quality."""

from app.core.config import Settings
from app.services.embedding_profile import EmbeddingProfile, InputPolicy


class LocalEncoderFake:
    dimension = 3
    max_input_tokens = 160

    def __init__(self):
        self.input_policy = InputPolicy.from_settings(Settings(_env_file=None))
        self.calls = []
        self.device = "cpu"

    def count_tokens(self, text, **kwargs):
        return len(text) + 2

    def preflight(self):
        return self.diagnostics()

    def embed_documents(self, texts):
        assert all(self.count_tokens(t) <= self.max_input_tokens for t in texts)
        self.calls.append(list(texts))
        return [[1.0, 0.5, 0.25] for _ in texts]

    def embedding_profile(self):
        return EmbeddingProfile.create(model="test-local", dimension=3,
                                       **self.input_policy.profile_fields(self.max_input_tokens))

    def require_compatible(self, profile):
        self.embedding_profile().require_compatible(profile)

    def diagnostics(self):
        return {"device": self.device, "batch_size": 2, "warnings": []}
