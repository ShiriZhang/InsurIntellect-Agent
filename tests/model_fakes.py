"""Minimal local tokenizer contract for offline model lifecycle tests."""

import json
from types import SimpleNamespace


class TokenizerStub:
    model_max_length = 512
    special_tokens_map = {"cls_token": "[CLS]", "sep_token": "[SEP]"}
    init_kwargs = {}

    def __init__(self, rule="characters"):
        self.rule = rule
        self.backend_tokenizer = self

    def to_str(self):
        return json.dumps({"model": {"type": self.rule}, "normalizer": None})

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        assert truncation is False
        return [1] * (len(text) + (2 if add_special_tokens else 0))


def client_stub():
    return SimpleNamespace(tokenizer=TokenizerStub(), max_seq_length=512)
