from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[.!?。！？；;\n]+")


class StylometricTransformer(BaseEstimator, TransformerMixin):
    """Extract language-light stylometric features for Chinese/English text."""

    feature_names_ = np.array(
        [
            "char_len",
            "token_count",
            "sentence_count",
            "avg_token_len",
            "avg_sentence_tokens",
            "unique_token_ratio",
            "punct_ratio",
            "digit_ratio",
            "upper_ratio",
            "newline_ratio",
            "comma_ratio",
            "quote_ratio",
        ],
        dtype=object,
    )

    def fit(self, X: Iterable[str], y=None):  # noqa: D401
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows = [self._extract(str(text)) for text in X]
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self.feature_names_

    @staticmethod
    def _extract(text: str) -> list[float]:
        chars = max(len(text), 1)
        tokens = TOKEN_RE.findall(text)
        token_count = len(tokens)
        token_lens = [len(t) for t in tokens] or [0]
        sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
        sentence_count = len(sentences)
        unique_ratio = len(set(t.lower() for t in tokens)) / max(token_count, 1)
        punct = sum(1 for c in text if not c.isalnum() and not c.isspace() and not "\u4e00" <= c <= "\u9fff")
        digits = sum(1 for c in text if c.isdigit())
        uppers = sum(1 for c in text if c.isupper())
        newlines = text.count("\n")
        commas = text.count(",") + text.count("，")
        quotes = text.count('"') + text.count("'") + text.count("“") + text.count("”")

        return [
            float(len(text)),
            float(token_count),
            float(sentence_count),
            float(np.mean(token_lens)),
            float(token_count / max(sentence_count, 1)),
            float(unique_ratio),
            float(punct / chars),
            float(digits / chars),
            float(uppers / chars),
            float(newlines / chars),
            float(commas / chars),
            float(quotes / chars),
        ]
