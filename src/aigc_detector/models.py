from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from features import StylometricTransformer


def _classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)


def build_model(name: str = "hybrid") -> Pipeline:
    """Build a named baseline detector."""
    name = name.lower()

    if name == "word_tfidf":
        return Pipeline(
            [
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=50000)),
                ("clf", _classifier()),
            ]
        )

    if name == "char_tfidf":
        return Pipeline(
            [
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=80000)),
                ("clf", _classifier()),
            ]
        )

    if name == "style":
        return Pipeline(
            [
                ("style", StylometricTransformer()),
                ("scale", StandardScaler()),
                ("clf", _classifier()),
            ]
        )

    if name == "hybrid":
        features = FeatureUnion(
            [
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, max_features=60000)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=30000)),
                (
                    "style",
                    Pipeline(
                        [
                            ("extract", StylometricTransformer()),
                            ("scale", StandardScaler()),
                        ]
                    ),
                ),
            ]
        )
        return Pipeline([("features", features), ("clf", _classifier())])

    raise ValueError(f"Unknown model '{name}'. Choose from word_tfidf, char_tfidf, style, hybrid.")
