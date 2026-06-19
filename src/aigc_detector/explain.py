from __future__ import annotations

import numpy as np


def _as_dense(row):
    if hasattr(row, "toarray"):
        return row.toarray().ravel()
    return np.asarray(row).ravel()


def explain_linear_text(model, text: str, top_k: int = 12) -> dict:
    """Return approximate feature contributions for sklearn linear pipelines."""
    if "clf" not in model.named_steps:
        return {"error": "Model does not expose a clf step."}
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {"error": "Classifier is not linear or has no coef_."}

    try:
        preprocessor = model[:-1]
        features = preprocessor.transform([text])
        names = preprocessor.get_feature_names_out()
    except Exception as exc:  # pragma: no cover - best-effort demo helper
        return {"error": f"Could not extract feature names: {exc}"}

    values = _as_dense(features[0])
    coef = clf.coef_[0]
    contributions = values * coef
    active = np.flatnonzero(values)
    ranked = sorted(active, key=lambda idx: abs(contributions[idx]), reverse=True)[:top_k]
    items = [
        {
            "feature": str(names[idx]),
            "value": float(values[idx]),
            "contribution": float(contributions[idx]),
            "direction": "AI" if contributions[idx] > 0 else "Human",
        }
        for idx in ranked
    ]
    return {"top_features": items}
