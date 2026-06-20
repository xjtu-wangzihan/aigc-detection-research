from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from data_utils import LABEL_COL, TEXT_COL


def positive_scores(model, texts) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(texts)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(texts), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    return np.asarray(model.predict(texts), dtype=float)


def metrics_from_predictions(y_true, scores, threshold: float = 0.5) -> OrderedDict:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = OrderedDict(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        human_fpr=float(fp / max(tn + fp, 1)),
        machine_tpr=float(tp / max(tp + fn, 1)),
    )
    result["auroc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None
    result["confusion_matrix"] = matrix.tolist()
    result["threshold"] = float(threshold)
    result["n"] = int(len(labels))
    return result


def evaluate_frame(model, df: pd.DataFrame, threshold: float = 0.5) -> OrderedDict:
    return metrics_from_predictions(
        df[LABEL_COL].to_numpy(), positive_scores(model, df[TEXT_COL].tolist()), threshold
    )


def evaluate_by_group(model, df: pd.DataFrame, group_col: str, threshold: float = 0.5) -> dict:
    if group_col not in df.columns:
        return {}
    output = {}
    for group, part in df.groupby(group_col):
        if len(part) >= 2:
            output[str(group)] = evaluate_frame(model, part, threshold)
    return output


def evaluate_scores_by_group(df: pd.DataFrame, scores, group_col: str, threshold: float = 0.5) -> dict:
    if group_col not in df.columns:
        return {}
    frame = df.copy()
    frame["_score"] = np.asarray(scores, dtype=float)
    output = {}
    for group, part in frame.groupby(group_col):
        if len(part) >= 2:
            output[str(group)] = metrics_from_predictions(
                part[LABEL_COL].to_numpy(), part["_score"].to_numpy(), threshold
            )
    return output
