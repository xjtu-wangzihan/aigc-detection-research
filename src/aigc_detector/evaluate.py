from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from datasets import LABEL_COL, TEXT_COL


def positive_scores(model, texts) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(texts)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(texts).astype(float)


def evaluate_frame(model, df: pd.DataFrame) -> OrderedDict:
    texts = df[TEXT_COL].tolist()
    y_true = df[LABEL_COL].to_numpy()
    y_pred = model.predict(texts)
    scores = positive_scores(model, texts)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    result = OrderedDict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        human_fpr=float(fp / max(tn + fp, 1)),
        machine_tpr=float(tp / max(tp + fn, 1)),
    )
    if len(set(y_true.tolist())) == 2:
        result["auroc"] = float(roc_auc_score(y_true, scores))
    else:
        result["auroc"] = None
    result["confusion_matrix"] = matrix.tolist()
    result["n"] = int(len(df))
    return result


def evaluate_by_group(model, df: pd.DataFrame, group_col: str) -> dict:
    if group_col not in df.columns:
        return {}
    output = {}
    for group, part in df.groupby(group_col):
        if len(part) >= 2:
            output[str(group)] = evaluate_frame(model, part)
    return output
