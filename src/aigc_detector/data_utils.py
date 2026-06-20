from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split


TEXT_COL = "text"
LABEL_COL = "label"


def normalize_label(value) -> int:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "ai", "aigc", "machine", "generated", "llm", "gpt"}:
            return 1
        if lowered in {"0", "human", "real", "authentic"}:
            return 0
    return int(value)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].map(normalize_label).astype(int)
    for optional in ["domain", "source", "attack"]:
        if optional not in df.columns:
            df[optional] = "unknown"
        df[optional] = df[optional].fillna("unknown").astype(str)
    return df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)


def stratified_split(df: pd.DataFrame, test_size: float, seed: int):
    if "group_id" in df.columns and df["group_id"].nunique() < len(df):
        groups = df["group_id"].fillna("").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(df, df[LABEL_COL], groups=groups))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
    y = df[LABEL_COL]
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=stratify)
