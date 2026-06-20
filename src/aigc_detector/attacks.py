from __future__ import annotations

import re

import pandas as pd

from data_utils import TEXT_COL


CONNECTOR_REPLACEMENTS = {
    "因此": "所以",
    "然而": "但是",
    "此外": "另外",
    "综上": "总的来看",
    "first": "to begin with",
    "therefore": "so",
    "however": "but",
    "moreover": "also",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def drop_light_punctuation(text: str) -> str:
    return re.sub(r"[,，;；:：]", " ", text)


def connector_swap(text: str) -> str:
    changed = text
    for src, dst in CONNECTOR_REPLACEMENTS.items():
        changed = re.sub(re.escape(src), dst, changed, flags=re.IGNORECASE)
    return changed


def sentence_shuffle(text: str) -> str:
    parts = [p.strip() for p in re.split(r"([。！？.!?])", text) if p.strip()]
    if len(parts) < 4:
        return text
    chunks = ["".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]
    if len(chunks) < 3:
        return text
    return chunks[1] + chunks[0] + "".join(chunks[2:])


ATTACKS = {
    "whitespace": normalize_whitespace,
    "punct_drop": drop_light_punctuation,
    "connector_swap": connector_swap,
    "sentence_shuffle": sentence_shuffle,
}


def make_attack_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attack_name, fn in ATTACKS.items():
        attacked = df.copy()
        attacked[TEXT_COL] = attacked[TEXT_COL].map(lambda text: fn(str(text)))
        attacked["attack"] = attack_name
        rows.append(attacked)
    return pd.concat(rows, ignore_index=True)
