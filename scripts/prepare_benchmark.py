from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASETS = {
    "hc3": ("Hello-SimpleAI/HC3", "all"),
    "hc3_zh": ("Hello-SimpleAI/HC3-Chinese", "all"),
    "mage": ("yaful/MAGE", None),
    "raid": ("liamdugan/raid", None),
}
OUTPUT_COLUMNS = ["text", "label", "domain", "source", "attack", "group_id", "benchmark", "split"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize AIGC detection benchmarks.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Benchmark adapter to use.")
    parser.add_argument("--split", default="train", help="Hugging Face split name.")
    parser.add_argument("--config", help="Optional Hugging Face dataset config; defaults are adapter-specific.")
    parser.add_argument("--output", help="Output CSV path.")
    parser.add_argument("--max-per-class", type=int, default=5000, help="Maximum rows for each label; 0 means all.")
    parser.add_argument("--min-chars", type=int, default=50, help="Discard shorter texts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-test", action="store_true", help="Test adapters without downloading data.")
    return parser.parse_args()


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _first(row: dict, *names: str, default=""):
    for name in names:
        value = row.get(name)
        if value is not None and _text(value):
            return value
    return default


def _hash_group(prefix: str, value) -> str:
    digest = hashlib.sha1(_text(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _answers(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_text(item) for item in value if _text(item)]
    value = _text(value)
    return [value] if value else []


def convert_hc3(row: dict, benchmark: str, split: str) -> list[dict]:
    question = _first(row, "question", "prompt", "id", default="unknown")
    domain = _text(_first(row, "source", "domain", "category", default="unknown"))
    group_id = _hash_group(benchmark, question)
    records = []
    for answer in _answers(row.get("human_answers")):
        records.append(_record(answer, 0, domain, "human", "none", group_id, benchmark, split))
    for answer in _answers(row.get("chatgpt_answers")):
        records.append(_record(answer, 1, domain, "chatgpt", "none", group_id, benchmark, split))
    return records


def _mage_source_parts(raw_source: str, label: int) -> tuple[str, str]:
    source = raw_source.lower().strip() or "unknown"
    if "_human" in source:
        return source.split("_human", 1)[0], "human"
    if "_machine" in source:
        domain, generator = source.split("_machine", 1)
        return domain or "unknown", generator.strip("_") or "machine"
    return "unknown", "human" if label == 0 else source


def convert_mage(row: dict, benchmark: str, split: str) -> list[dict]:
    raw_label = int(_first(row, "label", default=-1))
    if raw_label not in {0, 1}:
        return []
    # MAGE uses 0=machine and 1=human; this project uses the opposite convention.
    label = 1 if raw_label == 0 else 0
    raw_source = _text(_first(row, "src", "source", "text_source", "origin", default="unknown"))
    domain, source = _mage_source_parts(raw_source, label)
    attack = "paraphrase" if "para" in split.lower() or "para" in raw_source.lower() else "none"
    index = _first(row, "index", "id", "idx", default=_first(row, "text", default="unknown"))
    group_id = _hash_group(f"{benchmark}:{domain}", index)
    text = _text(_first(row, "text", "generation", "content"))
    return [_record(text, label, domain, source, attack, group_id, benchmark, split)] if text else []


def convert_raid(row: dict, benchmark: str, split: str) -> list[dict]:
    text = _text(_first(row, "generation", "text", "content"))
    model = _text(_first(row, "model", "source", "generator", default="unknown"))
    raw_label = row.get("label")
    if raw_label is not None and _text(raw_label) in {"0", "1"}:
        label = int(raw_label)
    elif model.lower() in {"human", "original", "real"}:
        label = 0
    elif model and model.lower() != "unknown":
        label = 1
    else:
        return []
    domain = _text(_first(row, "domain", "genre", default="unknown"))
    attack = _text(_first(row, "attack", default="none")) or "none"
    source = "human" if label == 0 else model
    item_id = _first(row, "id", "generation_id", "source_id", default=text)
    group_id = _hash_group(f"{benchmark}:{domain}", item_id)
    return [_record(text, label, domain, source, attack, group_id, benchmark, split)] if text else []


def _record(text, label, domain, source, attack, group_id, benchmark, split) -> dict:
    return {
        "text": _text(text),
        "label": int(label),
        "domain": _text(domain) or "unknown",
        "source": _text(source) or "unknown",
        "attack": _text(attack) or "none",
        "group_id": _text(group_id),
        "benchmark": benchmark,
        "split": split,
    }


CONVERTERS = {
    "hc3": convert_hc3,
    "hc3_zh": convert_hc3,
    "mage": convert_mage,
    "raid": convert_raid,
}


def _load_source(dataset_name: str, config: str | None, split: str, streaming: bool, seed: int):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: run `python -m pip install -r requirements-benchmark.txt`.") from exc

    kwargs = {"split": split, "streaming": streaming}
    try:
        dataset = load_dataset(dataset_name, config, **kwargs) if config else load_dataset(dataset_name, **kwargs)
    except Exception:
        if config:
            dataset = load_dataset(dataset_name, **kwargs)
        else:
            raise
    if streaming and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=10000)
    return dataset


def collect_rows(
    raw_rows: Iterable[dict], benchmark: str, split: str, max_per_class: int, min_chars: int
) -> pd.DataFrame:
    converter = CONVERTERS[benchmark]
    counts = {0: 0, 1: 0}
    records = []
    seen_texts = set()

    for raw in raw_rows:
        for record in converter(dict(raw), benchmark, split):
            label = record["label"]
            text = record["text"]
            if len(text) < min_chars or text in seen_texts:
                continue
            if max_per_class > 0 and counts[label] >= max_per_class:
                continue
            records.append(record)
            seen_texts.add(text)
            counts[label] += 1
        if max_per_class > 0 and all(counts[label] >= max_per_class for label in (0, 1)):
            break

    if not records:
        raise ValueError("No usable rows were produced. Check the dataset split and source schema.")
    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    print(f"Collected rows: human={counts[0]}, machine={counts[1]}, total={len(df)}")
    return df


def self_test() -> None:
    hc3 = collect_rows(
        [{"question": "q", "source": "qa", "human_answers": ["human answer"], "chatgpt_answers": ["machine answer"]}],
        "hc3",
        "train",
        10,
        1,
    )
    assert set(hc3["label"]) == {0, 1} and hc3["group_id"].nunique() == 1

    mage = collect_rows(
        [
            {"text": "machine", "label": 0, "src": "cmv_machine_gpt", "index": 1},
            {"text": "human", "label": 1, "src": "cmv_human", "index": 1},
        ],
        "mage",
        "train",
        10,
        1,
    )
    assert mage.loc[mage["source"] == "human", "label"].item() == 0

    raid = collect_rows(
        [
            {"generation": "human", "model": "human", "domain": "news", "attack": "none", "id": 1},
            {"generation": "machine", "model": "gpt4", "domain": "news", "attack": "paraphrase", "id": 2},
        ],
        "raid",
        "train",
        10,
        1,
    )
    assert set(raid["label"]) == {0, 1}
    print("BENCHMARK ADAPTER SELF-TEST: OK")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or not args.output:
        raise SystemExit("--dataset and --output are required unless --self-test is used.")

    dataset_name, default_config = DATASETS[args.dataset]
    config = args.config if args.config is not None else default_config
    raw_rows = _load_source(dataset_name, config, args.split, args.streaming, args.seed)
    df = collect_rows(raw_rows, args.dataset, args.split, args.max_per_class, args.min_chars)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved normalized benchmark to {output}")


if __name__ == "__main__":
    main()
