from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("hc3_zh", "hc3", "mage", "raid")
POOL_FILES = {"hc3_zh": "hc3_zh_train.csv", "hc3": "hc3_en_train.csv", "raid": "raid_train.csv"}
MAGE_FILES = {"train": "mage_train.csv", "val": "mage_validation.csv", "test": "mage_test.csv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build deterministic course-profile benchmark splits.")
    parser.add_argument("--profile", default="course", choices=["course"])
    parser.add_argument("--benchmark", default="all", choices=["all", *BENCHMARKS])
    parser.add_argument("--input-dir", default=str(PROJECT_ROOT / "data" / "benchmarks"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame.copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].astype(int)
    data = data[data["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    if "group_id" not in data:
        data["group_id"] = data["text"].map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:16])
    data["group_id"] = data["group_id"].fillna("").astype(str)
    return data


def balanced_sample(frame: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in (0, 1):
        part = frame[frame["label"] == label]
        if len(part) < per_class:
            raise ValueError(f"Label {label} has {len(part)} rows; expected {per_class}.")
        parts.append(part.sample(n=per_class, random_state=seed))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def split_grouped(frame: pd.DataFrame, seed: int):
    best = None
    for offset in range(100):
        current = seed + offset
        outer = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=current)
        train_idx, rest_idx = next(outer.split(frame, frame["label"], frame["group_id"]))
        rest = frame.iloc[rest_idx]
        inner = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=current)
        val_rel, test_rel = next(inner.split(rest, rest["label"], rest["group_id"]))
        splits = (frame.iloc[train_idx], rest.iloc[val_rel], rest.iloc[test_rel])
        if any(set(part["label"]) != {0, 1} for part in splits):
            continue
        score = sum(abs(len(part) / len(frame) - ratio) for part, ratio in zip(splits, (0.6, 0.2, 0.2)))
        if best is None or score < best[0]:
            best = (score, splits)
    if best is None:
        raise ValueError("Could not build group-disjoint splits with both labels.")
    return tuple(x.sample(frac=1, random_state=seed).reset_index(drop=True) for x in best[1])


def validate(parts, max_class_ratio_gap: float = 0.05):
    for name, frame in parts.items():
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"{name} does not contain both labels.")
        if frame["text"].duplicated().any():
            raise ValueError(f"{name} contains duplicate text.")
        proportions = frame["label"].value_counts(normalize=True)
        if abs(float(proportions[0]) - float(proportions[1])) > max_class_ratio_gap:
            raise ValueError(f"{name} is not class-balanced: {frame['label'].value_counts().to_dict()}")
    names = list(parts)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = set(parts[left]["group_id"]) & set(parts[right]["group_id"])
            if overlap:
                raise ValueError(f"group_id leakage between {left} and {right}: {len(overlap)}")
            duplicated = set(parts[left]["text"]) & set(parts[right]["text"])
            if duplicated:
                raise ValueError(f"text leakage between {left} and {right}: {len(duplicated)}")


def missing_instructions(benchmark: str, input_dir: Path) -> str:
    if benchmark == "mage":
        return ("Prepare MAGE train/validation/test with prepare_benchmark.py; expected files: "
                + ", ".join(str(input_dir / x) for x in MAGE_FILES.values()))
    if benchmark == "raid":
        return ("Prepare clean English RAID: python scripts/prepare_benchmark.py --dataset raid "
                "--split train --max-per-class 5000 --raid-english --attacks none "
                f"--output {input_dir / POOL_FILES['raid']}")
    return f"Missing normalized input for {benchmark} in {input_dir}."


def build_one(benchmark: str, input_dir: Path, output_root: Path, seed: int):
    if benchmark == "mage":
        paths = {name: input_dir / filename for name, filename in MAGE_FILES.items()}
        if any(not path.exists() for path in paths.values()):
            raise FileNotFoundError(missing_instructions(benchmark, input_dir))
        parts = {
            "train": balanced_sample(normalize(pd.read_csv(paths["train"])), 3000, seed),
            "val": balanced_sample(normalize(pd.read_csv(paths["val"])), 1000, seed),
            "test": balanced_sample(normalize(pd.read_csv(paths["test"])), 1000, seed),
        }
        sources = paths
    else:
        path = input_dir / POOL_FILES[benchmark]
        if not path.exists():
            raise FileNotFoundError(missing_instructions(benchmark, input_dir))
        pool = balanced_sample(normalize(pd.read_csv(path)), 5000, seed)
        train, val, test = split_grouped(pool, seed)
        parts, sources = {"train": train, "val": val, "test": test}, {"pool": path}

    validate(parts)
    target = output_root / benchmark
    target.mkdir(parents=True, exist_ok=True)
    for name, frame in parts.items():
        frame.to_csv(target / f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "profile": "course", "benchmark": benchmark, "seed": seed,
        "sources": {n: {"path": str(p), "sha256": file_hash(p)} for n, p in sources.items()},
        "splits": {
            n: {"path": str(target / f"{n}.csv"), "rows": int(len(f)),
                "labels": {str(k): int(v) for k, v in f["label"].value_counts().sort_index().items()},
                "groups": int(f["group_id"].nunique()), "sha256": file_hash(target / f"{n}.csv")}
            for n, f in parts.items()
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def self_test():
    frame = pd.DataFrame([{"text": f"text-{g}-{y}", "label": y, "group_id": f"g-{g}"}
                          for g in range(100) for y in (0, 1)])
    first = dict(zip(("train", "val", "test"), split_grouped(frame, 42)))
    second = dict(zip(("train", "val", "test"), split_grouped(frame, 42)))
    validate(first)
    assert {k: sorted(v.group_id) for k, v in first.items()} == {
        k: sorted(v.group_id) for k, v in second.items()
    }
    print("SPLIT SELF-TEST: OK")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    selected = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    failures = []
    for benchmark in selected:
        try:
            manifest = build_one(benchmark, input_dir, output_dir, args.seed)
            print(benchmark, {k: v["rows"] for k, v in manifest["splits"].items()})
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{benchmark}: {exc}")
    if failures:
        raise SystemExit("\n\n".join(failures))


if __name__ == "__main__":
    main()
