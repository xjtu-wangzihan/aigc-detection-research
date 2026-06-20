from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("hc3_zh", "hc3", "mage", "raid")
CLASSICAL = ("word_tfidf", "char_tfidf", "style", "hybrid")
DEEP = ("encoder", "fusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the course-profile experiment matrix.")
    parser.add_argument("--profile", default="course", choices=["course"])
    parser.add_argument("--benchmarks", default="all", help="all or comma-separated benchmark names")
    parser.add_argument("--families", default="classical,deep", help="classical,deep")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def complete_result(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("schema_version") == 1 and "overall" in data and "checkpoint" in data
    except (OSError, json.JSONDecodeError):
        return False


def split_paths(benchmark):
    base = ROOT / "data" / "splits" / benchmark
    paths = {name: base / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Build benchmark splits first. Missing: " + ", ".join(missing))
    return paths


def classical_commands(benchmark, paths):
    for method in CLASSICAL:
        for seed in (42, 43, 44):
            result = ROOT / "results" / benchmark / method / f"seed_{seed}.json"
            model = ROOT / "models" / benchmark / method / f"seed_{seed}.joblib"
            command = [
                sys.executable, str(ROOT / "src" / "aigc_detector" / "train.py"),
                "--data", str(paths["train"]), "--val-data", str(paths["val"]),
                "--test-data", str(paths["test"]), "--benchmark", benchmark,
                "--model", method, "--seed", str(seed), "--out", str(model),
                "--metrics", str(result), "--robust",
            ]
            yield command, result


def deep_commands(benchmark, paths):
    for mode in DEEP:
        suffix = "style" if mode == "fusion" else "encoder"
        method = ("roberta" if benchmark == "hc3_zh" else "xlmr") + f"_{suffix}"
        result = ROOT / "results" / benchmark / method / "seed_42.json"
        model = ROOT / "models" / "deep" / benchmark / method / "seed_42"
        command = [
            sys.executable, str(ROOT / "scripts" / "train_deep.py"),
            "--benchmark", benchmark, "--train-data", str(paths["train"]),
            "--val-data", str(paths["val"]), "--test-data", str(paths["test"]),
            "--mode", mode, "--seed", "42", "--out", str(model),
            "--metrics", str(result), "--robust",
        ]
        yield command, result


def main():
    args = parse_args()
    benchmarks = BENCHMARKS if args.benchmarks == "all" else tuple(
        item.strip() for item in args.benchmarks.split(",") if item.strip()
    )
    invalid = set(benchmarks) - set(BENCHMARKS)
    if invalid:
        raise SystemExit(f"Unknown benchmarks: {sorted(invalid)}")
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    invalid_families = families - {"classical", "deep"}
    if invalid_families:
        raise SystemExit(f"Unknown families: {sorted(invalid_families)}")
    if "deep" in families and not args.dry_run:
        missing = [name for name in ("torch", "transformers", "peft", "accelerate")
                   if importlib.util.find_spec(name) is None]
        if missing:
            raise SystemExit(
                "Missing deep dependencies: " + ", ".join(missing)
                + ". Install CUDA PyTorch and requirements-deep.txt."
            )

    jobs = []
    for benchmark in benchmarks:
        paths = split_paths(benchmark)
        if "classical" in families:
            jobs.extend(classical_commands(benchmark, paths))
        if "deep" in families:
            jobs.extend(deep_commands(benchmark, paths))

    completed, skipped = 0, 0
    for index, (command, result) in enumerate(jobs, 1):
        if args.resume and complete_result(result):
            print(f"[{index}/{len(jobs)}] SKIP {result}")
            skipped += 1
            continue
        print(f"[{index}/{len(jobs)}] " + subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            result.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                result.check_returncode()           
        completed += 1
    print(f"jobs={len(jobs)} executed_or_planned={completed} skipped={skipped}")


if __name__ == "__main__":
    main()
