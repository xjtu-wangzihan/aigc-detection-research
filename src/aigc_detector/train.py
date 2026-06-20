from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib

from attacks import make_attack_benchmark
from data_utils import LABEL_COL, TEXT_COL, load_dataset, stratified_split
from evaluate import evaluate_scores_by_group, metrics_from_predictions, positive_scores
from models import build_model
from result_utils import dump_json, environment_info, path_size_bytes, sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight AIGC text detection baselines.")
    parser.add_argument("--data", required=True, help="Training CSV, or a pool CSV when --test-data is omitted.")
    parser.add_argument("--val-data", help="Optional fixed validation CSV.")
    parser.add_argument("--test-data", help="Optional fixed test CSV.")
    parser.add_argument("--benchmark", help="Benchmark name stored in result metadata.")
    parser.add_argument("--model", default="hybrid", choices=["word_tfidf", "char_tfidf", "style", "hybrid"])
    parser.add_argument("--out", default="models/hybrid.joblib")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--robust", action="store_true")
    return parser.parse_args()


def infer_benchmark(df, explicit: str | None) -> str:
    if explicit:
        return explicit
    if "benchmark" in df.columns and df["benchmark"].nunique() == 1:
        return str(df["benchmark"].iloc[0])
    return "unknown"


def grouped_metrics(df, scores, threshold):
    return {
        "by_domain": evaluate_scores_by_group(df, scores, "domain", threshold),
        "by_source": evaluate_scores_by_group(df, scores, "source", threshold),
    }


def main() -> None:
    args = parse_args()
    train_df = load_dataset(args.data)
    val_df = load_dataset(args.val_data) if args.val_data else None
    if args.test_data:
        test_df = load_dataset(args.test_data)
        split_strategy = "fixed_files"
    else:
        train_df, test_df = stratified_split(train_df, test_size=args.test_size, seed=args.seed)
        split_strategy = "group_holdout" if "group_id" in train_df.columns else "stratified_random"

    model = build_model(args.model, seed=args.seed)
    started = time.perf_counter()
    model.fit(train_df[TEXT_COL].tolist(), train_df[LABEL_COL].to_numpy())
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test_scores = positive_scores(model, test_df[TEXT_COL].tolist())
    inference_seconds = time.perf_counter() - started
    overall = metrics_from_predictions(test_df[LABEL_COL].to_numpy(), test_scores, args.threshold)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    result = {
        "schema_version": 1,
        "benchmark": infer_benchmark(train_df, args.benchmark),
        "method": args.model,
        "encoder": None,
        "fusion": args.model == "hybrid",
        "tuning": "logistic_regression",
        "seed": args.seed,
        "split_strategy": split_strategy,
        "threshold": args.threshold,
        "data": {
            "train_path": str(Path(args.data)),
            "val_path": str(Path(args.val_data)) if args.val_data else None,
            "test_path": str(Path(args.test_data)) if args.test_data else None,
            "train_sha256": sha256_file(args.data),
            "val_sha256": sha256_file(args.val_data) if args.val_data else None,
            "test_sha256": sha256_file(args.test_data) if args.test_data else None,
            "train_size": int(len(train_df)),
            "val_size": int(len(val_df)) if val_df is not None else 0,
            "test_size": int(len(test_df)),
        },
        "samples": {"train": int(len(train_df)), "val": int(len(val_df)) if val_df is not None else 0,
                    "test": int(len(test_df))},
        "hyperparameters": {"class_weight": "balanced", "solver": "liblinear", "max_iter": 2000},
        "overall": overall,
        **grouped_metrics(test_df, test_scores, args.threshold),
        "timing": {
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "inference_ms_per_sample": inference_seconds * 1000.0 / max(len(test_df), 1),
        },
        "checkpoint": str(output_path),
        "model_size_bytes": 0,
        "environment": environment_info(),
    }

    if val_df is not None:
        val_scores = positive_scores(model, val_df[TEXT_COL].tolist())
        result["validation"] = metrics_from_predictions(
            val_df[LABEL_COL].to_numpy(), val_scores, args.threshold
        )

    if args.robust:
        attacked = make_attack_benchmark(test_df)
        attacked_scores = positive_scores(model, attacked[TEXT_COL].tolist())
        result["robustness"] = {
            "overall_attacked": metrics_from_predictions(
                attacked[LABEL_COL].to_numpy(), attacked_scores, args.threshold
            ),
            "by_attack": {
                "original": overall,
                **evaluate_scores_by_group(attacked, attacked_scores, "attack", args.threshold),
            },
        }

    result["model_size_bytes"] = path_size_bytes(output_path)
    dump_json(result, args.metrics)
    print(f"Overall: {dict(overall)}")
    print(f"Saved model to {output_path}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()
