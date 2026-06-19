# from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from attacks import make_attack_benchmark
from datasets import TEXT_COL, load_dataset, stratified_split
from evaluate import evaluate_by_group, evaluate_frame
from models import build_model


def parse_args() :
    parser = argparse.ArgumentParser(description="Train AIGC text detection baselines.")
    parser.add_argument("--data", required=True, help="CSV with text,label,domain,source,attack columns.")
    parser.add_argument("--test-data", help="Optional official test CSV. If omitted, --data is split locally.")
    parser.add_argument("--model", default="hybrid", choices=["word_tfidf", "char_tfidf", "style", "hybrid"])
    parser.add_argument("--out", default="models/hybrid.joblib")
    parser.add_argument("--metrics", default="results/metrics.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robust", action="store_true", help="Evaluate simple perturbation robustness.")
    return parser.parse_args()


def main() :
    args = parse_args()
    df = load_dataset(args.data)
    if args.test_data:
        train_df = df
        test_df = load_dataset(args.test_data)
        split_strategy = "official_train_test"
    else:
        train_df, test_df = stratified_split(df, test_size=args.test_size, seed=args.seed)
        split_strategy = "group_holdout" if "group_id" in df.columns and df["group_id"].nunique() < len(df) else "stratified_random"

    model = build_model(args.model)
    model.fit(train_df[TEXT_COL].tolist(), train_df["label"].to_numpy())

    metrics = {
        "model": args.model,
        "split_strategy": split_strategy,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "overall": evaluate_frame(model, test_df),
        "by_domain": evaluate_by_group(model, test_df, "domain"),
        "by_source": evaluate_by_group(model, test_df, "source"),
    }

    if args.robust:
        attack_df = make_attack_benchmark(test_df)
        metrics["robustness"] = {
            "overall_attacked": evaluate_frame(model, attack_df),
            "by_attack": evaluate_by_group(model, attack_df, "attack"),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print(f"Saved model to {out_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
