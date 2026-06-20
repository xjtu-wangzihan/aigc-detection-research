from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("accuracy", "precision", "recall", "f1", "auroc", "human_fpr", "machine_tpr")


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate schema-v1 experiment JSON files.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "summary"))
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No completed experiments.\n"
    values = frame.fillna("").astype(str)
    headers = list(values.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in values.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def load_records(results_dir: Path):
    main_rows, attack_rows = [], []
    for path in results_dir.glob("*/*/seed_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema_version") != 1 or "overall" not in data:
            continue
        overall = data["overall"]
        row = {
            "benchmark": data.get("benchmark"), "method": data.get("method"),
            "seed": data.get("seed"), "encoder": data.get("encoder"),
            "fusion": data.get("fusion"), "tuning": data.get("tuning"),
            **{metric: overall.get(metric) for metric in METRICS},
            "train_seconds": data.get("timing", {}).get("train_seconds"),
            "inference_ms_per_sample": data.get("timing", {}).get("inference_ms_per_sample"),
            "model_size_mb": (data.get("model_size_bytes", 0) or 0) / 1024 / 1024,
            "result_path": str(path),
        }
        main_rows.append(row)
        for attack, metrics in data.get("robustness", {}).get("by_attack", {}).items():
            attack_rows.append({
                "benchmark": data.get("benchmark"), "method": data.get("method"),
                "seed": data.get("seed"), "attack": attack,
                **{metric: metrics.get(metric) for metric in METRICS},
                "result_path": str(path),
            })
    return pd.DataFrame(main_rows), pd.DataFrame(attack_rows)


def mean_std_table(frame: pd.DataFrame, group_columns, metrics):
    if frame.empty:
        return frame
    rows = []
    for keys, part in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["runs"] = len(part)
        for metric in metrics:
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[metric] = "" if values.empty else (
                f"{values.mean():.4f}" if len(values) == 1
                else f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def main():
    args = parse_args()
    results_dir, output_dir = Path(args.results_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main_frame, attack_frame = load_records(results_dir)
    main_frame.to_csv(output_dir / "raw_main_results.csv", index=False, encoding="utf-8-sig")
    attack_frame.to_csv(output_dir / "raw_robustness_results.csv", index=False, encoding="utf-8-sig")

    main_summary = mean_std_table(
        main_frame, ["benchmark", "method"], list(METRICS) + [
            "train_seconds", "inference_ms_per_sample", "model_size_mb"
        ]
    )
    robust_summary = mean_std_table(
        attack_frame, ["benchmark", "method", "attack"], ["accuracy", "f1", "auroc", "human_fpr"]
    )
    ablation = main_summary[main_summary["method"].isin(
        ["style", "roberta_encoder", "roberta_style", "xlmr_encoder", "xlmr_style"]
    )] if not main_summary.empty else main_summary

    main_summary.to_csv(output_dir / "main_results.csv", index=False, encoding="utf-8-sig")
    robust_summary.to_csv(output_dir / "robustness_results.csv", index=False, encoding="utf-8-sig")
    (output_dir / "main_results.md").write_text(markdown_table(main_summary), encoding="utf-8")
    (output_dir / "robustness_results.md").write_text(markdown_table(robust_summary), encoding="utf-8")
    (output_dir / "ablation_results.md").write_text(markdown_table(ablation), encoding="utf-8")
    print(f"completed_runs={len(main_frame)}")
    print(f"main_groups={len(main_summary)} robustness_groups={len(robust_summary)}")
    print(f"saved summaries to {output_dir}")


if __name__ == "__main__":
    main()
