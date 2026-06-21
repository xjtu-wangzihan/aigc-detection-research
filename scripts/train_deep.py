from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src" / "aigc_detector"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from attacks import make_attack_benchmark
from data_utils import LABEL_COL, TEXT_COL, load_dataset
from deep_models import (
    DeepModelSpec, create_model, load_checkpoint, parameter_counts,
    require_deep_dependencies, save_checkpoint,
)
from evaluate import evaluate_scores_by_group, metrics_from_predictions
from features import StylometricTransformer
from result_utils import dump_json, environment_info, path_size_bytes, sha256_file


BENCHMARK_DEFAULTS = {
    "hc3_zh": {"encoder": "hfl/chinese-roberta-wwm-ext", "tuning": "full"},
    "hc3": {"encoder": "xlm-roberta-base", "tuning": "lora"},
    "mage": {"encoder": "xlm-roberta-base", "tuning": "lora"},
    "raid": {"encoder": "xlm-roberta-base", "tuning": "lora"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune encoder-only or encoder+style AIGC detectors.")
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARK_DEFAULTS))
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--mode", required=True, choices=["encoder", "fusion"])
    parser.add_argument("--encoder")
    parser.add_argument("--tuning", choices=["full", "lora"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--robust", action="store_true")
    return parser.parse_args()


def dependency_objects():
    torch, _, AutoTokenizer = require_deep_dependencies()
    try:
        from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("Install requirements-deep.txt.") from exc
    return torch, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed


class CsvTextDataset:
    def __init__(self, frame, tokenizer, max_length, style_values):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style_values = np.asarray(style_values, dtype=np.float32)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            str(self.frame.iloc[index][TEXT_COL]), truncation=True,
            max_length=self.max_length, padding=False,
        )
        encoded["labels"] = int(self.frame.iloc[index][LABEL_COL])
        encoded["style_features"] = self.style_values[index]
        return encoded


class StyleCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch
        clean = [dict(item) for item in features]
        styles = torch.tensor(np.stack([item.pop("style_features") for item in clean]), dtype=torch.float32)
        labels = torch.tensor([item.pop("labels") for item in clean], dtype=torch.long)
        batch = self.tokenizer.pad(clean, padding=True, return_tensors="pt")
        batch["style_features"], batch["labels"] = styles, labels
        return batch


def style_arrays(train_df, val_df, test_df, use_style):
    if not use_style:
        zeros = [np.zeros((len(frame), 12), dtype=np.float32) for frame in (train_df, val_df, test_df)]
        return None, *zeros
    extractor, scaler = StylometricTransformer(), StandardScaler()
    train = scaler.fit_transform(extractor.transform(train_df[TEXT_COL]))
    val = scaler.transform(extractor.transform(val_df[TEXT_COL]))
    test = scaler.transform(extractor.transform(test_df[TEXT_COL]))
    return scaler, train.astype(np.float32), val.astype(np.float32), test.astype(np.float32)


def scores_from_prediction(prediction):
    logits = np.asarray(prediction.predictions)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def main():
    args = parse_args()
    torch, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed = dependency_objects()
    set_seed(args.seed)
    defaults = BENCHMARK_DEFAULTS[args.benchmark]
    encoder_name, tuning = args.encoder or defaults["encoder"], args.tuning or defaults["tuning"]
    use_style = args.mode == "fusion"
    batch_size = args.batch_size or (8 if tuning == "full" else 4)
    accumulation = args.gradient_accumulation or (2 if tuning == "full" else 4)
    learning_rate = args.learning_rate or (2e-5 if tuning == "full" else 1e-4)

    train_df = load_dataset(args.train_data)
    val_df = load_dataset(args.val_data)
    test_df = load_dataset(args.test_data)
    scaler, train_style, val_style, test_style = style_arrays(train_df, val_df, test_df, use_style)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    train_set = CsvTextDataset(train_df, tokenizer, args.max_length, train_style)
    val_set = CsvTextDataset(val_df, tokenizer, args.max_length, val_style)
    test_set = CsvTextDataset(test_df, tokenizer, args.max_length, test_style)
    spec = DeepModelSpec(encoder_name=encoder_name, use_style=use_style, tuning=tuning)
    model = create_model(spec)

    def compute_metrics(prediction):
        return dict(metrics_from_predictions(prediction.label_ids, scores_from_prediction(prediction), args.threshold))

    class BestCheckpointCallback(TrainerCallback):
        def __init__(self):
            self.best = float("-inf")

        def on_evaluate(self, training_args, state, control, metrics=None, model=None, **kwargs):
            score = (metrics or {}).get("eval_auroc")
            if score is not None and score > self.best:
                self.best = score
                save_checkpoint(model, tokenizer, scaler, args.out)
            return control

    training_args = TrainingArguments(
        output_dir=str(Path(args.out) / "_trainer"), learning_rate=learning_rate,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation, num_train_epochs=args.epochs,
        weight_decay=0.01, eval_strategy="epoch", save_strategy="no",
        logging_strategy="steps", logging_steps=50, load_best_model_at_end=False,
        fp16=torch.cuda.is_available(), gradient_checkpointing=True, max_grad_norm=1.0,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
        seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set,
        data_collator=StyleCollator(tokenizer), compute_metrics=compute_metrics,
        callbacks=[BestCheckpointCallback()],
    )
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    if not (Path(args.out) / "model_spec.json").exists():
        save_checkpoint(model, tokenizer, scaler, args.out)

    best_model, tokenizer, scaler, spec = load_checkpoint(args.out)
    best_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    evaluator = Trainer(
        model=best_model, args=training_args, data_collator=StyleCollator(tokenizer),
        compute_metrics=compute_metrics,
    )
    val_prediction = evaluator.predict(val_set)
    started = time.perf_counter()
    test_prediction = evaluator.predict(test_set)
    inference_seconds = time.perf_counter() - started
    val_scores, test_scores = scores_from_prediction(val_prediction), scores_from_prediction(test_prediction)
    overall = metrics_from_predictions(test_df[LABEL_COL], test_scores, args.threshold)
    method = ("roberta" if args.benchmark == "hc3_zh" else "xlmr") + f"_{'style' if use_style else 'encoder'}"

    result = {
        "schema_version": 1, "benchmark": args.benchmark, "method": method,
        "encoder": encoder_name, "fusion": use_style, "tuning": tuning, "seed": args.seed,
        "split_strategy": "fixed_files", "threshold": args.threshold,
        "data": {
            "train_path": args.train_data, "val_path": args.val_data, "test_path": args.test_data,
            "train_sha256": sha256_file(args.train_data), "val_sha256": sha256_file(args.val_data),
            "test_sha256": sha256_file(args.test_data), "train_size": len(train_df),
            "val_size": len(val_df), "test_size": len(test_df),
        },
        "samples": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "hyperparameters": {
            "max_length": args.max_length, "epochs": args.epochs, "batch_size": batch_size,
            "gradient_accumulation": accumulation, "learning_rate": learning_rate,
            "weight_decay": 0.01, "fp16": bool(torch.cuda.is_available()),
            "lora_r": spec.lora_r if tuning == "lora" else None,
            "lora_alpha": spec.lora_alpha if tuning == "lora" else None,
        },
        "validation": metrics_from_predictions(val_df[LABEL_COL], val_scores, args.threshold),
        "overall": overall,
        "by_domain": evaluate_scores_by_group(test_df, test_scores, "domain", args.threshold),
        "by_source": evaluate_scores_by_group(test_df, test_scores, "source", args.threshold),
        "timing": {
            "train_seconds": train_seconds, "inference_seconds": inference_seconds,
            "inference_ms_per_sample": inference_seconds * 1000.0 / max(len(test_df), 1),
        },
        "parameters": parameter_counts(best_model), "checkpoint": args.out,
        "model_size_bytes": path_size_bytes(args.out),
        "environment": {**environment_info(), "cuda": bool(torch.cuda.is_available())},
    }

    if args.robust:
        attacked = make_attack_benchmark(test_df)
        raw = StylometricTransformer().transform(attacked[TEXT_COL])
        styles = scaler.transform(raw).astype(np.float32) if scaler is not None else np.zeros((len(attacked), 12), dtype=np.float32)
        attacked_set = CsvTextDataset(attacked, tokenizer, args.max_length, styles)
        attacked_scores = scores_from_prediction(evaluator.predict(attacked_set))
        result["robustness"] = {
            "overall_attacked": metrics_from_predictions(attacked[LABEL_COL], attacked_scores, args.threshold),
            "by_attack": {
                "original": overall,
                **evaluate_scores_by_group(attacked, attacked_scores, "attack", args.threshold),
            },
        }

    dump_json(result, args.metrics)
    print(f"Overall: {dict(overall)}")
    print(f"Saved checkpoint to {args.out}")
    print(f"Saved metrics to {args.metrics}")


if __name__ == "__main__":
    main()
