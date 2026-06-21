from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src" / "aigc_detector"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import metrics_from_predictions
from features import StylometricTransformer
from prepare_benchmark import collect_rows
from build_benchmark_splits import split_grouped, validate
from aggregate_results import single_run_table
from train_deep import style_arrays


class CoreTests(unittest.TestCase):
    def test_metrics(self):
        metrics = metrics_from_predictions([0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["human_fpr"], 0.5)
        self.assertAlmostEqual(metrics["machine_tpr"], 1.0)

    def test_style_shape(self):
        values = StylometricTransformer().transform(["human text.", "机器生成文本。"])
        self.assertEqual(values.shape, (2, 12))
        self.assertTrue(np.isfinite(values).all())

    def test_mage_label_flip(self):
        frame = collect_rows([
            {"text": "machine text", "label": 0, "src": "cmv_machine_gpt", "index": 1},
            {"text": "human text", "label": 1, "src": "cmv_human", "index": 1},
        ], "mage", "train", 10, 1)
        self.assertEqual(frame.loc[frame["source"] == "human", "label"].item(), 0)

    def test_hc3_pair_shares_group(self):
        frame = collect_rows([{
            "question": "same question", "source": "qa",
            "human_answers": ["human answer"], "chatgpt_answers": ["machine answer"],
        }], "hc3", "train", 10, 1)
        self.assertEqual(set(frame["label"]), {0, 1})
        self.assertEqual(frame["group_id"].nunique(), 1)

    def test_raid_english_filter_and_labels(self):
        frame = collect_rows([
            {"generation": "human news", "model": "human", "domain": "NYT News", "source_id": 1},
            {"generation": "machine news", "model": "gpt4", "domain": "NYT News", "source_id": 2},
            {"generation": "excluded code", "model": "gpt4", "domain": "Python Code", "source_id": 3},
        ], "raid", "train", 10, 1, raid_english=True)
        self.assertEqual(set(frame["label"]), {0, 1})
        self.assertEqual(set(frame["domain"]), {"news"})

    def test_style_scaler_fits_training_only(self):
        train = pd.DataFrame({"text": ["short.", "a much longer training sentence."], "label": [0, 1]})
        val = pd.DataFrame({"text": ["validation text " * 20], "label": [0]})
        test = pd.DataFrame({"text": ["test text " * 30], "label": [1]})
        scaler, train_values, _, _ = style_arrays(train, val, test, True)
        raw_train = StylometricTransformer().transform(train["text"])
        np.testing.assert_allclose(scaler.mean_, raw_train.mean(axis=0))
        np.testing.assert_allclose(train_values.mean(axis=0), np.zeros(12), atol=1e-6)

    def test_group_splits_are_disjoint_and_reproducible(self):
        frame = pd.DataFrame([
            {"text": f"text-{g}-{y}", "label": y, "group_id": f"group-{g}"}
            for g in range(100) for y in (0, 1)
        ])
        one = dict(zip(("train", "val", "test"), split_grouped(frame, 42)))
        two = dict(zip(("train", "val", "test"), split_grouped(frame, 42)))
        validate(one)
        self.assertEqual(
            {k: sorted(v.group_id) for k, v in one.items()},
            {k: sorted(v.group_id) for k, v in two.items()},
        )

    def test_single_run_summary_columns(self):
        frame = pd.DataFrame([{
            "benchmark": "hc3", "method": "style", "seed": 42,
            "accuracy": 0.9, "f1": 0.8,
        }])
        summary = single_run_table(frame, ["benchmark", "method"], ["accuracy", "f1"])
        self.assertEqual(list(summary.columns), ["benchmark", "method", "seed", "accuracy", "f1"])
        self.assertNotIn("runs", summary.columns)
        self.assertNotIn("±", summary.to_csv(index=False))


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft")),
    "deep dependencies are not installed",
)
class DeepShapeTests(unittest.TestCase):
    def test_encoder_and_fusion_forward_shapes(self):
        import torch
        from deep_models import DeepModelSpec, create_model

        class Config:
            hidden_size = 8

        class TinyEncoder(torch.nn.Module):
            config = Config()

            def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
                batch, length = input_ids.shape
                output = torch.zeros(batch, length, 8)
                return type("Output", (), {"last_hidden_state": output})()

        ids = torch.ones(2, 4, dtype=torch.long)
        mask = torch.ones_like(ids)
        for use_style in (False, True):
            spec = DeepModelSpec("tiny", use_style, "full")
            model = create_model(spec, encoder=TinyEncoder())
            result = model(
                input_ids=ids, attention_mask=mask,
                style_features=torch.zeros(2, 12), labels=torch.tensor([0, 1]),
            )
            self.assertEqual(tuple(result.logits.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
