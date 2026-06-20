# AIGC Detection Research

面向 HC3-Chinese、HC3-English、MAGE 和 RAID 的可复现 AIGC 文本检测实验。项目同时提供轻量特征模型、Chinese RoBERTa、XLM-R + LoRA，以及编码器与十二维风格特征融合方法。

## 1. 安装

## 2. 准备 benchmark

```powershell
python scripts/prepare_benchmark.py --dataset hc3_zh --split train --max-per-class 5000 --output data/benchmarks/hc3_zh_train.csv
python scripts/prepare_benchmark.py --dataset hc3 --split train --max-per-class 5000 --output data/benchmarks/hc3_en_train.csv
python scripts/prepare_benchmark.py --dataset mage --split train --max-per-class 3000 --output data/benchmarks/mage_train.csv
python scripts/prepare_benchmark.py --dataset mage --split validation --max-per-class 1000 --output data/benchmarks/mage_validation.csv
python scripts/prepare_benchmark.py --dataset mage --split test --max-per-class 1000 --output data/benchmarks/mage_test.csv
python scripts/prepare_benchmark.py --dataset raid --split train --max-per-class 5000 --raid-english --attacks none --output data/benchmarks/raid_train.csv
```

转换器统一输出 text、label、domain、source、attack、group_id、benchmark 和 split。项目标签固定为 0=human、1=machine。

## 3. 固定划分与批量实验

```powershell
python scripts/build_benchmark_splits.py --profile course
python scripts/run_experiments.py --profile course --resume
python scripts/aggregate_results.py
```

先检查任务而不训练：

```powershell
python scripts/run_experiments.py --profile course --dry-run
```

只运行轻量模型：

```powershell
python scripts/run_experiments.py --profile course --families classical --resume
```

只运行单个 benchmark 的深度模型：

```powershell
python scripts/run_experiments.py --profile course --benchmarks hc3_zh --families deep --resume
```

固定划分写入 data/splits/<benchmark>/，单次结果写入 results/<benchmark>/<method>/seed_<seed>.json，聚合表写入 results/summary/。

## 4. GPU 烟测

正式训练前先各取 100 条样本运行一轮：

```powershell
python scripts/train_deep.py --benchmark hc3_zh --train-data data/splits/hc3_zh/train.csv --val-data data/splits/hc3_zh/val.csv --test-data data/splits/hc3_zh/test.csv --mode encoder --epochs 1 --max-train-samples 100 --max-eval-samples 100 --out models/deep-smoke/hc3_zh_encoder --metrics results/deep-smoke/hc3_zh_encoder.json
python scripts/train_deep.py --benchmark hc3 --train-data data/splits/hc3/train.csv --val-data data/splits/hc3/val.csv --test-data data/splits/hc3/test.csv --mode fusion --epochs 1 --max-train-samples 100 --max-eval-samples 100 --out models/deep-smoke/hc3_xlmr_style --metrics results/deep-smoke/hc3_xlmr_style.json
```

## 5. 测试与边界

```powershell
python scripts/prepare_benchmark.py --self-test
python scripts/build_benchmark_splits.py --self-test
python -m unittest discover -s tests -v
python -m compileall scripts src tests
```

sample_dataset.csv 只用于烟测。报告中的正式结论必须来自固定划分和可追溯 JSON。旧 results/hc3_zh_metrics.json 使用 75/25 划分，只能作为初步结果，不能与统一主表混排。
