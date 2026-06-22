# AIGC Text Detection Research

一个面向中英文场景的 AIGC 文本检测实验项目，支持公开数据集标准化、无泄漏数据划分、传统与深度模型训练、文本扰动评估、结果可视化和 Streamlit 演示。

> 检测结果仅用于研究和辅助复核，不应作为处分、定责或学术不端认定的唯一依据。

## 功能

- 支持 HC3-Chinese、HC3-English、MAGE 和 RAID；
- 统一使用 `0=human`、`1=machine` 的标签格式；
- 基于 `group_id` 划分数据，降低关联样本泄漏风险；
- 提供 Word TF-IDF、Char TF-IDF、风格特征和混合特征基线；
- 支持 RoBERTa、XLM-R 以及编码器与风格特征融合模型；
- 评估空白、标点、连接词和句序扰动下的鲁棒性；
- 输出 Accuracy、F1、AUROC、Human FPR、推理时间和模型大小；
- 导出 PNG、SVG、PDF 图表，并提供交互式文本检测页面。

## 项目结构

```text
.
├── demo/app.py                    # Streamlit 演示
├── scripts/
│   ├── prepare_benchmark.py       # 下载并标准化数据集
│   ├── build_benchmark_splits.py  # 构建训练、验证和测试集
│   ├── run_experiments.py         # 执行实验矩阵
│   ├── train_deep.py              # 训练深度模型
│   ├── aggregate_results.py       # 聚合实验结果
│   └── plot_results.py            # 生成结果图表
├── src/aigc_detector/             # 特征、模型、评估与预测实现
├── data/                          # 数据文件，默认不纳入 Git
├── models/                        # 模型文件，默认不纳入 Git
├── results/                       # 实验结果，默认不纳入 Git
└── figures/generated/             # 自动生成的图表，默认不纳入 Git
```

## conda环境配置

```powershell
conda env create -f environment.yml
conda activate aigc-detector
```

## 快速开始

### 1. 运行自检

```powershell
python scripts/prepare_benchmark.py --self-test
python scripts/build_benchmark_splits.py --self-test
```

### 2. 准备一个中文 Benchmark

```powershell
python scripts/prepare_benchmark.py `
  --dataset hc3_zh `
  --split train `
  --max-per-class 5000 `
  --output data/benchmarks/hc3_zh_train.csv

python scripts/build_benchmark_splits.py --benchmark hc3_zh --seed 42
```

标准化数据至少包含 `text` 和 `label`，完整字段为：

```text
text, label, domain, source, attack, group_id, benchmark, split
```

### 3. 训练传统模型

```powershell
python scripts/run_experiments.py `
  --benchmarks hc3_zh `
  --families classical `
  --seed 42
```

可先加入 `--dry-run` 查看任务，或使用 `--resume` 跳过已有结果。

### 4. 文本预测

```powershell
python src/aigc_detector/predict.py `
  --model models/hc3_zh/hybrid/seed_42.joblib `
  --text "这里是一段需要检测的文本。" `
  --explain
```

## 完整实验

准备四个 Benchmark 后，可运行完整实验矩阵：

```powershell
python scripts/build_benchmark_splits.py --benchmark all --seed 42
python scripts/run_experiments.py --benchmarks all --families classical,deep --seed 42
```

深度实验默认配置：

| Benchmark | 编码器 | 微调方式 |
| --- | --- | --- |
| HC3-Chinese | `hfl/chinese-roberta-wwm-ext` | Full fine-tuning |
| HC3-English、MAGE、RAID | `xlm-roberta-base` | LoRA |

查看脚本的全部参数：

```powershell
python scripts/prepare_benchmark.py --help
python scripts/train_deep.py --help
python scripts/run_experiments.py --help
```

## 结果与图表

聚合指定随机种子的实验结果：

```powershell
python scripts/aggregate_results.py --seed 42
```

生成 PNG、SVG 和 PDF 图表：

```powershell
python scripts/plot_results.py
```

图表保存在 `figures/generated/`。如只需要 PDF：

```powershell
python scripts/plot_results.py --formats pdf
```

## Web 演示

完成至少一个模型训练后运行：

```powershell
streamlit run demo/app.py
```

页面会自动发现 `models/` 下的模型，并显示 AI 分数、风险区间、测试集指标和可用的特征解释。
