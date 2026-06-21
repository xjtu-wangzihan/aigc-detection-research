# AIGC 文本检测研究平台

面向中文与英文场景的可复现 AIGC 文本检测实验项目。项目围绕“检测效果、跨数据集泛化、改写扰动鲁棒性和人工复核风险”组织完整流程，支持传统机器学习、预训练编码器、风格特征融合、结果聚合和交互式演示。

> 本项目输出的是模型风险分数，不是文本来源的确定性证据。检测结果不应被单独用于处罚、定责、学术不端认定或内容下架。

## 1. 项目概览

项目实现了从公开数据集接入到模型演示的实验闭环：

```text
公开数据集
    ↓  统一字段与标签
data/benchmarks
    ↓  分组切分、平衡采样、泄漏检查
data/splits/{benchmark}
    ↓  传统模型 / 深度模型训练
models + results
    ↓  单随机种子与扰动结果汇总
results/summary
    ↓
命令行预测 / Streamlit Demo
```

核心能力包括：

- 统一接入 HC3-Chinese、HC3-English、MAGE 和 RAID；
- 约定 `0=human`、`1=machine`，修正不同数据集的标签差异；
- 按 `group_id` 切分数据，避免成对样本跨训练集与测试集泄漏；
- 对比词级 TF-IDF、字符级 TF-IDF、写作风格和混合特征模型；
- 支持 RoBERTa、XLM-R 编码器及“编码器 + 12 维风格特征”融合模型；
- 评估空白归一化、标点删除、连接词替换和句序扰动；
- 输出 Accuracy、F1、AUROC、Human FPR、推理耗时和模型体积等指标；
- 提供三档风险输出：低风险、不确定需复核、高风险。

## 2. 支持的数据集与方法

### 2.1 数据集

| 标识 | 数据集 | 主要用途 |
| --- | --- | --- |
| `hc3_zh` | HC3-Chinese | 中文问答场景与中文检测能力 |
| `hc3` | HC3-English | 英文问答场景与中英文差异比较 |
| `mage` | MAGE | 多领域、多生成器泛化评估 |
| `raid` | RAID | 多领域检测与扰动鲁棒性评估 |

### 2.2 模型

| 方法 | 实现 | 特点 |
| --- | --- | --- |
| `word_tfidf` | 词级 1–2 gram + Logistic Regression | 训练快，适合作为基础基线 |
| `char_tfidf` | 字符级 2–5 gram + Logistic Regression | 对中英文及局部改写较友好 |
| `style` | 12 维写作风格特征 + Logistic Regression | 模型极小，可解释性较强 |
| `hybrid` | 字符、词和风格特征融合 | 综合轻量级基线 |
| `encoder` | RoBERTa / XLM-R 编码器 | 端到端语义表示 |
| `fusion` | 编码器表示 + 12 维风格特征 | 用于检验风格信息的增益 |

深度模型默认配置如下：

| Benchmark | 编码器 | 微调方式 |
| --- | --- | --- |
| `hc3_zh` | `hfl/chinese-roberta-wwm-ext` | Full fine-tuning |
| `hc3`、`mage`、`raid` | `xlm-roberta-base` | LoRA |

## 3. 目录结构

```text
aigc-detection-research/
├─ data/                       # 原始标准化数据与固定划分（默认不纳入 Git）
│  ├─ benchmarks/             # 统一格式的数据集
│  └─ splits/                 # train / val / test 与 manifest
├─ models/                    # joblib 模型和深度学习 checkpoint（默认不纳入 Git）
├─ results/                   # 单次实验 JSON 与聚合表（默认不纳入 Git）
├─ scripts/
│  ├─ prepare_benchmark.py    # 下载并标准化公开数据集
│  ├─ build_benchmark_splits.py # 构建无组间泄漏的固定划分
│  ├─ run_experiments.py      # 生成并执行课程实验矩阵
│  ├─ train_deep.py           # 训练编码器或融合模型
│  ├─ predict_deep.py         # 深度模型单文本预测
│  └─ aggregate_results.py    # 聚合多次运行结果
├─ src/aigc_detector/
│  ├─ train.py                # 传统模型训练入口
│  ├─ predict.py              # 传统模型预测入口
│  ├─ models.py               # 传统模型定义
│  ├─ deep_models.py          # 深度模型、LoRA 与 checkpoint
│  ├─ features.py             # 12 维风格特征
│  ├─ attacks.py              # 文本扰动方法
│  ├─ evaluate.py             # 指标计算与分组评估
│  └─ explain.py              # 线性模型特征贡献解释
├─ demo/app.py                # Streamlit 交互演示
├─ tests/test_core.py         # 数据、指标、切分和模型形状测试
└─ docs/                      # Benchmark 与深度学习实验说明
```

## 4. 环境准备

建议使用 Python 3.10–3.12，并在项目根目录创建独立环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

安装传统模型与测试所需依赖：

```powershell
python -m pip install numpy pandas scikit-learn joblib
```

按需安装扩展依赖：

```powershell
# 下载公开 Benchmark
python -m pip install datasets

# 启动交互式 Demo
python -m pip install streamlit

# 深度模型：请先安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖
python -m pip install transformers peft accelerate safetensors
```

深度模型训练会下载 Hugging Face 编码器权重，需要可用网络；仅运行传统模型时不需要 GPU。

## 5. 快速开始

### 5.1 运行自检

以下命令不下载数据，可先验证数据适配、分组切分和核心逻辑：

```powershell
python scripts/prepare_benchmark.py --self-test
python scripts/build_benchmark_splits.py --self-test
python -m unittest discover -s tests -v
```

### 5.2 准备 Benchmark

```powershell
# HC3 中文与英文
python scripts/prepare_benchmark.py --dataset hc3_zh --split train --max-per-class 5000 --output data/benchmarks/hc3_zh_train.csv
python scripts/prepare_benchmark.py --dataset hc3 --split train --max-per-class 5000 --output data/benchmarks/hc3_en_train.csv

# MAGE 官方 train / validation / test
python scripts/prepare_benchmark.py --dataset mage --split train --max-per-class 3000 --output data/benchmarks/mage_train.csv
python scripts/prepare_benchmark.py --dataset mage --split validation --max-per-class 1000 --output data/benchmarks/mage_validation.csv
python scripts/prepare_benchmark.py --dataset mage --split test --max-per-class 1000 --output data/benchmarks/mage_test.csv

# RAID：保留干净英文样本
python scripts/prepare_benchmark.py --dataset raid --split train --max-per-class 5000 --raid-english --attacks none --output data/benchmarks/raid_train.csv
```

标准化后的 CSV 统一包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 待检测文本 |
| `label` | `0` 表示人类文本，`1` 表示机器生成文本 |
| `domain` | 文本领域或类型 |
| `source` | `human` 或具体生成模型 |
| `attack` | `none`、`paraphrase` 或其他攻击名称 |
| `group_id` | 用于防止关联样本跨集合泄漏的分组标识 |
| `benchmark` | 数据集标识 |
| `split` | 原始数据集划分名称 |

### 5.3 构建固定划分

```powershell
python scripts/build_benchmark_splits.py --profile course --benchmark all --seed 42
```

HC3 和 RAID 从平衡后的样本池按约 `60% / 20% / 20%` 构建训练集、验证集和测试集；MAGE 保留官方划分并做类别平衡。每个数据集都会生成 `manifest.json`，记录样本量、标签分布、分组数和 SHA-256，以便复现实验。

### 5.4 训练与评估传统模型

先用 `--dry-run` 检查将要执行的任务：

```powershell
python scripts/run_experiments.py --profile course --benchmarks hc3_zh --families classical --seed 42 --dry-run
```

正式运行：

```powershell
python scripts/run_experiments.py --profile course --benchmarks hc3_zh --families classical --seed 42
```

该命令会对四种传统方法分别运行一次 `seed=42`，保存模型、固定测试集指标和四种文本扰动结果。传统模型和深度模型共用 `--seed`；如需改变随机种子，只需在统一实验入口指定其他整数。也可直接训练单个模型：

```powershell
python src/aigc_detector/train.py `
  --data data/splits/hc3_zh/train.csv `
  --val-data data/splits/hc3_zh/val.csv `
  --test-data data/splits/hc3_zh/test.csv `
  --benchmark hc3_zh `
  --model hybrid `
  --seed 42 `
  --out models/hc3_zh/hybrid/seed_42.joblib `
  --metrics results/hc3_zh/hybrid/seed_42.json `
  --robust
```

### 5.5 训练深度模型

```powershell
# Encoder-only
python scripts/run_experiments.py --profile course --benchmarks hc3_zh --families deep --seed 42

# 同时规划全部传统与深度实验，但不执行
python scripts/run_experiments.py --profile course --benchmarks all --families classical,deep --seed 42 --dry-run
```

完整深度实验默认每个 Benchmark 运行一次 `encoder` 和一次 `fusion`。训练按验证集 AUROC 保存最佳 checkpoint；`--resume` 仅跳过结构完整的结果文件：

```powershell
python scripts/run_experiments.py --profile course --benchmarks all --families classical,deep --seed 42 --resume
```

显存有限时可直接调用 `train_deep.py`，减小 batch size 并增大梯度累积：

```powershell
python scripts/train_deep.py `
  --benchmark hc3 `
  --train-data data/splits/hc3/train.csv `
  --val-data data/splits/hc3/val.csv `
  --test-data data/splits/hc3/test.csv `
  --mode fusion `
  --seed 42 `
  --batch-size 2 `
  --gradient-accumulation 8 `
  --out models/deep/hc3/xlmr_style/seed_42 `
  --metrics results/hc3/xlmr_style/seed_42.json `
  --robust
```

### 5.6 聚合结果

```powershell
python scripts/aggregate_results.py --seed 42
```

聚合产物位于 `results/summary/`：

- `main_results.csv/.md`：指定随机种子的主测试集结果；
- `robustness_results.csv/.md`：不同扰动下的鲁棒性；
- `ablation_results.md`：风格特征与编码器融合消融；
- `raw_main_results.csv`、`raw_robustness_results.csv`：指定随机种子的可追溯明细。

## 6. 推理与演示

### 6.1 传统模型预测

```powershell
python src/aigc_detector/predict.py `
  --model models/hc3_zh/hybrid/seed_42.joblib `
  --text "这里是一段需要检测的文本。" `
  --explain
```

### 6.2 深度模型预测

```powershell
python scripts/predict_deep.py `
  --checkpoint models/deep/hc3_zh/roberta_style/seed_42 `
  --text "这里是一段需要检测的文本。"
```

### 6.3 启动 Web Demo

```powershell
streamlit run demo/app.py
```

Demo 会自动发现 `models/` 下结构完整的传统模型和深度 checkpoint，并展示 AI 分数、预测标签、风险区间、固定测试集记录以及可用的特征解释。

## 7. 评估设计

### 7.1 主要指标

- `Accuracy`：总体分类正确率；
- `Precision / Recall / F1`：以机器生成文本为正类；
- `AUROC`：衡量不同阈值下的整体区分能力；
- `Human FPR`：人类文本被误判为机器文本的比例，是部署时的重要安全指标；
- `Machine TPR`：机器文本被正确检出的比例；
- `Inference ms/sample` 与 `Model size`：衡量部署成本。

### 7.2 鲁棒性测试

启用 `--robust` 后，测试集会额外经过四类轻量扰动：

| 攻击 | 操作 |
| --- | --- |
| `whitespace` | 合并多余空白 |
| `punct_drop` | 删除部分轻量标点 |
| `connector_swap` | 替换中英文连接词 |
| `sentence_shuffle` | 轻度调整句子顺序 |

### 7.3 风险分级

| AI 分数 | 输出 |
| --- | --- |
| `≤ 0.25` | `low_ai_risk` |
| `(0.25, 0.75)` | `uncertain_review_needed` |
| `≥ 0.75` | `high_ai_risk` |

分类标签默认以 `0.5` 为阈值；风险分级用于提示复核优先级，两者用途不同。

## 8. 当前实验结果

仓库本地结果中，各 Benchmark 的代表性最高 AUROC 如下；传统模型和深度模型均为单次 `seed=42` 结果。

| Benchmark | 方法 | Accuracy | F1 | AUROC | Human FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| HC3-English | XLM-R Encoder / Fusion | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| HC3-Chinese | RoBERTa Encoder | 0.9895 | 0.9896 | 0.9990 | 0.0151 |
| MAGE | XLM-R + Style | 0.7460 | 0.6973 | 0.8533 | 0.0930 |
| RAID | Char TF-IDF | 0.9895 | 0.9895 | 0.9992 | 0.0090 |

完整结果请查看 `results/summary/main_results.md` 和 `results/summary/robustness_results.md`。不同数据集之间的分数不可直接视为模型能力排名；尤其是 MAGE，其多领域、多生成器分布明显更复杂。

## 9. 复现注意事项

1. `data/`、`models/`、`results/` 和 `chapters/` 默认被 `.gitignore` 排除，克隆项目后需要重新准备数据和训练模型。
2. HC3 中同一问题的人类答案和 ChatGPT 答案共享 `group_id`，切勿使用普通随机行切分。
3. MAGE 原始标签语义与本项目相反，必须通过 `prepare_benchmark.py` 转换。
4. RAID 类别分布不平衡，应限制每类样本量并单独报告 `Human FPR`。
5. `joblib.load` 可执行序列化代码，只加载可信来源的本地模型。
6. 深度训练的耗时、显存占用和结果会受到 GPU、PyTorch、Transformers 版本及随机性的影响。
7. 当前风险分数未针对真实部署场景做概率校准；短文本、非母语写作和强改写文本应优先人工复核。

## 10. 延伸文档

- `docs/benchmark_guide.md`：公开数据集接入、标签和泄漏边界；
- `docs/deep_learning_guide.md`：深度实验顺序、断点恢复和低显存建议；
- `docs/research_direction.md`：研究问题、实验故事线与相关文献线索；
- `figures/data-manifest.md`：论文图表的数据来源约定；
- `tables/table-schema.md`：结果表格字段约定。
