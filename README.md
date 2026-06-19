# AIGC Generation and Detection Research Prototype

本项目面向《网络与信息安全》课程大作业，主题为 **AIGC 生成与检测**。当前版本聚焦一个容易做深、也相对容易写成论文或专利交底的小方向：

> 面向跨域与轻度改写攻击的轻量级、可解释 AIGC 文本检测。

核心思想不是追求训练一个巨大的深度模型，而是建立可复现实验闭环：

- 公开数据集适配：HC3、MAGE、RAID、M4/M4GT-Bench 等。
- Baseline：词级 TF-IDF、字符 n-gram、风格统计、混合特征逻辑回归。
- 安全视角：跨域泛化、改写攻击鲁棒性、误报风险、置信度阈值。
- Demo：Streamlit 网页展示检测概率、风险等级和可解释特征。

## Quick Start

```powershell
cd aigc-detection-research
python src/aigc_detector/train.py --data data/sample_dataset.csv --model hybrid --out models/hybrid.joblib --metrics results/smoke_metrics.json --robust
python src/aigc_detector/predict.py --model models/hybrid.joblib --text "本文围绕网络安全中的AIGC内容鉴别问题展开分析。"
streamlit run demo/app.py
```

当前源码使用同目录绝对导入（如 `from datasets import ...`），因此请在项目根目录直接执行上述脚本，不要使用 `python -m aigc_detector.train`。

## Data Format

训练数据使用 CSV：

```csv
text,label,domain,source,attack
"...",0,essay,human,none
"...",1,essay,gpt,none
```

字段说明：

- `text`: 文本内容。
- `label`: `0` 表示人类文本，`1` 表示 AIGC 文本。
- `domain`: 领域，如 essay/news/qa/academic/security。
- `source`: 来源，如 human/gpt/llama/qwen。
- `attack`: 改写或扰动类型，无攻击填 `none`。

## Benchmark Quick Start

安装 benchmark 数据工具：

```powershell
python -m pip install -r requirements-benchmark.txt
```

首次建议接入 HC3 中文版，流式抽样每类 5000 条：

```powershell
python scripts/prepare_benchmark.py --dataset hc3_zh --split train --max-per-class 5000 --output data/benchmarks/hc3_zh_train.csv
python src/aigc_detector/train.py --data data/benchmarks/hc3_zh_train.csv --model hybrid --out models/hc3_zh_hybrid.joblib --metrics results/hc3_zh_metrics.json --robust
```

MAGE 和 RAID 子集：

```powershell
python scripts/prepare_benchmark.py --dataset mage --split train --max-per-class 10000 --output data/benchmarks/mage_train.csv
python scripts/prepare_benchmark.py --dataset raid --split train --max-per-class 10000 --output data/benchmarks/raid_train.csv
```

转换后会额外保留 `group_id`、`benchmark` 和 `split`。HC3 会按问题分组切分，避免同一问题的人类答案与 ChatGPT 答案同时出现在训练集和测试集。MAGE 官方标签是 `0=machine, 1=human`，转换器会自动翻转为本项目的 `0=human, 1=machine`。

## Project Layout

- `docs/research_direction.md`: 研究方向调研、baseline 和可投稿故事线。
- `plan/experiment-protocol.md`: 实验协议。
- `src/aigc_detector/`: 训练、评测、扰动、解释代码。
- `demo/app.py`: Streamlit 展示网页。
- `data/sample_dataset.csv`: 仅用于代码烟测的小样例数据，不可作为真实实验结果。

## Important Boundary

`data/sample_dataset.csv` 是手工构造的烟测数据，只用于验证代码能跑通。报告、论文或专利中的性能结论必须来自公开数据集或真实采集数据，不能使用该样例数据冒充实验结果。
