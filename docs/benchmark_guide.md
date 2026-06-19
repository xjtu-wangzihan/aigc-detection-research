# Benchmark 接入指南

## 推荐顺序

1. HC3-Chinese：中文课程作业场景的第一个正式 benchmark。
2. HC3-English：验证中英文差异。
3. MAGE：进行跨领域、跨生成模型与 paraphrase 实验。
4. RAID：只抽取子集，用于攻击鲁棒性评测。

## 统一格式

`prepare_benchmark.py` 将数据统一为：

| Field | Meaning |
|---|---|
| text | 待检测文本 |
| label | 0=human, 1=machine |
| domain | 领域或文本类型 |
| source | human 或生成模型 |
| attack | none/paraphrase/其他攻击 |
| group_id | 用于防止配对样本泄漏 |
| benchmark | 数据集名称 |
| split | 官方划分名称 |

## 命令

```powershell
python -m pip install -r requirements-benchmark.txt
python scripts/prepare_benchmark.py --self-test
python scripts/prepare_benchmark.py --dataset hc3_zh --split train --max-per-class 5000 --output data/benchmarks/hc3_zh_train.csv
python src/aigc_detector/train.py --data data/benchmarks/hc3_zh_train.csv --model hybrid --out models/hc3_zh_hybrid.joblib --metrics results/hc3_zh_metrics.json --robust
```

## 官方划分

如果数据集提供独立 train/test，分别转换后执行：

```powershell
python src/aigc_detector/train.py --data data/benchmarks/mage_train.csv --test-data data/benchmarks/mage_test.csv --model hybrid --out models/mage_hybrid.joblib --metrics results/mage_metrics.json
```

RAID 官方 test 集不公开标签，课程实验先使用带标签的 train/extra 子集，不要尝试用 test 集训练。

## 数据泄漏边界

- HC3 的人类和 ChatGPT 答案来自同一问题，必须使用 `group_id` 分组切分。
- MAGE 的标签定义与本项目相反，禁止跳过转换器直接训练。
- RAID 机器文本数量远大于人类文本，应当限制 `--max-per-class` 并单独报告 `human_fpr`。

## 官方来源

- HC3: https://github.com/Hello-SimpleAI/chatgpt-comparison-detection
- MAGE: https://github.com/yafuly/MAGE
- RAID: https://github.com/liamdugan/raid
