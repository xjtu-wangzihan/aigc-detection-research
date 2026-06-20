# 深度学习实验指南

## 方法映射

| Benchmark | Encoder-only | Encoder + Style | Tuning |
|---|---|---|---|
| hc3_zh | roberta_encoder | roberta_style | full |
| hc3 | xlmr_encoder | xlmr_style | LoRA |
| mage | xlmr_encoder | xlmr_style | LoRA |
| raid | xlmr_encoder | xlmr_style | LoRA |

融合模型只增加风格分支和融合分类头，编码器、训练集、验证集和超参数保持一致，因此可与 Encoder-only 做直接消融。

## 运行顺序

1. 运行 build_benchmark_splits.py，并检查每个 manifest.json。
2. 使用 100 条样本、1 epoch 做 Chinese RoBERTa 和 XLM-R 两次烟测。
3. 用 nvidia-smi 观察峰值显存，超过 6 GB 时先把 batch size 减半并相应增加 gradient accumulation。
4. 依次运行 hc3_zh、hc3、mage、raid 的 encoder 和 fusion。
5. 运行 aggregate_results.py，并检查每个表格单元能追溯到 seed_*.json。

## 断点与恢复

run_experiments.py --resume 仅在目标 JSON 同时含 schema_version、overall 和 checkpoint 时跳过任务。深度训练以 validation AUROC 保存最佳 adapter/encoder、分类头、tokenizer 和 style scaler。异常中断但结果 JSON 不完整时，该任务会重新训练。

## 6 GB 显存建议

- 保持 max_length=256，XLM-R batch size=4、gradient accumulation=4。
- 保持 FP16 和 gradient checkpointing。
- 不要同时运行两个深度训练进程。
- 仍溢出时使用 batch size=2、gradient accumulation=8；这不改变有效 batch size。
- 记录实际峰值显存后再填写报告效率表。
