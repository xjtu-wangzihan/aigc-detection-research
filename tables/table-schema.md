# Table Schema

| Table | Rows | Metrics | Generated source | Status |
|---|---|---|---|---|
| 数据统计 | benchmark/split/class | count, domain count | data/splits/*/manifest.json | HC3 可生成；MAGE/RAID 待数据 |
| 传统主结果 | benchmark × 4 classical methods | mean±std of Accuracy/F1/AUROC/Human FPR | results/summary/main_results.md | 待完整实验 |
| 深度消融 | Style-LR/Encoder/Encoder+Style | Accuracy/F1/AUROC/Human FPR | results/summary/ablation_results.md | 待深度实验 |
| 鲁棒性 | benchmark/method/attack | F1/AUROC/Human FPR and delta | results/summary/robustness_results.md | 待完整实验 |
| 分领域 | benchmark/method/domain | n/Accuracy/F1/AUROC/Human FPR | seed JSON by_domain | 待完整实验 |
| 效率 | benchmark/method | train/inference/model size/GPU memory | seed JSON + nvidia-smi log | GPU memory 待记录 |

提交前检查：不得将旧 75/25 HC3 结果放入统一主表；待实验标记只能由聚合产物中的真实数值替换。
