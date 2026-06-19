# Table Schema

| Table | Purpose | Rows | Metrics | Data source | Replacement owner |
|---|---|---|---|---|---|
| Table 1 | 同域随机划分主结果 | baselines | Accuracy, F1, AUROC, FPR | HC3/MAGE/RAID | user |
| Table 2 | 跨领域泛化 | held-out domains | F1, AUROC, FPR | MAGE/RAID/M4 | user |
| Table 3 | 改写攻击鲁棒性 | attack types | F1 drop, AUROC drop | RAID + local attacks | user |
| Table 4 | 消融实验 | feature ablations | F1, AUROC | selected dataset | user |
| Table 5 | 效率比较 | baselines | train time, infer ms, model size | local logs | user |

Note: PLANNING DATA - replace before submission.
