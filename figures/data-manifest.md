# Figure Data Manifest

| Figure | Input | Intended conclusion | Status |
|---|---|---|---|
| 四 benchmark 主结果 | results/summary/main_results.csv | 比较各表示与编码器 | [待完整实验] |
| 风格融合增益 | results/summary/ablation_results.md | 检验 Style 是否补充 Encoder | [待深度实验] |
| 扰动性能变化 | results/summary/robustness_results.csv | 定位最敏感攻击 | [待完整实验] |
| 分领域 Human FPR | seed JSON by_domain | 展示误报风险差异 | [待完整实验] |
| 性能-效率散点图 | main_results.csv | 比较精度与成本 | [待完整实验] |
| 融合模型结构图 | docs/deep_learning_guide.md | 展示 768+12 维融合路径 | 待绘制 |

绘图前必须先明确图的结论并核对输入行数。旧 HC3 初步结果如需作图，标题和图注必须标明 75/25 preliminary split。
