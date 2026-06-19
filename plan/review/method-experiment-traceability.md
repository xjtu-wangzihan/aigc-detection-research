# Method-Experiment Traceability

| Contribution | Method module | Experiment | Table/Figure | Allowed claim | Evidence status |
|---|---|---|---|---|---|
| 轻量级多视角检测 | hybrid feature pipeline | Random split, cross-domain split | Table 1, Table 2 | 混合特征可作为强 baseline | 代码完成，待真实数据 |
| 改写攻击鲁棒性 | attacks.py benchmark | Attack split | Table 3, Figure 1 | 可量化轻度扰动下性能下降 | 烟测完成，待真实数据 |
| 风险分级与复核 | 0.25/0.75 threshold | Coverage-risk curve | Figure 2 | 拒判机制可降低误报风险 | demo 完成，待实验 |
| 可解释输出 | linear contribution extraction | Case study | Table 4 | 可定位影响判断的表层特征 | demo 完成，待案例分析 |
