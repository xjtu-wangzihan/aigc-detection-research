# Progress

## 2026-06-12

- Stage: S0 Scope + S3 Experiments.
- 已完成课程模板读取：报告结构为实验简介目标、原理、过程结果、问题解决、总结、分工、参考文献。
- 已确定推荐方向：面向跨域与轻度改写攻击的轻量级可解释 AIGC 文本检测。
- 已创建 baseline 代码、demo、实验协议和表格/图数据契约。
- 2026-06-19：新增 HC3/HC3-Chinese/MAGE/RAID 流式 benchmark 转换器、分组防泄漏切分、官方 test CSV 入口和 human FPR 指标。

### Capability-use audit

- Required skills: paper-orchestration, brainstorming-research, experiment-results-planning, nature-academic-search。
- Skills actually used: paper-orchestration, brainstorming-research, experiment-results-planning, nature-academic-search。
- Inputs consumed: 课程实验报告模板、AIGC 检测公开论文/数据集线索、本地 Python 环境。
- Inputs not used and why: 尚未下载公开数据集，当前先完成可运行原型和实验协议。
- Artifacts produced: `docs/`, `plan/`, `src/`, `demo/`, `data/sample_dataset.csv`。
- Verification run: 
  - `python src/aigc_detector/train.py --data data/sample_dataset.csv --model hybrid --out models/hybrid.joblib --metrics results/smoke_metrics.json --robust`
  - `python -m compileall src demo`
  - `python src/aigc_detector/predict.py --model models/hybrid.joblib --text ... --explain`
- Remaining risk: 真实实验结果依赖公开数据集接入；样例数据只可用于烟测。
