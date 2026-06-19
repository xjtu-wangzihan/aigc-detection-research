# 实验协议

## Dataset and Split Strategy

第一阶段用 `data/sample_dataset.csv` 做代码烟测，不报告真实性能。正式实验建议：

1. HC3：问答领域，人类/ChatGPT 对照。
2. MAGE：多领域、多生成模型。
3. RAID：多模型、多领域、多攻击、多解码策略。
4. M4/M4GT-Bench：多语言、多领域、多生成器。
5. 自建中文课程作业小集：每类至少 200 条，记录来源、生成模型和提示词。

划分策略：

- Random split：基础可行性。
- Leave-one-domain-out：跨领域泛化。
- Leave-one-source-out：跨生成模型泛化。
- Attack split：原文训练，扰动/改写测试。

## Baselines

- Majority / length-only。
- Word TF-IDF + Logistic Regression。
- Char n-gram TF-IDF + Logistic Regression。
- Stylometric features + Logistic Regression。
- Hybrid features + Logistic Regression。
- Optional：RoBERTa/BERT、DetectGPT、Fast-DetectGPT、Ghostbuster。

## Metrics

- Accuracy、Precision、Recall、F1、AUROC。
- FPR on human texts：人类文本误报率，安全部署中必须单独报告。
- Attack degradation：攻击前后 AUROC/F1 下降。
- Coverage-risk curve：拒判阈值下覆盖率与误报率。
- Efficiency：训练时间、推理时间、模型大小。

## Main Comparison

比较 word、char、style、hybrid 在同域、跨域和攻击场景中的表现。

## Ablation Studies

- 去掉字符 n-gram。
- 去掉词级 TF-IDF。
- 去掉风格统计。
- 关闭 class_weight。
- 关闭拒判机制。

## Generalization and Robustness Checks

- 未见领域测试。
- 未见生成模型测试。
- 标点删除、连接词替换、句序轻扰动。
- 后续加入 LLM paraphrase 攻击。

## Explainability Evaluation

- 展示 top positive/negative feature contribution。
- 检查解释是否过度依赖长度、模板短语或标点。
- 对误报样本做人工案例分析。

## Claim Boundary

只有完成公开数据集实验后，才能声称“方法有效”。当前样例数据只能证明代码可运行。
