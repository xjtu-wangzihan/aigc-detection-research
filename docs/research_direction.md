# 研究方向调研与选题建议

## 推荐题目

**面向跨域与轻度改写攻击的轻量级可解释 AIGC 文本检测方法研究**

这个方向适合作为课程大作业继续扩展为专利或会议短文。理由是：训练成本低，实验可复现，安全属性明确，并且能避开“又训练了一个 RoBERTa 分类器”的拥挤路线。

## 研究问题

1. 现有 AIGC 检测器在训练集内往往表现不错，但遇到未见过的领域、模型、采样策略和改写攻击时会明显退化。
2. 直接输出“AI/人类”的二分类结论有误伤风险，尤其在短文本、非母语写作、模板化写作中更明显。
3. 课程大作业可做出一个实用闭环：检测概率、风险等级、解释特征、人工复核建议、攻击鲁棒性评测。

## 可投稿/可专利的切入点

推荐主线：

> 轻量级混合特征检测器 + 跨域鲁棒性评测 + 置信度拒判机制。

可以包装成三个贡献：

1. **多视角轻量检测**：字符 n-gram、词级 TF-IDF、风格统计特征融合，支持中英文，不依赖目标生成模型概率。
2. **鲁棒安全评测协议**：按领域、生成模型、攻击方式拆分，报告同域、跨域和扰动后的 AUROC/F1/误报率。
3. **风险分级与可解释输出**：不把检测结果当定责证据，而是输出高风险、低风险、需复核三类，并展示贡献特征。

专利表达可以偏系统方法：

> 一种面向生成式人工智能文本的跨域鲁棒检测与复核提示方法、系统、设备及存储介质。

## 推荐公开数据集

- **HC3**：人类专家与 ChatGPT 回答对比语料，适合做问答场景检测。
- **MAGE**：强调多领域、多 LLM 的“野外”检测，适合跨域泛化。
- **RAID**：包含多模型、多领域、多攻击和多解码策略，适合鲁棒性评测。
- **M4 / M4GT-Bench**：多语言、多领域、多生成器，适合扩展到中文/多语言。
- **Ghostbuster datasets**：包含学生作文、创意写作、新闻等领域，适合和 Ghostbuster 思路对齐。

## Baseline 清单

必须做：

- Majority / length-only baseline：证明任务不是被长度差异投机解决。
- Word TF-IDF + Logistic Regression。
- Char n-gram TF-IDF + Logistic Regression。
- Stylometric features + Logistic Regression。
- Hybrid features：字符、词、风格统计融合。

建议做：

- RoBERTa/BERT 分类器：有 GPU 或时间时补充。
- DetectGPT / Fast-DetectGPT：作为零样本概率曲率类方法，成本较高，可做小样本对比。
- Ghostbuster：作为较强公开检测系统的思想参考。

不建议主打：

- 单纯调用商业 AI 检测网站。不可复现，难以写论文。
- 只做水印检测。除非自己控制生成过程，否则真实场景覆盖有限。

## 关键文献线索

- DetectGPT 提出基于概率曲率的零样本检测思路，优点是不需要训练分类器，但计算成本较高。
- Fast-DetectGPT 改进了 DetectGPT 的效率，用条件概率曲率降低开销。
- Ghostbuster 强调黑盒检测，并在学生作文、创意写作、新闻等领域做跨域比较。
- RAID 明确指出共享 benchmark 需要覆盖采样策略、对抗攻击和开源生成模型。
- MAGE/M4/M4GT-Bench 都强调跨模型、跨领域、多语言检测仍然困难。
- Sadasivan 等工作指出递归改写会显著削弱检测器，说明鲁棒性是论文故事的核心。
- Liang 等工作指出 GPT 检测器可能误伤非母语英文写作者，说明必须报告误报风险。
- SynthID-Text 说明水印路线在生产系统中有价值，但需要控制生成过程，和本文的后验检测是互补关系。

## 实验故事线

主实验：

1. 同域随机划分：验证基础可行性。
2. Leave-one-domain-out：训练时留出一个领域，测试跨域泛化。
3. Leave-one-source-out：训练时留出一个生成模型，测试跨模型泛化。
4. 改写攻击：空白归一化、标点删除、连接词替换、句序轻扰动；后续可加入真实 LLM paraphrase。
5. 拒判机制：设置 0.25/0.75 阈值，中间区域输出“需人工复核”，比较误报率和覆盖率。

论文卖点：

- 如果 hybrid 在跨域/攻击下比单一 TF-IDF 稳定，可以写成“多视角特征提升鲁棒性”。
- 如果拒判机制显著降低人类文本误报，可以写成“面向教学/内容审核的安全部署策略”。
- 如果中文或中英混合场景数据充分，可以写成“面向中文课程作业的 AIGC 检测基准与系统”。

## 参考链接

- DetectGPT: https://arxiv.org/abs/2301.11305
- Ghostbuster: https://arxiv.org/abs/2305.15047
- Fast-DetectGPT: https://arxiv.org/abs/2310.05130
- HC3: https://arxiv.org/abs/2301.07597
- MAGE: https://arxiv.org/abs/2305.13242
- RAID: https://arxiv.org/abs/2405.07940
- M4GT-Bench: https://arxiv.org/abs/2402.11175
- Reliability of AI text detectors: https://arxiv.org/abs/2303.11156
- Bias against non-native English writers: https://arxiv.org/abs/2304.02819
- A Watermark for Large Language Models: https://arxiv.org/abs/2301.10226
- SynthID-Text: https://www.nature.com/articles/s41586-024-08025-4
