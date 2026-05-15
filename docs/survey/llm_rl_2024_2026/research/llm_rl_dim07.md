# 维度7: Benchmark体系与评估方法深度调研报告

> 调研范围: 2024-2025年LLM+RL领域使用的Benchmark和评估方法
> 覆盖领域: 数学推理、代码生成、通用推理、指令遵循、Agent评估、评估方法论

---

## 目录

1. [数学推理Benchmark](#1-数学推理benchmark)
2. [代码生成Benchmark](#2-代码生成benchmark)
3. [通用推理Benchmark](#3-通用推理benchmark)
4. [指令遵循Benchmark](#4-指令遵循benchmark)
5. [Agent评估Benchmark](#5-agent评估benchmark)
6. [评估方法论](#6-评估方法论)
7. [评估框架与工具](#7-评估框架与工具)
8. [Benchmark分类汇总表](#8-benchmark分类汇总表)
9. [参考文献](#9-参考文献)

---

## 1. 数学推理Benchmark

### 1.1 GSM8K (Grade School Math 8K)

| 属性 | 详情 |
|------|------|
| **全称** | Grade School Math 8K |
| **发布年份** | 2021 |
| **数据规模** | 训练集7,473题 / 测试集1,319题 |
| **难度** | 小学级别 |
| **评估指标** | Exact Match (精确匹配最终数值答案) |
| **答案格式** | 数值答案，通常使用 `\boxed{}` 包裹 |

**核心特征**: GSM8K是评估多步数学推理的基础benchmark，包含需要多步推理的小学数学文字题 [^537^]。尽管其难度相对较低，但仍是评估模型基础数学能力的重要基准。GPT-4使用DUP策略可达到97.1%准确率 [^525^]，显示该benchmark已趋于饱和。

**在LLM+RL研究中的使用情况**:
- 作为基础评估benchmark广泛用于RLVR训练后的模型评估 [^495^][^625^]
- 常与MATH-500、AIME等组合使用形成评估套件 [^107^][^498^]
- 训练时通常使用GSM8K-Aug（GPT-4扩增至385K样本的版本）[^544^]

---

### 1.2 MATH / MATH-500

| 属性 | MATH | MATH-500 |
|------|------|----------|
| **全称** | MATH (Hendrycks et al., 2021) | MATH-500 (Lightman et al., 2023) |
| **发布年份** | 2021 | 2023 |
| **数据规模** | 12,500题 (7,500训练/5,000测试) | 500题 |
| **难度** | 高中竞赛级别 | MATH测试集的代表性子集 |
| **评估指标** | Exact Match | Exact Match |
| **答案格式** | 数值或符号答案 | `\boxed{}` 格式 |

**核心特征**: MATH数据集包含高中数学竞赛级别的问题，涵盖代数、几何、数论、组合数学等七个领域，难度等级1-5 [^480^][^483^]。MATH-500是OpenAI在PRM800K工作中引入的500题评估子集，用于评估过程监督方法 [^483^]。

**代表性论文使用**:
- DeepScaleR使用MATH-500作为主要评估benchmark之一 [^107^][^498^]
- MATH-Beyond通过pass@1024筛选构建了181题的扩展benchmark [^600^]
- 评估设置通常采用4-shot prompting [^626^]

---

### 1.3 AIME 2024 / AIME 2025

| 属性 | 详情 |
|------|------|
| **全称** | American Invitational Mathematics Examination |
| **发布年份** | 2024 / 2025 |
| **数据规模** | 每年30题 (AIME I和AIME II各15题) |
| **难度** | 高中竞赛高级 (通往USAMO的邀请赛) |
| **评估指标** | Accuracy / Pass@1 / Avg@k |
| **答案格式** | 0-999的整数 |

**核心特征**: AIME是MAA主办的高难度数学竞赛，题目要求多步逻辑推理和创造性解题策略 [^482^]。答案为0-999范围内的整数，支持无歧义的自动评估。AIME 2025因发布时间较新，被常用作低数据污染风险的fresher benchmark [^107^]。

**评估实践**:
- 使用temperature=0.6, top_p=0.95的标准设置 [^488^][^615^]
- 小规模benchmark（30题）采用Avg@32或Avg@8减少方差 [^591^][^635^]
- 评估脚本多基于DeepScaleR代码库 [^488^]

---

### 1.4 AMC 2023

| 属性 | 详情 |
|------|------|
| **全称** | American Mathematics Competitions 2023 |
| **发布年份** | 2023 |
| **数据规模** | 40题 (AMC 12A和AMC 12B) |
| **难度** | 高中竞赛中级 |
| **评估指标** | Accuracy / Avg@8 |

**核心特征**: AMC 2023包含40道高中竞赛数学题，难度介于AIME和课堂练习之间 [^498^]。作为中等难度benchmark常用于评估模型鲁棒性和泛化能力。

---

### 1.5 OlympiadBench

| 属性 | 详情 |
|------|------|
| **全称** | OlympiadBench: Olympiad-level Bilingual Multimodal Scientific Benchmark |
| **发布年份** | 2024 |
| **数据规模** | 8,952题 (数学6,524 + 物理2,428) |
| **语言** | 中英双语 |
| **难度** | 国际奥赛级别 |
| **评估指标** | 多种（符号匹配、数值匹配）|

**核心特征**: OlympiadBench从国际数学奥林匹克、中国奥林匹克和高考最难部分收集的双语多模态科学benchmark [^524^]。包含逐步专家级推理注释。GPT-4V在数学部分仅得17.23%，物理部分11.28% [^524^]。

**使用方式**: 通常使用英语数学子集（约675题文本问题）[^530^][^494^]

---

### 1.6 GPQA / GPQA-Diamond

| 属性 | 详情 |
|------|------|
| **全称** | Graduate-Level Google-Proof Q&A |
| **发布年份** | 2024 (Rein et al.) |
| **数据规模** | GPQA Extended 546 / Main 448 / Diamond 198 |
| **领域** | 物理、化学、生物学 |
| **难度** | 研究生级别 |
| **评估指标** | Accuracy (多选题) |
| **答案格式** | 四选一多项选择 (A/B/C/D) |

**核心特征**: GPQA是由领域专家（PhD）撰写的问题集，故意设计为"Google-proof"——即使有无限制的网络访问，非专家也难以回答 [^482^][^483^]。GPQA-Diamond是最高质量的198题子集，仅保留独立专家验证者都正确回答而非专家验证者失败的问题 [^480^]。

**人类基线**: 领域专家准确率约65%，非专家约34%（相比随机基线25%）[^482^]

**在LLM+RL研究中的使用情况**:
- 作为OOD通用化能力的核心评估benchmark [^484^][^491^]
- 通常使用Avg@8或rollout=2进行多次采样取平均 [^635^][^486^]
- 使用OpenCompass框架进行评估 [^486^]

---

### 1.7 DeepScaleR

| 属性 | 详情 |
|------|------|
| **全称** | DeepScaleR-Preview-Dataset |
| **发布年份** | 2025 (Luo et al.) |
| **数据规模** | ~40,000题 |
| **来源** | AIME(1984-2023), AMC(<2023), Omni-MATH, STILL dataset |
| **用途** | 数学RLVR训练数据集兼参考benchmark |
| **难度分布** | 高中竞赛到高级数学推理 |

**核心特征**: DeepScaleR通过系统化的难度感知选择管道构建：先评估问题难度，然后渐进采样以强调需要多步推导的挑战性问题 [^498^]。它既是RL训练语料也是评估参考benchmark。

**关键创新**:
- 难度梯度精心校准
- 广泛覆盖代数、几何、数论、组合数学
- 在提升小规模模型推理边界方面表现出色

**代表性论文**:
- Luo et al., 2025 - DeepScaleR原始论文
- 被POPO [^107^], EAPO [^591^], HEAL [^487^], SPOT [^490^] 等方法广泛采用

---

### 1.8 FrontierMath

| 属性 | 详情 |
|------|------|
| **全称** | FrontierMath |
| **发布机构** | Epoch AI |
| **发布年份** | 2024 (Glazer et al.) |
| **特点** | 未发表的专家撰写问题，隐藏评估集 |
| **难度** | 研究级数学 |
| **评估指标** | Avg@8 |

**核心特征**: FrontierMath是Epoch AI设计的研究级数学benchmark，包含未发表的专家撰写问题，具有隐藏评估集以防止数据污染 [^508^]。从2024年6月到2025年12月，前沿模型在FrontierMath Tier 1-3上的准确率从1%提升至41% [^559^]。

---

### 1.9 Humanity's Last Exam (HLE)

| 属性 | 详情 |
|------|------|
| **全称** | Humanity's Last Exam |
| **发布年份** | 2025 (Phan et al.) |
| **数据规模** | ~3,000题 (文本子集2,158) |
| **领域** | 数学、物理、化学、生物、CS、工程、人文、社会科学 |
| **难度** | 前沿学术推理 |
| **评估指标** | Accuracy |

**核心特征**: HLE是由CAIS和Scale AI合作开发的多模态benchmark，旨在测试AI系统的前沿学术能力 [^554^]。与传统知识benchmark中前沿模型超过90%准确率不同，HLE中大多数模型得分低于10% [^554^]。

**题目分布** (文本子集):
- 数学: 45.23% (976题)
- 计算机科学/AI: 10.38%
- 生物/医学: 10.29%
- 物理: 9.36%

**人类vs模型表现**: 人类专家平均准确率>90%，Grok-4约25.4%，GPT-4和Claude-3<10% [^554^]

**使用情况**: 
- Nemotron-Cascade使用HLE文本子集(2,158题)评估推理能力 [^493^]
- 使用temperature=1.0, top-p=0.95, 128K token思考预算 [^493^]

---

### 1.10 LiveMathBench

| 属性 | 详情 |
|------|------|
| **全称** | LiveMathBench |
| **发布年份** | 2025 (Liu et al.) |
| **特点** | 持续更新的动态数学benchmark |
| **数据规模** | 202505 hard split含100题 |
| **难度** | 高难度 |

**核心特征**: LiveMathBench是持续更新的数学benchmark，提供高质量英语数学问题。Rectifying LLM Thought工作中使用了202505 hard split的100题 [^485^]。

---

### 1.11 Omni-MATH

| 属性 | 详情 |
|------|------|
| **全称** | Omni-MATH |
| **发布年份** | 2024 (Gao et al.) |
| **数据规模** | 4,428题 |
| **覆盖** | 33+数学子领域 |
| **难度** | 奥赛级别 |

**核心特征**: Omni-MATH是一个通用奥赛级数学benchmark，问题来源包括国家/国际奥林匹克、训练材料、在线论坛等 [^526^]。包含细粒度难度分层，强调问题多样性 [^536^]。

---

### 1.12 数学推理Benchmark演进关系

```
GSM8K (2021) → MATH (2021) → MATH-500 (2023) → AIME/AMC (竞赛级)
                                    ↓
                           OlympiadBench (2024) → Omni-MATH (2024)
                                    ↓
                           FrontierMath (2024) → HLE (2025)
                                    ↓
                           DeepScaleR (2025, 训练数据集)
```

---

## 2. 代码生成Benchmark

### 2.1 HumanEval / HumanEval+

| 属性 | 详情 |
|------|------|
| **全称** | HumanEval / HumanEval+ (EvalPlus) |
| **发布年份** | 2021 (Chen et al.) / 2023 (EvalPlus) |
| **数据规模** | 164题 / HumanEval+增强测试套件 |
| **语言** | Python |
| **评估指标** | Pass@k (功能正确性) |
| **难度** | 函数级代码生成 |

**核心特征**: HumanEval是评估Python函数合成的经典benchmark，每题包含docstring描述和单元测试。EvalPlus通过增加80倍以上的测试用例解决了原测试覆盖不足的问题 [^500^][^510^]。

---

### 2.2 MBPP (Mostly Basic Programming Problems)

| 属性 | 详情 |
|------|------|
| **全称** | Mostly Basic Programming Problems |
| **发布年份** | 2021 (Austin et al.) |
| **数据规模** | 974题 (sanitized version 427题) |
| **语言** | Python |
| **评估指标** | Pass@k |
| **难度** | 入门级编程 |

---

### 2.3 LiveCodeBench (LCB) v5 / v6

| 属性 | 详情 |
|------|------|
| **全称** | LiveCodeBench |
| **发布年份** | 2024 (Jain et al.) |
| **数据规模** | LCB v5: ~374-511题 / LCB v6: ~131-454题 (按时间窗口) |
| **语言** | Python (多平台) |
| **来源** | LeetCode, AtCoder, Codeforces |
| **评估指标** | Pass@1, avg@8 |
| **特点** | 时序分区防污染 |

**核心特征**: LiveCodeBench通过收集在线编程平台新发布的问题来最小化数据污染 [^510^]。以时间窗口进行版本划分（v5: 2024.8-2025.2, v6: 2025.2-2025.5），确保评估数据在模型训练截止日期之后 [^603^][^605^]。

**评估设置**:
- 标准设置: temperature=0.6, top_p=0.95, top_k=20 [^615^][^603^]
- 最大生成长度: 32,768-65,536 tokens [^607^]
- 通常报告avg@8（8次采样取平均）[^618^]

---

### 2.4 SWE-Bench 系列

| 变体 | 发布年份 | 规模 | 特点 |
|------|---------|------|------|
| SWE-Bench | 2024 (Jimenez et al.) | 原数据集 | 仓库级issue修复评估 |
| SWE-Bench Verified | 2024 (OpenAI) | 500题 | 人工验证子集，过滤问题 |
| SWE-Bench Pro | 2025 (Deng et al.) | 扩展 | 长程工程任务，多语言 |
| SWE-Bench Live | 2025 (Zhang et al.) | 动态 | 持续从GitHub收集新patch |

**核心特征**: SWE-Bench系列评估模型在真实代码库中修复GitHub issue的能力。通过仓库测试套件验证patch正确性 [^500^]。SWE-Bench Pro平均需要修改107行代码、跨越4个文件 [^521^]。

**代表性使用**:
- Nemotron-Cascade使用mini-SWE-agent scaffold在SWE-Bench Verified上评估 [^516^]
- 使用云评估工具sb-cli获取最终分数

---

### 2.5 BigCodeBench

| 属性 | 详情 |
|------|------|
| **全称** | BigCodeBench |
| **发布年份** | 2024 (Zhuo et al.) |
| **数据规模** | 1,140题 (Full) |
| **评估指标** | Pass@k |
| **特点** | 需要调用多样化第三方库的函数级代码生成 |

**核心特征**: BigCodeBench评估需要调用139个不同库的复杂功能实现，更贴近实际软件工程场景 [^589^][^590^]。

---

### 2.6 MultiPL-E

| 属性 | 详情 |
|------|------|
| **全称** | MultiPL-E |
| **发布年份** | 2023 (Cassano et al.) |
| **语言覆盖** | 18种编程语言 |
| **来源** | 将HumanEval和MBPP翻译到多语言 |
| **评估指标** | Pass@k |

---

### 2.7 其他代码Benchmark

| Benchmark | 描述 |
|-----------|------|
| **APPS** | 竞争性编程难度，2021 |
| **CodeContests** | Google Code竞赛问题，2022 |
| **CRUXEval** | 程序输入/输出预测，2024 |
| **DS-1000** | 数据科学代码生成，2023 |
| **EvalPlus** | HumanEval/MBPP的增强测试版本 |

---

### 2.8 代码Benchmark演进关系

```
HumanEval (2021) → EvalPlus (2023) → HumanEval+
MBPP (2021)
LiveCodeBench (2024) → v5 → v6 (时序防污染)
SWE-Bench (2024) → Verified → Pro → Live
BigCodeBench (2024) / MultiPL-E (2023)
```

---

## 3. 通用推理Benchmark

### 3.1 MMLU (Massive Multitask Language Understanding)

| 属性 | 详情 |
|------|------|
| **全称** | Massive Multitask Language Understanding |
| **发布年份** | 2021 (Hendrycks et al.) |
| **数据规模** | 57个学科, ~14,079测试题 |
| **评估指标** | Accuracy (多选) |
| **评估设置** | 5-shot |

---

### 3.2 MMLU-Pro

| 属性 | 详情 |
|------|------|
| **全称** | MMLU-Pro |
| **发布年份** | 2024 (Wang et al.) |
| **数据规模** | ~12,032题 (研究生级别) |
| **评估指标** | Accuracy |
| **关键改进** | 选项从4个增加到10个，移除简单/噪声问题 |

**核心特征**: MMLU-Pro通过增加选项数量、加入更多具有挑战性的大学级别问题、多轮专家审查消除噪声问题来提升区分度 [^512^][^635^]。GPT-4在MMLU-Pro上仅达72.6%准确率，相比MMLU显著下降，表明难度和区分度提升 [^512^]。

**代表性使用**:
- Nemotron-Cascade使用thinking模式，单generation评估 [^635^]
- 与MMLU相比，MMLU-Pro具有更好的题目区分度 [^511^]

---

### 3.3 BBH (BIG-Bench Hard)

| 属性 | 详情 |
|------|------|
| **全称** | BIG-Bench Hard |
| **发布年份** | 2023 (Suzgun et al.) |
| **数据规模** | 23个任务 |
| **评估指标** | 各任务Accuracy |
| **特点** | 先前LLM未能超过平均人类评分的任务子集 |

**核心特征**: BBH从BIG-Bench中筛选出23个最具挑战性的任务，覆盖算法推理、常识推理和多步推理 [^570^][^572^]。随着前沿模型在BBH上接近满分，区分度下降。

---

### 3.4 BBEH (BIG-Bench Extra Hard)

| 属性 | 详情 |
|------|------|
| **全称** | BIG-Bench Extra Hard |
| **发布年份** | 2025 (Kazemi et al., Google DeepMind) |
| **数据规模** | 23个任务 (BBH的加难版本) |
| **评估指标** | Harmonic Average Accuracy |

**核心特征**: BBEH针对BBH饱和问题，将每个BBH任务替换为探测相似推理能力但难度显著增加的新任务。最佳通用模型在BBEH上平均准确率仅9.8%，最佳推理专用模型44.8% [^601^]。

---

### 3.5 ARC (AI2 Reasoning Challenge)

| 属性 | 详情 |
|------|------|
| **全称** | AI2 Reasoning Challenge |
| **发布年份** | 2018 (Clark et al.) |
| **数据规模** | Challenge Set 2,590题 / Easy Set |
| **评估指标** | Accuracy |
| **评估设置** | 25-shot |

---

## 4. 指令遵循Benchmark

### 4.1 IFEval (Instruction Following Evaluation)

| 属性 | 详情 |
|------|------|
| **全称** | Instruction Following Evaluation |
| **发布年份** | 2023 (Zhou et al.) |
| **数据规模** | ~541 prompts, 25种约束类型 |
| **评估指标** | Prompt-level / Instruction-level Accuracy |
| **评估方式** | 程序化验证（无需LLM judge） |

**核心特征**: IFEval评估模型遵循可验证指令的能力（如输出长度限制、格式要求、关键词使用等）[^501^][^567^]。使用预定义的Python脚本进行客观评估，避免LLM-as-a-Judge的主观偏差。

**代表性使用**:
- Nemotron-Cascade使用thinking和non-thinking模式中的较高分 [^635^]
- 使用OpenCompass工具包进行评估 [^595^]

---

### 4.2 MT-Bench

| 属性 | 详情 |
|------|------|
| **全称** | Multi-Turn Benchmark |
| **发布年份** | 2023 (Zheng et al.) |
| **评估方式** | LLM-as-a-Judge (GPT-4打分) |
| **评分** | 1-10分 |

**核心特征**: MT-Bench使用强大的LLM（如GPT-4）作为评判者，对模型输出进行1-10分的评分。文档中存在rubric drift和judge bias问题 [^500^]。

---

### 4.3 AlpacaEval 2

| 属性 | 详情 |
|------|------|
| **全称** | AlpacaEval 2.0 (Length-Controlled) |
| **发布年份** | 2024 (Dubois et al.) |
| **评估方式** | LLM-as-a-Judge pairwise comparison |
| **关键指标** | LC Win Rate (长度控制胜率) |

**核心特征**: AlpacaEval 2.0引入长度控制机制减少评估器长度偏差。两两比较模型输出，反映哪个模型的输出更受欢迎 [^564^]。

---

### 4.4 其他指令遵循Benchmark

| Benchmark | 年份 | 规模 | 特点 |
|-----------|------|------|------|
| **FollowBench** | 2024 | 820 instructions | 5级难度约束跟随 [^509^] |
| **ArenaHard** | 2024 | 500题 | Chatbot Arena困难子集 |
| **IFBench** | 2025 | 58约束类型 | 更细粒度的指令遵循评估 |
| **WildBench** | 2025 | 1,024样本 | 真实世界对话日志 |
| **MT-Bench-101** | 2024 | - | 扩展多轮评估 |

---

## 5. Agent评估Benchmark

### 5.1 AgentBench

| 属性 | 详情 |
|------|------|
| **全称** | AgentBench |
| **发布年份** | 2023 (Wang et al.) |
| **评估维度** | 多步任务执行能力 |

**核心特征**: AgentBench评估LLM Agent在多种环境中的决策和执行能力，包括操作系统、数据库、知识图谱、数字卡牌游戏等 [^504^]。

---

### 5.2 ToolBench

| 属性 | 详情 |
|------|------|
| **全称** | ToolBench |
| **发布年份** | 2023 (Qin et al.) |
| **评估维度** | 工具使用能力 |

---

### 5.3 GAIA

| 属性 | 详情 |
|------|------|
| **全称** | GAIA |
| **发布年份** | 2023 (Mialon et al.) |
| **数据规模** | 165题 (validation split) |
| **难度级别** | Level 1-3 |

**核心特征**: GAIA通过工具密集型任务评估Agent行为，Level 1测试事实检索和浅层推理，Level 2强调多跳推理和规划，Level 3需要推理和工具之间的复杂协调 [^491^]。

---

### 5.4 τ²-Bench

| 属性 | 详情 |
|------|------|
| **全称** | τ²-Bench Telecom |
| **发布年份** | 2025 (Barres et al.) |
| **领域** | 电信领域客户服务交互 |

---

### 5.5 BFCL (Berkeley Function-Calling Leaderboard)

| 属性 | 详情 |
|------|------|
| **全称** | Berkeley Function-Calling Leaderboard |
| **发布年份** | 2025 v3 (Patil et al.) |
| **评估维度** | 函数调用和工具使用准确率 |

---

### 5.6 其他Agent Benchmark

| Benchmark | 描述 |
|-----------|------|
| **AgentDojo** | 多步执行安全评估 |
| **WebShop** | Web导航和交互 |
| **ALFWorld** | 具身导航和操作 |
| **Lifelong Agent Bench** | 持续学习设置中的OS和DB交互 |
| **Cybench** | 网络安全CTF评估 |

---

## 6. 评估方法论

### 6.1 评估指标定义

#### Pass@k
测量k个独立采样输出中至少有一个正确的概率：
```
Pass@k = E[1 - C(n-c, k) / C(n, k)]
```
其中n为总采样数，c为正确数。Pass@1衡量单次生成正确率 [^596^]。

#### Maj@k (Majority Vote @k)
从k个独立采样输出中进行多数投票，选择最常见的答案作为最终答案 [^254^]。

#### Avg@k
k次采样的平均正确率，通过多次采样减少评估方差。常用于小规模benchmark（如AIME 30题）[^487^][^635^]。

#### RM@k (Reward Model @k)
使用奖励模型从k个采样中选择最佳答案。策略是选择具有每条路径最小PRM分数最大值的推理路径 [^254^]。

---

### 6.2 测试时采样策略

#### 标准评估设置 (2024-2025共识)

| 参数 | 数学推理 | 代码生成 | 知识推理 |
|------|----------|----------|----------|
| **temperature** | 0.6 | 0.6 | 0.6-1.0 |
| **top_p** | 0.95 | 0.95 | 0.95 |
| **top_k** | 20 | 20 | - |
| **max_tokens** | 32,768 | 32,768-65,536 | - |
| **采样次数** | 4-32 (Avg@k) | 1-8 (avg@k) | 1-8 |

**关键参考**:
- DeepSeek-R1评估框架: temperature=0.6, top_p=0.95 [^615^]
- Qwen3-8B官方协议: temperature=0.7, top_p=0.8, top_k=20 [^481^]
- Nemotron-Cascade推理任务: temperature=0.6, top_p=0.95 [^635^]
- 数学benchmark常用: temperature=1.0, top_p=0.95（生成阶段）[^591^]

---

### 6.3 数据污染与防污染策略

#### 主要挑战
- **静态Benchmark污染**: 评估数据可能出现在预训练语料中 [^532^]
- **后训练污染**: SFT/RL阶段使用的合成数据可能 resembling 评估任务 [^540^]
- **检测困难**: 大多数前沿模型不公开训练数据 [^532^]

#### 防污染Benchmark设计
| 策略 | 代表Benchmark |
|------|--------------|
| 时序过滤 (Temporal) | LiveCodeBench, LiveBench |
| 动态更新 | LiveBench (月度更新), SWE-Bench Live |
| 隐藏测试集 | FrontierMath |
| 持续更新 | LiveMathBench |
| 新问题生成 | PPM, DyCodeEval |

#### 污染检测方法
1. **变体检测**: 生成benchmark问题的变体，检查性能是否下降 [^532^]
2. **时序分析**: 比较训练截止前后的题目表现 [^532^]
3. **n-gram匹配**: 检测训练数据中的精确匹配

---

### 6.4 评估框架演进

| 框架 | 机构 | 特点 |
|------|------|------|
| **lm-evaluation-harness** | EleutherAI | 标准化开源评估 [^522^] |
| **OpenCompass** | 上海AI Lab | 综合评估工具包 [^595^] |
| **Inspect** | UK AISI | 安全评估框架 [^611^] |
| **simple-evals** | OpenAI | 简单评估框架 [^528^] |
| **Math-Verify** | - | 数学评估验证框架 [^528^] |
| **lighteval** | HuggingFace | 轻量评估框架 [^484^] |
| **verl** | - | RL训练与评估框架 [^630^] |

---

## 7. 评估框架与工具

### 7.1 lm-evaluation-harness (EleutherAI)

**核心特点**:
- 最广泛使用的标准化LLM评估框架
- 支持多种benchmark的统一评估接口
- 支持few-shot评估配置
- 被Open LLM Leaderboard等采用 [^522^][^538^]

**支持的Benchmark**: MMLU, GSM8K, HellaSwag, WinoGrande, ARC, TruthfulQA等

---

### 7.2 OpenCompass

**核心特点**:
- 上海AI Lab开发的综合评估平台
- 支持多种模型和benchmark的并行评估
- 被Nemotron-Cascade等工作用于GPQA-Diamond和AIME评估 [^486^][^595^]

---

### 7.3 Inspect (UK AISI)

**核心特点**:
- 英国AI安全研究所开发的开源评估框架
- 支持动态prompt工程、数据集过滤、多种评分指标
- 支持exact match和choice-based评估 [^611^]
- 广泛用于安全评估和能力评估 [^606^][^613^]

---

### 7.4 Math-Verify

**核心特点**:
- 专门的数学答案验证框架
- 支持从模型输出中提取`\boxed{}`中的答案
- 被FlashThink等工作采用 [^528^]
- 支持精确的数值和符号匹配

---

### 7.5 simple-evals (OpenAI)

**核心特点**:
- OpenAI开发的轻量评估框架
- 支持0-shot和3-shot设置
- 被FlashThink用于GSM8K, MATH, GPQA Diamond评估 [^528^]

---

## 8. Benchmark分类汇总表

### 8.1 综合分类表

| 维度 | Benchmark | 规模 | 主要指标 | 难度 | 防污染 | 代表性论文 |
|------|-----------|------|----------|------|--------|------------|
| **数学推理** | GSM8K | 1,319测试 | Exact Match | 小学 | 否 | RLVR基线评估 |
| | MATH-500 | 500 | Exact Match | 高中竞赛 | 否 | DeepScaleR, EAPO |
| | AIME 2024 | 30 | Accuracy | 邀请赛 | 部分 | 几乎所有RLVR论文 |
| | AIME 2025 | 30 | Accuracy | 邀请赛 | 强 | DeepScaleR, POPO |
| | AMC 2023 | 40 | Accuracy | 竞赛中级 | 部分 | HEAL, EAPO |
| | OlympiadBench | 8,952 | 多种 | 奥赛级 | 否 | 多项工作 |
| | GPQA-Diamond | 198 | Accuracy | 研究生 | 是(Google-proof) | Nemotron-Cascade |
| | FrontierMath | 隐藏集 | Avg@8 | 研究级 | 强(隐藏集) | Epoch AI |
| | HLE | 3,000 | Accuracy | 前沿学术 | 是 | Nemotron-Cascade |
| | DeepScaleR | 40K训练 | - | 竞赛级 | - | Luo et al., 2025 |
| | LiveMathBench | 动态 | Accuracy | 高 | 强(动态) | Rectifying LLM Thought |
| | Omni-MATH | 4,428 | Accuracy | 奥赛级 | 否 | 多项工作 |
| **代码生成** | HumanEval | 164 | Pass@k | 函数级 | 否 | 基线评估 |
| | HumanEval+ | 164增强 | Pass@k | 函数级 | 否 | EvalPlus增强 |
| | MBPP | 500 | Pass@k | 入门级 | 否 | 基线评估 |
| | LiveCodeBench v5 | ~374-511 | Pass@1/avg@8 | 竞赛级 | 强(时序) | AceReason, Nemotron |
| | LiveCodeBench v6 | ~131-454 | Pass@1/avg@8 | 竞赛级 | 强(时序) | RefineRL, X-Coder |
| | SWE-Bench | 动态 | Pass(test) | 仓库级 | 部分 | SWE-agent |
| | SWE-Bench Verified | 500 | Pass(test) | 仓库级 | 增强 | OpenAI验证 |
| | SWE-Bench Pro | 扩展 | Pass(test) | 长程工程 | 增强 | 多语言支持 |
| | BigCodeBench | 1,140 | Pass@k | 库级 | 否 | 代码评估 |
| | MultiPL-E | 18语言 | Pass@k | 函数级 | 否 | 多语言评估 |
| **通用推理** | MMLU | 14,079 | Accuracy | 多领域 | 否 | 通用基线 |
| | MMLU-Pro | 12,032 | Accuracy | 研究生 | 否 | Nemotron-Cascade |
| | BBH | 23任务 | Accuracy | 困难 | 否 | 通用推理 |
| | BBEH | 23任务 | Harmonic Avg | 极难 | 是(新题) | Google DeepMind |
| | ARC-C | 2,590 | Accuracy | 科学问答 | 否 | 通用基线 |
| **指令遵循** | IFEval | 541 prompts | 严格/宽松准确率 | - | - | Nemotron-Cascade |
| | MT-Bench | - | 1-10分 | - | - | LLM-as-Judge |
| | AlpacaEval 2 | - | LC Win Rate | - | - | 指令遵循 |
| | FollowBench | 820 | HSR | 5级 | - | 约束跟随 |
| | ArenaHard | 500 | Win Rate | - | - | Chatbot Arena |
| **Agent** | AgentBench | 多环境 | 多指标 | - | - | Agent评估 |
| | ToolBench | - | 工具使用 | - | - | 工具评估 |
| | GAIA | 165 | 准确率 | L1-L3 | - | Agent推理 |
| | BFCL v3 | 100 | 函数调用 | - | - | 工具使用 |

### 8.2 评估指标汇总

| 指标 | 定义 | 适用场景 | 论文来源 |
|------|------|----------|----------|
| **Pass@1** | 单次生成正确率 | 标准评估 | Chen et al., 2021 |
| **Pass@k** | k次采样至少一个正确 | 探索能力 | Chen et al., 2021 |
| **Avg@k** | k次采样平均正确率 | 减少方差 | DeepScaleR系列 |
| **Maj@k** | k次采样多数投票正确率 | 自一致性 | Wang et al., 2023 |
| **RM@k** | 奖励模型选最佳@k | PRM辅助 | Step-level Verifier |

### 8.3 标准评估超参数

| Benchmark类别 | Temperature | Top-p | Top-k | 采样次数 | Max Tokens |
|---------------|-------------|-------|-------|----------|------------|
| 数学推理 | 0.6 | 0.95 | 20 | 4-32 | 32,768 |
| 代码生成 | 0.6 | 0.95 | 20 | 1-8 | 32K-65K |
| 知识/科学 | 0.6-1.0 | 0.95 | - | 1-8 | 可变 |
| 指令遵循 | - | - | - | 1 | 可变 |

---

## 9. 参考文献

### 核心论文与来源

**数学推理Benchmark**:
- Hendrycks et al. (2021). "Measuring Mathematical Problem Solving With the MATH Dataset." [^495^]
- Lightman et al. (2023). "Let's Verify Step by Step." PRM800K, MATH-500. [^483^]
- Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems." GSM8K. [^537^]
- Rein et al. (2024). "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." [^482^][^483^]
- He et al. (2024). "OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems." [^524^]
- Luo et al. (2025). "DeepScaleR: Scaling Up Difficulty in Math Reasoning." [^498^][^107^]
- Glazer et al. (2024). "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI." Epoch AI. [^508^]
- Phan et al. (2025). "Humanity's Last Exam." [^554^]
- Gao et al. (2024). "Omni-MATH: A Universal Olympiad Level Mathematic Benchmark." [^526^]
- Liu et al. (2025). "LiveMathBench." [^485^]

**代码生成Benchmark**:
- Chen et al. (2021). "Evaluating Large Language Models Trained on Code." HumanEval. [^500^]
- Austin et al. (2021). "Program Synthesis with Large Language Models." MBPP. [^500^]
- Jain et al. (2024). "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code." [^510^][^603^]
- Jimenez et al. (2024). "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" [^500^][^521^]
- Zhuo et al. (2024/2025). "BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Realistic Dependencies." [^589^][^590^]
- Cassano et al. (2023). "MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation." [^500^]
- Liu et al. (2023). "EvalPlus: Rigorous Evaluation of LLM-based Code Synthesis." [^510^]

**通用推理与指令遵循**:
- Hendrycks et al. (2021). "Measuring Massive Multitask Language Understanding." MMLU. [^627^]
- Wang et al. (2024). "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark." [^512^][^635^]
- Suzgun et al. (2023). "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them." BBH. [^570^]
- Kazemi et al. (2025). "BIG-Bench Extra Hard." Google DeepMind. [^601^]
- Zhou et al. (2023). "Instruction Following Evaluation for Large Language Models." IFEval. [^567^]
- Zheng et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." [^604^]
- Dubois et al. (2024). "Length-Controlled AlpacaEval 2.0." [^564^]
- Li et al. (2024). "ArenaHard." [^564^]

**Agent评估**:
- Wang et al. (2023). "AgentBench: Evaluating LLMs as Agents." [^504^]
- Qin et al. (2023). "ToolBench." [^504^]
- Mialon et al. (2023). "GAIA: A Benchmark for General AI Assistants." [^491^]
- Patil et al. (2025). "BFCL v3: Berkeley Function-Calling Leaderboard." [^516^]

**评估方法论与框架**:
- Wang et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning." Maj@k. [^604^]
- Gao et al. (2024). "lm-evaluation-harness." EleutherAI. [^522^]
- UK AI Safety Institute (2024). "Inspect: A Framework for Large Language Model Evaluations." [^611^]
- White et al. (2025). "LiveBench: A Benchmark for Contamination-Resistant LLM Evaluation." [^532^]

**代表性LLM+RL论文使用Benchmark**:
- POPO/Beyond Negative Rollouts (2026). DeepScaleR训练，AIME/MATH/AMC/Olympiad评估 [^107^]
- EAPO (2026). DeepScaleR训练，6个数学benchmark [^591^]
- Nemotron-Cascade 2 (2026). MMLU/MMLU-Pro/GPQA/HLE全面评估 [^493^][^635^]
- DCPO (2026). MATH-500/AIME/LiveCodeBench [^481^]
- AceReason-Nemotron (2025). LiveCodeBench v5/v6 + CodeForces Elo [^615^]
- RefineRL (2026). LiveCodeBench v5/v6竞争编程 [^607^]
- Scaf-GRPO (2026). DeepScaleR训练，8个评估benchmark [^494^]

---

## 附录: 各论文具体实现细节

### A. DeepScaleR系列 (Luo et al., 2025)
- **训练数据**: 40K问题 (AIME 1984-2023, AMC pre-2023, Omni-MATH, STILL)
- **评估Benchmark**: MATH-500, AMC 2023, AIME 2024, AIME 2025, OlympiadBench
- **基座模型**: Qwen2.5-Math-1.5B/7B, DeepSeek-R1-Distill-Qwen系列
- **训练设置**: GRPO, batch size 256, group size 8, 3 epochs [^107^]

### B. Nemotron-Cascade 2 (2026)
- **评估套件**: MMLU, MMLU-Pro, GPQA-Diamond, HLE, IFEval, IFBench, ArenaHard, LiveCodeBench v5/v6, SWE-Bench Verified, BFCL-V3
- **推理设置**: temperature=0.6, top-p=0.95, 64K/128K token预算
- **GPQA-Diamond**: thinking模式, avg@8
- **HLE**: thinking模式, temperature=1.0, top-p=0.95, 128K budget [^493^][^635^]

### C. DCPO - 校准强化学习 (2026)
- **数学训练**: DeepScalar dataset
- **数学评估**: MATH-500, AIME 2024/2025, AMC 2023/2024
- **代码训练**: PrimeIntellect-verifier dataset
- **代码评估**: LiveCodeBench v5, LiveCodeBench v6, HumanEval+
- **评估设置**: temperature=0.7, top-p=0.8, top-k=20 [^481^]

### D. EAPO (2026)
- **训练数据**: DeepScaleR dataset
- **训练解码**: temperature=1.0, top-p=0.95
- **评估Benchmark**: AIME24, AIME25, AMC, MATH, Minerva, OlympiadBench
- **评估解码**: vLLM, temperature=0.6, top-p=0.95
- **指标**: AIME系列Avg@32/Pass@32, 其他Avg@4/Pass@4 [^591^]

### E. AceReason-Nemotron (2025)
- **代码评估**: LiveCodeBench v5 (202408-202502), v6 (202502-202505)
- **额外评估**: LiveCodeBench Pro (Codeforces), EvalPlus
- **评估设置**: DeepSeek-R1框架 (template, temperature=0.6, top_p=0.95, max_len=32768)
- **指标**: avg@k [^615^]

---

> **报告生成说明**: 本报告基于2024-2025年LLM+RL领域40+篇顶会和前沿论文的深度调研，覆盖NeurIPS, ICML, ICLR, ACL, EMNLP等会议及arXiv预印本。所有引用均来自实际论文内容。

> **最后更新**: 2025年
