# 5. Benchmark体系与评估方法

LLM+RL领域的快速发展催生了庞大的评估需求。截至2025年，该领域已积累超过45个活跃使用的Benchmark，覆盖数学推理、代码生成、通用知识、指令遵循与Agent行为等多个维度 [^537^][^500^][^510^]。然而，Benchmark饱和与数据污染问题日益严峻——GSM8K已被主流模型推至97.1%的准确率上限 [^525^]，HumanEval等经典代码Benchmark大量泄露至训练语料 [^500^]，AIME每年仅30题的稀缺供给难以满足大规模评估需求 [^482^]。本章系统梳理LLM+RL领域使用的Benchmark层级体系，建立从基础验证到前沿挑战的评估谱系，并深入分析采样策略标准化、评估框架工具链与数据污染检测方法论，为研究者提供可操作的评估实践指南。

## 5.1 数学推理Benchmark层级

数学推理Benchmark构成了LLM+RL研究中最为成熟和细分的评估体系。根据题目难度、知识层级和当前模型表现，可将数学推理Benchmark划分为基础层、进阶层、专家层与前沿层四个层级。

### 5.1.1 基础层：GSM8K——已饱和的基础能力验证

GSM8K（Grade School Math 8K）发布于2021年，包含1,319道测试集题目，均为需要多步推理的小学数学文字题 [^537^]。其评估指标为精确匹配（Exact Match），即模型输出答案与标准答案的数值一致性。GSM8K是RLVR训练后模型评估的标配基准——几乎所有GRPO类训练的论文均将其作为基础验证项 [^495^][^625^][^107^]。然而，GPT-4通过DUP策略已达97.1%准确率 [^525^]，DeepSeek-R1系列在该基准上接近满分，表明GSM8K的区分度已基本丧失。当前研究中，GSM8K的主要作用是验证模型是否具备基础多步推理能力，而非区分前沿模型的性能差异。训练实践中，研究者通常使用GSM8K-Aug（GPT-4扩增至385K样本的版本）作为训练数据，而保留原始测试集用于评估 [^544^]。

### 5.1.2 进阶层：MATH-500与AIME——区分度仍存但题目稀缺

MATH-500是OpenAI在PRM800K工作中引入的500题评估子集，源自Hendrycks等人发布的MATH数据集（12,500题，高中竞赛级别）[^483^][^480^]。该子集涵盖代数、几何、数论、组合数学等七个领域，难度等级1-5，是当前评估模型高中竞赛级数学能力的核心基准。代表性工作如DeepScaleR [^107^][^498^]、EAPO [^591^]、HEAL [^487^]均将MATH-500作为主要评估项。评估设置通常采用4-shot prompting [^626^]，前沿模型在该基准上的准确率约为80%-85%，仍有提升空间。

AIME（American Invitational Mathematics Examination）是通往USAMO（美国数学奥林匹克）的邀请赛，每年发布AIME I和AIME II各15题，共30题，答案为0-999范围内的整数 [^482^]。AIME题目要求多步逻辑推理和创造性解题策略，是当前区分前沿推理模型的高敏感度Benchmark。DeepSeek-R1在AIME 2024上达到79.8%的pass@1准确率，成为推理能力的标志性成就。由于每年仅30题，AIME评估通常采用Avg@32或Avg@8采样策略以减少方差 [^591^][^635^]，并使用temperature=0.6、top_p=0.95的标准解码设置 [^488^][^615^]。AIME 2025因发布时间较新，被广泛用于低数据污染风险的"fresher"评估 [^107^]。

### 5.1.3 专家层：GPQA-Diamond——前沿模型与人类专家的差距

GPQA-Diamond（Graduate-Level Google-Proof Q&A Diamond子集）包含198道由PhD领域专家撰写的多选题，覆盖物理、化学和生物学 [^482^][^483^]。该Benchmark的设计核心在于"Google-proof"——即使有无限制的网络访问，非专家也难以正确回答。GPQA-Diamond仅保留那些独立专家验证者均能正确回答、而非专家验证者失败的问题，确保题目质量 [^480^]。人类基线数据显示，领域专家准确率约65%，非专家约34%（相比随机基线25%）[^482^]，为模型评估提供了清晰的人类能力参照。GPQA-Diamond是评估OOD（Out-of-Distribution）通用化能力的核心基准 [^484^][^491^]，通常使用Avg@8或rollout=2进行多次采样取平均 [^635^][^486^]。当前前沿模型在该基准上的准确率约为60%-65%，尚未超越人类专家水平，使其成为最具区分度的专家级评估工具之一。

### 5.1.4 前沿层：FrontierMath与LiveMathBench——抗污染动态更新机制

FrontierMath由Epoch AI于2024年发布，采用未发表的专家撰写问题与隐藏评估集设计，从根本上防止数据污染 [^508^]。该Benchmark的研究级数学问题覆盖多个前沿领域。数据显示，从2024年6月到2025年12月，前沿模型在FrontierMath Tier 1-3上的准确率从1%提升至41% [^559^]，进步显著但仍远低于饱和水平，是当前评估模型数学能力前沿的有效工具。

LiveMathBench是2025年推出的持续更新动态数学Benchmark，提供高质量英语数学问题 [^485^]。其202505 hard split包含100道高难度题目。动态更新机制确保了评估数据的新鲜度，降低了训练数据污染的风险。类似的抗污染设计理念也被LiveCodeBench所采用，后者通过时序分区实现版本迭代（v5: 2024.8-2025.2, v6: 2025.2-2025.5）[^603^][^605^]。

 Humanity's Last Exam（HLE）是2025年由CAIS与Scale AI合作开发的多模态Benchmark，文本子集约2,158题，其中数学占45.23%（976题）[^554^]。与传统知识Benchmark中前沿模型超过90%准确率不同，HLE中大多数模型得分低于10%，人类专家平均准确率则超过90% [^554^]。Nemotron-Cascade使用thinking模式、temperature=1.0与128K token思考预算进行评估，Grok-4约达25.4%，而GPT-4和Claude-3低于10% [^493^][^554^]。

**表1：数学推理Benchmark层级体系**

| Benchmark | 难度级别 | 题目数量 | 评估指标 | 代表性论文使用频率 | 是否已饱和 |
|:---|:---|:---|:---|:---|:---|
| GSM8K [^537^] | 小学 | 1,319 | Exact Match | 极高（几乎所有RLVR论文） | 是（97%+）[^525^] |
| MATH-500 [^483^] | 高中竞赛 | 500 | Exact Match | 高（DeepScaleR, EAPO, HEAL等） | 接近饱和（~85%） |
| AIME 2024 [^482^] | 邀请赛 | 30 | Accuracy / Avg@32 | 极高（推理模型标配） | 否（~80%区分度仍存） |
| AIME 2025 [^107^] | 邀请赛 | 30 | Accuracy / Avg@32 | 高（低污染优势） | 否 |
| AMC 2023 [^498^] | 竞赛中级 | 40 | Accuracy / Avg@8 | 中（HEAL, EAPO等） | 否 |
| OlympiadBench [^524^] | 国际奥赛 | 8,952 | 多种 | 中（多领域评估） | 否（GPT-4V仅17%） |
| Omni-MATH [^526^] | 奥赛级 | 4,428 | Accuracy | 中（33+子领域覆盖） | 否 |
| DeepScaleR Dataset [^498^] | 竞赛级 | ~40K训练 | — | 高（训练兼评估参考） | — |
| GPQA-Diamond [^482^] | 博士级 | 198 | Accuracy | 高（Nemotron-Cascade等） | 否（~65%，未超专家） |
| FrontierMath [^508^] | 研究级 | 隐藏集 | Avg@8 | 中（Epoch AI发布） | 否（41%且持续提升） |
| LiveMathBench [^485^] | 高难度 | 动态更新 | Accuracy | 低（新兴Benchmark） | 否（动态机制） |
| HLE [^554^] | 前沿学术 | ~3,000 | Accuracy | 中（Nemotron-Cascade） | 否（<10%多数模型） |

![数学推理Benchmark饱和度梯度与难度层级](fig_math_benchmark_saturation.png)

上表与图呈现了数学推理Benchmark从基础到前沿的完整层级谱系。GSM8K作为基础验证工具已趋于饱和，其核心价值在于排除模型是否存在基础推理缺陷，而非性能排序。MATH-500和AIME系列构成当前评估的主干，前者覆盖高中竞赛级知识广度，后者提供高区分度的推理深度测试。GPQA-Diamond以其博士级难度和人类专家基线成为衡量模型是否达到专家水平的关键门槛。FrontierMath、LiveMathBench和HLE则代表了抗污染评估的前沿方向，通过隐藏测试集、动态更新和极高难度设计，试图在模型能力快速进步的背景下维持评估的区分效度。值得注意的是，AIME每年仅30题的供给量构成了评估瓶颈——即便采用Avg@32采样策略，评估结果的统计置信度仍受限于题目池规模。这一结构性约束推动了LiveMathBench等动态更新机制的发展。

## 5.2 代码生成Benchmark

代码生成评估在LLM+RL研究中占据重要地位，尤其是RLVR训练在代码推理任务上的应用日益广泛。代码Benchmark面临的独特挑战在于评估数据更容易被泄露到训练语料中——LeetCode、Codeforces等平台的公开题目常被收录于预训练数据。

### 5.2.1 HumanEval/MBPP：经典但已被大量泄露

HumanEval发布于2021年，包含164道Python函数合成题目，每题包含docstring描述与单元测试，评估指标为Pass@k [^500^]。HumanEval是代码生成评估的开创性Benchmark，但由于其广泛使用和公开时间较长，题目内容已被大量收录于预训练语料和合成训练数据中，数据污染风险极高。EvalPlus框架通过增加80倍以上的测试用例解决了原测试覆盖不足的问题，形成HumanEval+增强版本 [^500^][^510^]，在一定程度上缓解了测试泄露导致的分数虚高问题。

MBPP（Mostly Basic Programming Problems）发布于同年，包含974题（sanitized版本427题），难度为入门级编程 [^500^]。与HumanEval类似，MBPP也面临数据污染问题，且其入门级难度对当前前沿模型的区分度有限。

### 5.2.2 LiveCodeBench：每月更新机制解决数据污染问题

LiveCodeBench（LCB）是2024年发布的动态代码评估Benchmark，通过收集LeetCode、AtCoder、Codeforces等在线编程平台新发布的问题来最小化数据污染 [^510^]。其核心创新在于时序分区版本划分：LCB v5覆盖2024年8月至2025年2月的新题目（约374-511题），LCB v6覆盖2025年2月至2025年5月（约131-454题）[^603^][^605^]。这种基于时间窗口的版本迭代机制确保了评估数据在模型训练截止日期之后发布，从根本上切断了数据污染的路径。

评估设置上，LiveCodeBench采用temperature=0.6、top_p=0.95、top_k=20的标准解码配置 [^615^][^603^]，最大生成长度为32,768至65,536 tokens [^607^]，通常报告avg@8指标 [^618^]。AceReason-Nemotron [^615^]、RefineRL [^607^]、DCPO [^481^]等代表性工作均采用LiveCodeBench v5/v6作为主要代码评估基准，使其成为当前代码推理模型评估的事实标准。

### 5.2.3 SWE-Bench系列：真实软件工程任务评估

SWE-Bench系列评估模型在真实代码库中修复GitHub issue的能力，是连接代码生成与实际软件工程实践的桥梁。该系列包含多个变体：SWE-Bench Original [^500^]评估仓库级issue修复，通过仓库测试套件验证patch正确性；SWE-Bench Verified [^500^]是OpenAI人工验证的500题子集，过滤了原数据集中的问题；SWE-Bench Pro [^521^]将评估扩展至长程工程任务与多语言支持，平均需要修改107行代码、跨越4个文件；SWE-Bench Live [^500^]则持续从GitHub收集新patch，实现动态更新。

Nemotron-Cascade使用mini-SWE-agent scaffold在SWE-Bench Verified上评估，通过云评估工具sb-cli获取最终分数 [^516^]。SWE-Bench系列的高实用性使其成为评估模型真实软件工程能力的黄金标准，但其评估环境复杂、运行成本高（需要执行仓库测试套件），限制了其在研究中的普及度。

BigCodeBench [^589^][^590^]是另一个值得关注的代码Benchmark，包含1,140道题目，评估需要调用139个不同第三方库的复杂功能实现，更贴近实际软件开发场景。MultiPL-E [^500^]则将HumanEval和MBPP翻译至18种编程语言，用于评估多语言代码生成能力。

**表2：代码生成Benchmark对比**

| Benchmark | 语言 | 题目数 | 评估指标 | 更新频率 | 泄露风险 | 主要特点 |
|:---|:---|:---|:---|:---|:---|:---|
| HumanEval [^500^] | Python | 164 | Pass@k | 静态 | 极高 | 开创性函数级代码评估 |
| HumanEval+ [^510^] | Python | 164增强 | Pass@k | 静态 | 高（增强测试套件缓解） | 80x+测试用例覆盖 |
| MBPP [^500^] | Python | 974 (sanitized 427) | Pass@k | 静态 | 极高 | 入门级编程 |
| LiveCodeBench v5 [^603^] | Python (多平台) | ~374-511 | Pass@1, avg@8 | 时序分区（2024.8-2025.2） | 低 | 动态防污染，事实标准 |
| LiveCodeBench v6 [^605^] | Python (多平台) | ~131-454 | Pass@1, avg@8 | 时序分区（2025.2-2025.5） | 低 | 最新版本 |
| SWE-Bench Verified [^500^] | 多语言 | 500 | Pass(test) | 静态（人工验证） | 中 | 真实issue修复评估 |
| SWE-Bench Pro [^521^] | 多语言 | 扩展 | Pass(test) | 静态 | 中 | 长程工程，107行/4文件均改 |
| SWE-Bench Live [^500^] | 多语言 | 动态 | Pass(test) | 持续更新（GitHub） | 低 | 动态收集新patch |
| BigCodeBench [^589^] | Python | 1,140 | Pass@k | 静态 | 中 | 139个第三方库调用 |
| MultiPL-E [^500^] | 18种语言 | — | Pass@k | 静态 | 高 | 多语言翻译评估 |

![代码生成Benchmark泄露风险与实用价值对比](fig_code_benchmark_risk_utility.png)

上表与图揭示了代码评估领域的结构性迁移趋势。HumanEval和MBPP作为经典Benchmark，虽然使用频率仍然很高，但其极高的泄露风险使其评估结果的信度受到质疑——模型在这些Benchmark上的高分可能反映的是记忆能力而非生成能力。LiveCodeBench通过时序分区机制实现了低泄露风险与高实用价值的双重优势，成为2024-2025年代码评估的首选工具。SWE-Bench系列则在任务真实性维度上处于领先地位，从Verified的人工验证到Pro的长程工程任务，再到Live的持续更新，覆盖了从可控研究评估到真实工程场景的完整光谱。研究者在实际评估中应当避免单独使用HumanEval/MBPP，而采用"LiveCodeBench（动态防污染）+ SWE-Bench Verified（真实任务）+ HumanEval+（增强测试）"的组合策略，以获得更全面和可信的评估结果。

## 5.3 通用能力与指令遵循

除数学推理和代码生成外，LLM+RL研究还需要评估模型的通用知识覆盖、符号推理能力与指令遵循质量。

### 5.3.1 MMLU/MMLU-Pro与BBH/BBEH

MMLU（Massive Multitask Language Understanding）是2021年发布的通用知识Benchmark，覆盖57个学科，约14,079道测试题，评估设置为5-shot [^627^]。MMLU长期作为通用模型能力评估的基线，但随着前沿模型在该基准上接近90%准确率，其区分度逐渐下降。MMLU-Pro [^512^][^635^]于2024年发布作为升级替代，将选项从4个增加到10个，移除简单和噪声问题，加入更多大学级别挑战性题目。GPT-4在MMLU-Pro上仅达72.6%准确率，相比MMLU显著下降，表明难度和区分度得到有效提升 [^512^]。Nemotron-Cascade使用thinking模式、单generation评估MMLU-Pro [^635^]。

BBH（BIG-Bench Hard）从BIG-Bench中筛选出23个最具挑战性的任务，覆盖算法推理、常识推理和多步推理 [^570^][^572^]。随着前沿模型在BBH上接近满分，其区分度也在下降。BBEH（BIG-Bench Extra Hard）是Google DeepMind于2025年发布的加难版本，将每个BBH任务替换为探测相似推理能力但难度显著增加的新任务 [^601^]。最佳通用模型在BBEH上平均准确率仅9.8%，最佳推理专用模型44.8% [^601^]，重新建立了有效的区分梯度。ARC（AI2 Reasoning Challenge）Challenge Set包含2,590题科学问答题目 [^500^]，作为通用基线评估使用。

### 5.3.2 IFEval/AlpacaEval 2.0：指令遵循与对齐评估

IFEval（Instruction Following Evaluation）评估模型遵循可验证指令的能力，包含约541个prompts，覆盖25种约束类型（如输出长度限制、格式要求、关键词使用等）[^501^][^567^]。其核心优势在于使用预定义的Python脚本进行客观评估，完全避免LLM-as-a-Judge的主观偏差。Nemotron-Cascade在IFEval评估中采用thinking和non-thinking模式中的较高分 [^635^]，使用OpenCompass工具包进行评估 [^595^]。

AlpacaEval 2.0通过LLM-as-a-Judge两两比较机制评估指令遵循质量，引入长度控制（Length-Controlled）机制减少评估器的长度偏差 [^564^]。其核心指标LC Win Rate反映模型输出在与基线模型对比中的胜率。MT-Bench [^500^][^604^]使用GPT-4等强大LLM作为评判者对模型输出进行1-10分评分，覆盖多轮对话能力，但存在rubric drift和judge bias问题。其他值得关注的指令遵循Benchmark包括FollowBench [^509^]（820条instructions，5级难度约束跟随）、ArenaHard [^564^]（500题Chatbot Arena困难子集）和WildBench [^500^]（1,024样本真实世界对话日志）。

在通用能力评估实践中，前沿模型通常采用MMLU/MMLU-Pro + BBH/BBEH + ARC的组合来覆盖通用知识与符号推理，配合IFEval + AlpacaEval 2.0评估指令遵循质量。这种多维组合避免了单一Benchmark饱和导致的评估失真。

## 5.4 评估方法论与框架

标准化评估方法论是确保Benchmark结果可比性和可复现性的基础。2024-2025年间，LLM+RL领域在采样策略、评估框架和数据污染检测三个维度上形成了日益成熟的方法论共识。

### 5.4.1 采样策略标准化

评估采样策略的标准化是确保不同研究结果可比性的关键。在数学推理和代码生成评估中，temperature=0.6、top_p=0.95已成为广泛接受的标准解码配置 [^615^][^603^][^635^]，部分工作增加top_k=20的限制 [^481^]。这一设置平衡了采样多样性与结果稳定性——较低的temperature确保输出质量，较高的top_p保留合理的采样空间。

对于小规模Benchmark（如AIME仅30题），多次采样取平均是减少评估方差的必要手段。Avg@k（k次采样平均正确率）是当前数学推理评估的首选指标 [^487^][^635^]，相比Pass@k更稳定。AIME系列通常采用Avg@32 [^591^]或Avg@8 [^635^]，GPQA-Diamond使用Avg@8或rollout=2 [^486^]。Pass@1衡量单次生成正确率，反映模型在实际部署场景中的期望性能。Maj@k（Majority Vote @k）从k个采样中进行多数投票 [^254^]，RM@k使用奖励模型从k个采样中选择最佳答案 [^254^]。

最大生成长度方面，数学推理通常设置32,768 tokens [^615^]，代码生成因需要更长输出设置32K-65K tokens [^607^]。Qwen3-8B使用temperature=0.7、top_p=0.8、top_k=20的配置 [^481^]，与DeepSeek-R1标准设置略有不同，反映了不同模型对解码超参数的敏感度差异。

### 5.4.2 评估框架工具链

标准化的评估框架是确保Benchmark可复现执行的基础设施。当前领域形成了多个互补的评估框架。

lm-evaluation-harness由EleutherAI开发，是最广泛使用的标准化开源LLM评估框架，支持MMLU、GSM8K、HellaSwag、WinoGrande、ARC、TruthfulQA等多种Benchmark的统一评估接口和few-shot评估配置 [^522^][^538^]，被Open LLM Leaderboard等平台采用。OpenCompass由上海AI Lab开发，支持多种模型和Benchmark的并行评估 [^595^]，被Nemotron-Cascade用于GPQA-Diamond和AIME评估 [^486^]。Inspect由英国AI安全研究所（UK AISI）开发，支持动态prompt工程、数据集过滤和多种评分指标（exact match、choice-based）[^611^][^606^]，广泛用于安全评估和能力评估。Math-Verify是专门的数学答案验证框架，支持从模型输出中提取`\boxed{}`中的答案并进行精确的数值和符号匹配 [^528^]，被FlashThink等工作采用。simple-evals由OpenAI开发，支持0-shot和3-shot设置 [^528^]。lighteval由HuggingFace开发，定位为轻量评估框架 [^484^]。verl是结合RL训练与评估的综合框架 [^630^]。

**表3：评估框架工具链对比**

| 框架 | 开发机构 | 支持算法/模型 | 支持Benchmark数量 | 主要特点 |
|:---|:---|:---|:---|:---|
| lm-evaluation-harness [^522^] | EleutherAI | 通用LLM | 50+（MMLU, GSM8K, ARC等） | 最广泛使用的标准化开源评估，few-shot配置，Open LLM Leaderboard采用 |
| OpenCompass [^595^] | 上海AI Lab | 通用LLM | 100+ | 多模型并行评估，被Nemotron-Cascade用于GPQA/AIME [^486^] |
| Inspect [^611^] | UK AISI | 通用LLM | 30+ | 动态prompt工程，安全评估专用，exact match + choice-based评分 |
| Math-Verify [^528^] | 独立 | 数学推理模型 | 数学专用 | 数学答案验证，`\boxed{}`提取，精确数值/符号匹配 |
| simple-evals [^528^] | OpenAI | OpenAI模型为主 | 10+ | 轻量，0-shot和3-shot，被FlashThink采用 |
| lighteval [^484^] | HuggingFace | HuggingFace模型 | 20+ | 轻量快速评估，与HF生态深度集成 |
| verl [^630^] | 独立 | RL训练模型 | 10+ | RL训练与评估一体化，支持GRPO类算法评估 |

上述框架形成了互补的评估工具链。lm-evaluation-harness和OpenCompass是通用评估的主力，覆盖从基础到高级的多种Benchmark。Math-Verify填补了数学答案精确验证的空白，其`\boxed{}`提取和符号匹配功能对于数学推理评估至关重要。Inspect在安全评估领域具有独特优势。研究者在实际评估中通常组合使用多个框架——例如使用OpenCompass执行GPQA-Diamond评估 [^486^]，使用Math-Verify处理数学答案验证 [^528^]，使用lm-evaluation-harness执行MMLU等通用Benchmark [^522^]。

### 5.4.3 数据污染检测

数据污染是LLM+RL评估面临的根本性挑战，污染来源包括预训练语料重叠和SFT/RL阶段合成数据的意外泄露 [^532^][^540^]。主要检测方法包括三类：基于n-gram匹配的精确文本重叠检测，是最直接但容易被改写绕过的方法；基于embedding相似度的语义检测，可捕获改写和释义后的污染；变体检测方法通过生成Benchmark问题的变体版本，比较模型在原始题与变体题上的表现差异，若性能显著下降则暗示污染 [^532^]。时序分析方法通过比较模型在训练截止前后发布的题目表现来推断污染程度 [^532^]。

防污染Benchmark设计策略呈现多样化趋势。时序过滤（Temporal Filtering）以LiveCodeBench [^510^]和LiveBench [^532^]为代表，仅使用模型训练截止日期之后发布的新题目。动态更新机制以SWE-Bench Live [^500^]和LiveMathBench [^485^]为代表，持续从真实来源收集新数据。隐藏测试集以FrontierMath [^508^]为代表，评估集不对外公开。此外，PPM和DyCodeEval等工作探索了自动生成新问题的方法论。由于大多数前沿模型不公开训练数据 [^532^]，完全消除污染评估在实践中极为困难，因此组合使用多种防污染策略（动态更新 + 隐藏集 + 变体检测）成为当前的最佳实践。

