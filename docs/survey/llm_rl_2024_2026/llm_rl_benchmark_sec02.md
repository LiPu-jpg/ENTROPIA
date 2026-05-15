## 2. 核心算法体系：从PPO到GRPO

LLM（Large Language Model，大语言模型）与RL（Reinforcement Learning，强化学习）的融合经历了三次范式跃迁：从PPO（Proximal Policy Optimization，近端策略优化）主导的多阶段RLHF流程，到DPO（Direct Preference Optimization，直接偏好优化）开启的离线偏好优化时代，再到GRPO（Group Relative Policy Optimization，组相对策略优化）引领的无Critic强化学习浪潮。本章系统梳理各方法的核心公式、典型配置与适用场景，为算法选择提供定量依据。

![LLM+RL核心算法演进时间线](fig_sec02_algorithm_timeline.png)

*图2-1 LLM+RL核心算法演进时间线。自2022年PPO应用于RLHF以来，算法演进呈现三大趋势：离线化（PPO→DPO）、简化（DPO→SimPO/ORPO）和去Critic化（PPO→GRPO）。GRPO及其变体在2024年后成为推理训练的主流范式。*

### 2.1 PPO与经典RLHF

#### 2.1.1 PPO在LLM中的标准实现

PPO是OpenAI在InstructGPT [^14^] 中确立的RLHF标准算法，其标准实现采用Actor-Critic架构。Actor模型（即被训练的语言模型）负责生成响应，Critic模型（通常是与Actor同规模或更小规模的独立网络）负责评估每个状态的价值函数。两者的协同通过GAE（Generalized Advantage Estimation，广义优势估计）实现：

$$\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 为时序差分残差，$\gamma$ 为折扣因子，$\lambda \in [0,1]$ 控制偏差-方差权衡。在LLM场景中，由于生成任务通常被视为episodic（每个序列独立），$\gamma$ 和 $\lambda$ 常被设为1 [^193^]。

PPO的核心创新在于Clipped Surrogate Objective，限制策略更新的幅度以避免训练崩溃：

$$\mathcal{L}^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为重要性采样比率，$\epsilon$ 通常为0.2。此外，KL散度约束 $\beta \cdot D_{KL}[\pi_\theta \| \pi_{ref}]$ 被添加到奖励函数中，防止策略偏离参考模型过远 [^14^]。

#### 2.1.2 ICML 2024全面对比：PPO与DPO的适用边界

Xu等人在ICML 2024发表的系统性对比研究表明，PPO在需要探索的复杂任务（如代码生成）上显著优于DPO，但训练稳定性更差 [^193^]。实验覆盖HH-RLHF（对话安全）、SafeRLHF（安全对齐）、APPS和CodeContest（代码生成）四个任务领域。结果显示，PPO在APPS和CodeContest上的pass@1分数平均高出DPO 8-12个百分点，这归因于PPO的在线采样机制允许策略在训练过程中探索超出静态偏好数据分布的新解法。相比之下，DPO在HH-RLHF等简单偏好对齐任务上收敛更快、训练更稳定，且无需维护Critic模型，工程实现更为简洁 [^193^]。

值得注意的是，PPO的训练不稳定性在超参数敏感性和奖励黑客（reward hacking）方面表现突出——不当的KL系数可能导致策略过早收敛到次优解，而优势估计的方差在序列长度超过512 tokens时显著增大 [^229^]。

#### 2.1.3 典型配置

基于ICML 2024的大规模实验和后续开源实现（如OpenRLHF [^327^]、TRL），PPO在LLM训练中的典型超参数配置如下表所示：

**表2-1 PPO在LLM训练中的典型超参数配置**

| 参数 | 典型设置 | 说明 |
|------|----------|------|
| Actor学习率 | $1\times10^{-5}$ [^193^] | 通常高于Critic以加速策略更新 |
| Critic学习率 | $5\times10^{-6}$ [^193^] | 较低学习率稳定价值估计 |
| GAE $\lambda$ | 1.0 [^193^] | LLM生成任务通常设为1（无折扣） |
| 折扣因子 $\gamma$ | 1.0 [^193^] | Episodic任务无时间折扣 |
| KL系数 $\beta$ | 0.01-0.1 [^193^] | 0.1为常见默认值，控制与参考模型的偏离 |
| 全局Batch Size | 512 [^193^] | 含多个prompt的采样结果 |
| 裁剪阈值 $\epsilon$ | 0.2 [^14^] | PPO标准设置 |
| 采样温度 | 1.0 [^193^] | 训练时采样温度 |
| Top-k | 200 [^193^] | 限制采样空间 |
| 奖励裁剪 | 20 [^193^] | 防止极端奖励值 |
| 最大生成长度 | 256-1024 [^193^] | 对话任务256，代码任务1024 |
| 训练Epoch | 3-5 [^193^] | 视数据量调整 |

该配置在7B-13B参数规模的模型上经过广泛验证 [^193^]。其中最关键的超参数是KL系数$\beta$：当$\beta$过小时策略可能快速偏离参考模型导致奖励黑客；过大则策略更新缓慢，难以获得显著提升。ICML 2024的实验建议$\beta=0.1$作为安全起点，根据任务复杂度向下调整 [^193^]。

### 2.2 DPO及其变体家族

#### 2.2.1 DPO核心公式：偏好学习作为分类问题

DPO是RLHF领域的里程碑式工作，其核心洞察在于：KL正则化奖励最大化问题的最优策略与奖励函数之间存在闭式关系 [^74^]。通过反转该关系，DPO用策略参数化奖励函数，将原本需要奖励模型和在线采样的两阶段训练简化为单阶段离线优化。

DPO的隐式奖励函数定义为：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

其中 $\pi_\theta$ 为策略模型，$\pi_{ref}$ 为参考模型（通常为SFT后的模型），$\beta$ 为温度参数控制策略偏离参考模型的程度。基于Bradley-Terry模型，DPO将偏好学习转化为二分类问题，目标是最大化首选响应（win）相对于非首选响应（lose）的log-likelihood margin：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

该公式的优雅之处在于：无需显式训练奖励模型，偏好数据直接驱动策略更新。$\beta$参数的典型值为0.1，数据质量较低时可增大至0.3-0.5以加强正则化 [^74^]。

然而，DPO存在三个已知局限。其一，长度偏差：DPO倾向于增加chosen响应的likelihood，而chosen响应通常更长，导致模型倾向于生成冗长输出 [^99^]。其二，参考模型开销：训练期间需同时加载策略模型和参考模型，显存需求翻倍。其三，过拟合风险：在确定性偏好数据上（即win和lose之间的差距始终一致），DPO可能发生过拟合导致策略崩溃 [^326^]。

#### 2.2.2 DPO改进版：R-DPO、IPO与$\beta$-DPO

针对DPO的局限，2024年涌现出一系列改进方法。**R-DPO**（Regularized DPO，ACL 2024 Findings）通过在目标函数中添加长度归一化正则化项，显式解耦响应长度与质量的影响 [^99^]：

$$\mathcal{L}_{R\text{-}DPO} = -\mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \alpha(|y_w| - |y_l|) \right) \right]$$

其中$\alpha$为长度正则化系数，典型值0.05-1.0。实验表明，R-DPO在MT-Bench和AlpacaEval上能在不牺牲质量的前提下显著减少冗长生成的倾向 [^99^]。

**IPO**（Identity Preference Optimization，AISTATS 2024）从理论上统一了RLHF和DPO，提出$\Psi$PO通用框架，将DPO和PPO视为特例 [^326^]。IPO使用平方损失替代DPO的logistic损失，有效避免了确定性偏好数据上的过拟合问题：

$$\mathcal{L}_{IPO} = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta} \right)^2 \right]$$

$\beta$在IPO中的含义与DPO不同，典型值更小（0.01-0.1），因为平方损失对large margin的惩罚更强烈 [^326^]。

**$\beta$-DPO**则从另一个角度切入，发现最优$\beta$值随批次级别数据的信息量动态变化，提出根据数据质量自适应调整$\beta$的机制 [^187^]。其核心思想是：高信息量偏好对（模型预测与标签不一致）应使用较小$\beta$以允许更大更新，低信息量对则使用较大$\beta$以加强正则化。

#### 2.2.3 简化方法：SimPO与ORPO

SimPO（Simple Preference Optimization，NeurIPS 2024/ICLR 2025）完全移除了参考模型，使用长度归一化的平均log概率作为奖励函数 [^240^]：

$$r_{SimPO}(x,y) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$$

$$\mathcal{L}_{SimPO} = -\mathbb{E} \left[ \log \sigma\left( r_{SimPO}(x, y_w) - r_{SimPO}(x, y_l) - \gamma \right) \right]$$

其中$\gamma$为目标奖励间隔（target reward margin），典型值为$\gamma/\beta \in [0.3, 1.6]$。SimPO解决了DPO隐式奖励与生成阶段实际使用指标之间的错位问题——DPO的奖励是policy与reference的log-ratio，而实际生成使用average log-likelihood [^240^]。实验显示，Llama-3-8B-Instruct配合SimPO在AlpacaEval 2上达到40.2% LC Win Rate，相比DPO提升2-5个百分点，同时生成更短、更少冗余的输出 [^240^]。

ORPO（Odds Ratio Preference Optimization，EMNLP 2024）将SFT和偏好对齐合并为单阶段训练，使用odds ratio惩罚项 [^148^]：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} - \lambda \cdot \mathbb{E} \left[ \log \sigma\left( \beta \log \frac{\text{odds}_{\pi_\theta}(y_w|x)}{\text{odds}_{\pi_\theta}(y_l|x)} \right) \right]$$

其中$\text{odds}_{\pi}(y|x) = \pi(y|x) / (1 - \pi(y|x))$。ORPO的核心洞察是：SFT本身对于偏好对齐的成功收敛至关重要，只需对不喜欢的生成样式施加轻微惩罚即可 [^148^]。

#### 2.2.4 KTO：二元反馈信号即可匹配DPO性能

KTO（Kahneman-Tversky Optimization）基于前景理论，仅需二元信号（"好"/"坏"）即可进行对齐，无需成对偏好数据 [^149^]。KTO的效用函数直接最大化生成结果的价值：

$$\mathcal{L}_{KTO} = \mathbb{E}_{(x,y)\sim D} \left[ w(y) \cdot \left(1 - v_{KTO}(x,y) \right) \right]$$

其中$v_{KTO} = \sigma(r_{KTO}(x,y) - z_{ref})$对期望输出，$v_{KTO} = \sigma(z_{ref} - r_{KTO}(x,y))$对非期望输出，$r_{KTO}(x,y) = \beta \log(\pi_\theta(y|x)/\pi_{ref}(y|x))$，$z_{ref}$为参考点。KTO的数据效率极高——可利用成对偏好数据中的单个响应或仅有二元标签的数据，在1B到30B参数规模上匹配或超过DPO性能 [^149^]。

**表2-2 DPO及其变体算法对比**

| 方法 | 年份 | 核心创新 | 优势 | 局限 | 典型超参数 |
|------|------|----------|------|------|------------|
| DPO [^74^] | NeurIPS 2023 | 闭式奖励参数化，分类损失 | 简单稳定，无需奖励模型 | 长度偏差，参考模型开销，过拟合 | $\beta$=0.1，lr=1e-6，bs=32-512 |
| R-DPO [^99^] | ACL 2024 | 长度归一化正则化 | 减少冗长生成长度 | 引入额外超参数$\alpha$ | $\alpha$=0.05-1.0，$\beta$=0.01-0.1 |
| IPO [^326^] | AISTATS 2024 | $\Psi$PO框架，平方损失 | 避免确定性偏好上过拟合 | 部分任务收敛较慢 | $\beta$=0.01-0.1，lr=5e-7 |
| $\beta$-DPO [^187^] | 2024 | 动态温度参数校准 | 自适应数据质量 | 额外计算开销 | 动态$\beta$调整 |
| SimPO [^240^] | NeurIPS 2024 | 无参考模型，长度归一化奖励 | 更简单高效，与生成目标对齐 | $\gamma$超参数敏感 | $\beta$=2.0-10，$\gamma/\beta$=0.3-1.6 |
| ORPO [^148^] | EMNLP 2024 | SFT+PO单阶段，odds ratio | 无需参考模型和SFT预热 | 偏好信号较弱 | $\beta$=0.1，$\lambda$=1.0，lr=5e-7 |
| KTO [^149^] | 2024 | 前景理论，二元信号 | 数据获取最容易，处理不平衡数据 | 依赖前景理论假设 | $\beta$=0.1，lr=1e-5，bs=8 |

上表揭示了一条清晰的简化路径：从DPO需要参考模型和偏好对，到SimPO/ORPO去除参考模型，再到KTO仅需二元反馈。每一步简化都伴随着工程效率的提升，但在特定场景下也有性能折衷。研究表明，SimPO在AlpacaEval等对话评估上表现最佳，KTO在数据受限场景下最具优势，而IPO在高确定性偏好数据上最稳定 [^240^] [^149^] [^326^]。

### 2.3 GRPO：Critic-Free的革命

#### 2.3.1 GRPO核心机制：组内相对奖励归一化

GRPO由DeepSeek-AI在DeepSeekMath论文中提出 [^1^]，其核心创新在于完全消除了PPO中需要单独训练的Critic网络。GRPO针对每个问题$q$采样一组回答$\{o_i\}_{i=1}^G$，通过组内相对奖励计算优势函数：

$$\hat{A}_{i} = \frac{r_i - \text{mean}(\{r_i\}_{i=1}^G)}{\text{std}(\{r_i\}_{i=1}^G)}$$

其中$G$为组大小（group size），$r_i$为第$i$个回答的奖励（对于数学推理通常为答案正确性的二值信号）。该优势函数对组内所有回答的奖励进行标准化（减去均值、除以标准差），并将同一优势值赋给回答中每个token [^1^]。这一设计的直觉在于：对于同一问题，一组回答中表现相对较好的应被鼓励，表现相对较差的应被抑制——无需外部价值模型提供绝对评估。

GRPO的完整目标函数为：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}\hat{A}_i, \ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i \right) - \beta D_{KL}[\pi_\theta \| \pi_{ref}] \right]$$

其中$\rho_{i,t} = \pi_\theta(o_{i,t}|q, o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})$为token级重要性采样比率。与PPO的关键区别在于：GRPO将KL散度在loss函数中显式添加，而非PPO中在奖励函数中隐式处理 [^1^]。

#### 2.3.2 DeepSeekMath原始实验

GRPO的原始实验在DeepSeekMath-Instruct 7B模型上进行，使用144K条GSM8K和MATH的chain-of-thought问题数据，关键超参数包括：学习率$1\times10^{-6}$，KL系数$\beta=0.04$，组大小$G=64$，最大序列长度1024 tokens [^1^]。训练后的模型在多个数学推理基准上取得显著提升：

| Benchmark | Instruct基线 | GRPO训练后 | 绝对提升 |
|-----------|-------------|-----------|----------|
| GSM8K | 82.9% [^1^] | 88.2% [^1^] | +5.3pp |
| MATH | 46.8% [^1^] | 51.7% [^1^] | +4.9pp |
| CMATH | 84.6% [^1^] | 88.8% [^1^] | +4.2pp |

这些结果在当时代表了7B级别开源模型的数学推理新SOTA，更重要的是证明了无需Critic模型的强化学习在LLM训练中的可行性。

#### 2.3.3 GRPO vs PPO系统性对比

**表2-3 GRPO与PPO系统性对比**

| 对比维度 | PPO | GRPO | 影响 |
|----------|-----|------|------|
| Critic网络 | 需要单独训练 | 不需要（组内归一化替代） | GRPO内存减半 |
| 模型数量 | 4个（actor/critic/reference/reward） | 2个（policy/reference） | GRPO工程复杂度大幅降低 |
| 优势估计 | GAE + Value Model | 组内相对归一化 | GRPO无需价值函数学习 |
| 内存消耗 | ~2倍模型参数 | ~1倍模型参数 | GRPO可使用更大batch或更大模型 |
| KL散度 | 在奖励函数中隐式添加 | 在loss函数中显式添加 | GRPO KL控制更直接 |
| 在线采样 | 需要实时奖励模型评估 | 仅需要可验证奖励 | GRPO适合可验证任务（数学/代码） |
| 训练稳定性 | 超参数敏感，容易崩溃 | 更稳定，但存在熵崩塌风险 | 两者均需 careful tuning |
| 适用任务 | 通用偏好对齐 | 可验证推理任务 | 任务类型决定方法选择 |

GRPO相比PPO的核心优势在于资源效率：省去Critic模型意味着显存消耗减半，无需训练Value model意味着减少了超参数调优空间。然而，GRPO并非在所有场景下都优于PPO。GRPO依赖组内回答的方差来获取学习信号——当所有回答都正确或都错误时（即零方差组），梯度信号消失，这是DAPO等后续改进重点解决的问题 [^2^]。此外，PPO在需要精细偏好建模的开放域对话任务上仍具优势，因为这类任务难以获得简单的二值可验证奖励。

### 2.4 GRPO变体算法群

GRPO的成功催生了大量变体算法，形成了一个活跃的改进方向。这些变体围绕六个核心挑战展开：熵崩塌、零方差组、长度偏差、离群值敏感、MoE（Mixture of Experts，混合专家模型）兼容性和Value model需求。

#### 2.4.1 DAPO：四大改进推动训练效率提升50%

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）是当前GRPO最具影响力的变体之一 [^2^]，通过四项关键技术使Qwen2.5-32B在AIME 2024上达到50分，仅需GRPO约50%的训练步数。

**改进一：Dynamic Sampling（动态采样）**。过滤掉所有回答都正确或都错误的"零方差组"，确保每个batch中所有prompt都有有效梯度信号。约束条件为$0 < |\{o_i \mid \text{is\_equivalent}(a, o_i)\}| < G$ [^2^]。

**改进二：Clip-Higher（非对称裁剪）**。将裁剪上下界解耦为$\epsilon_{low}=0.20$和$\epsilon_{high}=0.28$，允许更大的正向策略更新幅度，有效缓解熵崩塌：

$$\text{clip}\left(\rho_{i,t}, \ 1-\epsilon_{low}, \ 1+\epsilon_{high}\right)$$

**改进三：Token-Level Policy Gradient Loss**。用总token数归一化损失，替代GRPO中按序列数归一化：

$$\mathcal{J}_{DAPO} = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^{G}|o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \min\left( \rho_{i,t}\hat{A}_{i}, \ \text{clip}(\rho_{i,t}, 1-\epsilon_{low}, 1+\epsilon_{high})\hat{A}_{i} \right) \right]$$

**改进四：Overlong Reward Shaping（过长奖励塑形）**。对超过预设长度阈值的回答施加soft惩罚，鼓励生成足够长的CoT（Chain-of-Thought，思维链）以容纳完整推理过程，但避免过度冗长 [^2^]。

![DAPO渐进式技术添加效果](fig_sec02_dapo_ablation.png)

*图2-2 DAPO渐进式技术添加效果。从Naive GRPO的30分出发，每一步改进都带来稳定提升：过长过滤+6分、非对称裁剪+2分、Soft过长惩罚+3分、Token级损失+1分、动态采样+8分，最终达到50分，超越DeepSeek-R1-Zero-Qwen-32B的47分 [^2^]。*

#### 2.4.2 Dr.GRPO：长度归一化修正

Dr.GRPO识别出GRPO中两个导致训练偏差的源头 [^3^]。**长度偏差**：GRPO对每个回答的损失除以$|o_i|$（回答长度），导致长回答中每个token的梯度被缩小，模型倾向于生成更长但不一定更好的回答。**标准差偏差**：除以组内标准差使得问题难度影响梯度大小，极端奖励分布的问题被过度加权。

Dr.GRPO的修正方案是直接移除$\frac{1}{|o_i|}$和$\text{std}(R)$两项归一化，优势函数简化为：

$$\hat{A}_i = r_i - \text{mean}(R)$$

使用7B模型在AIME 2024上达到43.3%，显著优于GRPO基线 [^3^]。Dr.GRPO的优势在于实现简单——仅需修改几行代码即可从GRPO升级，同时有效提升token效率。

#### 2.4.3 GMPO/GSPO：几何平均与序列级优化

**GMPO**（Geometric-Mean Policy Optimization）用几何平均替代GRPO中的算术平均来聚合token级奖励 [^4^]：

$$\mathcal{J}_{GMPO}(\theta) = \mathbb{E}\left[\left(\prod_{t=1}^{|o_i|} \rho_{i,t} \cdot \mathbf{1}_{[\text{unclipped}]}\right)^{\frac{1}{|o_i|}} \cdot \hat{A}_i \right]$$

等价于对数空间中的平均：$\log \mathcal{J}_{GMPO} \propto \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \log \rho_{i,t}$。几何平均对离群值不敏感（outlier robust），使重要性采样比率的范围更稳定，允许使用更宽的裁剪阈值而不失去稳定性 [^4^]。GMPO-7B在AIME 2024、AMC、MATH-500等多个数学基准上平均超越GRPO 4.1个百分点。

**GSPO**（Group Sequence Policy Optimization）则从另一维度改进——将优化粒度从token级提升到序列级 [^5^]。GSPO定义序列级重要性比率：

$$s_i(\theta) = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \right)$$

并对整个序列统一裁剪：$\mathcal{J}_{GSPO} = \mathbb{E} [ \min(s_i\hat{A}_i, \ \text{clip}(s_i, 1-\epsilon, 1+\epsilon)\hat{A}_i) ]$。GSPO是Qwen3模型RL训练的核心算法，在MoE模型上展现出显著优于GRPO的训练稳定性 [^5^]。

#### 2.4.4 其他变体：VAPO、SAPO、M-GRPO、Lambda-GRPO

**VAPO**（Value-based Augmented Proximal Policy Optimization）是唯一回归使用Value model的变体，通过Value Pretraining（用reward model初始化value model）避免了vanilla PPO中value model学习崩塌的问题 [^6^]。VAPO还引入解耦GAE、自适应GAE、正例LM损失和组采样等七项改进，在Qwen-32B上达到AIME 2024的60.4分新SOTA [^6^]。消融实验显示，仅正例LM损失一项就贡献6分提升，证明在RL训练中维持正样本的likelihood至关重要。

**SAPO**（Soft Adaptive Policy Optimization）用平滑、温度控制的门控函数替代硬裁剪，通过Sigmoid函数实现token级自适应衰减 [^7^]。其核心设计哲学是对正优势token允许更大更新幅度，对负优势token施加更严格约束，非对称温度设计进一步区分正负token的处理。

**M-GRPO**使用缓慢演变的momentum模型提供稳定训练目标，结合IQR自适应过滤方法动态剪除低熵轨迹，解决长程训练中的策略崩塌问题 [^10^]。**Lambda-GRPO**通过可学习token偏好参数统一GRPO框架，动态调整长度惩罚和偏好权重 [^11^]。

**表2-4 GRPO变体算法系统性对比**

| 方法 | 年份 | 核心创新 | 解决的核心问题 | 局限 | 典型超参数 |
|------|------|----------|--------------|------|------------|
| GRPO [^1^] | 2024 | 组内相对归一化 | 无需Critic | 熵崩塌，零方差，长度偏差 | $G$=64，$\beta$=0.04，lr=1e-6 |
| DAPO [^2^] | 2025 | 动态采样+Clip-Higher+Token级损失+过长惩罚 | 零方差，熵崩塌，训练效率 | 实现复杂度中等 | $\epsilon_{low}$=0.20，$\epsilon_{high}$=0.28，KL=0 |
| Dr.GRPO [^3^] | 2025 | 移除长度/std归一化 | 长度偏差，难度偏差 | 标准差信息丢失 | $G$=8-16，lr=1e-6 |
| GMPO [^4^] | 2025 | 几何平均替代算术平均 | 离群值敏感 | 实现需特殊处理 | $G$=8-16，lr=1e-6 |
| GSPO [^5^] | 2025 | 序列级重要性比率与裁剪 | MoE兼容，熵稳定 | 信息粒度较粗 | $G$=8-16，lr=1e-6 |
| VAPO [^6^] | 2025 | Value预训练+7项改进 | Value崩塌，长CoT | 需维护Critic，复杂度高 | Actor lr=1e-6，Critic lr=2e-6 |
| SAPO [^7^] | 2025 | 软门控替代硬裁剪 | 硬裁剪信号丢失 | 温度参数需调 | $\tau_{pos}$/$\tau_{neg}$非对称 |
| M-GRPO [^10^] | 2025 | Momentum锚定+IQR过滤 | 长程策略崩塌 | Momentum更新频率 | Momentum decay=0.99 |
| Lambda-GRPO [^11^] | 2025 | 可学习token偏好参数 | 统一框架 | 额外参数量 | 可学习$\lambda$参数 |

### 2.5 方法选择决策框架

#### 2.5.1 场景-方法匹配

基于上述分析，以下决策框架帮助研究者根据任务特性选择最合适的算法：

**表2-5 场景-方法匹配决策表**

| 应用场景 | 推荐方法 | 备选方法 | 不推荐方法 | 关键考量 |
|----------|----------|----------|------------|----------|
| 通用偏好对齐（对话/指令跟随） | SimPO [^240^] | DPO [^74^], KTO [^149^] | GRPO | SimPO无参考模型且与生成目标对齐 |
| 数学/科学推理训练 | DAPO [^2^] | GRPO [^1^], Dr.GRPO [^3^] | PPO | 需可验证奖励信号；DAPO效率最高 |
| 代码生成与验证 | GRPO/DAPO | PPO [^193^] | DPO/SimPO | PPO在复杂代码探索上仍有优势 |
| 多目标对齐（安全+有用） | MODPO [^105^] | MO-ODPO [^97^] | 单一目标方法 | 需Pareto最优trade-off |
| 数据极度受限（仅二元标签） | KTO [^149^] | - | DPO/SimPO | KTO仅需二元信号 |
| 长链推理（30K+ tokens） | VAPO [^6^] | DAPO [^2^] | 标准GRPO | VAPO的Value model对长CoT有帮助 |
| MoE模型训练 | GSPO [^5^] | SAPO [^7^] | GRPO | 序列级优化对MoE更稳定 |
| 快速原型验证/资源受限 | Dr.GRPO [^3^] | GRPO [^1^] | VAPO/PPO | Dr.GRPO实现最简单 |
| 端到端单阶段训练 | ORPO [^148^] | CPO [^85^] | DPO | ORPO合并SFT和偏好优化 |

该框架的核心理念是：**任务类型决定方法选择**。对于可验证的推理任务（数学、代码），GRPO系列方法因无需人工标注偏好数据且训练高效而成为首选；对于开放域的偏好对齐任务（对话质量、风格调整），DPO系列方法因更贴合人类主观偏好评估而更适合 [^193^] [^1^]。

#### 2.5.2 各方法Benchmark覆盖矩阵与性能边界

**表2-6 核心算法Benchmark覆盖矩阵**

| 方法 | AlpacaEval 2 | MT-Bench | GSM8K | MATH-500 | AIME 2024 | 代码生成 |
|------|-------------|----------|-------|----------|-----------|----------|
| PPO | 中等 | 中等 | - | - | - | **优** [^193^] |
| DPO | 良好 | 良好 | - | - | - | 差 |
| SimPO | **40.2%** [^240^] | 良好 | - | - | - | 差 |
| ORPO | 12.20% [^148^] | 7.32 | - | - | - | 差 |
| KTO | 匹配DPO [^149^] | 匹配DPO | - | - | - | 差 |
| GRPO | - | - | 88.2% [^1^] | 51.7% [^1^] | ~30 | 良好 |
| DAPO | - | - | - | - | **50** [^2^] | 良好 |
| Dr.GRPO | - | - | - | - | 43.3 [^3^] | 良好 |
| VAPO | - | - | - | - | **60.4** [^6^] | 良好 |
| GMPO | - | - | +4.1% [^4^] | +4.1% [^4^] | +4.1% [^4^] | 良好 |

上表中的数据揭示了算法性能与评估任务之间的结构性关联。偏好优化方法（DPO/SimPO/ORPO/KTO）在对话评估基准（AlpacaEval、MT-Bench）上表现突出，但在需要多步推理的数学基准上几乎无覆盖。相反，GRPO系列方法专注于数学推理基准，在GSM8K和AIME上取得显著进展。这种"任务-方法绑定"现象说明当前LLM+RL领域已形成两个相对独立的技术生态：偏好对齐生态以DPO系列为核心，推理训练生态以GRPO系列为核心 [^1^] [^74^]。

值得注意的是，VAPO在AIME 2024上达到的60.4分是目前公开报告中的最高分 [^6^]，但这一结果依赖于Value model的使用，增加了训练复杂度。对于大多数实际应用场景，DAPO以更低的实现复杂度（无需Value model）提供了极具竞争力的性能——50分在Qwen2.5-32B上超越DeepSeek-R1-Zero-Qwen-32B的47分 [^2^]，且训练步数仅为GRPO的50%。对于资源受限的研究者，Dr.GRPO仅需修改GRPO的几行归一化代码即可获得约43.3分的强基线结果 [^3^]，是性价比最高的入门选择。
