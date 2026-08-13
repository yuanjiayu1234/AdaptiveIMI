R1W1:
感谢您的评论。我们同意，诸如 KMeans 聚类、缓存、异步数据移动和 GPU 执行等技术本身并不新颖，我们也不声称它们是独立的贡献。我们的贡献在于算法系统协同设计，其核心是 RoPE 感知子空间划分。这种划分改变了倒排多索引的几何结构和粒度，因此需要一个相应的运行时流水线，包括层/头级索引元数据、预算感知集群选择、令牌/页级缓存规划、GPU 端检索计划生成、异步键值获取和缓存准入，以及执行缓冲区调度。这些机制围绕所提出的划分结构设计，并将离线学习的索引与在线检索注意力机制连接起来。我们将修改论文，以区分核心算法创新点和标准实现原语，并阐明我们的贡献在于它们在特定任务中的集成和端到端协同设计。

R1W2:
我们补充了RULER实验16k-96k，例如16k full/TopK/Magic/PQ/ParisKV/Retro/our 分数分别为91.31/91.82/81.68/85.67/87.84/87.78/89.26，更多长度见newResults/RULER.

R1W3:
1）论文图1表明需要更大的预算。实际测试中发现，5% 的预算精度不足。此外，RetroInfer 实际上采用了两部分 KV 缓存——检索区和 estimation 区，二者来源不同。因此，虽然检索区选取的 KV 较少，但最终参与 attention 计算的 KV 预算实际达到了 25%。
2）Full attention并非在所有长上下文任务中都是最优选择——当上下文过长时，强制关注每一个 token 反而可能引入无关或干扰性信息。以 Long/Easy 场景为例，正确答案往往仅依赖于上下文中一个相对较小的子集；此时，AdaptiveIMI 基于相关性的检索机制能够有效抑制无关 token，起到隐式降噪的作用，从而在该任务上取得更高的分数。需要强调的是，我们并不认为 AdaptiveIMI 在整体准确性上优于Full attention。总体而言，充分注意力仍然是一种强大且表现更为稳定的基线方法。上述例子仅说明：在特定任务场景下，选择性地聚焦上下文可能是有益的。
3）由于 Llama3、Qwen3 支持的最大上下文长度为 128K 个 token，对于超出该限制的样本，我们对所有方法统一采用中间截断策略，保留输入的开头和结尾部分。

R2W1:
我们的 learned partition 是针对每个 layer-head 独立优化的受约束向量量化结构，其目标是最小化 KV 向量的几何重构误差，并满足 cell-size 上限，而不是学习显式的语义类别。因此，不同层和不同 attention head 的分区可能反映各自的 RoPE 变换和 KV 分布差异，但不一定具有稳定、直观的语义可解释性。我们将在修订稿中明确这一点，并补充分区学习目标及其几何含义，同时将“可解释结构分析”列为当前工作的局限，而不对尚未验证的跨层或跨头模式作过度解读。

R2W2:
我们澄清，大多数主流的LLM模型，包括我们实验中使用的Llama-3.1和Qwen3模型，都采用了半分割（NeoX风格）的RoPE布局，其中第(i)个RoPE对的两个坐标为((i, i+d/2))，而不是相邻坐标，例如((2i,
2i+1))。因此，不可学习的IMI基线模型使用的原始序列划分会将RoPE对分割到不同的子空间中。我们的设计包含两个相互独立的目的：首先，将每个 RoPE 对的两个坐标放在同一子空间中，以保持其旋转结构；其次，在保持 RoPE 对完整性的约束下，通过离线优化学习更有效的 RoPE 对分组，从而进一步提升 IMI 的聚类质量。也就是说，保持 RoPE 对完整性是为了保护结构，而离线优化则是为了改善聚类，两者承担不同的作用。

R2W3:
<!-- W3. 仅测试了8B模型；未覆盖⻓⽣成和数学推理场景；缺少超⻓上下⽂检索基准测试 （如RULER、NIAH）。 -->

我们增加了32B模型和mistral模型，详情见newResults/models
增加AIME26（R2D4），增加RULER（R1W2）

R2D1:
我们的分阶段分析表明，AdaptiveIMI 在 128K 数据量下 TPOT 降低近 50% 并非主要归功于缓存命中率的提升，而是主要得益于更高效的 GPU 端检索计划生成，从而避免了 RetroInfer 的 CPU 端查找以及相关的主机-设备开销（平均每个解码步骤 CPU 查找耗时 9.09 毫秒，选择 D2H 耗时 11.57 毫秒），同时还降低了键值获取/收集成本，并实现了更高的稳态缓存命中率。详细的实验结果已在仓库的 newResults/R2D1 目录下提供。
Our per-stage breakdown indicates that AdaptiveIMI’s nearly 50% TPOT reduction at 128K is not primarily due to cache hit ratio alone, but mainly comes from more efficient GPU-side retrieval-plan generation, which avoids RetroInfer’s CPU-side lookup and associated Host-Device overhead (9.09 ms for CPU lookup and 11.57 ms for selection D2H per decode step on average), while also reducing KV fetch/gather cost and achieving a higher steady-state cache hit ratio. Detailed experimental results are provided in the repository under newResults/R2D1.

R2D2:
<!-- 不同检索预算下缺乏系统性对⽐**：该论⽂仅在10%的预算下对所有⽅法进⾏了对⽐。建议调整不同的检索预算并展示准确率变化趋势，因为部分基准⽅法在其原始论⽂中报告的预算低于10%。 加一个不同检索预算下缺乏系统性对⽐ -->

PQCache的预算为10%和20%，Retroinfer的实际attention预算大于10%，见R1W3，同时我们补充更多预算的分数见newResults/Budget.

R2D3:
<!-- 加一个70B模型？学习到的分区的跨模型可迁移性值得探讨 -->
为了验证跨模型泛化性，我们增加更多模型，见newResults/models。

R2D4:
<!-- 1）⻓⽣成场景下的索引增量更新未得到验证
2）也缺乏在数学推理或其他⻓⽣成任务上的准确性数据 -->
我们使用AIME26 pass@8（数学推理任务集同时能保证输出长度达到几千个token）和Full attention对比，结果没有下降，证明……，详细见newResults/AIME26

3）原因见R1W3第二条回复

R3W1:
我们的离线预处理按层进行：每层总耗时合计约12.9 min。校准数据规模、KV 向量数量、收敛行为、候选搜索数量及内存开销等详细结果见 newResults/R3W1。

R3W2:
<!-- 效率实验将输出⻓度固定为128个token，因此评估主要覆盖⻓输⼊、短⽣成的⼯作 负载。在⻓⽂本⽣成过程中，新的KV token会持续插⼊，需要重复进⾏集群分配、质⼼更新、缓冲token合并以及GPU缓存维护。这可能导致质⼦陈旧、集群碎⽚化、查询/KV分布漂移以及缓存频繁变动。检索错误也可能⾃回归式地累积，进⽽影响⽣成质量。输出数千个token的实验应报告每步召回率、TPOT、缓存命中率、索引更新开销以及⻓⽂ 
本⽣成质量。 -->

见R2D4的AIME26长输出数据集，我们给出Full attention准确度对比（没有下降）。以及tpot依然很低，这是因为对于长输出，索引是使用CPU后台更新，所以不会增加开销。

R3W3:
RetroInfer 已经代表 CPU–GPU KV offloading + retrieval 这一类方法的强竞争基线；但是谢谢您的意见，我们将补充更多方法在仓库，见newResults/methods

R3W4:
<!-- 本实验仅在配备 PCIe Gen4 接⼝和 64GB 主机内存的单台 A100-40GB 服务器上进⾏。由于 AdaptiveIMI 可能对 GPU 显存容量、PCIe 带宽以及 CPU 内存带宽较为敏感，因此在消费级 GPU、更⼩主机内存配置、H100/H200 等新型加速器以及 SSD 分层环境下开展评估，将进⼀步增强其实⽤性结论。此外，⼦空间数量 $m=8$、⽬标聚类规模$n_c=16$、检索⽐例 10% 以及热度⽐例 0.3 等关键参数均基本固定。需要开展更全⾯ 
的敏感性分析，以明确这些参数在不同模型、上下⽂⻓度和⼯作负载下对准确率、召回 率、TPOT 指标及内存占⽤的影响。 -->
为了验证……，我们的补充实验在4090（24G）上进行，具体硬件参数见……。
论文参数解释：部分可在论文中找到（Fig8、Fig11），聚类规模为16考虑到：，更详细的解释见newResults/parameter
