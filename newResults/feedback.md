R1W1: Thank you for the comment. We agree that techniques such as KMeans clustering, caching, asynchronous data movement, and GPU execution are not novel by themselves, and we do not claim them as independent contributions. Our contribution lies in the algorithm–system co-design centered on **RoPE-aware subspace partitioning**. This partition changes the geometry and granularity of the inverted multi-index, requiring a corresponding runtime pipeline, including layer/head-level index metadata, budget-aware cluster selection, token/page-level cache planning, GPU-side retrieval planning, asynchronous KV fetching and cache admission, and execution-buffer scheduling. These mechanisms are designed around the proposed partitioning structure, connecting the offline-learned index with online retrieval attention. We will revise the paper to distinguish the core algorithmic innovation from standard implementation primitives, and clarify that our contribution lies in their integration and end-to-end co-design for the target workload.


R1W2: RULER experiments at 16K–96K sequence lengths were added. For Llama-3-8B at 96K, the scores of Full, Magic, PQ, ParisKV, Retro, and our method are **80.32, 76.35, 65.97, 69.75, 77.46, and 79.65**, respectively.


R1W3:
1）Fig. 1 shows larger budgets are needed; 5% is insufficient. RetroInfer’s KV comes from retrieval and estimation, totaling 25% in the paper.
2）Full attention is not always optimal for long-context tasks; attending to all tokens may introduce noise. In Long/Easy, AdaptiveIMI focuses on relevant tokens, suppressing irrelevant ones and improving the score. We do not claim overall superiority over full attention, which remains a strong and stable baseline.
3）Since Llama3 and Qwen3 support up to 128K tokens, we apply middle truncation to samples exceeding this limit for all methods.

R2W1: AdaptiveIMI’s learned partition is not intended to learn explicit semantic categories. Instead, for each layer-head, it optimizes the vector quantization structure of the KV representations under the cell-size constraint, so that the partition better matches the layer-head-specific KV geometry and the representation structure induced by RoPE. Therefore, its key significance is not to discover a unified or intuitively interpretable semantic pattern across different layers/heads, but to adaptively construct a search space that is well matched to the KV geometry of each layer-head.


R2W2: We clarify that most mainstream LLMs, including Llama and Qwen, use half-split (NeoX-style) RoPE, where the (i)-th RoPE pair is ((i,i+d/2)), rather than adjacent coordinates ((2i,2i+1)). Thus, the sequential partition used by the non-learned IMI baseline splits RoPE pairs across subspaces. Our design has two distinct goals: (1) keeping each RoPE pair in the same subspace to preserve its rotational structure; and (2) learning more effective RoPE-pair groupings offline, subject to this constraint, to improve IMI clustering. In short, RoPE-pair integrity preserves structure, while offline optimization improves clustering.


R2W3:
For scalability to more models, experimental results are shown in R2D3.
For long generation and mathematical reasoning scenarios, see R2D4. 
RULER results are shown in R1W2.


R2D1: Our per-stage breakdown indicates that AdaptiveIMI’s nearly 50% TPOT reduction at 128K is not primarily due to cache hit ratio alone, but mainly comes from more efficient GPU-side retrieval-plan generation, which avoids RetroInfer’s CPU-side lookup and associated Host-Device overhead (9.09 ms for CPU lookup and 11.57 ms for selection D2H per decode step on average), while also reducing KV fetch/gather cost and achieving a higher steady-state cache hit ratio. 

R2D2:
For PQCache, we use 10% and 20% budgets, while RetroInfer uses an effective attention budget greater than 10% (see R1W3). We also provide additional results under smaller budgets, showing that overly small budgets can lead to accuracy degradation:
* **Qasper Full:** 44.82
* **2% budget:** MagicPIG/PQCache/ParisKV/RetroInfer/Our = 42.68/41.56/42.03/42.11/42.15
* **5% budget:** 43.60/42.78/43.30/42.74/43.77
* **10% budget:** 44.44/43.32/43.04/45.16/45.30

R2D3:


R2D4:
<!-- 1）⻓⽣成场景下的索引增量更新未得到验证
2）也缺乏在数学推理或其他⻓⽣成任务上的准确性数据 -->
We further evaluate AdaptiveIMI on **AIME26 pass@8**, a mathematical reasoning benchmark with outputs of several thousand tokens. Compared with Full Attention, **AdaptiveIMI shows no accuracy degradation**, validating the effectiveness of our incremental index updates for long-generation scenarios. 

3）原因见R1W3第二条回复

R3W1: Our offline preprocessing is performed layer by layer. We use 11 calibration samples of approximately 128K tokens each, retaining 131,072 K vectors per layer/head, and process the layers sequentially. The complete preprocessing takes **12.9 min/layer** on average. The full-dimensional KMeans teacher takes only **0.35 s/head**, while RoPE-pair grouping search dominates the cost at **92.3 s/head** over approximately **12.5K candidates**. Notably, the search typically converges early: after about **29.6 s/head**, the objective is already within **1%** of the final result, suggesting that early stopping could further reduce offline preprocessing cost.
The peak memory usage is **5.39 GiB RAM / 10.36 GiB GPU memory** during KV collection, and **1.26 GiB RAM / 6.47 GiB GPU memory** during grouping search.




R3W2:
<!-- 效率实验将输出⻓度固定为128个token，因此评估主要覆盖⻓输⼊、短⽣成的⼯作 负载。在⻓⽂本⽣成过程中，新的KV token会持续插⼊，需要重复进⾏集群分配、质⼼更新、缓冲token合并以及GPU缓存维护。这可能导致质⼦陈旧、集群碎⽚化、查询/KV分布漂移以及缓存频繁变动。检索错误也可能⾃回归式地累积，进⽽影响⽣成质量。输出数千个token的实验应报告每步召回率、TPOT、缓存命中率、索引更新开销以及⻓⽂ 
本⽣成质量。 -->

See the long-output AIME26 results in R2D4. We compare Full Attention accuracy, with no degradation. TPOT also remains low, as the index is updated in the background on the CPU, adding no runtime overhead.


R3W3: RetroInfer already represents a strong baseline for CPU–GPU KV offloading + retrieval methods. Nevertheless, we thank the reviewer and will add more methods to the repository; see `newResults/methods`.


R3W4:
<!-- 本实验仅在配备 PCIe Gen4 接⼝和 64GB 主机内存的单台 A100-40GB 服务器上进⾏。由于 AdaptiveIMI 可能对 GPU 显存容量、PCIe 带宽以及 CPU 内存带宽较为敏感，因此在消费级 GPU、更⼩主机内存配置、H100/H200 等新型加速器以及 SSD 分层环境下开展评估，将进⼀步增强其实⽤性结论。此外，⼦空间数量 $m=8$、⽬标聚类规模$n_c=16$、检索⽐例 10% 以及热度⽐例 0.3 等关键参数均基本固定。需要开展更全⾯ 
的敏感性分析，以明确这些参数在不同模型、上下⽂⻓度和⼯作负载下对准确率、召回 率、TPOT 指标及内存占⽤的影响。 -->
To further validate the practicality across hardware configurations, we conducted additional experiments on an **RTX 4090 (24GB)**; detailed hardware specifications are provided in `newResults/hardware`. Regarding parameter settings, sensitivity analyses for key parameters are already included in **Fig. 8 and Fig. 11**. In particular, we set the target cluster size to **16** to balance retrieval granularity, index overhead, and memory efficiency. We further provide detailed parameter analyses covering different settings in `newResults/parameter`.

