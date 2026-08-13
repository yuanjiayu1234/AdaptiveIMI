R1W1: Thank you for the comment. We agree that techniques such as KMeans clustering, caching, asynchronous data movement, and GPU execution are not novel by themselves, and we do not claim them as independent contributions. Our contribution lies in the algorithm–system co-design centered on **RoPE-aware subspace partitioning**. This partition changes the geometry and granularity of the inverted multi-index, requiring a corresponding runtime pipeline, including layer/head-level index metadata, budget-aware cluster selection, token/page-level cache planning, GPU-side retrieval planning, asynchronous KV fetching and cache admission, and execution-buffer scheduling. These mechanisms are designed around the proposed partitioning structure, connecting the offline-learned index with online retrieval attention. We will revise the paper to distinguish the core algorithmic innovation from standard implementation primitives, and clarify that our contribution lies in their integration and end-to-end co-design for the target workload.


R1W2: We added RULER results at 16K–96K; at 16K, Full/TopK/Magic/PQ/ParisKV/Retro/ours score 91.31/91.82/81.68/85.67/87.84/87.78/89.26. More results are in `newResults/RULER`.

R1W3:
1）Fig. 1 shows larger budgets are needed; 5% is insufficient. RetroInfer’s KV comes from retrieval and estimation, totaling 25% in the paper.
2）Full attention is not always optimal for long-context tasks; attending to all tokens may introduce noise. In Long/Easy, AdaptiveIMI focuses on relevant tokens, suppressing irrelevant ones and improving the score. We do not claim overall superiority over full attention, which remains a strong and stable baseline.
3）Since Llama3 and Qwen3 support up to 128K tokens, we apply middle truncation to samples exceeding this limit for all methods.

R2W1: Our learned partition is independently optimized for each layer-head as a constrained vector quantization structure, minimizing KV reconstruction error under a cell-size constraint, rather than learning explicit semantic classes. Thus, partitions may reflect layer/head-specific RoPE transformations and KV distributions, but need not have stable semantic interpretability. We will clarify this, explain its objective and geometric meaning, and note interpretable structure analysis as a limitation without overinterpreting unverified cross-layer/head patterns.


R2W2: We clarify that most mainstream LLMs, including Llama and Qwen, use half-split (NeoX-style) RoPE, where the (i)-th RoPE pair is ((i,i+d/2)), rather than adjacent coordinates ((2i,2i+1)). Thus, the sequential partition used by the non-learned IMI baseline splits RoPE pairs across subspaces. Our design has two distinct goals: (1) keeping each RoPE pair in the same subspace to preserve its rotational structure; and (2) learning more effective RoPE-pair groupings offline, subject to this constraint, to improve IMI clustering. In short, RoPE-pair integrity preserves structure, while offline optimization improves clustering.


R2W3&R2D3: To further validate cross-model generalization, we added experiments on 32B and Mistral models. As shown in `newResults/models`, AdaptiveIMI consistently preserves near-Full-Attention accuracy with only a 10% retrieval budget across these additional models, demonstrating that the proposed approach generalizes well across different model scales and architectures.


增加AIME26（R2D4），增加RULER（R1W2）

R2D1: Our per-stage breakdown indicates that AdaptiveIMI’s nearly 50% TPOT reduction at 128K is not primarily due to cache hit ratio alone, but mainly comes from more efficient GPU-side retrieval-plan generation, which avoids RetroInfer’s CPU-side lookup and associated Host-Device overhead (9.09 ms for CPU lookup and 11.57 ms for selection D2H per decode step on average), while also reducing KV fetch/gather cost and achieving a higher steady-state cache hit ratio. Detailed experimental results are provided in the repository under newResults/R2D1.

R2D2:
<!-- 不同检索预算下缺乏系统性对⽐**：该论⽂仅在10%的预算下对所有⽅法进⾏了对⽐。建议调整不同的检索预算并展示准确率变化趋势，因为部分基准⽅法在其原始论⽂中报告的预算低于10%。 加一个不同检索预算下缺乏系统性对⽐ -->
In the paper, PQCache uses 10% and 20% budgets, while RetroInfer’s effective attention budget exceeds 10% (see R1W3). We also provide results under larger budgets in `newResults/Budget`.



R2D4:
<!-- 1）⻓⽣成场景下的索引增量更新未得到验证
2）也缺乏在数学推理或其他⻓⽣成任务上的准确性数据 -->
We further evaluate AdaptiveIMI on **AIME26 pass@8**, a mathematical reasoning benchmark with outputs of several thousand tokens. Compared with Full Attention, **AdaptiveIMI shows no accuracy degradation**, validating the effectiveness of our incremental index updates for long-generation scenarios. Details are provided in `newResults/AIME26`.


3）原因见R1W3第二条回复

R3W1: Our offline preprocessing is performed layer-wise, with a total time of ~12.9 min per layer. Details on calibration data size, KV vector count, convergence, candidate search, and memory overhead are provided in `newResults/R3W1`.



R3W2:
<!-- 效率实验将输出⻓度固定为128个token，因此评估主要覆盖⻓输⼊、短⽣成的⼯作 负载。在⻓⽂本⽣成过程中，新的KV token会持续插⼊，需要重复进⾏集群分配、质⼼更新、缓冲token合并以及GPU缓存维护。这可能导致质⼦陈旧、集群碎⽚化、查询/KV分布漂移以及缓存频繁变动。检索错误也可能⾃回归式地累积，进⽽影响⽣成质量。输出数千个token的实验应报告每步召回率、TPOT、缓存命中率、索引更新开销以及⻓⽂ 
本⽣成质量。 -->

See the long-output AIME26 results in R2D4. We compare Full Attention accuracy, with no degradation. TPOT also remains low, as the index is updated in the background on the CPU, adding no runtime overhead.


R3W3: RetroInfer already represents a strong baseline for CPU–GPU KV offloading + retrieval methods. Nevertheless, we thank the reviewer and will add more methods to the repository; see `newResults/methods`.


R3W4:
<!-- 本实验仅在配备 PCIe Gen4 接⼝和 64GB 主机内存的单台 A100-40GB 服务器上进⾏。由于 AdaptiveIMI 可能对 GPU 显存容量、PCIe 带宽以及 CPU 内存带宽较为敏感，因此在消费级 GPU、更⼩主机内存配置、H100/H200 等新型加速器以及 SSD 分层环境下开展评估，将进⼀步增强其实⽤性结论。此外，⼦空间数量 $m=8$、⽬标聚类规模$n_c=16$、检索⽐例 10% 以及热度⽐例 0.3 等关键参数均基本固定。需要开展更全⾯ 
的敏感性分析，以明确这些参数在不同模型、上下⽂⻓度和⼯作负载下对准确率、召回 率、TPOT 指标及内存占⽤的影响。 -->
To further validate the practicality across hardware configurations, we conducted additional experiments on an **RTX 4090 (24GB)**; detailed hardware specifications are provided in `newResults/hardware`. Regarding parameter settings, sensitivity analyses for key parameters are already included in **Fig. 8 and Fig. 11**. In particular, we set the target cluster size to **16** to balance retrieval granularity, index overhead, and memory efficiency. We further provide detailed parameter analyses covering different settings in `newResults/parameter`.

