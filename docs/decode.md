# AdaptiveIMI Decode 路径分析与后续优化计划

本文档记录当前 AdaptiveIMI 在 long-context decode 阶段的真实数据流、已经完成的 GPU cache-aware retrieval 改造、当前 `/root/test_gpu_cache_aware_v2_llama.log` 暴露出的瓶颈，以及下一步如何把 `other` 从临界路径中消掉。

目标读者是后续继续修改 decode/retrieval/cache promotion 的开发者。请先读完本文，再改 `cache_hub/adaptive_imi/retrieval.py` 和 `library/AdaptiveIMI/cpp_extensions/cuda/gather_copy.cu`。

## 目标

我们只优化 `flat` decode 路径。不要为了降低时间而修改用户未允许的压缩率、retrieval budget、cache ratio、上下文长度或模型参数。

理想 decode 临界路径应该只剩：

```text
flat_search
  -> GPU plan / descriptor build
  -> gather current-token KV into execution_buffer
  -> sparse attention
  -> dense model residual work
```

cache promotion/update 是给未来 token 提高 cache hit 率的工作，不应该完整阻塞当前 token 的 attention，也不应该在下一个同 layer 前形成大的 wait。如果无法完全后台化，至少要有上限和调度策略，避免 `other` 随 miss blocks 线性增长。

当前理想时间估计：

```text
flat_search        ~8 ms
GPU plan           ~3-4 ms
concat/gather      ~6-7 ms
sparse attention   ~3 ms
non_sparse_model   ~19 ms
--------------------------------
理想 full TPOT     ~40-45 ms
```

因此，当前 `other` 不是不可避免的模型计算，它主要是我们 cache/promotion 路径引入的额外等待，应该继续消除。

## 当前实现入口

主要代码位置：

- `cache_hub/adaptive_imi/retrieval.py`
  - `_run_retrieval_full_gpu_cache_aware()`
  - `_schedule_gpu_cache_promotion()`
- `library/AdaptiveIMI/cpp_extensions/cuda/gather_copy.cu`
  - `build_cache_aware_retrieval_plan_kernel`
  - `promote_hot_cold_cache_pages_from_execution`
  - `gather_copy_static_chunks_profile`
  - `gather_copy_miss_chunks_profile`
  - `gather_copy_hit_chunks_profile`
- `library/AdaptiveIMI/cpp_extensions/__init__.py`
  - Python extension export

当前正式路径就是 GPU cache-aware flat retrieval。它不再需要通过环境变量打开；`raw_sum`
flat search 是默认路径，`async_update` 保留给生成新 token 的增量更新。

当前 token 不再走：

```text
selected.to(cpu)
adpimi_index.batch_access()
CPU hit/miss lookup
CPU descriptor generation
adpimi_index.sync()
gather_copy_and_scatter() current path
```

而是走 GPU cache-aware path。

## 当前 Decode 数据流

当前正式路径如下：

```text
1. flat_search
   query_grouped + centroids + cluster_sizes
   -> selected_clusters [GPU]

2. GPU cache-aware plan
   selected_clusters [GPU]
   cluster_offsets / cluster_sizes [GPU]
   GPU cache hash table [GPU]
   -> miss descriptors [GPU]
   -> hit descriptors [GPU]
   -> valid_lengths [GPU]

3. gather current-token KV
   steady_zone KV [GPU]
   list KV [CPU pinned 或 GPU]
   GPU cache KV [GPU]
   descriptors [GPU]
   -> execution_buffer [GPU]

4. sparse attention
   execution_buffer + valid_lengths
   -> sparse attention output

5. cache promotion for future tokens
   miss descriptors + execution_buffer
   -> GPU cache KV + GPU hash table
   runs on promotion CUDA stream
```

注意第 5 步不是当前 attention 的必需输入。当前 token 的 attention 只依赖第 3 步写好的 `execution_buffer`。

## 已完成的关键改造

### 1. GPU plan kernel 并行化

当前 `build_cache_aware_retrieval_plan_kernel` 已经从早期的一组一个线程串行扫描，改成：

```text
one CUDA block / group
256 threads / group
threads parallel scan selected clusters/pages
```

plan kernel 输出：

```text
miss_offsets
miss_sizes
miss_cumsum
miss_counts
miss_page_keys

hit_offsets
hit_sizes
hit_cumsum
hit_counts

valid_lengths
```

重要正确性点：

- `miss_cumsum/hit_cumsum` 必须和 descriptor 顺序一致。
- 并行实现不能分别对 `count` 和 `tokens` 做独立 atomic 后乱序写 cumsum。
- 当前实现用 packed 64-bit atomic 同时预留 `block_count + token_count`，保证 cumsum 单调一致。

这点不能删。否则 `gather_copy_and_concat` 会用错误的最后 descriptor 推导 miss/hit 总长度，导致 execution buffer 拼接错位。

### 2. Promotion 当前读取 execution_buffer

当前实现会在 attention 后从 `execution_buffer` promotion 到 cache：

```text
promote_hot_cold_cache_pages_from_execution()
```

这避免了第二次从 CPU pinned `list_keys/list_values` 读取相同 miss pages，但也让 `execution_buffer`
生命周期和 promotion stream 绑定。当前代码用 execution-buffer ring event 避免覆盖仍在被
promotion 读取的 slot；如果 ring slot 都忙，会退回到 fallback buffer 并跳过本轮 cache admission。

需要继续关注的问题：

- retrieval 会读取同一套 GPU cache hash/queue metadata。
- promotion 在后台 stream 更新这些 metadata。
- 目前还没有 committed metadata view 或版本隔离。

### 3. GPU cache hit 已经生效

当前 logs 里：

```text
hit_blk 约 290k
miss_blk 约 18k
hit ratio 约 94%
concat_copy 去首 token 后约 6.5 ms
```

这说明 GPU cache-aware gather 确实在读 cache hit，不是 list-only。

## 当前测试结果

测试日志：

```text
/root/test_gpu_cache_aware_v2_llama.log
```

配置：

```text
model: Llama-3.1-8B-Instruct
length: 98304
batch: 1
max_new_tokens: 64
retrieval_budget: 0.1
flat mode
```

正式 decode summary：

```text
Prefilling latency: 24.1242 s
Decoding latency: 4.5069 s
TPOT: 71.54 ms
Throughput: 13.98 tok/s
```

decode detail：

```text
total             71.54 ms
non_sparse_model  19.49 ms
sparse_total      52.05 ms
flat_search        8.38 ms
retrieval         11.35 ms
flash_attn         3.27 ms
sparse_other      29.05 ms
```

去掉正式 decode 首 token 后：

```text
total mean        50.23 ms  (sparse 部分统计口径)
search mean        8.31 ms
gather mean       10.82 ms
attn mean          3.26 ms
other mean        27.84 ms
```

step >= 7 后：

```text
total mean        49.12 ms  (sparse 部分统计口径)
search mean        8.32 ms
gather mean       10.64 ms
attn mean          3.27 ms
other mean        26.88 ms
```

gather detail 去掉首 token 后：

```text
selection_d2h      0.88 ms
GPU plan           3.43 ms   # 日志字段仍叫 cpu_lookup
concat_copy        6.51 ms
cache_sync         0.00 ms
scatter            0.00 ms
```

结论：

- GPU plan 目标已经达到：约 `3.4 ms`。
- current-token gather 已经降到约 `6-7 ms`。
- CPU lookup spike 已经消失。
- 当前最大的额外开销是 `sparse_other`，约 `27-29 ms`。

## `other` 是什么

在 `retrieval.py` 当前 profile 里：

```text
other_ms = total_ms - search_ms - gather_ms - attn_ms
```

它不是一个独立 kernel 名，而是所有没有归入 search/gather/attention 的等待和执行时间。

在当前 GPU cache-aware path 下，`other` 主要来自：

```text
下一次同 layer 开始前等待上一轮 promotion event
promotion stream 与当前 stream/copy engine/GPU memory bandwidth 竞争
Python/profile 同步残留
少量 model glue / reshape / dispatch overhead
```

最关键的是第一项。

当前 `_run_retrieval_full_gpu_cache_aware()` 开头有：

```python
prior_event = self.gpu_cache_promote_events[layer_idx]
if prior_event is not None:
    torch.cuda.current_stream(device).wait_event(prior_event)
```

这个 wait 是必要的：如果当前 plan 要查 GPU cache hash table，就不能和上一轮还在修改 hash/cache 的 promotion 并发，否则会读到不一致的 cache metadata。

但这也说明：虽然我们把 promotion 从当前 attention 后挪到了独立 stream，它仍然可能在下一次同 layer 前被等回来。日志中的 `other` 就是这个等待的主要体现。

## 为什么确认 `other` 来自 promotion

从 `/root/test_gpu_cache_aware_v2_llama.log` 统计：

```text
miss_blk vs other correlation: 0.835
miss_blk vs total correlation: 0.816
miss_blk vs gather correlation: 0.761
```

miss/update blocks 多的 step，`other` 明显变高：

```text
step 2   miss=78640  other=48.46 ms
step 3   miss=51591  other=44.56 ms
step 4   miss=47385  other=42.73 ms
step 42  miss=30947  other=40.97 ms
step 29  miss=30733  other=39.89 ms
step 55  miss=24007  other=39.59 ms
```

而 plan 和 concat 在这些 step 没有同等幅度增长：

```text
GPU plan:    usually 3.1-3.8 ms
concat_copy: usually 5.5-8.9 ms, first token exception
```

所以当前瓶颈不是 flat search，也不是 GPU plan，而是 miss promotion/update policy。

## 当前方案的安全边界

当前实现为了 correctness 使用单一 per-layer cache state：

```text
cache_keys/cache_values
gpu_cache_hash_keys/gpu_cache_hash_values
gpu_cache_slot_keys
gpu_cache_next_slots
gpu_cache_promote_events[layer]
```

安全规则：

1. plan 查 hash table 之前必须看到上一轮 promotion 的一致状态。
2. promotion 写 cache slot 和 hash table 必须按同一个事件发布。
3. attention 只读 execution_buffer，不直接读 cache metadata。
4. promotion 不允许读 execution_buffer。
5. 如果 future token 使用 cache hit，就必须避免 hash table 写读竞态。

这就是为什么当前代码还需要 `wait_event(prior_event)`。要消除 `other`，不能简单删掉 wait；必须改变 cache promotion 的发布策略或双缓冲/版本化策略。

## 后续优化目标：完全消除 `other`

“完全消除 other”在工程上应理解为：

```text
当前 token 临界路径中不再包含大块 cache promotion wait
next same layer 不因为上一轮全部 miss promotion 未完成而阻塞
profile 中 sparse_other 接近 dispatch/glue 下限
```

理想上：

```text
sparse_other <= 3-5 ms
```

而不是当前 `27-29 ms`。

下面按优先级列出后续方案。

## P0: 给 promotion 加硬上限

当前问题是每个 step 会尝试把所有 miss pages 都 promote：

```text
miss_blk 平均约 18k
spike 可到 78k
```

这会导致下一次同 layer 等很久。最直接的优化是：

```text
每个 layer / step 最多 promote N blocks
超过上限的 miss 只用于当前 attention，不进入 cache
```

建议先加环境变量：

```bash
IMI_GPU_PROMOTE_MAX_BLOCKS_PER_GROUP
IMI_GPU_PROMOTE_MAX_BLOCKS_PER_LAYER
IMI_GPU_PROMOTE_MAX_BLOCKS_PER_STEP
```

实现方式：

- plan kernel 仍然生成完整 miss descriptors 给当前 gather。
- 另生成 promotion descriptors，但只保留前 N 个或高价值 N 个。
- `promote_cache_pages_from_list` 只消费 promotion descriptors，不消费完整 miss descriptors。

注意：current attention 的 recall 不变，因为 gather 仍读取全部 selected miss/hit。变化只影响未来 cache hit 率。

预期：

- `other` 立刻下降。
- `concat_copy` 可能小幅上升，因为 cache hit 率下降。
- 需要找到最优点，让 `other` 降得多于 `concat_copy` 上升。

这是最稳的第一步。

## P1: promotion descriptor 与 current miss descriptor 分离

当前 miss descriptors 同时服务两件事：

```text
1. current gather 从 list 读 miss pages
2. future promotion 把 miss pages 写 cache
```

这导致 promotion 默认继承 current miss 的完整规模。应该拆成：

```text
miss descriptors for current attention
promotion descriptors for future cache
```

新增 buffers：

```text
promote_offsets
promote_sizes
promote_counts
promote_page_keys
```

plan kernel 输出完整 miss/hit，同时根据策略输出 promote subset。

策略可以从简单到复杂：

1. first-N miss pages
2. 按 selected cluster rank 取前 N
3. 按 cluster size/page count 取前 N
4. 按历史热度/hit frequency 取前 N
5. 只 promote 当前 step 中高 rank 且未在 cache 的 pages

先做 1 或 2，验证 `other` 是否下降。

## P2: promotion 双缓冲 / versioned cache publish

当前 `wait_event(prior_event)` 的根因是 plan 和 promotion 共享同一份 hash table/cache slot metadata。

可以改成 versioned publish：

```text
active hash/cache metadata: 当前 plan 只读
building hash/cache metadata: promotion 写
publish event 完成后 swap/version++
```

关键设计：

- 当前 token 如果上一轮 promotion 没完成，可以继续使用旧版本 cache metadata。
- 不等待未完成 promotion。
- promotion 完成后，未来 token 才看到新版本。

数据结构：

```text
gpu_cache_hash_keys[2]
gpu_cache_hash_values[2]
gpu_cache_slot_keys[2]
cache_next_slots[2] 或 versioned next slot
active_version[layer]
building_version[layer]
publish_event[layer]
```

注意 cache KV storage 本身也要处理版本一致性：

- 简单版本：cache KV slot 不双缓冲，但 hash table 只有在 slot 写完后才 publish。
- 如果旧版本 hash 仍指向某 slot，promotion 不能覆盖该 slot，否则旧版本 plan 会读到新数据。
- 因此只双缓冲 hash 不够，还需要 slot ownership/version 或 cache slot 双缓冲。

更安全的版本：

```text
cache KV slots 也分 active/building pool
promotion 写 building pool
publish 后 active_version 切换
```

缺点是显存翻倍。可以先只给 hot cache 小池做双缓冲，而不是整个 cache。

## P3: promotion ring buffer，不覆盖当前 active slots

如果不想 cache KV 双缓冲，可以用 append-only ring + delayed reclamation：

```text
promotion 只写当前没有被 active hash 引用的 free slots
写满则停止 promotion，不 eviction
下一轮安全点再回收旧 slots
```

这能避免 plan 等 promotion，也避免旧 hash 指向的 slot 被覆盖。

要求：

- GPU 上维护 free list 或 monotonically increasing slot cursor。
- active hash publish 后，旧版本引用计数归零才能回收。
- cache 容量不足时宁可少 promote，不要阻塞当前 token。

这个方案比完整双缓冲省显存，但实现复杂。

## P4: 融合 current gather 与 cache fill，避免重复读 list miss

当前 miss page 可能被读两次：

```text
list -> execution_buffer   # 当前 attention 必需
list -> cache              # future promotion
```

如果 list 在 CPU pinned memory，这就是两次 H2D/PCIe 读；如果 list 在 GPU，也是两次 global memory copy。

更理想的 gather kernel：

```text
for selected miss page:
    load list page once
    write execution_buffer
    if selected for promotion:
        also write cache slot
        record publish metadata
```

这样 promotion 不再是独立的大 copy kernel，至少数据搬运不重复。

安全点：

- cache metadata publish 仍必须在 cache slot 写完后。
- 如果不想等 publish，使用 P2/P3 的 versioned/delayed publish。
- fused kernel 不应增加 current gather 太多负担；只对 bounded promotion subset 做 cache fill。

这一步适合在 P0/P1 验证 promotion 上限有效后再做。

## P5: 不要每 token 都 promotion

当前每 decode step 都可能 promotion。长上下文下 selected clusters 有局部性，不一定需要每 token 更新 cache。

可选策略：

```text
IMI_GPU_PROMOTE_INTERVAL=K
```

例如：

- 每 2 或 4 个 token promotion 一次。
- 非 promotion step 只 gather，不更新 cache。
- promotion step 使用 bounded promotion descriptors。

预期：

- `other` 大幅下降。
- cache hit 率可能下降，但因为 query 相邻 token 通常相似，下降可能可控。

必须用日志验证：

```text
hit_blk
miss_blk
concat_copy
other
TPOT
```

## P6: 对 miss spike 做 backpressure

从日志看，`other` spike 来自 miss spike。即使平均 miss 可接受，单步 30k-80k miss blocks 会拉高 p95。

应该在 plan 后统计：

```text
miss_blocks_per_step
miss_blocks_per_layer
```

并做：

```text
if miss_blocks > threshold:
    reduce promotion quota for this step
    or skip promotion for low-rank clusters
```

这不会影响当前 attention，因为当前 gather 仍完整读取 selected miss。

## P7: 更细的 profile

现在 `other` 是剩余项，不够细。下一步应该继续拆：

```text
prior_promotion_wait
promotion_submit
post_attention_glue
weighted_flash_dispatch_gap
model_layer_residual
```

最重要的是单独记录：

```text
prior_promotion_wait_ms
promotion_blocks
promotion_bytes
promotion_kernel_ms
promotion_h2d_bytes
```

当前日志里 `cache_sync=0, scatter=0` 是对的，因为这些已经不在 current gather 内；但 `other` 仍需拆开，否则后续很容易误判。

## 推荐下一步实现顺序

### Step 1: 加 promotion quota

新增环境变量：

```bash
IMI_GPU_PROMOTE_MAX_BLOCKS_PER_GROUP=...
IMI_GPU_PROMOTE_MAX_BLOCKS_PER_LAYER=...
```

plan kernel 输出完整 current miss/hit，同时输出 bounded promotion descriptors。

成功标准：

```text
other mean 明显下降
TPOT 下降
concat_copy 不显著上升
hit_blk/miss_blk 仍保持可接受
```

### Step 2: 拆 profile

在 `_run_retrieval_full_gpu_cache_aware()` 开头 wait event 前后计时：

```text
prior_promotion_wait_ms
```

在 `_schedule_gpu_cache_promotion()` 内记录：

```text
promotion_submit_ms
promotion_blocks
```

成功标准：

```text
other 能解释到具体 promotion wait
```

### Step 3: 只 publish 完成的 promotion

如果上一轮 promotion 没完成，不等它，当前 token 使用旧 cache view。

这一步需要 versioning，不能直接删 wait。

最小安全版本：

```text
if promotion event not ready:
    skip consuming new cache state
    use last published hash/cache state
else:
    publish new state
```

如果 hash/cache state 没有双缓冲，这一步不安全。必须先保证 promotion 不会覆盖 active hash 指向的 cache slots。

### Step 4: fused gather + bounded cache fill

把 selected promotion subset 的 cache fill 合并进 gather kernel，避免重复读 miss list source。

成功标准：

```text
concat_copy + promotion_wait 总和下降
```

## 不要做的事

不要做以下修改：

1. 不要删掉 `wait_event(prior_event)` 来“消除 other”。这会制造 hash/cache 读写竞态。
2. 不要回到 `selected.to(cpu)` 和 CPU `batch_access()`。
3. 不要把 `cache_sync/scatter` 改名或挪字段来隐藏时间。
4. 不要降低 retrieval budget、压缩率、cache ratio 来换时间。
5. 不要让 promotion 读 `execution_buffer`。
6. 不要重新引入 dense mirror `[groups, n_centroids, buffer_size]`。
7. 不要让 plan kernel 的 cumsum 乱序。

## 当前结论

当前 GPU cache-aware retrieval 已经证明：

```text
GPU plan:      ~3.4 ms
concat_copy:   ~6.5 ms
attention:     ~3.3 ms
flat_search:   ~8.4 ms
```

这些已经接近目标。现在 `TPOT=71.54ms` 的主要原因不是 retrieval kernel 本身，而是：

```text
sparse_other ~= 29 ms
```

它与 miss/update blocks 强相关，说明 cache promotion 的后台工作仍然被下一次同 layer 等回来。

后续优化必须围绕：

```text
bounded promotion
promotion descriptors 与 current miss descriptors 分离
versioned/delayed publish
fused gather + cache fill
更细 profile
```

最终目标是让 `other` 接近 0，或者至少降到 `3-5ms` 的调度/dispatch 下限。当前日志显示，只要消掉这部分，整体 decode 才有机会稳定进入 `40-50ms` 区间。
