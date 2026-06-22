# AdaptiveIMI Decode Follow-up Notes

Date: 2026-06-17

This note records the current state after the recent decode/cache-copy changes, plus the next low-risk analysis and cleanup items. It is intentionally kept in the project root instead of `docs/` or `todo/`, because those directories may contain legacy notes.

## 2026-06-18 CPU staging pack 复测结论

### 本次对比命令

Baseline：不启用 CPU staging，走原始 miss copy 路径。

```bash
cd /root/AdaptiveIMI
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH_OVERRIDE=/root/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
NUM_SAMPLES=3 BATCH_SIZE=1 RULER_RESUME=0 \
IMI_PROFILE_DECODE=1 IMI_PROFILE_STEPS=5 \
IMI_PROFILE_GATHER_DETAIL=1 IMI_PROFILE_CACHE_DETAIL=1 \
bash /root/AdaptiveIMI/benchmark/ruler/ruler_run.sh qwen3-8b synthetic AdaptiveIMI 98304 niah_single_1 bf16 \
> /root/AdaptiveIMI/test_topk_admit1.log 2>&1
```

CPU staging：启用 `IMI_MISS_COPY_STAGING=1`，先在 CPU pinned staging buffer 中 pack，再用大块 `cudaMemcpy2DAsync` 拷到 GPU。

```bash
cd /root/AdaptiveIMI
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH_OVERRIDE=/root/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
NUM_SAMPLES=3 BATCH_SIZE=1 RULER_RESUME=0 \
IMI_MISS_COPY_STAGING=1 \
IMI_PROFILE_DECODE=1 IMI_PROFILE_STEPS=5 \
IMI_PROFILE_GATHER_DETAIL=1 IMI_PROFILE_CACHE_DETAIL=1 \
bash /root/AdaptiveIMI/benchmark/ruler/ruler_run.sh qwen3-8b synthetic AdaptiveIMI 98304 niah_single_1 bf16 \
> /root/AdaptiveIMI/test_staging.log 2>&1
```

### 正确性

`/root/AdaptiveIMI/test_staging.log` 当前正确性已经恢复：

- `Score 100.0`
- `Nulls 0/3`

这说明这次 CPU staging 路径没有再复现之前“输出乱码/全是 !”的问题。

之前容易导致错误结果的风险点已经修复：

- 不能用最后一个 run descriptor 推导 miss 区域长度，因为 run descriptor 不是按目标 cumsum 排序生成的。当前用 C++ 并行扫描 `max(buffer_offset + size)` 计算 staging 需要的真实宽度。
- 2D copy 使用所有 group 的全局最大宽度；短 group 的 staging 尾部如果未初始化，可能把垃圾拷到 execution buffer。当前 pack 前会清零本次 staging copy window，避免未初始化尾部污染结果。
- 旧的半成品 `pack_miss_chunks_to_pinned` 已经不再从 Python 导出，也不再 pybind 暴露，staging 路径只保留 `compute_miss_staging_lengths` 和 `pack_miss_chunks_to_pinned_and_copy` 两个 C++ 入口。

### 性能结果

Baseline `/root/AdaptiveIMI/test_topk_admit1.log`：

- sample 2: `Decoding latency 58.73 ms/step`
- sample 3: `Decoding latency 58.81 ms/step`
- steady miss copy: 大约 `64 ms`
- steady `miss_bw`: 大约 `14 GiB/s`

CPU staging `/root/AdaptiveIMI/test_staging.log`：

- sample 2: `Decoding latency 135.57 ms/step`
- sample 3: `Decoding latency 135.99 ms/step`
- steady miss copy: 大约 `188-236 ms`
- `stage_cpu`: 大约 `178-225 ms`
- `stage_wait`: 只有大约 `0.1 ms`
- `Score 100.0`

结论：当前 CPU staging 方向在这版实现上明显变慢，主要不是等待 staging ring，也不是 attention，而是 CPU 端 pack 本身太慢。

### 为什么慢

当前 CPU staging 做了两段搬运：

1. CPU pinned `list_keys/list_values` 随机 run copy 到 CPU pinned staging buffer。
2. CPU pinned staging buffer 再用 `cudaMemcpy2DAsync` 拷到 GPU execution buffer。

原始 baseline 虽然是大量小 run，但 GPU kernel 直接从 pinned host list KV 读到 GPU execution buffer，避免了 CPU 先 pack 一遍。CPU staging 额外引入了：

- CPU 侧随机源地址读取；
- CPU 侧 staging 写入；
- 为 correctness 清零 staging window；
- 第二段 H2D 传输；
- CPU cache/TLB 压力；
- 大量小 run 的 `memcpy` 调用/调度成本。

从日志看，`stage_cpu` 几乎等于 `miss_copy`：

- sample 3 avg: `miss_copy=188.338ms`，`stage_cpu=178.393ms`
- sample 2 avg: `miss_copy=235.929ms`，`stage_cpu=225.196ms`

所以瓶颈非常明确：不是 H2D 大块 copy 没跑起来，而是 CPU pack 在 H2D 之前已经把时间花掉了。

这里的 `miss_bw` 在 staging 模式下不应该再当作真实 PCIe 带宽解释，因为它用 miss pages 估算字节数，却把 CPU pack 时间也算进了 `miss_copy`。它只能说明整体 miss-copy 阶段吞吐很低，不能说明 PCIe 本身只有 `3-5 GiB/s`。

### 对 CPU pack 方向的判断

CPU pack 这个思路不是算法上错误；它解决的是“小随机 H2D copy 带宽低”的问题。但是当前形态不适合直接替换 baseline：

- 如果 pack 在 CPU 上做，它必须足够快，否则会把 GPU copy 瓶颈换成 CPU 随机内存瓶颈。
- 当前 workload 的 run 数仍然非常多：每步平均 `7.2-7.5 万` 个 miss runs。
- coalescing 只有大约 `3.1-3.3 pages/run`，说明 run-level 合并有效但不够大，CPU 需要处理的 run 数仍然太多。
- baseline 的 GPU kernel 虽然带宽只有约 `14 GiB/s`，但它利用了 GPU 并行度，实际仍比 CPU pack 快很多。

因此当前版本不应默认开启 `IMI_MISS_COPY_STAGING=1`。它只能作为实验开关保留。

### 下一步建议

优先方向不是继续堆 Python/C++ CPU memcpy，而是减少 CPU pack 的根本成本：

1. 把 staged 路径拆分 profile：
   - descriptor D2H 时间；
   - CPU length scan 时间；
   - CPU memset 时间；
   - CPU pack memcpy 时间；
   - H2D 2D copy 时间。

2. 如果继续 CPU pack，必须降低 run 数或降低 CPU memcpy 调用开销：
   - 按 group/run 长度分桶，避免按全局最大宽度清零和 2D copy；
   - 对很短 run 使用更粗粒度 page/block copy 策略，减少 `memcpy` 调用数量；
   - 评估是否可以按 cluster/page order 重新组织 pinned list layout，让 CPU 源地址更连续；
   - 使用固定线程池/NUMA 亲和，避免 `at::parallel_for` 调度和跨 NUMA 带来的抖动。

3. 更值得优先尝试的是 GPU 侧 batch gather 改造：
   - 保留 pinned host list KV，不把完整 KV 放 GPU；
   - 让 GPU kernel 继续直接读取 pinned host；
   - 但改 kernel 的 work partition、向量化、run 合并和访问顺序，提升 PCIe read 利用率；
   - 避免 CPU 先复制一遍。

4. staged CPU pack 的保留条件：
   - correctness 必须继续 `Score 100.0 / Nulls 0/3`；
   - `stage_cpu + H2D` 必须低于 baseline `miss_copy`；
   - 如果 `stage_cpu` 仍然超过 `50ms`，这条路径就不值得作为主线。

## 2026-06-17 Current State 旧记录

- Latest Qwen correctness log: `/root/AdaptiveIMI/test_topk_admit1.log`.
- Correctness has been restored:
  - `Score 100.0`
  - `Nulls 0/3`
  - The three expected magic numbers were generated correctly.
- Stable decode performance in the latest Qwen log is roughly:
  - sample 2: total about `50.703 ms/step`, gather about `43.329 ms`, miss copy about `35.069 ms`, cpu lookup about `7.030 ms`.
  - sample 3: total about `50.407 ms/step`, gather about `42.985 ms`, miss copy about `34.916 ms`, cpu lookup about `6.843 ms`.
- The first sample still contains cold/warmup effects and is not the best signal for steady-state analysis.

## Recently Fixed

- Foreground miss copy now uses run-level descriptors generated during cache-aware retrieval planning.
- Page-level miss descriptors are still kept for cache promotion and hit layout.
- The important correctness fix was to keep execution-buffer layout based on page-level miss destination offsets:
  - `miss_copy_cumsum` must follow page-level `miss_dst`.
  - Hit copy must use page-level miss descriptors when computing final output layout.
- The non-profile path was fixed to use the same static/miss/hit copy sequence as the profile path.
- A parameter-order bug in the non-profile miss-copy path was fixed.
- A targeted smoke test passed for layout:
  - expected layout: `[static][miss page 0][miss page 2][hit page 1]`
  - verified output: `layout_ok`.

Relevant files:

- `library/AdaptiveIMI/cpp_extensions/cuda/gather_copy.cu`
- `cache_hub/adaptive_imi/core.py`
- `cache_hub/adaptive_imi/runtime.py`
- `cache_hub/adaptive_imi/indexing.py`
- `cache_hub/adaptive_imi/retrieval.py`

## Main Performance Bottleneck

The current decode bottleneck is still foreground gather, especially miss copy.

- `miss_copy` is about `35 ms` in the stable Qwen samples.
- This is roughly 69% of the profiled total decode step time.
- `cpu_lookup` is now about `6.8-7.0 ms`, which is not the largest cost but is large enough to matter after miss copy is reduced.
- The run-level miss-copy change did not yet show a clear large speedup, so we should measure whether it is actually helping before adding more complexity.

Before further tuning, add diagnostics for:

- `miss_pages`
- `miss_copy_runs`
- coalescing ratio: `miss_pages / miss_copy_runs`
- foreground `miss_copy_ms`
- `cpu_lookup_ms`

If `miss_copy_runs` is close to `miss_pages`, run-level descriptors provide little benefit and may only increase planning overhead. If the ratio is much smaller, the idea is valid but the copy path still needs optimization.

## Logic Risks To Check Carefully

### 1. Promotion and retrieval metadata race

There is a potential race between background GPU cache promotion and the next foreground retrieval.

Observed structure:

- `sparse_attention()` starts retrieval and reads cache metadata.
- `_schedule_gpu_cache_promotion()` checks the previous promotion event near the end of the layer flow.
- If a previous promotion is still updating hash/cache metadata while current retrieval reads the same metadata, retrieval may observe partially updated state.

Relevant locations:

- `cache_hub/adaptive_imi/retrieval.py`
  - `sparse_attention()`
  - `_schedule_gpu_cache_promotion()`
- `library/AdaptiveIMI/cpp_extensions/cuda/gather_copy.cu`
  - cache-aware retrieval planning
  - promotion planning/finalization/copy kernels

Do not blindly add a wait unless correctness requires it, because waiting may directly hurt decode latency.

Safer next analysis:

- Add a debug counter at retrieval start: `promotion_inflight_at_retrieval`.
- If this counter is frequently nonzero, the current async promotion design may be racing with foreground retrieval.
- Then choose a clear consistency design:
  - conservative mode: wait or skip use of in-flight cache state.
  - better mode: expose only committed cache metadata to retrieval, using versioned or double-buffered metadata.

### 2. Large-cluster descriptor path

The cache-aware plan kernel has different behavior based on cluster size:

- small clusters can use local cached slot state.
- larger clusters may require additional hash lookups.

For clusters larger than the local fast path, concurrent promotion can make repeated hash lookups observe changed state. This overlaps with the promotion/retrieval metadata race above.

Need diagnostic:

- count clusters whose page count is greater than the local fast-path threshold.
- check whether those clusters are common in 96k/128k workloads.

### 3. Hot-Cold hit update duplicate work

The Hot-Cold policy should be preserved as a core algorithmic idea:

- miss inserts into cold queue.
- cold hit promotes to hot queue.
- hot hit refreshes position.
- hot overflow demotes to cold.
- cold tail eviction happens when capacity is exceeded.

One implementation issue to re-check:

- `last_hit_cluster` only skips consecutive duplicate clusters.
- Hit descriptors are allocated with atomics, so the same cluster may appear multiple times non-consecutively.
- This is mostly redundant metadata work, but it can still waste promotion bandwidth.

Need to keep O(1) queue operations and avoid turning cache maintenance into a heavy foreground task.

## Code Cleanup Items

### 1. Legacy fused copy wrapper

`gather_copy_and_concat_chunk_parallel` is effectively legacy now.

Reason:

- The current path needs separate page-level miss descriptors and run-level miss-copy descriptors.
- The old fused wrapper cannot safely express this split.
- Leaving it exported makes accidental future misuse more likely.

Places to clean:

- `library/AdaptiveIMI/cpp_extensions/cuda/gather_copy.cu`
- `library/AdaptiveIMI/cpp_extensions/__init__.py`
- build/import smoke checks that still reference this function.

### 2. Misleading function names

These functions are now used in both profile and non-profile paths:

- `gather_copy_static_chunks_profile`
- `gather_copy_miss_chunks_profile`
- `gather_copy_hit_chunks_profile`

Their names are misleading. Rename or wrap them with neutral names after the current correctness/performance path is stable.

### 3. Profiling fields that always print zero

Some cache profile fields are currently unfilled or no longer meaningful on the current GPU-cache-aware path.

Examples:

- `evict_blk`
- `max_upd_blk`
- `max_evict_blk`
- `free_blocks_before`
- `free_blocks_after`

These do not affect hot-path performance much, but they make logs harder to reason about. Either remove them from printing or implement the stats correctly.

### 4. Duplicate profile and non-profile copy sequence

Profile and non-profile paths now both perform static, miss, and hit copy explicitly. This duplication is acceptable temporarily, but after the logic is stable we should factor it into a small helper to avoid future parameter-order bugs.

## Suggested Next Order

1. Add diagnostics only:
   - `miss_pages`
   - `miss_copy_runs`
   - coalescing ratio
   - `promotion_inflight_at_retrieval`
   - optional large-cluster count

2. Use those diagnostics to decide whether run-level miss copy is worth keeping:
   - keep it if coalescing ratio is high and planning overhead is acceptable.
   - remove or simplify it if it mostly adds `cpu_lookup` overhead.

3. Fix promotion/retrieval metadata consistency with a clear design:
   - avoid blind synchronization if possible.
   - retrieval should see only committed cache entries.
   - background promotion should remain background and should not corrupt or stall the foreground path.

4. Clean legacy exports and misleading names:
   - remove old fused wrapper export.
   - rename profile-named copy wrappers.
   - remove or repair zero-only profile fields.

5. Continue bottleneck work:
   - reduce miss volume.
   - improve cache hit rate.
   - keep Hot-Cold queue operations O(1).
   - make foreground miss copy and background cache admission match the algorithm, instead of patching around old execution-buffer assumptions.
