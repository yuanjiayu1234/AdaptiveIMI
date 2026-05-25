# Extreme-Performance CUDA / LLM Inference Engineering Guidelines

---

# Core Philosophy

Modern LLM inference systems are usually bottlenecked by:

- HBM memory bandwidth
- Cache locality
- Kernel launch overhead
- CPU-GPU synchronization
- PCIe / NVLink transfer
- Memory fragmentation
- Random memory access

NOT by raw FLOPs.

Therefore the optimization priority is:

```text
Reduce memory movement
    >
Improve memory locality
    >
Reduce synchronization
    >
Fuse kernels
    >
Improve arithmetic efficiency
```

The most important principle:

```text
Keep data on GPU.
Move data as little as possible.
Read memory sequentially.
Launch as few kernels as possible.
```

---

# 1. Eliminate Python from Hot Paths

## NEVER use Python loops inside:

- decoding loop
- attention computation
- KV retrieval
- ANN traversal
- top-k selection
- quantization/dequantization
- routing logic
- cache management

Bad:

```python
for token in tokens:
    scores.append(q @ k[token])
```

Good:

```python
scores = torch.matmul(q, k.T)
```

Python loops cause:

- kernel launch explosion
- GIL overhead
- CPU scheduling latency
- GPU starvation

Even tiny Python loops become catastrophic in long-context decoding.

---

# 2. Prefer CUDA Operators

Always prefer:

```python
torch.matmul
torch.bmm
torch.topk
torch.gather
torch.scatter
torch.index_select
flash_attn
fused kernels
```

Avoid:

- Python-side indexing
- scalar tensor operations
- repeated `.item()`
- CPU-side filtering

---

# 3. Avoid CPU-GPU Synchronization

Bad:

```python
if scores.max().item() > threshold:
```

This forces GPU synchronization.

Also avoid:

```python
tensor.cpu()
tensor.numpy()
print(cuda_tensor)
```

Good principle:

```text
GPU -> GPU -> GPU
```

Avoid:

```text
GPU -> CPU -> GPU
```

---

# 4. Memory Access Dominates Compute

A single HBM access can cost more than dozens of FLOPs.

Therefore:

```text
Optimize memory before compute.
```

Modern LLM inference is usually:

```text
Memory-bound > Compute-bound
```

---

# 5. Tensor Layout Is Critical

Tensor layout determines:

- coalesced access
- L2 hit rate
- cache reuse
- vectorized loads
- TLB efficiency

Prefer layouts optimized for access pattern.

Typical KV layout:

```text
[layer][head][token][dim]
```

Paged KV layout:

```text
[layer][page][head][token][dim]
```

Avoid layouts requiring:

- repeated transpose
- non-contiguous gather
- stride-heavy access

---
# 6. Preserve Contiguity

Always prefer contiguous tensors.

Check:

```python
tensor.is_contiguous()
```

Avoid:

```python
x = x.permute(...).contiguous()
```

inside decoding loops.

Avoid repeated:

- transpose
- reshape
- layout conversion
- tensor copies

Design the storage format once and keep it stable.

---

# 7. Avoid Hidden Tensor Copies

These operations may trigger copies:

```python
.contiguous()
.clone()
.float()
.half()
.cuda()
.cpu()
.permute()
```

Hidden copies destroy bandwidth efficiency.

---

# 8. Coalesced Memory Access

Good GPU access pattern:

```text
Thread i accesses address i
```

Bad:

```text
Thread i accesses random address
```

Random access causes:

- DRAM inefficiency
- TLB misses
- L2 miss amplification
- warp stalls

---

# 9. Minimize Gather/Scatter

Bad:

```python
x = kv_cache[random_indices]
```

Prefer:

- block retrieval
- chunk retrieval
- sorted indices
- grouped access

ANN systems easily become random-access machines if poorly designed.

---

# 10. Optimize for L2 Cache

L2 reuse is extremely important.

Strategies:

- centroid caching
- shared-memory staging
- query batching
- grouped retrieval
- block reuse

Bad:

```text
Repeatedly read same centroid from HBM
```

Good:

```text
Load once into shared memory.
Reuse across threads.
```

---

# 11. Shared Memory Optimization

Use shared memory for:

- centroids
- LUTs
- partial reductions
- local top-k buffers
- attention tiles

Avoid:

- bank conflicts
- oversized buffers
- unaligned access

Shared memory is vastly faster than HBM.

---

# 12. Avoid Warp Divergence

Bad:

```python
if thread_id % 2 == 0:
```

Different execution paths inside a warp reduce throughput.

Prefer:

- branch-free kernels
- predication
- tensor masking

---

# 13. Reduce Kernel Launch Count

Bad:

```text
100 tiny kernels per token
```

Good:

```text
Few large fused kernels
```

Kernel launch overhead becomes massive during decoding.

---

# 14. Fuse Everything Possible

Fuse:

- matmul + softmax
- dequantize + attention
- quantize + pack
- score + top-k
- routing + gather
- RoPE + attention
- RMSNorm + linear

Every unfused stage:

- writes HBM
- rereads HBM
- launches another kernel

---

# 15. Attention Is Memory-Bound

Attention optimization should prioritize:

- KV bandwidth
- cache reuse
- reduced reads

NOT FLOPs.

Key principle:

```text
Avoid reading unnecessary KV.
```

---

# 16. KV Cache Engineering

KV cache is the real bottleneck.

Goals:

- contiguous storage
- aligned loads
- low fragmentation
- block allocation
- page reuse

Avoid:

- token-wise allocation
- fragmented storage
- scattered retrieval

---

# 17. Use Paged KV Allocation

Avoid:

```text
malloc/free per token
```

Prefer:

- slab allocator
- page allocator
- preallocated memory pools

Benefits:

- less fragmentation
- stable addresses
- CUDA graph compatibility

---

# 18. Avoid Tiny GPU Allocations

Tiny allocations destroy performance.

Prefer:

- arena allocators
- static workspace buffers
- persistent memory pools

---

# 19. Use CUDA Graphs

CUDA graphs reduce:

- launch overhead
- CPU scheduling latency
- runtime jitter

Requirements:

- static shapes
- stable addresses
- fixed execution graph

---

# 20. Dynamic Shapes Are Expensive

Dynamic shapes reduce:

- fusion opportunities
- graph capture
- compiler optimization

Prefer:

- padded batches
- fixed page size
- fixed block size
- static retrieval chunk

---

# 21. Batch Everything

Never process:

- one token
- one head
- one query

individually if batching is possible.

Prefer:

```python
[B, H, T, D]
```

operations.

Avoid:

```python
for h in range(H):
```

Batching improves:

- occupancy
- throughput
- kernel efficiency

---

# 22. ANN Retrieval Optimization

ANN retrieval becomes dangerous when it causes:

- random memory access
- excessive candidate expansion
- pointer chasing
- heap-heavy top-k
- CPU-side traversal

---