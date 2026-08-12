# Offline Partition-Learning Cost Summary

## Scope

This supplementary measurement profiles the offline learning of head-specific RoPE-pair partitions for **Llama-3.1-8B-Instruct**. It uses the same fixed mixed calibration subset as the partition-learning procedure, with no evaluation examples included. All reported timing statistics are aggregated across profiled layer-level runs and their layer-head optimization scopes.

Each profiling run optimizes all eight KV heads for one layer. The profile characterizes the per-layer and per-head offline cost. The full-model number below is an explicitly labeled projection, not a direct 32-layer measurement.

## Calibration Data and Configuration

| Item | Value |
|---|---:|
| Calibration documents | 11 |
| BookSum documents | 4 |
| NoLiMa documents | 4 |
| BABILong documents | 3 |
| Total calibration input characters | 5,231,886 |
| Maximum input characters in one document | 524,285 |
| Collected tokens per layer/head after prompt processing | 1,323,074 |
| Retained K vectors per layer/head | 131,072 |
| KV heads / head dimension | 8 / 128 |
| Teacher clusters per layer-head | 8,192 |
| Teacher KMeans iterations | 10 |
| Learned subspaces | 8 |
| Search starts | 32 random + 1 sequential |
| Search steps per start | 384 |
| Objective | constrained MSE, hard cell cap 96 |
| Collection mode | `LAYER_BATCH_SIZE=1`, `device=auto` |
| Hardware | 2 × NVIDIA GeForce RTX 4090 (24 GB each) |

Each saved layer K tensor has shape `[1, 8, 131072, 128]` and occupies 256 MiB in FP16. The layer-wise collection mode keeps only one such tensor at a time, then removes it after the partition for that layer is written.

## Measured Offline Cost

| Quantity | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
| Key collection time per profiled layer | 32.48 s | 32.68 s | 44.42 s | 45.73 s |
| Partition-search process time per profiled layer | 742.37 s | 740.57 s | 745.53 s | 746.09 s |
| Teacher construction time per profiled layer | 2.80 s | 2.80 s | 2.80 s | 2.80 s |
| Discrete partition search time per profiled layer | 738.50 s | 736.73 s | 741.63 s | 742.19 s |
| Final maximum cell size per profiled layer | 94.67 | 96 | 96 | 96 |

Across the 24 measured layer-head scopes:

| Quantity | Mean | Median | P95 |
|---|---:|---:|---:|
| Teacher construction time per head | 0.350 s | 0.318 s | 0.574 s |
| Partition search time per head | 92.31 s | 92.52 s | 95.45 s |
| End-to-end scope time per head | 92.70 s | 93.02 s | 95.80 s |
| Unique candidate partitions evaluated per head | 12,477 | 12,475 | 12,492 |
| Final normalized centroid MSE | 0.256 | 0.247 | 0.335 |

The teacher is inexpensive relative to partition search: it accounts for about 0.35 s per head, whereas the discrete partition search takes about 92.3 s per head under the final search budget.

## Peak Memory

| Stage | Peak CPU RSS | Peak GPU reserved |
|---|---:|---:|
| Key collection, maximum over profiling runs | 5.39 GiB | 10.36 GiB |
| Partition search, maximum over profiling runs | 1.26 GiB | 6.47 GiB |

The collection peak includes model loading and the per-layer FP16 K reservoir. The search peak includes the FP32 teacher/input tensors and chunked KMeans assignment workspace. Memory is reported as the maximum PyTorch reserved memory on a GPU and the maximum process RSS during the corresponding process.

## Convergence

Best-so-far traces were recorded every 128 newly evaluated candidate partitions. Across all profiled layer-head searches, the best objective reached within 1% of its final value after:

| Statistic | Candidate evaluations | Search time |
|---|---:|---:|
| Mean | 3,976 | 29.59 s |
| Median | 4,320 | 32.68 s |
| P95 | 8,771 | 66.20 s |

The final search budget evaluates approximately 12,477 unique candidates per head. All measured heads finish with zero hard-cap violations; their final maximum cell size is at most 96.

## Full-Model Projection (Not Directly Measured)

The search time is stable across profiled runs. Multiplying the measured mean search-process time by 32 Llama layers gives a projected **6.60 h** for full-model partition search under the same configuration.

Collection time grows with the target-layer depth because `device=auto` executes the model prefix up to the target layer. A linear fit to the collection measurements projects **2.01 h** for all 32 layer-wise collection passes. Combining these two components yields a planning estimate of **8.61 h** for the current `LAYER_BATCH_SIZE=1` implementation.

This 8.61 h figure should be described as an **estimate based on aggregated per-layer costs**, not as a full-model wall-clock measurement. The directly measured results above are sufficient for reporting per-layer/head cost, teacher cost, memory, and convergence. A direct full-model run is required if the paper claims an exact total preprocessing time.

## Source Artifacts

Raw profiles and traces are stored under:

`outputs/train_data/partitions/booksum_nolima_babilong_offline_profile_b1/tmp_layer_partitions/llama3_booksum_nolima_babilong_offline_profile_b1/`

The machine-readable aggregate is `offline_preprocessing_summary.csv`; raw per-run profiles and traces remain in the source artifact directory for reproducibility.
