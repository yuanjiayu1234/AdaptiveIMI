# AdaptiveIMI: Retrieval-Guided Sparse Attention for Long-Context Inference

AdaptiveIMI is a research repository for experimenting with **retrieval-guided sparse attention** and **IMI-based KV-cache indexing** for long-context LLM inference.

TLDR; AdaptiveIMI combines Python model wrappers, a custom KV-cache pipeline, and C++/CUDA kernels to accelerate long-context decoding and evaluate the results on LongBench, InfiniteBench and RULER benchmarks.

## Installation

This repository has been tested with dependencies from `requirements.txt`, including:

```txt
torch==2.5.1
vllm==0.6.5
transformers==4.49.0
pybind11==2.12.0
```

Clone the repository or use your existing local checkout:

```bash
git clone <repo-url>
cd AdaptiveIMI
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Build notes

- `build.sh` tries to detect `CUDA_HOME` automatically.
- You can override target GPU architectures with `IMI_CUDA_ARCH_LIST` or `TORCH_CUDA_ARCH_LIST`.
- The build script prefers CUDA libraries from the active conda environment when available.

## Key Features

- **AdaptiveIMI KV cache** for retrieval-guided decoding over long contexts.
- **CPU/GPU cooperative indexing pipeline** with custom C++ and CUDA extensions.
- **Model wrappers for Llama, Qwen, and Mistral**.
- **Configurable retrieval budget and IMI partitioning** via `retrieval_budget`, `budget_ratio`, and `subspace_parts`.
- **Benchmark support** for LongBench, InfiniteBench and RULER evaluation.



## Quick Start

### 1. Build the extensions

```bash
cd library/AdaptiveIMI/cpp_extensions
python setup.py build_ext --inplace
cd ../../..
```

This step builds the C++/CUDA extensions used by AdaptiveIMI, including the IMI index manager, gather/scatter kernels and decode kernels.

### 2. Run a single LongBench task

Main wrapper:
- `benchmark/LongBench/pred.sh`
- Script usage: `bash pred.sh <model_name> <task_name> <attn_type> <dtype> <budget_ratio> [--subspace_parts <N>] [--fixed_budget]`

Example:

```bash
cd benchmark/LongBench
bash pred.sh llama-3-8b-1048k qasper AdaptiveIMI bf16 0.1 --subspace_parts 2 --fixed_budget
```

Parameter meaning:
- `llama-3-8b-1048k`: the model name or model alias to evaluate.
- `qasper`: the LongBench task name.
- `AdaptiveIMI`: the attention backend.
- `bf16`: the inference dtype.
- `0.1`: the retrieval budget / budget ratio.
- `--subspace_parts 2`: split the IMI index into 2 subspaces.
- `--fixed_budget`: keep the attended token budget fixed during decoding.

### 3. Run a RULER evaluation

Main wrapper:
- `benchmark/ruler/ruler_run.sh`
- Script usage: `bash ruler_run.sh <model_name> <benchmark_name> <attn_type> <context_length> <task> <dtype> <budget_ratio>`

Example:

```bash
cd benchmark/ruler
bash ruler_run.sh llama-3-8b-1048k niah AdaptiveIMI 131072 all bf16 0.1
```

Parameter meaning:
- `llama-3-8b-1048k`: the model name or model alias to evaluate.
- `niah`: the benchmark name.
- `AdaptiveIMI`: the attention backend.
- `131072`: the context length used for evaluation.
- `all`: run all tasks under the selected benchmark setting.
- `bf16`: the inference dtype.
- `0.1`: the retrieval budget / budget ratio.

