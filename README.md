# AdaptiveIMI: Retrieval-Guided Sparse Attention for Long-Context Inference

AdaptiveIMI is a research repository for experimenting with **retrieval-guided sparse attention** and **IMI-based KV-cache indexing** for long-context LLM inference.

TLDR; AdaptiveIMI combines Python model wrappers, a custom KV-cache pipeline, and C++/CUDA kernels to accelerate long-context decoding and evaluate the results on LongBench, InfiniteBench and RULER benchmarks.

## Installation

The recommended setup is a CPython 3.10 conda environment with CUDA 12.4.
Do not use GraalPy/GraalVM Python: PyTorch, FlashInfer, FlashAttention and the
local CUDA extensions require CPython binary wheels or CPython extension builds.

Clone the repository or use an existing checkout:

```bash
git clone <repo-url>
cd AdaptiveIMI
```

Create and activate the conda environment:

```bash
conda create -n adaptiveimi -c conda-forge "python=3.10.*=*cpython*" pip -y
conda activate adaptiveimi
```

Install CUDA, build tools and Python packages:

```bash
conda install -y -c nvidia cuda-toolkit=12.4
conda install -y -c conda-forge git cmake ninja gcc_linux-64=11 gxx_linux-64=11

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt

cd /root/AdaptiveIMI/library/AdaptiveIMI/cpp_extensions
python setup.py build_ext --inplace
```
PyTorch packages are installed from the PyTorch CUDA 12.4 wheel index above and
are intentionally not listed in `requirements.txt`. This avoids accidentally
installing a CPU-only or mismatched PyTorch build when installing the remaining
Python dependencies. Do not install `torch` from the default PyPI or a mirror
index for this project; use the CUDA 12.4 index command above.

Verify the core dependency set before running experiments:

```bash
python - <<'PY'
import torch
import numpy
import transformers
import tokenizers
import filelock
import sympy

print("torch:", torch.__version__, torch.version.cuda, torch.cuda.is_available())
print("numpy:", numpy.__version__, numpy.__file__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("filelock:", filelock.__version__)
print("sympy:", sympy.__version__)
PY
```

Expected key values:

```text
torch: 2.5.1+cu124 ... True
numpy: 1.26.4 ...
transformers: 4.57.1
tokenizers: 0.22.1
filelock: 3.16.1
sympy: 1.13.1
```

Install FlashInfer for CUDA 12.4 and PyTorch 2.5:

```bash
pip install flashinfer-python \
  -i https://flashinfer.ai/whl/cu124/torch2.5/ \
  --no-deps
```

Install the weighted FlashAttention kernel used by `weighted_flash_decoding`:

```bash
MAX_JOBS=8 pip install \
  "git+https://github.com/Starmys/flash-attention.git@weighted" \
  --no-build-isolation
```

Install CUTLASS headers required by the local CUDA extension build:

```bash
git clone --depth 1 --branch v3.4.1 https://github.com/NVIDIA/cutlass.git library/cutlass
```

Optional import checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import flashinfer; print(flashinfer.__version__)"
python -c "from weighted_flash_decoding import weighted_flash_decoding; print('weighted_flash_decoding ok')"
```

### Repairing A Broken Existing Environment

If an existing `adaptiveimi` environment has been mixed with CPU PyTorch wheels
or partially overwritten packages, import errors may appear before project code
runs. Common symptoms include:

- `undefined symbol: iJIT_NotifyEvent`
- `Torch not compiled with CUDA enabled`
- `module 'numpy' has no attribute 'ndarray'`
- `No module named 'idna'`
- `cannot import name 'BaseFileLock' from 'filelock'`
- `tokenizers>=0.22.0,<=0.23.0 is required`
- `No module named 'sympy'`

The fastest reliable fix is to clean the broken package directories and reinstall
the pinned versions:

```bash
source /root/anaconda3/etc/profile.d/conda.sh
conda activate adaptiveimi

pip uninstall -y torch torchvision torchaudio numpy tokenizers filelock sympy mpmath
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/torch"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/torchvision"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/torchaudio"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/numpy"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/tokenizers"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/filelock"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/idna"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/jinja2"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/Jinja2"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/pillow"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/Pillow"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/sympy"*
rm -rf "$CONDA_PREFIX/lib/python3.10/site-packages/mpmath"*

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install --no-cache-dir --force-reinstall -r requirements.txt
```

If only the offline `experiments/imi_partition` scripts are needed, `torchvision`
is not used by the experiment itself. However, installing the matching
`torchvision==0.20.1` CUDA wheel keeps packages such as `vllm==0.6.5` satisfied
and prevents mismatched `torchvision::nms` import errors.

For Qwen3 calibration collection, `transformers==4.57.1` and
`tokenizers==0.22.1` are pinned because Qwen3 is not recognized by older
Transformers versions and newer Tokenizers releases may violate Transformers'
version bounds.

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
