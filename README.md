# AdaptiveIMI: Retrieval-Guided Sparse Attention for Long-Context Inference

AdaptiveIMI is a research repository for experimenting with **retrieval-guided sparse attention** and **IMI-based KV-cache indexing** for long-context LLM inference.

TLDR; AdaptiveIMI combines Python model wrappers, a custom KV-cache pipeline, and C++/CUDA kernels to accelerate long-context decoding and evaluate the results on LongBench, InfiniteBench and RULER benchmarks.

## Installation

Our python environment management is based on miniconda3. After installing
miniconda3, execute commands below to setup.

```bash
cd /root/AdaptiveIMI
source /home/vipuser/miniconda3/etc/profile.d/conda.sh

conda create -n adaptiveimi python=3.10 pip -y
conda activate adaptiveimi

pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

aria2c -c -x 8 -s 8 --dir=/tmp \
  --out=flashinfer_python-0.2.5+cu124torch2.5-cp38-abi3-linux_x86_64.whl \
  "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5%2Bcu124torch2.5-cp38-abi3-linux_x86_64.whl"
pip install --no-deps /tmp/flashinfer_python-0.2.5+cu124torch2.5-cp38-abi3-linux_x86_64.whl

pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
git clone --depth 1 --branch v3.4.1 https://github.com/NVIDIA/cutlass.git library/cutlass

aria2c -c -x 4 -s 4 --dir=/tmp --out=flash-attention-weighted.tar.gz \
  "https://codeload.github.com/Starmys/flash-attention/tar.gz/refs/heads/weighted"
rm -rf /tmp/flash-attention-weighted
tar -xzf /tmp/flash-attention-weighted.tar.gz -C /tmp
rmdir /tmp/flash-attention-weighted/csrc/cutlass
ln -s /root/AdaptiveIMI/library/cutlass /tmp/flash-attention-weighted/csrc/cutlass
pip install /tmp/flash-attention-weighted --no-build-isolation

cd /root/AdaptiveIMI/library/AdaptiveIMI/cpp_extensions
python setup.py build_ext --inplace
```

Current implementation requires CUDA 12.4 and a GPU with compute capability
at least 8.0. On this machine, use `/home/vipuser/miniconda3`; the
`/root/anaconda3/envs/adaptiveimi` environment is broken.
