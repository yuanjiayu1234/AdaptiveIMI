# AIME26 Evaluation

This directory evaluates `math-ai/aime26` with Qwen3-8B using the same
problem-level pass@k convention as the ParisKV AIME runner.

Each problem generates `k` independent attempts. A problem passes when at least
one attempt produces the exact integer answer.

The dataset is stored at:

```text
benchmark/AIME26/data/aime2026.jsonl
```

## Pass@1

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=/path/to/Qwen3-8B \
./benchmark/AIME26/run_aime26.sh Full_Flash_Attn 1 0.0 0.1 1
```

## AdaptiveIMI Pass@1

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=/path/to/Qwen3-8B \
./benchmark/AIME26/run_aime26.sh AdaptiveIMI 1 0.0 0.1 1
```

The positional GPU argument and `CUDA_VISIBLE_DEVICES` should describe the
same physical GPU. Inside the process the selected GPU is `cuda:0`.

## Pass@8

```bash
CUDA_VISIBLE_DEVICES=1 \
MODEL_PATH=/path/to/Qwen3-8B \
./benchmark/AIME26/run_aime26.sh Full_Flash_Attn 8 0.7 0.1 1
```

Use the same `temperature`, `top_p`, `top_k`, seed policy, and generation
length for Full Attention and AdaptiveIMI.
