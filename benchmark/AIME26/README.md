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

## Fixed 30-Problem Full vs AdaptiveIMI Run

Run the fixed comparable experiment with:

```bash
./benchmark/AIME26/run_full_vs_adaptiveimi_30x8.sh
```

It sequentially runs all 30 dataset problems with pass@8 for Full Flash
Attention Offload and AdaptiveIMI. The fixed AdaptiveIMI `retrieval_budget` and
`cache_ratio` are both `0.1`. Both methods use the same sampling settings and
a `65536` model context / `49152` generation safety cap, and stop naturally at
the model EOS token; the evaluator extracts the last valid integer answer.

## Replay an Existing Full-Attention Attempt Set

The checked-in Full Attention result currently contains **84 attempt records**
over 12 problem IDs; it is not 84 complete pass@8 problems. To make an
AdaptiveIMI comparison, replay the same `(id, attempt_index)` whitelist with
the original seed and per-attempt generation limits:

```bash
python benchmark/AIME26/continue_aime26_adaptiveimi.py \
  --seed_path benchmark/AIME26/results/adaptiveimi_pass@8_clean.jsonl \
  --patch_path benchmark/AIME26/adaptiveimi_target84_patch.jsonl \
  --merged_path benchmark/AIME26/results/adaptiveimi_pass@8_target84.jsonl \
  --target_path benchmark/AIME26/results/full_flash_attn_pass@8_clean.jsonl \
  --model_path /path/to/Qwen3-8B \
  --device cuda:0
```

Run the same command with `--dry_run` first. The tool reuses only existing
AdaptiveIMI attempts whose seed, sampling parameters, thinking mode, and
`max_model_len` / `max_new_tokens` match the Full Attention reference.
Length-limited outputs are retained as valid attempts rather than rerun with a
different seed or a larger generation limit.
