#!/usr/bin/env bash
set -euo pipefail

export MODEL="${MODEL:-/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
export PARTITION="${PARTITION:-outputs/imi_partition/llama3_partition.json}"
export OUTDIR="${OUTDIR:-outputs/imi_partition/recall_llama3_longbench_r100}"
export METHODS="${METHODS:-learned,sequential,pair_sequential,full_kmeans}"

bash "$(dirname "$0")/run_qwen3_recall.sh"
