#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-python}"
CALIB_DIR="${CALIB_DIR:-outputs/imi_partition}"
CALIB_JSONLS="${CALIB_JSONLS:-}"
EXTERNAL_CALIB_JSONL="${EXTERNAL_CALIB_JSONL:-}"
EXTERNAL_CALIB_NAME="${EXTERNAL_CALIB_NAME:-}"
OUT_DIR="${OUT_DIR:-outputs/imi_partition}"
PARTITION_TMP_DIR="${PARTITION_TMP_DIR:-${OUT_DIR}/tmp_layer_partitions}"
MERGED_CALIB_NAME="${MERGED_CALIB_NAME:-all_domains_longbench_long}"
KEEP_KEYS="${KEEP_KEYS:-0}"
CALIB_ORDER="${CALIB_ORDER:-longest}"
CALIB_MIN_CHARS="${CALIB_MIN_CHARS:-20000}"
CALIB_REQUIRE_MAX_CHARS="${CALIB_REQUIRE_MAX_CHARS:-200000}"
AUTO_LONGBENCH_CALIB="${AUTO_LONGBENCH_CALIB:-1}"
LONGBENCH_JSON="${LONGBENCH_JSON:-/root/LongBench/data.json}"
LONGBENCH_CALIB_JSONL="${LONGBENCH_CALIB_JSONL:-${OUT_DIR}/calib_longbench_long.jsonl}"
LONGBENCH_CALIB_LIMIT="${LONGBENCH_CALIB_LIMIT:-8}"
LONGBENCH_CALIB_MIN_CHARS="${LONGBENCH_CALIB_MIN_CHARS:-200000}"
LONGBENCH_CALIB_MAX_CHARS="${LONGBENCH_CALIB_MAX_CHARS:-800000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE TRANSFORMERS_OFFLINE PYTORCH_CUDA_ALLOC_CONF

LLAMA3_MODEL="${LLAMA3_MODEL:-/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
QWEN3_MODEL="${QWEN3_MODEL:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
MODELS="${MODELS:-llama3,qwen3}"
MODEL_TAGS=()
MODEL_PATHS=()
for model_name in ${MODELS//,/ }; do
  case "$model_name" in
    llama3)
      MODEL_TAGS+=(llama3)
      MODEL_PATHS+=("$LLAMA3_MODEL")
      ;;
    qwen3)
      MODEL_TAGS+=(qwen3)
      MODEL_PATHS+=("$QWEN3_MODEL")
      ;;
    *)
      echo "[error] unknown model in MODELS=${MODELS}: ${model_name}" >&2
      echo "[hint] use MODELS=llama3,qwen3 or MODELS=llama3 or MODELS=qwen3" >&2
      exit 1
      ;;
  esac
done
if [[ "${#MODEL_TAGS[@]}" -eq 0 ]]; then
  echo "[error] MODELS did not select any model" >&2
  exit 1
fi

CALIB_PREVIEW_ROWS="${CALIB_PREVIEW_ROWS:-12}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
MAX_TOKENS_PER_PROMPT="${MAX_TOKENS_PER_PROMPT:-131072}"
STATIC_START="${STATIC_START:-32}"
STATIC_END="${STATIC_END:-64}"
DTYPE="${DTYPE:-bf16}"
DEVICE="${DEVICE:-cuda:0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
MLP_CHUNK_SIZE="${MLP_CHUNK_SIZE:-8192}"
LAYER_BATCH_SIZE="${LAYER_BATCH_SIZE:-1}"
ANSWER_PREFIX_FIELD="${ANSWER_PREFIX_FIELD:-answer_prefix}"
COLLECT_BATCH_SIZE="${COLLECT_BATCH_SIZE:-1}"

LAYER_GROUPS="${LAYER_GROUPS:-3}"
SCOPE="${SCOPE:-layer_head}"
M="${M:-8}"
SAMPLE_TOKENS="${SAMPLE_TOKENS:-131072}"
TARGET_CLUSTER_SIZE="${TARGET_CLUSTER_SIZE:-16}"
MAX_FULL_CLUSTERS="${MAX_FULL_CLUSTERS:-2048}"
KMEANS_ITERS="${KMEANS_ITERS:-6}"
STARTS="${STARTS:-2}"
STEPS="${STEPS:-8}"
RANDOM_SEED="${RANDOM_SEED:-0}"

calib_name_from_path() {
  local path="$1"
  local base="${path##*/}"
  base="${base%.jsonl}"
  base="${base#calib_}"
  base="${base//[^[:alnum:]_.-]/_}"
  echo "$base"
}

require_calibration_file() {
  local path="$1"
  local label="$2"
  if [[ ! -s "$path" ]]; then
    echo "[error] missing or empty calibration file for ${label}: ${path}" >&2
    echo "[hint] build one with: ${PYTHON} experiments/imi_partition/build_external_calib.py --out ${path} --limit-per-source 64 --min-chars 2000" >&2
    exit 1
  fi
}

run_step() {
  local output="$1"
  shift
  if [[ -s "$output" ]]; then
    echo "[skip] $output"
    return
  fi
  echo "[run] $*"
  "$@"
}

keep_intermediate_keys() {
  case "${KEEP_KEYS,,}" in
    1|true|yes|y|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

cleanup_keys() {
  local keys_path="$1"
  local learned_path="$2"
  if keep_intermediate_keys; then
    echo "[keep] $keys_path"
    return
  fi
  if [[ -s "$learned_path" && -f "$keys_path" ]]; then
    rm -f "$keys_path"
    echo "[cleanup] removed intermediate keys: $keys_path"
  fi
}

migrate_legacy_partition_outputs() {
  local model_tag="$1"
  local model_tmp_dir="$2"
  local prefix="${model_tag}_${MERGED_CALIB_NAME}"
  local src
  local dst_name
  local dst

  mkdir -p "$model_tmp_dir"
  shopt -s nullglob
  for src in \
    "$OUT_DIR"/"${prefix}"_centroid_partition_log.json \
    "$OUT_DIR"/"${prefix}"_layer*_centroid_partition.json \
    "$OUT_DIR"/"${prefix}"_layer*_centroid_partition_log.json \
    "$OUT_DIR"/"${prefix}"_layer*_keys.pt; do
    if [[ ! -e "$src" ]]; then
      continue
    fi
    dst_name="${src##*/}"
    dst_name="${dst_name#${prefix}_}"
    dst="${model_tmp_dir}/${dst_name}"
    if [[ -e "$dst" ]]; then
      echo "[migrate] tmp already has $dst; leaving $src"
      continue
    fi
    mv "$src" "$dst"
    echo "[migrate] $src -> $dst"
  done
  shopt -u nullglob
}

compact_json_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    return
  fi
  "$PYTHON" - "$path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
tmp_path = path.with_suffix(path.suffix + ".tmp")
tmp_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
tmp_path.replace(path)
print(f"[compact] {path}")
PY
}

migrate_final_partition_output() {
  local model_tag="$1"
  local final_path="$2"
  local legacy_path="${OUT_DIR}/${model_tag}_${MERGED_CALIB_NAME}_centroid_partition.json"

  if [[ -s "$final_path" || ! -s "$legacy_path" ]]; then
    return
  fi
  mv "$legacy_path" "$final_path"
  echo "[migrate] $legacy_path -> $final_path"
}

mkdir -p "$OUT_DIR" "$PARTITION_TMP_DIR"

if [[ -n "$CALIB_JSONLS" || -n "$EXTERNAL_CALIB_JSONL" ]]; then
  echo "[calib] explicit calibration source provided; skipping AUTO_LONGBENCH_CALIB"
else
  case "${AUTO_LONGBENCH_CALIB,,}" in
    1|true|yes|y|on)
    rebuild_longbench_calib=0
    if [[ ! -s "$LONGBENCH_CALIB_JSONL" ]]; then
      rebuild_longbench_calib=1
    elif ! "$PYTHON" - "$LONGBENCH_CALIB_JSONL" "$LONGBENCH_CALIB_LIMIT" "$LONGBENCH_CALIB_MIN_CHARS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
required_rows = int(sys.argv[2])
required_max = int(sys.argv[3])
rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(int(row.get("chars") or len(str(row.get("input") or ""))))
if len(rows) < required_rows or max(rows, default=0) < required_max:
    raise SystemExit(1)
print(f"[calib] long LongBench supplement exists: {path} rows={len(rows)} chars_max={max(rows)}")
PY
    then
      rebuild_longbench_calib=1
    fi

    if [[ "$rebuild_longbench_calib" -eq 1 ]]; then
      if [[ ! -s "$LONGBENCH_JSON" ]]; then
        echo "[error] AUTO_LONGBENCH_CALIB=1 but LongBench JSON is missing: $LONGBENCH_JSON" >&2
        exit 1
      fi
      echo "[calib] building long LongBench supplement -> $LONGBENCH_CALIB_JSONL"
      "$PYTHON" experiments/imi_partition/build_longbench_calib.py \
        --input "$LONGBENCH_JSON" \
        --out "$LONGBENCH_CALIB_JSONL" \
        --limit "$LONGBENCH_CALIB_LIMIT" \
        --min-chars "$LONGBENCH_CALIB_MIN_CHARS" \
        --max-chars "$LONGBENCH_CALIB_MAX_CHARS"
    fi
      ;;
  esac
fi

CALIB_PATHS=()
if [[ -n "$CALIB_JSONLS" ]]; then
  for path in ${CALIB_JSONLS//,/ }; do
    CALIB_PATHS+=("$path")
  done
elif [[ -n "$EXTERNAL_CALIB_JSONL" ]]; then
  CALIB_PATHS+=("$EXTERNAL_CALIB_JSONL")
  if [[ -n "$EXTERNAL_CALIB_NAME" ]]; then
    MERGED_CALIB_NAME="$EXTERNAL_CALIB_NAME"
  else
    MERGED_CALIB_NAME="$(calib_name_from_path "$EXTERNAL_CALIB_JSONL")"
  fi
else
  mapfile -t CALIB_PATHS < <(
    find "$CALIB_DIR" -maxdepth 1 -type f -name 'calib_*.jsonl' \
      ! -name "calib_${MERGED_CALIB_NAME}.jsonl" | sort
  )
fi

if [[ "${#CALIB_PATHS[@]}" -eq 0 ]]; then
  echo "[error] no calibration jsonl files found in ${CALIB_DIR}" >&2
  echo "[hint] expected files like ${CALIB_DIR}/calib_external_mix.jsonl" >&2
  exit 1
fi

NONEMPTY_CALIB_PATHS=()
for path in "${CALIB_PATHS[@]}"; do
  if [[ -s "$path" ]]; then
    NONEMPTY_CALIB_PATHS+=("$path")
  else
    echo "[skip] empty calibration file: ${path}" >&2
  fi
done
CALIB_PATHS=("${NONEMPTY_CALIB_PATHS[@]}")

if [[ "${#CALIB_PATHS[@]}" -eq 0 ]]; then
  echo "[error] no non-empty calibration jsonl files found in ${CALIB_DIR}" >&2
  echo "[hint] build one with: ${PYTHON} experiments/imi_partition/build_external_calib.py --out ${CALIB_DIR}/calib_external_mix.jsonl --limit-per-source 64 --min-chars 2000" >&2
  exit 1
fi

echo "[calib] ${#CALIB_PATHS[@]} source file(s)"
for path in "${CALIB_PATHS[@]}"; do
  echo "[calib] source -> ${path}"
  require_calibration_file "$path" "$(calib_name_from_path "$path")"
done

merged_calib="${OUT_DIR}/calib_${MERGED_CALIB_NAME}.jsonl"
echo "[calib] merging sources -> ${merged_calib}"
"$PYTHON" - "$merged_calib" "$RANDOM_SEED" "$CALIB_ORDER" "$CALIB_MIN_CHARS" "$CALIB_REQUIRE_MAX_CHARS" "${CALIB_PATHS[@]}" <<'PY'
import json
import random
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
seed = int(sys.argv[2])
order = sys.argv[3].strip().lower()
min_chars = max(int(sys.argv[4]), 0)
require_max_chars = max(int(sys.argv[5]), 0)
inputs = [Path(p) for p in sys.argv[6:]]
seen = set()
rows = []

for path in inputs:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row.get("source_dataset"),
                row.get("source_id"),
                row.get("input"),
            )
            marker = json.dumps(key, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            if min_chars > 0 and int(row.get("chars") or len(str(row.get("input") or ""))) < min_chars:
                continue
            seen.add(marker)
            rows.append(row)

if order in ("shuffle", "random"):
    random.Random(seed).shuffle(rows)
elif order in ("longest", "length_desc"):
    rows.sort(key=lambda row: int(row.get("chars") or len(str(row.get("input") or ""))), reverse=True)
elif order not in ("original", "none"):
    raise ValueError(f"unsupported CALIB_ORDER={order!r}; use shuffle, longest, or original")

if require_max_chars > 0:
    observed_max = max((int(row.get("chars") or len(str(row.get("input") or ""))) for row in rows), default=0)
    if observed_max < require_max_chars:
        raise SystemExit(
            f"[calib] max chars {observed_max} < required {require_max_chars}; "
            "build longer external calibration data or lower CALIB_REQUIRE_MAX_CHARS"
        )

out_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
with tmp_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
tmp_path.replace(out_path)
lengths = [int(row.get("chars") or len(str(row.get("input") or ""))) for row in rows]
if lengths:
    ordered = sorted(lengths)
    print(
        f"[calib] wrote {out_path} rows={len(rows)} order={order} min_chars={min_chars} "
        f"chars_min={ordered[0]} chars_p50={ordered[len(ordered)//2]} chars_max={ordered[-1]}"
    )
else:
    print(f"[calib] wrote {out_path} rows=0 order={order} min_chars={min_chars}")
PY
require_calibration_file "$merged_calib" "$MERGED_CALIB_NAME"

echo "[models] ${MODEL_TAGS[*]}"
echo "[config] MAX_LENGTH=$MAX_LENGTH MAX_TOKENS_PER_PROMPT=$MAX_TOKENS_PER_PROMPT SAMPLE_TOKENS=$SAMPLE_TOKENS"
echo "[config] LAYER_BATCH_SIZE=$LAYER_BATCH_SIZE MLP_CHUNK_SIZE=$MLP_CHUNK_SIZE ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "[config] ANSWER_PREFIX_FIELD=$ANSWER_PREFIX_FIELD COLLECT_BATCH_SIZE=$COLLECT_BATCH_SIZE"
echo "[config] CALIB_ORDER=$CALIB_ORDER CALIB_MIN_CHARS=$CALIB_MIN_CHARS CALIB_REQUIRE_MAX_CHARS=$CALIB_REQUIRE_MAX_CHARS"
echo "[config] PARTITION_TMP_DIR=$PARTITION_TMP_DIR"

"$PYTHON" - "$merged_calib" "$CALIB_PREVIEW_ROWS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
limit = int(sys.argv[2])
print(f"[calib] first {limit} training sample(s) after ordering/filtering:")
with path.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= limit:
            break
        row = json.loads(line)
        text = str(row.get("input") or "")
        chars = int(row.get("chars") or len(text))
        print(
            f"[calib] train[{idx}] chars={chars} "
            f"source={row.get('source_dataset')} id={row.get('source_id')}"
        )
PY

for model_idx in "${!MODEL_TAGS[@]}"; do
  model_tag="${MODEL_TAGS[$model_idx]}"
  model_path="${MODEL_PATHS[$model_idx]}"

  if [[ ! -e "$model_path" ]]; then
    echo "[error] missing model path for ${model_tag}: $model_path" >&2
    exit 1
  fi

  echo
  echo "#################### ${model_tag} ####################"
  echo "[model] $model_path"

  keys="${OUT_DIR}/${model_tag}_${MERGED_CALIB_NAME}_keys.pt"
  learned="${OUT_DIR}/${model_tag}_partition.json"
  model_tmp_dir="${PARTITION_TMP_DIR}/${model_tag}_${MERGED_CALIB_NAME}"
  mkdir -p "$model_tmp_dir"
  migrate_legacy_partition_outputs "$model_tag" "$model_tmp_dir"
  migrate_final_partition_output "$model_tag" "$learned"

  echo
  echo "========== ${model_tag} / ${MERGED_CALIB_NAME} =========="

  if [[ -s "$learned" ]]; then
    compact_json_file "$learned"
    echo "[skip] $learned"
    cleanup_keys "$keys" "$learned"
    continue
  fi

  if [[ "$LAYER_BATCH_SIZE" -gt 0 ]]; then
    num_layers="$("$PYTHON" - "$model_path" <<'PY'
import sys
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1])
print(int(cfg.num_hidden_layers))
PY
)"
    partial_jsons=()
    for ((layer_start=0; layer_start<num_layers; layer_start+=LAYER_BATCH_SIZE)); do
      layer_end=$((layer_start + LAYER_BATCH_SIZE - 1))
      if [[ "$layer_end" -ge "$num_layers" ]]; then
        layer_end=$((num_layers - 1))
      fi
      if [[ "$layer_start" -eq "$layer_end" ]]; then
        layers_arg="${layer_start}"
        layer_tag="layer${layer_start}"
      else
        layers_arg="${layer_start}-${layer_end}"
        layer_tag="layers${layer_start}-${layer_end}"
      fi

      layer_keys="${model_tmp_dir}/${layer_tag}_keys.pt"
      layer_learned="${model_tmp_dir}/${layer_tag}_centroid_partition.json"
      legacy_layer_keys="${OUT_DIR}/${model_tag}_${MERGED_CALIB_NAME}_${layer_tag}_keys.pt"
      legacy_layer_learned="${OUT_DIR}/${model_tag}_${MERGED_CALIB_NAME}_${layer_tag}_centroid_partition.json"
      partial_jsons+=("$layer_learned")

      if [[ ! -s "$layer_learned" && -s "$legacy_layer_learned" ]]; then
        cp "$legacy_layer_learned" "$layer_learned"
        echo "[migrate] $legacy_layer_learned -> $layer_learned"
      fi
      if [[ ! -s "$layer_keys" && -s "$legacy_layer_keys" ]]; then
        cp "$legacy_layer_keys" "$layer_keys"
        echo "[migrate] $legacy_layer_keys -> $layer_keys"
      fi

      if [[ -s "$layer_learned" ]]; then
        echo "[skip] $layer_learned"
        cleanup_keys "$layer_keys" "$layer_learned"
        continue
      fi

      run_step "$layer_keys" "$PYTHON" -m experiments.imi_partition.collect_llama3_rope_keys \
        --model "$model_path" \
        --jsonl "$merged_calib" \
        --text-field input \
        --answer-prefix-field "$ANSWER_PREFIX_FIELD" \
        --collect-batch-size "$COLLECT_BATCH_SIZE" \
        --limit 0 \
        --max-length "$MAX_LENGTH" \
        --max-tokens-per-prompt "$MAX_TOKENS_PER_PROMPT" \
        --sample-tokens "$SAMPLE_TOKENS" \
        --sample-seed "$RANDOM_SEED" \
        --layers "$layers_arg" \
        --static-start "$STATIC_START" \
        --static-end "$STATIC_END" \
        --dtype "$DTYPE" \
        --device "$DEVICE" \
        --attn-implementation "$ATTN_IMPLEMENTATION" \
        --mlp-chunk-size "$MLP_CHUNK_SIZE" \
        --out "$layer_keys"

      run_step "$layer_learned" "$PYTHON" -m experiments.imi_partition.search_centroid_partitions \
        --keys "$layer_keys" \
        --layout lhtd \
        --scope "$SCOPE" \
        --layer-groups "$LAYER_GROUPS" \
        --layers "$layers_arg" \
        --m "$M" \
        --sample-tokens "$SAMPLE_TOKENS" \
        --target-cluster-size "$TARGET_CLUSTER_SIZE" \
        --max-full-clusters "$MAX_FULL_CLUSTERS" \
        --kmeans-iters "$KMEANS_ITERS" \
        --starts "$STARTS" \
        --steps "$STEPS" \
        --random-seed "$RANDOM_SEED" \
        --device "$DEVICE" \
        --out-json "$layer_learned" \
        --compact-json

      cleanup_keys "$layer_keys" "$layer_learned"
    done

    echo "[merge] layer partitions -> $learned"
    "$PYTHON" - "$learned" "$model_tmp_dir" "${partial_jsons[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_json = Path(sys.argv[1])
tmp_dir = Path(sys.argv[2])
inputs = [Path(p) for p in sys.argv[3:]]
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
if not payloads:
    raise SystemExit("[merge] no partial partitions")

base = payloads[0]
layer_groups = []
log_rows = []
for payload in payloads:
    layer_groups.extend(payload.get("layer_groups", []))
    log_rows.extend(payload.get("metadata", {}).get("log_rows", []))

layer_groups.sort(key=lambda group: (int(group.get("layer_start", 0)), int(group.get("layer_end", 0))))
log_rows.sort(key=lambda row: (int(row.get("layer", -1)), int(row.get("head", -1)), str(row.get("scope", ""))))

merged = dict(base)
metadata = dict(base.get("metadata", {}))
metadata["source_keys"] = "layerwise_intermediate_keys"
metadata["layerwise_partitions_dir"] = str(tmp_dir)
metadata["log_rows"] = log_rows
merged["metadata"] = metadata
merged["layer_groups"] = layer_groups

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(merged, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
print(f"[merge] wrote {out_json} layer_groups={len(layer_groups)} log_rows={len(log_rows)}")
PY
    continue
  fi

  run_step "$keys" "$PYTHON" -m experiments.imi_partition.collect_llama3_rope_keys \
    --model "$model_path" \
    --jsonl "$merged_calib" \
    --text-field input \
    --answer-prefix-field "$ANSWER_PREFIX_FIELD" \
    --collect-batch-size "$COLLECT_BATCH_SIZE" \
    --limit 0 \
    --max-length "$MAX_LENGTH" \
    --max-tokens-per-prompt "$MAX_TOKENS_PER_PROMPT" \
    --sample-tokens "$SAMPLE_TOKENS" \
    --sample-seed "$RANDOM_SEED" \
    --static-start "$STATIC_START" \
    --static-end "$STATIC_END" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --attn-implementation "$ATTN_IMPLEMENTATION" \
    --mlp-chunk-size "$MLP_CHUNK_SIZE" \
    --out "$keys"

  run_step "$learned" "$PYTHON" -m experiments.imi_partition.search_centroid_partitions \
    --keys "$keys" \
    --layout lhtd \
    --scope "$SCOPE" \
    --layer-groups "$LAYER_GROUPS" \
    --m "$M" \
    --sample-tokens "$SAMPLE_TOKENS" \
    --target-cluster-size "$TARGET_CLUSTER_SIZE" \
    --max-full-clusters "$MAX_FULL_CLUSTERS" \
    --kmeans-iters "$KMEANS_ITERS" \
    --starts "$STARTS" \
    --steps "$STEPS" \
    --random-seed "$RANDOM_SEED" \
    --device "$DEVICE" \
    --out-json "$learned" \
    --compact-json

  cleanup_keys "$keys" "$learned"
done

echo
echo "[done] centroid-structural partition experiments finished"
