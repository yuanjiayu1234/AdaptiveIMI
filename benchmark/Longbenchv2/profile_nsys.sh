#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NSYS="${NSYS:-/home/yjy/tools/nsight-systems-2026.4.1/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64/nsys}"

TAG="${TAG:-prefill_profile}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  TAG="$1"
  shift
fi
EXTRA_BENCH_ARGS=("$@")

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/prof/longbenchv2_verify}"
RUN_DIR="${OUT_ROOT}/${TAG}"
REP_PREFIX="${RUN_DIR}/${TAG}"
REP="${REP_PREFIX}.nsys-rep"
DB="${REP_PREFIX}.sqlite"
SUMMARY_DIR="${RUN_DIR}/summary"
RESULT_JSONL="${RUN_DIR}/${TAG}.jsonl"
mkdir -p "${SUMMARY_DIR}"
if [[ "${PROFILE_RESUME:-0}" != "1" ]]; then
  rm -f "${RESULT_JSONL}"
fi

export MODEL_FAMILY="${MODEL_FAMILY:-llama3}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export CONFIG_MODEL_NAME="${CONFIG_MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
export DATA_PATH="${DATA_PATH:-/home/yjy/LongBench/data.json}"
export OUT_PATH="${OUT_PATH:-${RESULT_JSONL}}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export ATTN_TYPE="${ATTN_TYPE:-AdaptiveIMI}"
export DTYPE="${DTYPE:-bf16}"
export DEVICE="${DEVICE:-auto}"
export IMI_ENGINE="${IMI_ENGINE:-hybrid_8way_v2}"
export HYBRID_POLICY="${HYBRID_POLICY:-gpu_only}"
export HYBRID_START="${HYBRID_START:-gpu}"

echo "[profile] tag=${TAG}"
echo "[profile] rep=${REP}"
echo "[profile] out=${OUT_PATH}"

"${NSYS}" profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=200 \
  --force-overwrite=true \
  --output "${REP_PREFIX}" \
  bash "${SCRIPT_DIR}/run.sh" "${RUN_DIR}" \
    --max_samples "${MAX_SAMPLES:-1}" \
    --save_context_chars "${SAVE_CONTEXT_CHARS:-0}" \
    "${EXTRA_BENCH_ARGS[@]}"

"${NSYS}" stats --force-export=true --timeunit=msec -r cuda_api_sum "${REP}" \
  | tee "${SUMMARY_DIR}/cuda_api_sum.txt"
"${NSYS}" stats --force-export=true --timeunit=msec -r cuda_gpu_kern_sum "${REP}" \
  | tee "${SUMMARY_DIR}/cuda_gpu_kern_sum.txt"
"${NSYS}" stats --force-export=true --timeunit=msec -r cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum "${REP}" \
  | tee "${SUMMARY_DIR}/cuda_gpu_mem.txt"
"${NSYS}" stats --force-export=true --timeunit=msec -r osrt_sum "${REP}" \
  | tee "${SUMMARY_DIR}/osrt_sum.txt"

sqlite3 "${DB}" <<'SQL' | tee "${SUMMARY_DIR}/nvtx_focus.tsv"
.headers on
.mode tabs
WITH nvtx AS (
  SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id=textId)) AS name
  FROM NVTX_EVENTS
  WHERE end IS NOT NULL
)
SELECT name,
       count(*) AS calls,
       round(sum(end-start)/1000000.0, 3) AS total_ms,
       round(avg(end-start)/1000000.0, 3) AS avg_ms,
       round(max(end-start)/1000000.0, 3) AS max_ms
FROM nvtx
WHERE name LIKE 'imi.prefill.L%.raw_d2h.%'
   OR name LIKE 'imi.prefill.L%.materialize_decode_metadata.%'
   OR name LIKE 'imi.prefill.L%.finish.reorder_raw_kv'
   OR name LIKE 'imi.prefill.L%.finish.ensure_list_storage'
   OR name LIKE 'imi.prefill.L%.load_metadata.install_gpu'
   OR name LIKE 'imi.prefill.L%.load_metadata.install_decode_metadata'
GROUP BY name
ORDER BY total_ms DESC, name;
SQL

sqlite3 "${DB}" <<'SQL' | tee "${SUMMARY_DIR}/nvtx_phase_rollup.tsv"
.headers on
.mode tabs
WITH nvtx AS (
  SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id=textId)) AS name
  FROM NVTX_EVENTS
  WHERE end IS NOT NULL
)
SELECT
  CASE
    WHEN name LIKE 'imi.prefill.L%.raw_d2h.%' THEN 'raw_d2h'
    WHEN name LIKE 'imi.prefill.L%.materialize_decode_metadata.%' THEN 'decode_metadata'
    WHEN name LIKE 'imi.prefill.L%.finish.reorder_raw_kv' THEN 'reorder_raw_kv'
    WHEN name LIKE 'imi.prefill.L%.finish.ensure_list_storage' THEN 'ensure_list_storage'
    WHEN name LIKE 'imi.prefill.L%.load_metadata.install_gpu' THEN 'install_gpu'
    WHEN name LIKE 'imi.prefill.L%.load_metadata.install_decode_metadata' THEN 'install_decode_metadata'
    ELSE 'other'
  END AS phase,
  count(*) AS calls,
  round(sum(end-start)/1000000.0, 3) AS total_ms,
  round(avg(end-start)/1000000.0, 3) AS avg_ms,
  round(max(end-start)/1000000.0, 3) AS max_ms
FROM nvtx
WHERE name LIKE 'imi.prefill.L%.raw_d2h.%'
   OR name LIKE 'imi.prefill.L%.materialize_decode_metadata.%'
   OR name LIKE 'imi.prefill.L%.finish.reorder_raw_kv'
   OR name LIKE 'imi.prefill.L%.finish.ensure_list_storage'
   OR name LIKE 'imi.prefill.L%.load_metadata.install_gpu'
   OR name LIKE 'imi.prefill.L%.load_metadata.install_decode_metadata'
GROUP BY phase
ORDER BY total_ms DESC, phase;
SQL

sqlite3 "${DB}" <<'SQL' | tee "${SUMMARY_DIR}/raw_d2h_cuda_api_by_range.tsv"
.headers on
.mode tabs
WITH nvtx0 AS (
  SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id=textId)) AS name
  FROM NVTX_EVENTS
  WHERE end IS NOT NULL
), nvtx AS (
  SELECT * FROM nvtx0
  WHERE name LIKE 'imi.prefill.L%.raw_d2h.copy_keys'
     OR name LIKE 'imi.prefill.L%.raw_d2h.copy_values'
     OR name LIKE 'imi.prefill.L%.raw_d2h.enqueue_copy'
)
SELECT nvtx.name AS range_name,
       s.value AS cuda_api,
       count(*) AS calls,
       round(sum(r.end-r.start)/1000000.0, 3) AS total_ms,
       round(avg(r.end-r.start)/1000000.0, 3) AS avg_ms,
       round(max(r.end-r.start)/1000000.0, 3) AS max_ms
FROM nvtx
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.start >= nvtx.start AND r.end <= nvtx.end
JOIN StringIds s ON r.nameId=s.id
GROUP BY nvtx.name, s.value
ORDER BY total_ms DESC, range_name
LIMIT 200;
SQL

sqlite3 "${DB}" <<'SQL' | tee "${SUMMARY_DIR}/raw_d2h_memcpy_by_range.tsv"
.headers on
.mode tabs
WITH nvtx0 AS (
  SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id=textId)) AS name
  FROM NVTX_EVENTS
  WHERE end IS NOT NULL
), nvtx AS (
  SELECT * FROM nvtx0
  WHERE name LIKE 'imi.prefill.L%.raw_d2h.copy_keys'
     OR name LIKE 'imi.prefill.L%.raw_d2h.copy_values'
     OR name LIKE 'imi.prefill.L%.raw_d2h.enqueue_copy'
)
SELECT nvtx.name AS range_name,
       ck.label AS copy_kind,
       sk.label AS src_kind,
       dk.label AS dst_kind,
       count(*) AS copies,
       round(sum(m.bytes)/1000000.0, 3) AS total_mb,
       round(sum(m.end-m.start)/1000000.0, 3) AS total_ms,
       round(max(m.end-m.start)/1000000.0, 3) AS max_ms
FROM nvtx
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.start >= nvtx.start AND r.end <= nvtx.end
JOIN CUPTI_ACTIVITY_KIND_MEMCPY m
  ON m.correlationId = r.correlationId
JOIN ENUM_CUDA_MEMCPY_OPER ck ON m.copyKind=ck.id
JOIN ENUM_CUDA_MEM_KIND sk ON m.srcKind=sk.id
JOIN ENUM_CUDA_MEM_KIND dk ON m.dstKind=dk.id
GROUP BY nvtx.name, ck.label, sk.label, dk.label
ORDER BY total_ms DESC, range_name
LIMIT 200;
SQL

echo "[profile] done"
echo "[profile] summary dir: ${SUMMARY_DIR}"
