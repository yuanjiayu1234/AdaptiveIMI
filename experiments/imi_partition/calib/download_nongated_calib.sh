#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

OUT_DIR="${OUT_DIR:-outputs/imi_partition}"
PYTHON="${PYTHON:-python3}"
LIMIT_PER_SOURCE="${LIMIT_PER_SOURCE:-64}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-300}"
export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT

mkdir -p "$OUT_DIR"

run_source() {
  local source_name="$1"
  local out_name="$2"
  local min_chars="$3"
  shift 3

  local out_path="${OUT_DIR}/${out_name}"
  if [[ -s "$out_path" ]]; then
    echo "[skip] ${out_path}"
    return
  fi

  echo
  echo "========== ${source_name} -> ${out_path} =========="
  if ! "$PYTHON" experiments/imi_partition/calib/build_external_calib.py \
      --out "$out_path" \
      --sources "$source_name" \
      --limit-per-source "$LIMIT_PER_SOURCE" \
      --min-chars "$min_chars" \
      "$@"; then
    echo "[warn] source failed: ${source_name}" >&2
  fi
}

# Already downloaded sources are skipped automatically if the files exist.
run_source longalign calib_external_mix.jsonl 2000
run_source wikisql calib_external_mix_v2.jsonl 200 --trust-remote-code
run_source wikitablequestions calib_wikitablequestions.jsonl 200 --trust-remote-code

# Non-gated sources that may need retrying depending on network/datasets version.
run_source longalpaca calib_longalpaca.jsonl 1000
run_source scrolls_gov_report calib_scrolls_gov_report.jsonl 2000 --trust-remote-code
run_source scrolls_contract_nli calib_scrolls_contract_nli.jsonl 1000 --trust-remote-code
run_source scrolls_summ_screen calib_scrolls_summ_screen.jsonl 2000 --trust-remote-code
run_source multi_news calib_multi_news.jsonl 2000 --trust-remote-code
run_source codesearchnet calib_codesearchnet.jsonl 500 --trust-remote-code

"$PYTHON" - <<'PY'
import json
from collections import Counter
from pathlib import Path

out_dir = Path("outputs/imi_partition")
inputs = [
    out_dir / "calib_external_mix.jsonl",
    out_dir / "calib_external_mix_v2.jsonl",
    out_dir / "calib_wikitablequestions.jsonl",
    out_dir / "calib_longalpaca.jsonl",
    out_dir / "calib_scrolls_gov_report.jsonl",
    out_dir / "calib_scrolls_contract_nli.jsonl",
    out_dir / "calib_scrolls_summ_screen.jsonl",
    out_dir / "calib_multi_news.jsonl",
    out_dir / "calib_codesearchnet.jsonl",
]
out_path = out_dir / "calib_external_nongated_final.jsonl"
seen = set()
rows = []
for path in inputs:
    if not path.exists() or path.stat().st_size == 0:
        print(f"[merge] skip {path}")
        continue
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("source_dataset"), row.get("source_id"), row.get("input")
            marker = json.dumps(key, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            rows.append(row)

with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[merge] wrote {out_path} rows={len(rows)}")
print("[merge] sources", dict(Counter(row.get("source_dataset") for row in rows)))
PY

echo
echo "[done] non-gated calibration download finished"
