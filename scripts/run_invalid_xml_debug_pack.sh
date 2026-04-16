#!/usr/bin/env bash
# Compile src_sort, then run invalid_xml files and check stderr for Invalid XML.
# Writes dataset/xml_debug_pack/reports/invalid_run_*.log and invalid_results_*.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if command -v nvcc >/dev/null 2>&1; then
  NVCC=(nvcc)
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=(/usr/local/cuda/bin/nvcc)
else
  echo "nvcc not found in PATH or /usr/local/cuda/bin/nvcc" >&2
  exit 2
fi

"${NVCC[@]}" -DDEBUG_MODE=0 -O3 -o output.exe ./src_sort/main.cu -w -gencode=arch=compute_61,code=sm_61

exec python3 "$SCRIPT_DIR/invalid_pack_runner.py" --repo-root "$REPO_ROOT" --exe "$REPO_ROOT/output.exe" "$@"
