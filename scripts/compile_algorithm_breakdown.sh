#!/usr/bin/env bash
# Build StructX for algorithm breakdown scripts. Binary: StructX/output.exe
#
# DEBUG_MODE (default 3): timing breakdown + validation ms lines.
#   DEBUG_MODE=5 enables GPU memory usage prints (Used = ... MB); use with two-pass collector.
# Override GPU arch (default sm_61):
#   SM=75 ./compile_algorithm_breakdown.sh
# Or full gencode:
#   NVCC_GENCODE="arch=compute_75,code=sm_75" ./compile_algorithm_breakdown.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${STRUCTX_ROOT}"

SM="${SM:-61}"
if [[ -n "${NVCC_GENCODE:-}" ]]; then
  GENCODE_FLAG="-gencode=${NVCC_GENCODE}"
else
  GENCODE_FLAG="-gencode=arch=compute_${SM},code=sm_${SM}"
fi

DEBUG_MODE="${DEBUG_MODE:-3}"

echo "StructX root: ${STRUCTX_ROOT}"
echo "DEBUG_MODE=${DEBUG_MODE}"
echo "nvcc ${GENCODE_FLAG}"

nvcc -DDEBUG_MODE="${DEBUG_MODE}" -O3 -o output.exe ./src/main.cu -w "${GENCODE_FLAG}"

echo "Built: ${STRUCTX_ROOT}/output.exe"
