#!/usr/bin/env bash
# Default: two-pass collection — DEBUG_MODE=3 timings, then DEBUG_MODE=5 for
# gpu_memory_used_mb_max (see collect_algorithm_breakdown.py). Compiles happen inside Python.
# Use --single-pass for timing-only (faster; GPU memory row stays NA).
# Passes extra args to collect_algorithm_breakdown.py (e.g. --dry-run, -o path).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python3 "${SCRIPT_DIR}/collect_algorithm_breakdown.py" --structx-root "${STRUCTX_ROOT}" "$@"
