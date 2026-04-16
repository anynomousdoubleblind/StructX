#!/usr/bin/env bash
# Same as run_algorithm_breakdown.sh (two-pass is the default since 2026).
# Kept as a stable alias for scripts that referenced this name.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRUCTX_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec python3 "${SCRIPT_DIR}/collect_algorithm_breakdown.py" --structx-root "${STRUCTX_ROOT}" "$@"
