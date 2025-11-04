#!/usr/bin/env bash

set -euo pipefail

cd .. || exit 1
PYTHON=$(command -v python || command -v python3)
if [ -z "$PYTHON" ]; then
  echo "python not found" >&2
  exit 2
fi

echo "Starting benchmark..."
$PYTHON -m benchmarks.benchmark_cylinder
exit $?
