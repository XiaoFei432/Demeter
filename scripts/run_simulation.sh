#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/demeter.yml}"
JOBS="${2:-64}"
MODE="${3:-normal}"
OUTPUT="${4:-results/simulation.csv}"

python -m MARL.cli simulate --config "$CONFIG" --jobs "$JOBS" --mode "$MODE" --output "$OUTPUT"
