#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITELLM_BASE_URL="http://localhost:8088/v1"
export LITELLM_API_KEY="sk-anything"
export MODEL="llama-glimmer"
python3 "$SCRIPT_DIR/main.py" --workspace "$HOME" "$@"
