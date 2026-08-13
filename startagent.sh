#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITELLM_BASE_URL="http://localhost:4000/v1"
export LITELLM_API_KEY="sk-master-change-me"
export MODEL="mistral-large"
python3 "$SCRIPT_DIR/main.py" --workspace "$HOME" "$@"
