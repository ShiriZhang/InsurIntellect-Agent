#!/usr/bin/env bash
# Force demo-safe embed settings (shell env can pollute .env).
# Usage: bash scripts/run_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Clash proxy often breaks local HF offline loads; demo uses cached bge.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy || true

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8766}"
export DEBUG="${DEBUG:-true}"
export SIMPLE_RAG_MODE="${SIMPLE_RAG_MODE:-true}"
export ENABLE_QUERY_REWRITING="${ENABLE_QUERY_REWRITING:-false}"
export ENABLE_QUERY_ROUTING="${ENABLE_QUERY_ROUTING:-false}"
export CHROMA_ANONYMIZED_TELEMETRY=false
export CHROMA_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OPENAI_EMBEDDING_MODEL="${OPENAI_EMBEDDING_MODEL:-hf:BAAI/bge-small-zh-v1.5}"
export SIMILARITY_THRESHOLD="${SIMILARITY_THRESHOLD:-0.32}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Load local .env if present, then RE-FORCE embed (shell/.env pollution guard)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export OPENAI_EMBEDDING_MODEL="hf:BAAI/bge-small-zh-v1.5"
export SIMILARITY_THRESHOLD="0.32"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PY="$ROOT/.venv/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then
  PY="$ROOT/.venv/bin/python"
fi
if [[ ! -x "$PY" ]]; then
  echo "Missing venv python. Run: uv venv .venv && uv pip install -r requirements.txt" >&2
  exit 1
fi

echo "Demo: http://${HOST}:${PORT}/"
echo "Embed: $OPENAI_EMBEDDING_MODEL thr=$SIMILARITY_THRESHOLD offline=$HF_HUB_OFFLINE"
echo "Ingestion/profile rebuild: see docs/document-ingestion.md; --reset is no longer supported."
exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$PORT"
