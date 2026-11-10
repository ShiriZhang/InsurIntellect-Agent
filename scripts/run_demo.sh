#!/usr/bin/env bash
# Settings reads .env as data; inherited environment takes precedence.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/Scripts/python.exe"
if [[ ! -x "$PY" ]]; then PY="$ROOT/.venv/bin/python"; fi
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. See docs/runtime-and-migration.md" >&2
  exit 1
fi
exec "$PY" -m scripts.launch
