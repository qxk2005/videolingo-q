#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PYTHON_BIN="python3"
if [ -x "/Users/niuzhidao/.pyenv/versions/Project/bin/python" ]; then
  PYTHON_BIN="/Users/niuzhidao/.pyenv/versions/Project/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" backend/mcp_server.py "$@"
