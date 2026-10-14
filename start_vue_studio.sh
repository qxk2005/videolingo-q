#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

PYTHON_BIN="python"
if [ -x "/Users/niuzhidao/.pyenv/versions/Project/bin/python" ]; then
  PYTHON_BIN="/Users/niuzhidao/.pyenv/versions/Project/bin/python"
fi

if ! "$PYTHON_BIN" -c "import fastapi, uvicorn, multipart, websockets" >/dev/null 2>&1; then
  echo "📦 Installing FastAPI Studio dependencies..."
  "$PYTHON_BIN" -m pip install "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" "python-multipart>=0.0.9" "websockets>=12.0"
fi

echo "🚀 Launching VideoLingo Q 2.0 Vue Studio on http://127.0.0.1:8502 ..."
exec "$PYTHON_BIN" backend/server.py
